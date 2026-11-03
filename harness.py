"""One robot, one episode: observe -> policy -> validate -> execute -> observe.

The policy is injected (e.g. an existing Kimi client). Physical action semantics,
reset and native success belong to the benchmark adapter. No automatic physical
retry, controller, planner, parallel scheduler, or simulator dependency lives here.
"""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from tcp_overlay import CameraCalibration, RobotGeometry, render_tcp_overlay
from visual_query import QUERY_NOTICE, QUERY_STAGES, query


@dataclass
class Observation:
    external_rgb: Image.Image
    wrist_rgb: Image.Image
    wrist_camera: CameraCalibration
    robot: RobotGeometry
    state: dict  # Measured orientation, gripper width, previous execution, etc.
    query_views: dict = field(default_factory=dict)  # Optional synchronized QueryView per camera.


class Benchmark(Protocol):
    def observe(self) -> Observation: ...
    def validate_action(self, action: dict, observation: Observation) -> None:
        """Reject with ValueError before any physics; verify schema, units and bounds."""
        ...
    def execute(self, action: dict) -> dict:
        """Execute exactly once and return measured feedback; do not retry on failure."""
        ...
    def success(self) -> bool:
        """Original benchmark success, not a model prediction."""
        ...


def text(value):
    return {"type": "text", "text": value if isinstance(value, str) else json.dumps(value)}


def image_part(path):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def parse_action(content):
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    action = json.loads(content)
    if not isinstance(action, dict):
        raise ValueError("one JSON object required")
    return action


def run(benchmark: Benchmark, policy, *, instruction, action_prompt, output_dir,
        max_calls=120, max_actions=40, keep_observations=8, max_queries_per_observation=4):
    """Policy(messages) returns a complete chat-completions response.

    Caller resets the benchmark and closes it afterwards. Use a fresh output
    directory; never silently resume an episode and repeat an uncertain action.
    Policy must enforce its own request timeout. This deliberately small loop
    has call/action bounds, not the original batch's wall-time/tick/memory guards.
    """
    if any(type(n) is not int or n <= 0 for n in
           (max_calls, max_actions, keep_observations, max_queries_per_observation)):
        raise ValueError("budgets must be positive integers")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    messages = [{"role": "system", "content": action_prompt + "\n" + QUERY_NOTICE}]
    calls = actions = rejected = queries = local_queries = 0
    observation_messages = []
    observation = None

    def event(kind, **data):
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": kind, **data}, allow_nan=False) + "\n")

    def finish(reason):
        result = dict(reason=reason, success=bool(benchmark.success()), calls=calls,
                      actions=actions, rejections=rejected, queries=queries)
        (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    try:
        while True:
            if benchmark.success():
                return finish("native_success")
            if calls >= max_calls or actions >= max_actions:
                return finish("call_budget" if calls >= max_calls else "action_budget")
            if observation is None:
                local_queries = 0
                observation = benchmark.observe()
                overlay, metadata = render_tcp_overlay(
                    observation.wrist_rgb, observation.wrist_camera, observation.robot
                )
                folder = out / f"observation-{actions:03d}"
                folder.mkdir()
                external_path, wrist_path = folder / "external.png", folder / "wrist_overlay.png"
                observation.external_rgb.save(external_path)
                observation.wrist_rgb.save(folder / "wrist_raw.png")
                overlay.save(wrist_path)
                (folder / "overlay.json").write_text(json.dumps(metadata, indent=2) + "\n")
                context = dict(task=instruction, observation_id=observation.robot.observation_id,
                               reference_frame=observation.robot.reference_frame,
                               measured_state=observation.state, overlay_notice=metadata["notice"])
                messages.append({"role": "user", "content": [
                    text("External RGB"), image_part(external_path),
                    text("Wrist RGB with measured TCP overlay"),
                    image_part(wrist_path), text(context),
                ]})
                observation_messages.append([len(messages) - 1])
                # Evict image bytes only. Text and returned reasoning remain.
                for group in observation_messages[:-keep_observations]:
                    for index in group:
                        messages[index]["content"] = [
                            text("[Older image omitted; text retained.]")
                            if part["type"] == "image_url" else part
                            for part in messages[index]["content"]
                        ]
                event("observation", context=context, folder=str(folder.name))

            response = policy(copy.deepcopy(messages))
            calls += 1
            event("response", response=response)
            choice = response["choices"][0]
            message = choice["message"]
            messages.append({"role": "assistant", **{
                key: message[key] for key in ("content", "reasoning_content") if key in message
            }})
            try:
                if choice["finish_reason"] != "stop":
                    raise ValueError("incomplete generation")
                action = parse_action(message.get("content") or "")
                if action.get("observation_id") != observation.robot.observation_id:
                    raise ValueError("stale or missing observation_id")
                if action.get("stage") == "finish":
                    if set(action) != {"stage", "observation_id"}:
                        raise ValueError("finish requires exactly stage and observation_id")
                    return finish("model_finish")
                if action.get("stage") in QUERY_STAGES:
                    if local_queries >= max_queries_per_observation:
                        raise ValueError("query budget for this observation exhausted")
                    annotated, query_feedback = query(action, observation)
                else:
                    benchmark.validate_action(action, observation)
            except (ValueError, TypeError, KeyError) as error:
                rejected += 1
                event("rejection", error=str(error), executed=False)
                messages.append({"role": "user", "content": [text("Rejected: " + str(error))]})
                continue
            if action.get("stage") in QUERY_STAGES:
                queries += 1
                local_queries += 1
                query_path = folder / f"query-{local_queries:02d}.png"
                annotated.save(query_path)
                event("visual_query", request=action, feedback=query_feedback,
                      image=str(query_path.relative_to(out)), executed_ticks=0)
                messages.append({"role": "user", "content": [
                    text(query_feedback), image_part(query_path),
                ]})
                # Associate all query images with the same observation for eviction.
                observation_messages[-1].append(len(messages) - 1)
                continue
            # Deliberately outside the validation handler. A physical failure is
            # never converted into a zero-execution validation error or retried.
            event("action_started", action=action)
            feedback = benchmark.execute(action)
            actions += 1
            event("execution", feedback=feedback)
            messages.append({"role": "user", "content": [text({"execution": feedback})]})
            observation = None
    except Exception:
        # Avoid copying exception strings that might contain network credentials.
        event("infrastructure_error", actions_completed=actions)
        raise
