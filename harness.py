"""One robot, one episode: observe -> policy -> validate -> execute -> observe.

The policy is injected (e.g. an existing Kimi client). Physical action semantics,
reset and native success belong to the benchmark adapter. No automatic physical
retry, controller, planner, parallel scheduler, or simulator dependency lives here.
"""

from __future__ import annotations

import base64
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from rollout_context import bind_observation_examples
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
    """Adapters may expose a cached `completion` receipt for the current physical state.

    Use a pending receipt before native evidence exists and None (or omit the
    attribute) to disable the tool. Reading a receipt must not step the simulator.
    """

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


def run(
    benchmark: Benchmark,
    policy,
    *,
    instruction,
    action_prompt,
    output_dir,
    max_calls=120,
    max_actions=40,
    keep_observations=8,
    max_queries_per_observation=4,
    context_compactor=None,
    bind_observation_ids=False,
    native_finish_feedback=False,
    hierarchy=None,
    wall_time_s=None,
    budget=None,
    progress_access="default",
    initial_observation=None,
    attempt_context=(),
):
    """Policy(messages) returns a complete chat-completions response.

    Caller resets the benchmark and closes it afterwards. Use a fresh output
    directory; never silently resume an episode and repeat an uncertain action.
    Policy must enforce its own request timeout. This deliberately small loop
    has call/action bounds, not the original batch's wall-time/tick/memory guards.
    Optional context_compactor changes inference history only, never the audit.
    In native_finish_feedback mode a premature model finish receives measured
    failure feedback and continues within the same call/action budgets.
    """
    if any(
        type(n) is not int or n <= 0
        for n in (max_calls, max_actions, keep_observations, max_queries_per_observation)
    ):
        raise ValueError("budgets must be positive integers")
    if context_compactor is not None and context_compactor.task != instruction:
        raise ValueError("compactor task must match the original instruction")
    if hierarchy is not None and hierarchy.calls:
        raise ValueError("create a fresh hierarchy per episode")
    if wall_time_s is not None and wall_time_s <= 0:
        raise ValueError("wall_time_s must be positive")
    deadline = time.monotonic() + wall_time_s if wall_time_s is not None else float("inf")

    def out_of_time():
        if budget:
            return budget.clock() >= budget.lower_deadline()
        return time.monotonic() >= deadline

    transition = trigger = None
    executing = False
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    messages = [{"role": "system", "content": action_prompt + "\n" + QUERY_NOTICE}]
    if progress_access not in {"default", "on_request", "hidden"}:
        raise ValueError("unknown progress access mode")
    query_available = (
        progress_access != "hidden" and getattr(benchmark, "completion", None) is not None
    )
    if query_available:
        messages[0]["content"] += (
            "\nOptional task progress tool: to see which goal conditions the environment "
            "last observed as satisfied, send "
            '{"stage":"check_task_completion","observation_id":"CURRENT"}. '
            "Pending or null means unknown. Choose your next action from the current "
            "observation. Reading the same observation returns the same evidence, advances "
            "zero ticks, and shares the visual query limit."
        )
    calls = actions = rejected = queries = completion_queries = local_queries = 0
    observation_messages = []
    observation = None

    def event(kind, **data):
        with (out / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"event": kind, "time_unix": time.time(), **data}, allow_nan=False)
                + "\n"
            )

    def finish(reason, detail=None):
        result = dict(
            reason=reason,
            success=bool(benchmark.success()),
            calls=calls,
            actions=actions,
            rejections=rejected,
            queries=queries,
            completion_queries=completion_queries,
        )
        if hierarchy is not None:
            result["upper_calls"] = hierarchy.calls
        if budget:
            result.update(budget.metrics())
        if detail:
            result["detail"] = detail
        (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    try:
        while True:
            if benchmark.success():
                if hierarchy is not None and actions and observation is None:
                    final = benchmark.observe()
                    terminal = out / "terminal"
                    terminal.mkdir()
                    final.external_rgb.save(terminal / "external.png")
                    final.wrist_rgb.save(terminal / "wrist_raw.png")
                    event(
                        "terminal_observation",
                        folder="terminal",
                        observation_id=final.robot.observation_id,
                        measured_state=final.state,
                    )
                return finish("native_success")
            if out_of_time():
                return finish("wall_time_budget")
            if calls >= max_calls or actions >= max_actions:
                return finish("call_budget" if calls >= max_calls else "action_budget")
            if observation is None:
                local_queries = 0
                observation = (
                    initial_observation if initial_observation is not None else benchmark.observe()
                )
                initial_observation = None
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
                context = dict(
                    task=instruction,
                    observation_id=observation.robot.observation_id,
                    reference_frame=observation.robot.reference_frame,
                    measured_state=observation.state,
                    overlay_notice=metadata["notice"],
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            text("External RGB"),
                            image_part(external_path),
                            text("Wrist RGB with measured TCP overlay"),
                            image_part(wrist_path),
                            text(context),
                        ],
                    }
                )
                observation_messages.append([len(messages) - 1])
                # Evict image bytes only. Text and returned reasoning remain.
                for group in observation_messages[:-keep_observations]:
                    for index in group:
                        messages[index]["content"] = [
                            text("[Older image omitted; text retained.]")
                            if part["type"] == "image_url"
                            else part
                            for part in messages[index]["content"]
                        ]
                event("observation", context=context, folder=str(folder.name))
                completion = getattr(benchmark, "completion", None)
                if completion is not None:
                    context = dict(context, completion=completion)
                    if not actions:
                        event("completion_evidence", receipt=completion)

            inference = messages
            if hierarchy is not None:
                from hierarchy import ControlStop, Frame

                try:
                    guidance = hierarchy.prepare(
                        instruction, Frame(folder, context, transition), actions, trigger, event
                    )
                except ControlStop as stop:
                    return finish(stop.reason, stop.detail)
                trigger = None
                if guidance and getattr(hierarchy, "advisory_only", False):
                    # Keep the action contract intact. Guidance lives only in this request.
                    latest = messages[-1]
                    inference = [
                        *messages[:-1],
                        dict(latest, content=[text(guidance), *latest["content"]]),
                    ]
                elif guidance:
                    inference = [
                        dict(role="system", content=messages[0]["content"] + "\n" + guidance),
                        *messages[1:],
                    ]
            if out_of_time():
                return finish("wall_time_budget")
            if attempt_context:
                # Attach frozen examples only to the latest decision, inside tokenizer accounting.
                latest = inference[-1]
                inference = [
                    *inference[:-1],
                    dict(latest, content=[*attempt_context, *latest["content"]]),
                ]
            wire = (
                context_compactor.prepare(inference)
                if context_compactor is not None
                else copy.deepcopy(inference)
            )
            if bind_observation_ids:
                wire = bind_observation_examples(wire)
            response = policy(wire)
            calls += 1
            event("response", response=response)
            choice = response["choices"][0]
            message = choice["message"]
            messages.append(
                {
                    "role": "assistant",
                    **{
                        key: message[key]
                        for key in ("content", "reasoning", "reasoning_content")
                        if key in message
                    },
                }
            )
            try:
                if choice["finish_reason"] != "stop":
                    raise ValueError("incomplete generation")
                action = parse_action(message.get("content") or "")
                if action.get("observation_id") != observation.robot.observation_id:
                    raise ValueError("stale or missing observation_id")
                if action.get("stage") == "finish":
                    if set(action) != {"stage", "observation_id"}:
                        raise ValueError("finish requires exactly stage and observation_id")
                    if hierarchy is not None:
                        trigger = "executor_finish"
                    if native_finish_feedback or hierarchy is not None:
                        successful = bool(benchmark.success())
                        feedback = dict(
                            stage="finish_feedback",
                            native_success=successful,
                            observation_id=observation.robot.observation_id,
                            executed_ticks=0,
                        )
                        if completion is not None and progress_access == "default":
                            feedback["completion"] = completion
                        event("finish_feedback", feedback=feedback)
                        if successful:
                            return finish("native_success")
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    text(feedback),
                                    text(
                                        "The environment has not judged this task successful. "
                                        "Continue from the current measured observation."
                                    ),
                                ],
                            }
                        )
                        continue
                    return finish("model_finish")
                if action.get("stage") == "check_task_completion":
                    if not query_available or completion is None:
                        raise ValueError("task progress tool unavailable in this episode")
                    if set(action) != {"stage", "observation_id"}:
                        raise ValueError(
                            "completion query requires exactly stage and observation_id"
                        )
                    if local_queries >= max_queries_per_observation:
                        raise ValueError("query budget for this observation exhausted")
                    queries += 1
                    completion_queries += 1
                    local_queries += 1
                    event("completion_query", request=action, receipt=completion, executed_ticks=0)
                    messages.append({"role": "user", "content": [text(completion)]})
                    continue
                if action.get("stage") in QUERY_STAGES:
                    if local_queries >= max_queries_per_observation:
                        raise ValueError("query budget for this observation exhausted")
                    annotated, query_feedback = query(action, observation)
                else:
                    benchmark.validate_action(action, observation)
            except (ValueError, TypeError, KeyError) as error:
                trigger = "rejected_action: " + str(error)
                rejected += 1
                event("rejection", error=str(error), executed=False)
                messages.append({"role": "user", "content": [text("Rejected: " + str(error))]})
                continue
            if action.get("stage") in QUERY_STAGES:
                queries += 1
                local_queries += 1
                query_path = folder / f"query-{local_queries:02d}.png"
                annotated.save(query_path)
                event(
                    "visual_query",
                    request=action,
                    feedback=query_feedback,
                    image=str(query_path.relative_to(out)),
                    executed_ticks=0,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            text(query_feedback),
                            image_part(query_path),
                        ],
                    }
                )
                # Associate all query images with the same observation for eviction.
                observation_messages[-1].append(len(messages) - 1)
                continue
            # Deliberately outside the validation handler. A physical failure is
            # never converted into a zero-execution validation error or retried.
            if out_of_time():
                return finish("wall_time_budget")
            event("action_started", action=action)
            executing = True
            feedback = benchmark.execute(action)
            executing = False
            actions += 1
            transition = dict(action=action, feedback=feedback)
            event("execution", feedback=feedback)
            if getattr(benchmark, "completion", None) is not None:
                event("completion_evidence", receipt=benchmark.completion)
            if isinstance(feedback, dict) and any(
                feedback.get(k) for k in ("interrupted", "error")
            ):
                trigger = "controller_feedback"
            messages.append({"role": "user", "content": [text({"execution": feedback})]})
            observation = None
    except TimeoutError:
        if not executing and out_of_time():
            return finish("wall_time_budget")
        event("infrastructure_error", actions_completed=actions)
        raise
    except Exception:
        # Avoid copying exception strings that might contain network credentials.
        event("infrastructure_error", actions_completed=actions)
        raise
