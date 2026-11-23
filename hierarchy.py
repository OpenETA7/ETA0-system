"""Episode-scoped planner, visual supervisor and optional one-shot teacher."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from harness import image_part, parse_action, text

PLAN = """Interpret the user's goal using the current views and any clarification. Preserve object
identities, relations, order constraints and all requested end states. If ambiguity would change a
physical action, return {"question":"one specific question"}. Otherwise return {"steps":[
{"instruction":"one short grounded task for the robot","done_when":"observable completion test"}]}.
Use 1-12 steps. Group actions the robot can already perform. Do not assume hidden state."""
JUDGE = """Review the ACTIVE step using ordered frames, executed actions and measured feedback.
Return {"decision":"continue|advance|replan|stop","reason":"visual evidence or uncertainty",
"evidence":["one or more supplied frame labels"],"hint":"short advice for executor"}.
advance must cite the latest frame and accepts only the active subgoal. For replan include "steps"
(same step schema) to replace the unfinished suffix and preserve all constraints. If the final
goal appears complete but native success is false, inspect the missing condition and give a
concrete hint or replan. stop halts execution when continued motion is inappropriate or
futile."""
TEACH = """Generate one hypothetical worked demonstration for the active subgoal. Return
{"trajectory":[{"cue":"observable precondition","skill":"short action in words",
"expected":"observable consequence"}]}. Use 2-6 sequential transitions, including verification.
Ground the example in these views. Do not invent measured poses, object contact, pixel coordinates
or future images. The executor computes actions from live state using its action schema."""


class ControlStop(Exception):
    """End an episode before dispatching another action."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def phrase(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 1500:
        raise ValueError("expected a nonempty text field up to 1500 characters")
    return value.strip()


def rows(data, key, fields, minimum, maximum):
    items = data.get(key)
    if not isinstance(items, list) or not minimum <= len(items) <= maximum:
        raise ValueError(f"expected {minimum}-{maximum} entries in {key}")
    return [{field: phrase(item[field]) for field in fields} for item in items]


@dataclass
class Frame:
    folder: Path
    context: dict
    transition: dict | None


class VisualAdvisor:
    """Shared transport and frame audit for upper-model requests."""

    def __init__(self, upper, max_upper_calls):
        self.upper = upper
        self.max_upper_calls = max_upper_calls
        self.calls = 0
        self.frames = []

    def selected_frames(self):
        last = len(self.frames) - 1
        return [self.frames[i] for i in sorted({0, max(0, last - 1), last})]

    def ask(self, role, prompt, data, event):
        if self.calls >= self.max_upper_calls:
            raise ControlStop("upper_call_budget")
        selected = self.selected_frames()
        frame_ids = [f.folder.name for f in selected]
        content = [text(data)]
        for frame in selected:
            content.extend(
                [
                    text(
                        dict(
                            frame=frame.folder.name,
                            **frame.context,
                            preceding_execution=frame.transition,
                        )
                    ),
                    text("External RGB"),
                    image_part(frame.folder / "external.png"),
                    text("Wrist RGB / TCP"),
                    image_part(frame.folder / "wrist_overlay.png"),
                ]
            )
        self.calls += 1
        event(
            "upper_request",
            role=role,
            call=self.calls,
            frames=frame_ids,
            input=data,
            prompt=prompt,
            timestamp=time.time(),
        )
        start = time.monotonic()
        reply = self.upper(
            [
                {"role": "system", "content": prompt + " Return one JSON object."},
                {"role": "user", "content": content},
            ]
        )
        choice = reply["choices"][0]
        event(
            "upper_response",
            role=role,
            call=self.calls,
            elapsed_s=time.monotonic() - start,
            content=choice["message"].get("content"),
            finish_reason=choice["finish_reason"],
            model=reply.get("model"),
            usage=reply.get("usage"),
        )
        if choice["finish_reason"] != "stop":
            raise ValueError(f"incomplete {role} reply")
        return parse_action(choice["message"]["content"])


class Hierarchy(VisualAdvisor):
    """Create a fresh instance for each episode. Every review is a synchronous action barrier."""

    def __init__(
        self,
        upper,
        *,
        teacher=True,
        supervise=True,
        review_every=3,
        max_upper_calls=60,
        clarify=None,
    ):
        if review_every < 1 or max_upper_calls < 1:
            raise ValueError("budgets must be positive")
        super().__init__(upper, max_upper_calls)
        self.teacher = teacher
        self.supervise = supervise
        self.review_every = review_every
        self.clarify = clarify
        self.steps = []
        self.completed = []
        self.demo = []
        self.hint = ""
        self.answer = ""
        self.last_review = 0

    def review(self, state, event):
        labels = [frame.folder.name for frame in self.selected_frames()]
        request = dict(state, allowed_evidence_labels=labels, latest_frame_label=labels[-1])
        prompt = (
            JUDGE + " Copy labels verbatim from allowed_evidence_labels into evidence. "
            "Place all descriptions of what you see in reason."
        )
        for attempt in range(3):
            result = self.ask("supervisor", prompt, request, event)
            evidence = result.get("evidence")
            if (
                not isinstance(evidence, list)
                or not evidence
                or any(not isinstance(label, str) or label not in labels for label in evidence)
            ):
                error = "judge must cite a supplied frame using the exact label"
            elif result.get("decision") == "advance" and labels[-1] not in evidence:
                error = "advance must cite the latest frame"
            else:
                return result
            event(
                "invalid_citation",
                call=self.calls,
                error=error,
                received=evidence,
                allowed=labels,
                executed=False,
            )
            if attempt == 2:
                raise ValueError(error)
            request = dict(request, correction=error, previous_evidence=evidence)

    def prepare(self, instruction, frame, actions, trigger, event):
        fresh = not self.frames or self.frames[-1].folder != frame.folder
        if fresh:
            self.frames.append(frame)
        state = dict(
            user_goal=instruction,
            clarification=self.answer,
            completed=self.completed,
            remaining=self.steps,
            trigger=trigger,
            native_success=False,
        )
        changed = False
        if not self.steps:
            plan = self.ask("planner", PLAN, state, event)
            if "question" in plan:
                question = phrase(plan["question"])
                event("clarification", question=question)
                if self.clarify is None:
                    raise ControlStop("needs_clarification", question)
                self.answer = phrase(self.clarify(question))
                state["question"] = question
                state["clarification"] = self.answer
                plan = self.ask("planner", PLAN, state, event)
                if "question" in plan:
                    raise ControlStop("needs_clarification", phrase(plan["question"]))
            self.steps = rows(plan, "steps", ("instruction", "done_when"), 1, 12)
            changed = True
        elif self.supervise and (
            trigger or (fresh and actions - self.last_review >= self.review_every)
        ):
            self.last_review = actions
            result = self.review(state, event)
            evidence = result["evidence"]
            reason = phrase(result["reason"])
            decision = result["decision"]
            self.hint = phrase(result["hint"]) if result.get("hint") else reason
            if decision == "stop":
                raise ControlStop("supervisor_stop", reason)
            if decision == "advance" and len(self.steps) > 1:
                self.completed.append(dict(**self.steps.pop(0), evidence=evidence))
                changed = True
            elif decision == "replan":
                self.steps = rows(result, "steps", ("instruction", "done_when"), 1, 12)
                changed = True
            elif decision not in {"continue", "advance"}:
                raise ValueError("unknown supervisor decision")
            elif decision == "advance":
                self.hint += " Native goal still false. Inspect remaining goal conditions."
        elif not self.supervise and trigger == "executor_finish" and len(self.steps) > 1:
            self.completed.append(self.steps.pop(0))
            changed = True
        if changed:
            self.demo = []
            if self.teacher:
                demo = self.ask(
                    "teacher",
                    TEACH,
                    dict(user_goal=instruction, clarification=self.answer, active=self.steps[0]),
                    event,
                )
                self.demo = rows(demo, "trajectory", ("cue", "skill", "expected"), 2, 6)
            event(
                "subgoal",
                completed=list(self.completed),
                active=self.steps[0],
                remaining=list(self.steps),
                demonstration=self.demo,
            )
        payload = dict(
            user_goal=instruction,
            clarification=self.answer,
            active=self.steps[0],
            completed=[step["instruction"] for step in self.completed],
            hint=self.hint,
            hypothetical_demonstration=self.demo,
        )
        return (
            "Execute only the active subgoal. Use the demonstration as an example. Ground "
            "every action in the current observation. Use finish to request a subgoal review. "
            "Return the usual benchmark action JSON.\n" + json.dumps(payload, ensure_ascii=False)
        )
