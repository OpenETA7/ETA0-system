"""Advisory control and disjoint wall-clock accounting for one executor-led episode."""

from __future__ import annotations

import http.client
import json
import math
import signal
import threading
import time
from collections import deque
from contextlib import contextmanager

from hierarchy import VisualAdvisor

PLAN = """You advise a robot that already solves short manipulation tasks independently.
Give one brief plan with at most three outcome-level milestones. Preserve the user's object
identities, frame of reference, ordering and end state. Group routine pick-and-place skills.
Leave perception, motion, grasp geometry and verification to the executor.
Return {"plan":"one short sentence"}. Never request confirmation or assign an active subtask."""
SUPERVISE = """The executor has encountered repeated difficulty. Compare the latest views,
controller feedback and available native completion evidence. Offer one specific, optional
correction that preserves the original task. Use uncertainty when evidence is unavailable.
Leave action selection to the robot. Return {"hint":"one short sentence"}.
Do not issue a stop, split the task into steps, ask the user, or declare success."""
TEACH = """Give one hypothetical recovery example applicable to the current obstacle.
Return {"cue":"observable situation","skill":"one high-level skill","expected":"observable result"}.
The example is advisory. Leave poses, trajectory knots, gripper commands and timing to the robot.
Do not claim the imagined result already occurred or ask the user for clarification."""


class EpisodeBudget:
    def __init__(self, lower_s, upper_s, clock=time.monotonic):
        if lower_s <= 0 or upper_s < 0:
            raise ValueError("invalid wall budget")
        self.clock = clock
        self.start = clock()
        self.lower_s, self.upper_s = lower_s, upper_s
        self.upper_elapsed = 0.0
        self.upper_deadline = None

    def lower_deadline(self):
        return self.start + self.lower_s + self.upper_elapsed

    @contextmanager
    def upper_interval(self):
        start = self.clock()
        remaining = self.upper_s - self.upper_elapsed
        if remaining <= 0:
            raise TimeoutError("upper wall budget exhausted")
        self.upper_deadline = start + remaining

        # The Linux episode runner owns its main thread. Bound even a server that
        # keeps a socket alive by trickling bytes through each read timeout.
        arm_timer = (
            threading.current_thread() is threading.main_thread()
            and hasattr(signal, "setitimer")
            and not signal.getitimer(signal.ITIMER_REAL)[0]
        )

        def expire(signum, frame):
            raise TimeoutError("upper wall budget exhausted")

        if arm_timer:
            previous = signal.signal(signal.SIGALRM, expire)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield
        finally:
            if arm_timer:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            self.upper_elapsed += self.clock() - start
            self.upper_deadline = None

    def metrics(self):
        total = self.clock() - self.start
        return dict(
            episode_wall_s=total,
            lower_wall_s=total - self.upper_elapsed,
            upper_wall_s=self.upper_elapsed,
        )


def tool_configuration(state):
    pose = state.get("eef_pose_base") or {}
    xyz, quat, width = (
        pose.get("position_mm"),
        pose.get("quaternion_wxyz"),
        state.get("gripper_open_width_mm"),
    )
    if not (
        isinstance(xyz, (list, tuple))
        and len(xyz) == 3
        and isinstance(quat, (list, tuple))
        and len(quat) == 4
        and all(isinstance(x, (float, int)) and math.isfinite(x) for x in (*xyz, *quat, width))
    ):
        return None
    norm = math.hypot(*quat)
    if not norm:
        return None
    return tuple(xyz), tuple(value / norm for value in quat), width


def same_configuration(a, b):
    # Quaternion sign is arbitrary. The absolute dot product is cos(angle / 2).
    return (
        math.dist(a[0], b[0]) <= 8
        and abs(a[2] - b[2]) <= 3
        and abs(sum(x * y for x, y in zip(a[1], b[1], strict=True))) >= math.cos(math.radians(4))
    )


class StallDetector:
    """Consume one new executor outcome at most once, before the next decision."""

    tracking_window = 8

    def __init__(self, failures=3, plateau=8, *, tracking_mm=30, tracking_count=3):
        if min(failures, plateau) < 1:
            raise ValueError("stall thresholds must be positive")
        self.failures, self.plateau = failures, plateau
        self.failed = self.flat = 0
        if tracking_mm <= 0 or not 1 <= tracking_count <= self.tracking_window:
            raise ValueError("invalid tracking thresholds")
        self.tracking_mm, self.tracking_count = tracking_mm, tracking_count
        self.poses = deque(maxlen=3 * plateau)
        self.tracking_errors = deque(maxlen=self.tracking_window)
        self.actions_since_progress = 0
        self.tracking_latched = False
        self.previous = None
        self.latched = False

    def persistent_revisit(self, pose):
        # Three comparable configurations, each separated by a full window of actions.
        last_match = 0
        matches = 0
        for age, old in enumerate(reversed(self.poses), 1):
            if age - last_match >= self.plateau and old and same_configuration(pose, old):
                matches += 1
                last_match = age
                if matches == 2:
                    return True
        return False

    def persistent_tracking_error(self, feedback, gained):
        if gained:
            self.actions_since_progress = 0
            self.tracking_errors.clear()
            self.tracking_latched = False
            return False
        self.actions_since_progress += 1
        error = feedback.get("final_goal_error")
        distance = error.get("position_error_mm") if isinstance(error, dict) else None
        usable = (
            feedback.get("termination") == "path_complete"
            and feedback.get("executed_ticks") != 0
            and not (feedback.get("interrupted") or feedback.get("error"))
            and type(distance) in (int, float)
            and math.isfinite(distance)
            and distance >= 0
        )
        high = distance >= self.tracking_mm if usable else None
        self.tracking_errors.append(high)
        if len(self.tracking_errors) < self.tracking_window or None in self.tracking_errors:
            return False
        count = sum(self.tracking_errors)
        if count == 0:
            self.tracking_latched = False
        return (
            self.actions_since_progress >= 2 * self.tracking_window
            and high
            and count >= self.tracking_count
            and not self.tracking_latched
        )

    def observe(self, kind, feedback=None, completion=None, state=None):
        if kind not in {"execution", "rejection", "finish_feedback"}:
            return None
        feedback = feedback or {}
        failed = kind != "execution" or bool(feedback.get("interrupted") or feedback.get("error"))
        revisited = tracking = False
        if kind == "execution":
            failed |= feedback.get("termination") == "tracking_lag_reobserve"
            failed |= feedback.get("executed_ticks") == 0
            conditions = ((completion or {}).get("feedback") or {}).get("conditions", [])
            known = (
                (completion or {}).get("status") == "checked"
                and conditions
                and all(type(c.get("satisfied_in_last_evaluation")) is bool for c in conditions)
            )
            current = (
                frozenset(
                    c["requirement_id"] for c in conditions if c["satisfied_in_last_evaluation"]
                )
                if known
                else None
            )
            gained = (
                current is not None and self.previous is not None and bool(current - self.previous)
            )
            tracking = self.persistent_tracking_error(feedback, gained)
            if current is None or self.previous is None:
                self.flat = 0
            elif gained:
                self.flat = self.failed = 0
                self.latched = False
                failed = False
            else:
                self.flat += 1
            if self.flat == 0:
                self.poses.clear()
            pose = tool_configuration(state or {})
            if pose and current is not None:
                revisited = self.persistent_revisit(pose)
            self.poses.append(pose)
            self.previous = current
            if not failed and self.flat == 0:
                self.latched = False
        self.failed = self.failed + 1 if failed else 0
        reason = (
            "consecutive_failures"
            if self.failed >= self.failures
            else "unchanged_goal_conditions"
            if self.flat >= self.plateau and revisited
            else None
        )
        if reason and not self.latched:
            self.latched = True
            if tracking:
                self.tracking_latched = True
            return reason
        if tracking:
            self.tracking_latched = True
            return "repeated_tracking_error"
        return None


def short_field(data, key, limit):
    value = data[key]
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError("invalid advisory text")
    return value.strip()


class ExecutorLed(VisualAdvisor):
    """Reuse the audited visual request writer. The executor always retains the full goal."""

    advisory_only = True

    def __init__(
        self,
        upper,
        budget,
        *,
        planner,
        supervise,
        teacher,
        long_task,
        failures=3,
        plateau=8,
        max_upper_calls=8,
    ):
        super().__init__(upper, max_upper_calls)
        self.supervise, self.teacher = supervise, teacher
        self.budget = budget
        self.planner, self.long_task = planner, long_task
        self.detector = StallDetector(failures, plateau)
        self.plan = ""
        self.last_action = 0
        self.clarification = ""

    def advisory(self, role, prompt, data, event):
        if self.calls >= self.max_upper_calls or self.budget.upper_elapsed >= self.budget.upper_s:
            event("upper_skipped", role=role, reason="upper_budget")
            return ""
        with self.budget.upper_interval():
            try:
                result = self.ask(role, prompt, data, event)
                if "question" in result:
                    receipt = self.frames[-1].context.get("completion") or {"status": "unavailable"}
                    event(
                        "environment_clarification",
                        role=role,
                        receipt=receipt,
                        question=short_field(result, "question", 240),
                        executed_ticks=0,
                    )
                    self.clarification = "Latest environment completion evidence: " + json.dumps(
                        receipt, ensure_ascii=False
                    )
                    return ""
                if role == "teacher":
                    return "Hypothetical example: " + " → ".join(
                        short_field(result, key, 120) for key in ("cue", "skill", "expected")
                    )
                return short_field(result, "plan" if role == "planner" else "hint", 240)
            except (
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                OSError,
                RuntimeError,
                EOFError,
                http.client.HTTPException,
            ) as exc:
                event("upper_unavailable", role=role, error_type=type(exc).__name__)
                return ""

    def prepare(self, goal, frame, actions, trigger, event):
        self.clarification = ""
        initial = not self.frames
        if initial or self.frames[-1].folder != frame.folder:
            self.frames.append(frame)
        request = {"original_goal": goal}
        if initial and self.planner and self.long_task:
            self.plan = self.advisory("planner", PLAN, request, event)
        reason = None
        if actions > self.last_action:
            self.last_action = actions
            reason = self.detector.observe(
                "execution",
                frame.transition["feedback"],
                frame.context.get("completion"),
                frame.context.get("measured_state"),
            )
        elif trigger:
            kind = "finish_feedback" if trigger == "executor_finish" else "rejection"
            reason = self.detector.observe(kind)
        advice = []
        if reason and (self.supervise or self.teacher):
            request.update(
                trigger=reason,
                latest_executor_feedback=trigger,
                consecutive_failures=self.detector.failed,
                plateau_transitions=self.detector.flat,
                tracking_errors_above_threshold=self.detector.tracking_errors.count(True),
                tracking_window=self.detector.tracking_window,
                tracking_threshold_mm=self.detector.tracking_mm,
            )
            event("stall_intervention", **request)
            if self.supervise:
                advice.append(self.advisory("supervisor", SUPERVISE, request, event))
            if self.teacher:
                advice.append(self.advisory("teacher", TEACH, request, event))
        guidance = [self.plan, *advice, self.clarification]
        return (
            "Optional guidance for the original goal: " + " ".join(filter(None, guidance))
            if any(guidance)
            else ""
        )
