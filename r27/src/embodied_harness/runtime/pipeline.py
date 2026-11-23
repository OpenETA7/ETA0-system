"""Overlap supervision with preparation, preserving the physical dependency barrier."""
import asyncio
import contextlib
import time

from .milestones import MilestoneManager
from .predicates import evaluate_condition
from embodied_harness.agents.prompts import FAST, execution_base_prompt


READ_ONLY = frozenset({"observe", "mark_point", "locate_point", "select_candidate"})
TERMINAL = frozenset({"wall_time_budget", "decision_budget", "checkpoint_unavailable"})

PREPARATION_SCOPE = """
PREPARATION-ONLY SCOPE OVERRIDE:
The Slow supervisor is preparing or assessing outcome contracts for the authorized mission.
Use ONE read-only observation/measurement to identify the requested target or destination from
current images. Available read-only choices are observe, select_candidate, locate_point, mark_point.
Never command physical motion, claim task completion, or use a fabricated reference. Return
one JSON {"kind":"tool","tool":"...","args":{...},"reason":"..."}, or {"kind":"wait"}.
This is preparation only; the eventual executor still must follow the Slow contract.
Keep the regular executor's geometry conventions: top_surface for a can/box, one rim wall for
a wide bowl, and placement_center for a destination region. Read-only scouting is not a reason
to choose a different grasp part or substitute an unsupported contact strategy.
"""
SCOUT = FAST + PREPARATION_SCOPE


def build_scout_prompt(*, parity=False, rim_grasp_guidance=False, view_recovery_guidance=False):
    """Reuse executor geometry wording; preparation still grants read-only work only."""
    if not parity:
        return SCOUT
    return execution_base_prompt(rim_grasp_guidance=rim_grasp_guidance,
                                 view_recovery_guidance=view_recovery_guidance) + PREPARATION_SCOPE


class PipelinedMilestoneManager(MilestoneManager):
    pipelined = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_task = None
        self.preparation_used = False
        self.budget_reviews = set()
        self.initial_task = None
        self.scout_task = None
        self.scout_decisions = 0
        self.readonly_parents = {}
        self.context.defer_routine_advice = True
        self.preparation_points = set()
        self.preparation_goal = None
        self.fast_model = None
        self.scout_prompt_parity = False
        self.rim_grasp_guidance = False
        self.view_recovery_guidance = False

    def note_preparation_points(self, prior_ids):
        if self.checkpoint_task is not None and self.preparation_used:
            points = getattr(self.coordinator.adapter, "points", {})
            created = set(points) - set(prior_ids)
            self.preparation_points.update(created)
            if created:
                self.store.emit("pipeline.preparation_measurements", {
                    "point_ids": sorted(created), "preview_goal": self.preparation_goal,
                    "scope": "unapproved_next_goal_preparation"})

    def finish_preparation(self, outcome):
        keep = (outcome == "advanced" and self.context.milestone is not None
                and self.context.milestone.goal == self.preparation_goal)
        if not keep and self.preparation_used:
            adapter = self.coordinator.adapter
            if hasattr(adapter, "retire_points"):
                adapter.retire_points(self.preparation_points)
            else:
                for point_id in self.preparation_points:
                    getattr(adapter, "points", {}).pop(point_id, None)
            retired = set(self.context.local_state.get("retired_point_ids", []))
            retired.update(self.preparation_points)
            self.context.local_state["retired_point_ids"] = sorted(retired)
            if self.fast_model is not None and hasattr(self.fast_model, "history"):
                self.fast_model.history.clear()
            self.context.local_state["preparation_invalidated"] = {
                "point_ids": sorted(self.preparation_points), "outcome": outcome,
                "instruction": "The next-goal preview was not authorized. Its measurements are retired. Re-ground any needed target under the current contract; a preparation intent is not completed work."}
            self.store.emit("pipeline.preparation_invalidated", dict(self.context.local_state["preparation_invalidated"]))
        elif keep:
            self.context.local_state.pop("preparation_invalidated", None)
        self.preparation_points.clear()
        self.preparation_goal = None

    async def query(self, prompt, extra):
        if extra.get("purpose") in {"initial_plan", "remaining_work_replan"}:
            prompt += "\nPIPELINED SUPERVISION: decision_budget is a background review threshold, not a hard stage stop. The unchanged episode decision limit remains hard. Pauses, explicit replans and stage-entry conditions remain binding. Keep contracts outcome-based.\n"
        return await super().query(prompt, extra)

    def matches_stamp(self, expected):
        # Only explicitly observed read-only refresh chains are exempt. An
        # unsolicited sensor update/cut-in still invalidates the assessment.
        current = self.stamp()
        if not all(expected[i] == current[i] for i in (0, 1, 3, 4, 5, 6)):
            return False
        observation_id = current[2]
        for _ in range(len(self.readonly_parents)+1):
            if observation_id == expected[2]:
                return True
            observation_id = self.readonly_parents.get(observation_id)
            if observation_id is None:
                break
        return False

    def note_readonly(self, before, after):
        # Called only around a whitelisted inspect() while holding the executor
        # lock. Do not bridge an intervening unclassified observation, physics
        # step, or robot-state change even if its generation counter stayed equal.
        before_sensor, after_sensor = before.state.get("sensor") or {}, after.state.get("sensor") or {}
        same_physics_clock = (all(before_sensor.get(k) == after_sensor.get(k)
                                  for k in ("sim_time", "control_steps"))
                              and all(k in before_sensor and k in after_sensor
                                      for k in ("sim_time", "control_steps")))
        physical_sensor = lambda sensor: {k: v for k, v in sensor.items()
                                          if k not in {"captured_at", "visual_sample"}}
        if (self.context.latest.id == before.id and before.generation == after.generation
                and before.action_seq == after.action_seq
                and same_physics_clock
                and physical_sensor(before_sensor) == physical_sensor(after_sensor)
                and before.state.get("tactile") == after.state.get("tactile")
                and before.state.get("robot") == after.state.get("robot")):
            self.readonly_parents[after.id] = before.id
            if len(self.readonly_parents) > 256:
                self.readonly_parents.pop(next(iter(self.readonly_parents)))

    async def initialize_with_scout(self, model):
        self.fast_model = model
        started = time.monotonic()
        self.initial_task = asyncio.create_task(self.initialize())
        projection = self.context.projection("fast")
        projection["mission"] = self.coordinator.mission_authority["mission"]
        self.scout_decisions = 1  # Includes cancelled model work in the same policy budget.
        prompt = build_scout_prompt(parity=self.scout_prompt_parity,
                                   rim_grasp_guidance=self.rim_grasp_guidance,
                                   view_recovery_guidance=self.view_recovery_guidance)
        self.store.emit("pipeline.scout_prompt_prepared", {
            "phase": "initial", "parity": self.scout_prompt_parity,
            "rim_grasp_guidance": self.scout_prompt_parity and self.rim_grasp_guidance,
            "view_recovery_guidance": self.scout_prompt_parity and self.view_recovery_guidance})
        self.scout_task = asyncio.create_task(model.decide(prompt, projection, self.context.latest.images))
        self.store.emit("pipeline.initial_overlap_started", {"scout_decisions": 1})
        try:
            done, _ = await asyncio.wait((self.initial_task, self.scout_task), return_when=asyncio.FIRST_COMPLETED)
            if self.scout_task in done:
                try:
                    decision = self.scout_task.result()
                    self.store.emit("pipeline.scout_decision", decision)
                    name, args = decision.get("tool"), decision.get("args")
                    if decision.get("kind") == "tool" and name in READ_ONLY and isinstance(args, dict):
                        self.coordinator.adapter.validate_args(name, args)
                        async with self.coordinator.lock:
                            before = self.context.latest
                            observation = await self.coordinator.adapter.inspect(name, args, self.context.action_seq)
                            self.note_readonly(before, observation)
                            self.context.ingest(observation)
                        self.store.emit("pipeline.scout_completed", {"tool": name, "physical_actions": 0})
                    elif decision.get("kind") != "wait":
                        self.store.emit("pipeline.scout_rejected", {"reason": "read_only_preparation_required"})
                except Exception as exc:
                    self.store.emit("pipeline.scout_error", {"type": type(exc).__name__})
            else:
                self.scout_task.cancel()
            wait_started = time.monotonic()
            result = await self.initial_task
            self.store.emit("pipeline.initial_ready", {"elapsed_s": time.monotonic()-started,
                "blocking_wait_s": time.monotonic()-wait_started, "scout_decisions": self.scout_decisions})
            return result
        finally:
            for task in (self.scout_task, self.initial_task):
                if task and not task.done():
                    task.cancel()
                if task:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    def request_local_review(self, reason):
        self.slow.request_review("local_recovery: " + str(reason))
        self.store.emit("pipeline.recovery_requested", {"reason": reason,
            "action_seq": self.context.action_seq, "blocking": False})

    async def recover(self, reason, *, allow_resume=True):
        hard = bool(self.pending_reason or self.coordinator.contract_review_pending or self.coordinator.paused)
        review_epoch = self.coordinator.contract_review_epoch
        if allow_resume:
            outcome = await super().recover(reason)
        else:
            outcome = await super().recover(reason, allow_resume=False)
        self.context.local_state["last_help"] = {"review_delivered": self.last_recovery_valid, "reason": reason}
        hard = hard or self.coordinator.paused or self.coordinator.contract_review_epoch != review_epoch
        if hard and not self.last_recovery_valid:
            # A rejected/exhausted planner cannot silently clear an accepted
            # contract conflict and resume execution under the disputed plan.
            self.pending_reason = str(reason)
            self.coordinator.contract_review_pending = True
            self.publish()
            self.store.emit("pipeline.hard_recovery_unavailable", {"reason": reason})
            return outcome or "supervisor_unavailable"
        return outcome

    async def checkpoint(self, trigger="fast_completion_proposal"):
        if self.checkpoint_task is None:
            self.preparation_used = False
            self.preparation_points.clear()
            self.preparation_goal = self.stages[self.index+1]["goal"] if self.index+1 < len(self.stages) else None
            self.checkpoint_task = asyncio.create_task(super().checkpoint(trigger))
            self.context.local_state["pipeline"] = {
                "phase": "checkpoint_pending", "allowed": "one read-only preparation decision",
                "next_goal_preview": self.stages[self.index+1]["goal"] if self.index+1 < len(self.stages) else None,
                "rule": "Current contract remains authoritative. Observe/measure for upcoming work only; no dependent physical action until assessment finishes."}
            self.store.emit("pipeline.checkpoint_started", {"trigger": trigger})
        return "checkpoint_pending"

    async def wait_checkpoint(self):
        task = self.checkpoint_task
        if task is None:
            return None
        started = time.monotonic()
        outcome = "aborted"
        try:
            outcome = await task
            self.store.emit("pipeline.checkpoint_ready", {"outcome": outcome,
                "blocking_wait_s": time.monotonic()-started, "preparation_used": self.preparation_used})
            return outcome
        finally:
            self.finish_preparation(outcome)
            self.checkpoint_task = None
            self.context.local_state.pop("pipeline", None)

    async def cancel_checkpoint(self, reason):
        if self.checkpoint_task:
            self.checkpoint_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.checkpoint_task
            self.finish_preparation("cancelled: " + reason)
            self.checkpoint_task = None
            self.context.local_state.pop("pipeline", None)
            self.store.emit("pipeline.checkpoint_cancelled", {"reason": reason})

    async def guard_proposal(self, kind, tool=None):
        if self.checkpoint_task is None:
            return True, None
        if self.pending_reason or self.coordinator.paused:
            await self.cancel_checkpoint("hard_control")
            return False, None
        if not self.checkpoint_task.done() and not self.preparation_used and kind == "tool" and tool in READ_ONLY:
            self.preparation_used = True
            self.store.emit("pipeline.preparation_allowed", {"tool": tool, "physical_actions": 0})
            return True, None
        # Do not execute a proposal produced before a new assessment/contract.
        outcome = await self.wait_checkpoint()
        return False, outcome if outcome in TERMINAL else None

    async def before_decision(self, remaining):
        self.remaining = remaining
        if remaining <= 0:
            return "decision_budget"
        if time.monotonic() >= self.deadline:
            return "wall_time_budget"
        if self.pending_reason or self.coordinator.paused:
            await self.cancel_checkpoint("hard_control")
            result = await self.recover(self.pending_reason or "supervisor paused execution; reassess before resuming")
            if result:
                return result
        if self.checkpoint_task and (self.checkpoint_task.done() or self.preparation_used):
            outcome = await self.wait_checkpoint()
            if outcome in TERMINAL:
                return outcome
            if outcome == "replan":
                return await self.recover(self.pending_reason or "checkpoint requires a revised contract")
        m = self.context.milestone
        if self.stage_decisions >= m.decision_budget and m.id not in self.budget_reviews:
            self.budget_reviews.add(m.id)
            self.request_local_review("stage review threshold reached; assess progress while Fast stays within current outcome and constraints")
        self.publish()
        sensor_conditions = [c for c in m.success_conditions if c["predicate"] != "visual"]
        checkpoint_key = (m.id, self.context.action_seq)
        if (self.checkpoint_task is None and not m.final_stage and sensor_conditions
                and checkpoint_key != self.last_sensor_checkpoint
                and all(evaluate_condition(c, self.context)["status"] == "satisfied" for c in sensor_conditions)):
            self.last_sensor_checkpoint = checkpoint_key
            await self.checkpoint("sensor_conditions_candidate")
        return None

    async def close(self):
        await self.cancel_checkpoint("episode_shutdown")
        for task in (self.scout_task, self.initial_task):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
