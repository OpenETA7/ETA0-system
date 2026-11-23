"""Outcome contracts: Slow plans/assesses; the persistent Fast owns every action."""
import asyncio
import copy
import time
from dataclasses import asdict

import jsonschema

from embodied_harness.agents.milestone_prompts import CHECKPOINT, PLAN
from embodied_harness.agents.model import ModelError
from embodied_harness.domain.models import Advice, Milestone, uid
from .predicates import SENSOR_PREDICATES, evaluate, evaluate_condition


TEXT = {"type": "string", "minLength": 1, "maxLength": 1200}
CONSTRAINT = {"anyOf": [TEXT, {"type": "object", "additionalProperties": False,
    "required": ["description", "scope"], "properties": {
        "description": TEXT, "scope": {"enum": ["throughout", "on_stage_completion", "on_mission_completion"]}}}]}
CONDITION = {"type": "object", "additionalProperties": False,
             "required": ["predicate"], "properties": {
                 "predicate": {"enum": ["visual", *SENSOR_PREDICATES]}, "description": TEXT,
                 "when": {"enum": ["current", "during_stage"]}}}
ENTRY_CONDITION = {**CONDITION, "properties": {
    **CONDITION["properties"], "predicate": {"enum": list(SENSOR_PREDICATES)}, "when": {"const": "current"}}}
STAGE = {"type": "object", "additionalProperties": False,
         "required": ["goal", "success_conditions"], "properties": {
             "goal": TEXT, "constraints": {"type": "array", "items": CONSTRAINT, "maxItems": 8},
             "preconditions": {"type": "array", "items": ENTRY_CONDITION, "maxItems": 6},
             "success_conditions": {"type": "array", "items": CONDITION, "minItems": 1, "maxItems": 8},
             "max_retries": {"type": "integer", "minimum": 1, "maximum": 3},
             "decision_budget": {"type": "integer", "minimum": 1, "maximum": 40}}}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["kind", "fundamental_constraints", "milestones"], "properties": {
                   "kind": {"const": "plan"},
                   "reason": TEXT,
                   "fundamental_constraints": {"type": "array", "items": CONSTRAINT, "maxItems": 12},
                   "milestones": {"type": "array", "items": STAGE, "minItems": 1, "maxItems": 5}}}
STATUS = {"enum": ["satisfied", "contradicted", "unknown"]}


class PlanError(ValueError):
    pass


class MilestoneManager:
    def __init__(self, slow, context, coordinator, store, max_decisions, deadline, max_replans=3):
        self.slow, self.context, self.coordinator, self.store = slow, context, coordinator, store
        self.max_decisions, self.deadline, self.max_replans = max_decisions, deadline, max_replans
        self.faithful_initial_referents = False
        self.plan_id = None
        self.stages = []
        self.index = 0
        self.fundamental = []
        self.fundamental_scopes = {}
        self.user_constraints = frozenset(coordinator.mission_authority["constraints"])
        self.completed = []
        self.stage_decisions = 0
        self.remaining = max_decisions
        self.replans = self.recoveries = self.checkpoints = 0
        self.pending_reason = None
        self.pending_milestone = None
        self.last_sensor_checkpoint = None
        self.official_terminal = False
        self.last_negative_checkpoint = None
        self.last_recovery_valid = False
        self.lock = asyncio.Lock()
        self.slow.milestones = self

    def stamp(self):
        m, o = self.context.milestone, self.context.latest
        return (m.id if m else None, m.revision if m else None,
                o.id if o else None, self.context.action_seq,
                self.coordinator.adapter.generation, self.coordinator.pause_epoch,
                self.coordinator.contract_review_epoch)

    def matches_stamp(self, expected):
        return expected == self.stamp()

    def publish(self):
        self.context.mission_plan = {
            "id": self.plan_id, "active_index": self.index,
            "fundamental_constraints": list(self.fundamental),
            "fundamental_constraint_scopes": dict(self.fundamental_scopes),
            "stages": [{"goal": s["goal"], "success_conditions": s["success_conditions"]} for s in self.stages],
            "completed": copy.deepcopy(self.completed[-8:]),
            "stage_decisions": self.stage_decisions, "episode_decisions_remaining": self.remaining,
            "pending_replan": self.pending_reason,
        }

    async def query(self, prompt, extra):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("milestone episode deadline")
        async with asyncio.timeout(remaining):
            async with self.slow.foreground(extra["purpose"]):
                stamp = self.stamp()
                projection = copy.deepcopy(self.context.projection("slow"))
                projection.update(extra)
                images = tuple(self.context.latest.images)
                self.store.emit("milestone.inference_started", {"purpose": extra["purpose"], "stamp": stamp})
                value = await self.slow.model.decide(prompt, projection, images)
                self.store.emit("milestone.inference_finished", {
                    "purpose": extra["purpose"], "stamp": stamp, "current_stamp": self.stamp(),
                    "decision": value, "provenance": "model_proposal"})
                if not self.matches_stamp(stamp):
                    raise PlanError("stale milestone inference; obtain fresh evidence")
                return value, stamp

    def validate_plan(self, value):
        jsonschema.validate(value, PLAN_SCHEMA)
        for description, scope in self.constraint_items(value["fundamental_constraints"]):
            if description in self.user_constraints and scope != "throughout":
                raise PlanError("user constraints apply throughout the mission")
            if description in self.fundamental_scopes and scope != self.fundamental_scopes[description]:
                raise PlanError("retained fundamental constraint scope cannot be changed: " + description)
        goals = [s["goal"].strip() for s in value["milestones"]]
        if len(set(goals)) != len(goals):
            raise PlanError("duplicate milestone goals")
        for stage in value["milestones"]:
            if any(c["predicate"] == "visual" and c.get("when", "current") != "current"
                   for c in stage["success_conditions"]):
                raise PlanError("visual outcomes need current semantic assessment; temporal occurrence is sensor-only")
            visual = [c for c in stage["success_conditions"] if c["predicate"] == "visual"]
            if not visual or any(not c.get("description", "").strip() for c in visual):
                raise PlanError("each milestone needs a concrete visual outcome; contact is not identity")
            if any(c["predicate"] == "visual" for c in stage.get("preconditions", [])):
                raise PlanError("entry preconditions must be public sensor predicates; put visual requirements in success conditions")
        for c in value["milestones"][0].get("preconditions", []):
            if evaluate(c["predicate"], self.context)["status"] != "satisfied":
                raise PlanError("first-stage entry condition is not currently supported: " + c["predicate"])

    @staticmethod
    def constraint_items(items):
        return [(item, "throughout") if isinstance(item, str) else (item["description"], item["scope"])
                for item in items]

    def activate(self, index):
        if self.remaining <= 0:
            self.pending_reason = "episode decision budget exhausted"
            self.publish()
            return False
        stage = self.stages[index]
        for condition in stage.get("preconditions", []):
            evidence = evaluate(condition["predicate"], self.context)
            if evidence["status"] != "satisfied":
                self.pending_reason = "next milestone entry unsupported: " + condition["predicate"]
                self.store.emit("milestone.entry_blocked", {"index": index, "evidence": evidence})
                self.publish()
                return False
        authority = self.coordinator.mission_authority
        previous = self.context.milestone
        final = index == len(self.stages) - 1
        stage_constraints = dict(self.constraint_items(stage.get("constraints", [])))
        scopes = {**stage_constraints, **self.fundamental_scopes,
                  **{c: "throughout" for c in self.user_constraints}}
        constraints = tuple(dict.fromkeys([*authority["constraints"], *self.fundamental, *stage_constraints]))
        milestone = Milestone(
            uid(), 1, authority["mission"], stage["goal"],
            capabilities=tuple(sorted(authority["capabilities"])),
            constraints=constraints, constraint_scopes={c: scopes.get(c, "throughout") for c in constraints},
            generation=self.coordinator.adapter.generation,
            success_predicate="environment_success" if final else "stage_conditions",
            preconditions=tuple(copy.deepcopy(stage.get("preconditions", []))),
            success_conditions=tuple(copy.deepcopy(stage["success_conditions"])),
            max_retries=stage.get("max_retries", 2),
            decision_budget=min(stage.get("decision_budget", 12), max(1, self.remaining)),
            plan_id=self.plan_id, stage_index=index, final_stage=final,
        )
        self.coordinator.delegate(milestone, expected_revision=previous.revision if previous else None)
        self.context.begin_stage_evidence(milestone.id)
        self.index, self.stage_decisions = index, 0
        self.pending_reason = self.pending_milestone = None
        self.coordinator.contract_review_pending = False
        # Advice for the old contract remains in the event log, never in the new stage.
        self.context.advice.clear()
        self.context.target_plan = {}
        self.slow.last_review_started_at = time.monotonic()
        self.publish()
        return True

    def advise(self, text):
        m = self.context.milestone
        if m and text:
            self.coordinator.apply_advice(Advice(
                m.id, m.revision, m.generation, "advice", str(text)[:1200], time.monotonic() + 120,
                expected_action_seq=self.context.action_seq, expected_action_id=self.context.action_id,
                expected_action_phase=self.context.action_phase))

    async def plan(self, reason, initial=False, *, allow_resume=True):
        error = None
        for attempt in range(2):
            if self.coordinator.paused and not allow_resume:
                raise PlanError("pause supersedes optional completion contract repair")
            try:
                prompt = PLAN
                if self.slow.review_economy:
                    from embodied_harness.agents.milestone_prompts import ECONOMICAL_PLAN
                    prompt = PLAN.replace("Simple pick-and-place usually needs two outcomes: the correct source is securely acquired, then the\nrequested final placement is established.", "") + ECONOMICAL_PLAN
                if initial and self.faithful_initial_referents:
                    from embodied_harness.agents.milestone_prompts import FAITHFUL_MISSION_REFERENTS
                    prompt += FAITHFUL_MISSION_REFERENTS
                    self.store.emit("milestone.referent_prompt_prepared", {
                        "initial": True, "attempt": attempt + 1,
                        "policy": "user_referents_without_inferred_appearance"})
                value, stamp = await self.query(prompt, {
                    "purpose": "initial_plan" if initial else "remaining_work_replan",
                    "mission": self.coordinator.mission_authority["mission"],
                    "plan_schema": PLAN_SCHEMA, "reason": reason, "validation_error": error,
                    "retained_fundamental_constraints": list(self.fundamental),
                    "retained_fundamental_constraint_scopes": dict(self.fundamental_scopes),
                    "episode_decisions_remaining": self.remaining,
                    "continue_allowed": not initial and not self.coordinator.paused,
                })
                if value.get("kind") == "continue" and not initial and not self.coordinator.paused:
                    if set(value) != {"kind", "advice"} or not isinstance(value["advice"], str) or not value["advice"].strip():
                        raise PlanError("continue requires a nonempty advice string only")
                    self.advise(value["advice"])
                    self.pending_reason = self.pending_milestone = None
                    self.coordinator.contract_review_pending = False
                    self.stage_decisions = 0
                    self.publish()
                    self.store.emit("milestone.recovery_continued", {"reason": reason, "advice": value["advice"]})
                    return "continue"
                self.validate_plan(value)
                async with self.coordinator.lock:
                    if (not self.matches_stamp(stamp) or self.coordinator.running or self.coordinator.unknown
                            or (self.coordinator.paused and not allow_resume)):
                        raise PlanError("plan handoff lost its verified action boundary")
                    proposed_constraints = dict(self.constraint_items(value["fundamental_constraints"]))
                    new_fundamental = list(dict.fromkeys([*self.fundamental, *proposed_constraints]))
                    self.fundamental = new_fundamental
                    self.fundamental_scopes.update(proposed_constraints)
                    self.coordinator.mission_authority["constraints"] |= frozenset(new_fundamental)
                    self.stages = copy.deepcopy(value["milestones"])
                    self.plan_id = uid()
                    if not initial:
                        self.replans += 1
                    self.activate(0)
                    self.store.emit("milestone.plan_accepted", {
                        "plan_id": self.plan_id, "reason": reason, "initial": initial,
                        "fundamental_constraints": self.fundamental, "stages": self.stages,
                        "provenance": "slow_model_outcome_plan", "motion_authority": "unchanged_user_mission"})
                if self.coordinator.paused and allow_resume:
                    await self.coordinator.resume()
                return "planned"
            except (jsonschema.ValidationError, PlanError, ModelError) as exc:
                error = str(exc)[:1600]
                self.store.emit("milestone.plan_rejected", {"attempt": attempt + 1, "error": error})
        raise PlanError(error or "invalid milestone plan")

    async def initialize(self):
        async with self.lock:
            return await self.plan("decompose the authorized mission into observable outcomes", initial=True)

    def request_replan(self, reason, milestone, expected_stamp=None):
        current = self.context.milestone
        if current != milestone or (expected_stamp is not None and not self.matches_stamp(expected_stamp)):
            self.store.emit("milestone.replan_stale", {"reason": reason})
            return False
        self.pending_reason = str(reason)[:1200]
        self.pending_milestone = current.id
        self.coordinator.contract_review_pending = True
        self.coordinator.contract_review_epoch += 1
        self.context.signal_control("replan", 80, current.id)
        self.publish()
        self.store.emit("milestone.replan_requested", {"milestone_id": current.id, "reason": self.pending_reason})
        return True

    async def recover(self, reason, *, allow_resume=True):
        async with self.lock:
            self.last_recovery_valid = False
            if self.coordinator.paused and not allow_resume:
                return "paused"
            if self.coordinator.running or self.coordinator.unknown:
                return "state_unknown"
            if self.recoveries >= self.max_replans:
                self.store.emit("milestone.recovery_exhausted", {"reason": reason})
                # A paused executor must stay stopped; ordinary failures remain locally recoverable.
                self.pending_reason = self.pending_milestone = None
                self.coordinator.contract_review_pending = False
                self.stage_decisions = 0
                self.advise("Supervisor recovery budget exhausted. Keep the current goal and constraints; choose a different local method or report inability.")
                self.publish()
                return "paused" if self.coordinator.paused else None
            self.recoveries += 1
            try:
                if allow_resume:
                    await self.plan(reason)
                else:
                    await self.plan(reason, allow_resume=False)
                self.last_recovery_valid = True
            except TimeoutError:
                return "wall_time_budget"
            except Exception as exc:
                self.store.emit("milestone.recovery_error", {"type": type(exc).__name__, "message": str(exc)[:600]})
                self.pending_reason = self.pending_milestone = None
                self.coordinator.contract_review_pending = False
                self.stage_decisions = 0
                self.publish()
                return "paused" if self.coordinator.paused else None
            return None

    async def before_decision(self, remaining):
        self.remaining = remaining
        if remaining <= 0:
            return "decision_budget"
        if time.monotonic() >= self.deadline:
            return "wall_time_budget"
        reason = self.pending_reason
        if self.coordinator.paused:
            reason = reason or "supervisor requested a stop; reassess the current evidence before resuming"
        if self.stage_decisions >= self.context.milestone.decision_budget:
            reason = reason or "current stage decision budget used; revise remaining work or give a substantive correction"
        self.publish()
        if reason:
            outcome = await self.recover(reason)
            if outcome:
                return outcome
        m = self.context.milestone
        sensor_conditions = [c for c in m.success_conditions if c["predicate"] != "visual"]
        checkpoint_key = (m.id, self.context.action_seq)
        if (not m.final_stage and sensor_conditions and checkpoint_key != self.last_sensor_checkpoint
                and all(evaluate_condition(c, self.context)["status"] == "satisfied" for c in sensor_conditions)):
            # Public sensors nominate a candidate boundary; they never certify object identity.
            # This generic condition trigger prevents an executor from silently skipping review.
            self.last_sensor_checkpoint = checkpoint_key
            outcome = await self.checkpoint(trigger="sensor_conditions_candidate")
            if outcome in {"wall_time_budget", "decision_budget", "checkpoint_unavailable"}:
                return outcome
            if outcome == "replan":
                return await self.recover(self.pending_reason or "checkpoint requires a revised contract")
        return None

    def record_decision(self):
        self.stage_decisions += 1
        self.remaining -= 1
        self.publish()

    @staticmethod
    def assessments(value, key, id_key, expected):
        entries = value.get(key)
        if not isinstance(entries, list) or len(entries) != len(expected):
            raise PlanError("checkpoint must assess every " + key)
        ids = [entry.get(id_key) for entry in entries if isinstance(entry, dict)]
        if len(ids) != len(entries) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise PlanError("checkpoint has missing/duplicate/unknown " + id_key)
        for entry in entries:
            if entry.get("status") not in STATUS["enum"] or not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
                raise PlanError("checkpoint status needs explicit supporting evidence")
        return all(entry["status"] == "satisfied" for entry in entries)

    def note_terminal(self, evidence):
        """Record benchmark termination separately from full contract certification."""
        self.official_terminal = evidence.get("status") == "verified"
        m = self.context.milestone
        self.store.emit("milestone.environment_terminal", {
            "official_success": self.official_terminal, "active_milestone_id": m.id,
            "final_stage_active": m.final_stage, "completed_intermediate_stages": len(self.completed),
            "sensor_conditions": [evaluate_condition(c, self.context)
                                  for c in m.success_conditions if c["predicate"] != "visual"],
            "semantic_constraints": "not independently certified by benchmark termination",
            "source": evidence.get("source"),
        })

    async def checkpoint(self, trigger="fast_completion_proposal"):
        # Repeating a rejected claim on exactly the same evidence is not a new
        # semantic assessment. Never cache an approval or bypass an unknown.
        if (self.slow.review_economy and self.last_negative_checkpoint == self.stamp()
                and not (self.pending_reason or self.coordinator.paused or self.coordinator.unknown
                         or self.coordinator.running or self.coordinator.contract_review_pending)):
            self.advise("This exact evidence was already insufficient. Obtain a new observation of the missing condition or make a meaningful corrective action before proposing completion again.")
            self.store.emit("milestone.checkpoint_coalesced", {"trigger": trigger, "reason": "unchanged_rejected_evidence"})
            return "continue"
        # Unavailable/invalid evidence is not a semantic instruction to continue.
        # Retry once with fresh public evidence, without letting Fast dispatch
        # another action between attempts. An exhausted assessment stops the run.
        for attempt in range(2):
            outcome = await self._checkpoint_once(trigger)
            if outcome != "checkpoint_retry":
                if outcome == "continue":
                    self.last_negative_checkpoint = self.stamp()
                return outcome
            if attempt == 0:
                self.store.emit("milestone.checkpoint_retry", {"attempt": 2, "trigger": trigger})
                if not (self.pending_reason or self.coordinator.paused or self.coordinator.unknown):
                    try:
                        await self.coordinator.observe()
                    except Exception as exc:
                        self.store.emit("milestone.checkpoint_refresh_error", {"type": type(exc).__name__})
                        break
        self.store.emit("milestone.checkpoint_unavailable", {"milestone_id": self.context.milestone.id})
        return "checkpoint_unavailable"

    async def _checkpoint_once(self, trigger):
        async with self.lock:
            m = self.context.milestone
            if self.pending_reason or self.coordinator.contract_review_pending:
                return "replan"
            if self.coordinator.running or self.coordinator.unknown or self.coordinator.paused:
                return "blocked"
            conditions = [{"condition_id": f"success-{i}", **c} for i, c in enumerate(m.success_conditions)]
            all_invariants = [{"constraint_id": f"constraint-{i}", "description": c,
                               "scope": m.constraint_scopes.get(c, "throughout")} for i, c in enumerate(m.constraints)]
            invariants = [c for c in all_invariants if c["scope"] != "on_mission_completion" or m.final_stage]
            deferred = [c for c in all_invariants if c not in invariants]
            sensors = {c["condition_id"]: evaluate_condition(c, self.context)
                       for c in conditions if c["predicate"] != "visual"}
            self.checkpoints += 1
            self.store.emit("milestone.completion_proposed", {"milestone_id": m.id, "stage_index": self.index, "trigger": trigger})
            try:
                value, stamp = await self.query(CHECKPOINT, {
                    "purpose": "checkpoint", "checkpoint_conditions": conditions,
                    "checkpoint_invariants": invariants, "sensor_assessments": sensors,
                    "deferred_mission_completion_constraints": deferred,
                })
                if value.get("kind") != "checkpoint" or value.get("decision") not in {"advance", "continue", "replan"}:
                    raise PlanError("invalid checkpoint decision")
                condition_ok = self.assessments(value, "assessments", "condition_id", [c["condition_id"] for c in conditions])
                invariant_ok = self.assessments(value, "invariants", "constraint_id", [c["constraint_id"] for c in invariants])
                self.store.emit("milestone.assessment", {"milestone_id": m.id, "decision": value,
                    "sensor_assessments": sensors, "provenance": "model_judgement_plus_separate_sensor_evidence"})
                if value["decision"] == "replan":
                    self.request_replan(value.get("reason", "checkpoint requests revised outcomes"), m)
                    return "replan"
                supported = condition_ok and invariant_ok and all(s["status"] == "satisfied" for s in sensors.values())
                if value["decision"] != "advance" or not supported or m.final_stage:
                    reason = value.get("advice") or value.get("reason") or "Stage conditions are not all supported; inspect missing evidence."
                    if m.final_stage and value["decision"] == "advance":
                        reason = "Final completion remains unverified. Reinspect the actual final relation and choose a local correction."
                    self.advise(reason)
                    self.store.emit("milestone.completion_rejected", {"milestone_id": m.id, "supported": supported, "final_stage": m.final_stage})
                    return "continue"
                async with self.coordinator.lock:
                    if (not self.matches_stamp(stamp) or self.coordinator.running or self.coordinator.paused
                            or self.pending_reason or self.coordinator.contract_review_pending):
                        raise PlanError("checkpoint handoff became stale")
                    completion = {"milestone_id": m.id, "plan_id": self.plan_id, "goal": m.goal,
                                  "observation_id": stamp[2], "action_seq": self.context.action_seq,
                                  "evidence": value["assessments"], "provenance": "model_assessed_stage_completion"}
                    if not any(c["milestone_id"] == m.id for c in self.completed):
                        self.completed.append(completion)
                        self.store.emit("milestone.completed", completion)
                    if self.remaining <= 0:
                        self.publish()
                        return "decision_budget"
                    activated = self.activate(self.index + 1)
                return "advanced" if activated else "replan"
            except TimeoutError:
                return "wall_time_budget"
            except Exception as exc:
                self.store.emit("milestone.checkpoint_error", {"type": type(exc).__name__, "message": str(exc)[:800]})
                return "checkpoint_retry"
