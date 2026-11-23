import asyncio
import time
import json
import math
import copy
from collections import deque
from embodied_harness.domain.models import ActionRequest, uid
from embodied_harness.domain.tool_errors import GraspOrientationError
from .prompts import FAST, execution_base_prompt
from .model import InferencePreempted
from embodied_harness.runtime.coordinator import GateError


class FastAgent:
    def __init__(self, model, context, coordinator, store, slow=None, mode="dual_async", review_every=3, *,
                 progress_recovery=False, async_completion_recovery=False,
                 completion_rendezvous=False, completion_review_timeout=30.0,
                 final_stage_prompt=False, recovery_reasoning=False, rim_grasp_guidance=False,
                 view_recovery_guidance=False, completion_contract_repair=False,
                 completion_schema_guard=False, recovery_executor_model=None,
                 physical_failure_executor=False, scout_prompt_parity=False):
        self.model, self.context, self.coordinator, self.store = model, context, coordinator, store
        self.slow, self.mode, self.review_every = slow, mode, review_every
        self.scout_prompt_parity = scout_prompt_parity
        self.steps = 0
        self.errors = 0
        self.false_claims = 0
        self.failed_signatures = {}
        self.readonly_signatures = {}
        self.help_counts = {}
        self.last_help_key = None
        self.seen_advice = set()
        self.seen_diagnostics = set()
        self.identity_checker = None
        self.destination_checker = None
        self.rejected_completion_seq = None
        self.rejected_completion_count = 0
        self.milestones = None
        self.active_contract_id = None
        from embodied_harness.runtime.recovery import ProgressRecovery
        self.progress_recovery = ProgressRecovery() if progress_recovery else None
        self.blocked_proposals = {}
        self.async_completion_recovery = async_completion_recovery
        self.completion_rendezvous = completion_rendezvous
        self.completion_review_timeout = completion_review_timeout
        self.completion_review_ticket = None
        self.completion_waited_seq = None
        self.final_stage_prompt = final_stage_prompt
        self.recovery_reasoning = recovery_reasoning
        self.recovery_reasoning_scope = None
        self.recovery_reasoning_calls = 0
        self.rim_grasp_guidance = rim_grasp_guidance
        self.view_recovery_guidance = view_recovery_guidance
        self.completion_contract_repair = completion_contract_repair
        # A new contract must not grant another automatic repair at an unchanged
        # physical revision. Episode decision/deadline/replan limits still apply.
        self.completion_repaired_actions = set()
        self.completion_contract_claim_counts = {}
        self.completion_schema_guard = completion_schema_guard
        self.completion_excluded_scope = None
        self.recovery_executor_model = recovery_executor_model
        self.recovery_executor_scope = None
        self.recovery_executor_armed = False
        self.recovery_executor_remaining = 0
        self.recovery_executor_requests = 0
        self.physical_failure_executor = physical_failure_executor
        self.executor_failure_scope = None
        self.executor_failure_receipts = {}

    def executor_contract_scope(self):
        m = self.context.milestone
        return (m.id, m.revision, m.generation, m.mission) if m else None

    def arm_recovery_executor(self, evidence, *, physical_evidence=None):
        """Replace at most four future actor calls; never add decisions or tools."""
        m, o = self.context.milestone, self.context.latest
        if not (self.recovery_executor_model and not self.recovery_executor_armed
                and self.mode == 'dual_async' and self.async_completion_recovery
                and self.milestones and getattr(self.milestones, 'pipelined', False)
                and m and m.final_stage and o and o.generation == m.generation
                and o.action_seq == self.context.action_seq
                and not self.coordinator.paused and not self.coordinator.running
                and not self.coordinator.unknown
                and evidence.get('status') == 'not_verified'
                and evidence.get('source') == 'official_libero_check_success'
                and evidence.get('observation_id') == o.id):
            return
        # record_decision already consumed this claim. Keep the episode budget.
        if self.milestones.remaining < 2:
            self.store.emit('executor_recovery.skipped', {'reason': 'fewer_than_two_future_decisions'})
            return
        self.recovery_executor_armed = True
        self.recovery_executor_scope = self.executor_contract_scope()
        self.recovery_executor_remaining = min(4, self.milestones.remaining)
        payload = {
            'contract_scope': list(self.recovery_executor_scope), 'action_seq': self.context.action_seq,
            'observation_id': o.id, 'max_requests': self.recovery_executor_remaining,
            'episode_decisions_remaining': self.milestones.remaining,
            'model': self.recovery_executor_model.model, 'role': 'recovery_executor',
            'semantics': 'replace_next_actor_requests_within_existing_decision_budget'}
        if physical_evidence is not None:
            payload.update(trigger='repeated_physical_failure', physical_evidence=physical_evidence)
        self.store.emit('executor_recovery.armed', payload)

    def note_executor_physical_failure(self, request, receipt, feedback, before_sim_time, evidence):
        """Use two real same-key failed actions, not rejected or blocked proposals.

        The existing detector defines equivalence and improvement. Its false
        measured_progress flag is not proof of zero motion or a collision.
        Only routing changes; it grants no action authority or extra budget.
        """
        if not (self.physical_failure_executor and self.progress_recovery
                and self.recovery_executor_model and not self.recovery_executor_armed
                and self.mode == 'dual_async' and self.async_completion_recovery
                and self.milestones and getattr(self.milestones, 'pipelined', False)):
            return
        m, o = self.context.milestone, self.context.latest
        if not (m and m.final_stage and o and o.generation == m.generation
                and o.action_seq == self.context.action_seq
                and request.milestone_id == m.id and request.revision == m.revision
                and request.generation == m.generation
                and receipt.request_id == request.request_id and receipt.action_id
                and receipt.status == 'completed' and receipt.observation_id == o.id
                and not self.coordinator.paused and not self.coordinator.running
                and not self.coordinator.unknown and not self.coordinator.contract_review_pending
                and not self.milestones.pending_reason
                and not (self.milestones.checkpoint_task and not self.milestones.checkpoint_task.done())
                and evidence.get('status') == 'not_verified'
                and evidence.get('source') == 'official_libero_check_success'
                and evidence.get('observation_id') == o.id):
            return
        if (type(before_sim_time) not in (int, float) or type(o.sim_time) not in (int, float)
                or not math.isfinite(before_sim_time) or not math.isfinite(o.sim_time)
                or o.sim_time <= before_sim_time or receipt.detail.get('host_error')
                or receipt.detail.get('state_unknown')):
            return
        if not (isinstance(feedback, dict) and isinstance(feedback.get('failure_key'), str)
                and feedback['failure_key'] and feedback.get('tool') == request.capability
                and type(feedback.get('stagnant_attempts')) is int
                and feedback['stagnant_attempts'] >= 1
                and feedback.get('failure_kind') not in {None, 'depth_discontinuity', 'unresolved_held_object'}):
            return
        scope = self.executor_contract_scope()
        if self.executor_failure_scope != scope:
            self.executor_failure_scope = scope
            self.executor_failure_receipts.clear()
        key = feedback['failure_key']
        if feedback.get('measured_progress') is not False:
            self.executor_failure_receipts.pop(key, None)
            return
        rows = self.executor_failure_receipts.get(key, [])
        if any(row['action_id'] == receipt.action_id or row['request_id'] == receipt.request_id
               or row['action_seq'] == o.action_seq for row in rows):
            return
        rows = (rows + [{'action_id': receipt.action_id, 'request_id': receipt.request_id,
            'action_seq': o.action_seq, 'observation_id': o.id,
            'before_sim_time': before_sim_time, 'after_sim_time': o.sim_time}])[-2:]
        self.executor_failure_receipts[key] = rows
        trigger = {'failure_key': key, 'failure_kind': feedback['failure_kind'],
            'stagnant_attempts': feedback['stagnant_attempts'], 'measured_progress': False,
            'contract_scope': list(scope), 'control_version': self.context.control_version,
            'physical_receipts': copy.deepcopy(rows)}
        self.store.emit('executor_recovery.physical_failure_recorded', trigger)
        if feedback['stagnant_attempts'] >= 2 and len(rows) == 2:
            self.arm_recovery_executor(evidence, physical_evidence=trigger)

    async def decide_executor(self, prompt, projection, images, inference_options):
        use_recovery = bool(self.recovery_executor_model and self.recovery_executor_remaining > 0)
        if use_recovery and self.executor_contract_scope() != self.recovery_executor_scope:
            self.store.emit('executor_recovery.expired', {
                'reason': 'contract_changed', 'unused_requests': self.recovery_executor_remaining})
            self.recovery_executor_remaining = 0
            use_recovery = False
        if use_recovery and (not self.context.milestone.final_stage
                or self.coordinator.paused or self.coordinator.running or self.coordinator.unknown
                or self.coordinator.contract_review_pending or self.milestones.pending_reason
                or (self.milestones.checkpoint_task and not self.milestones.checkpoint_task.done())):
            use_recovery = False
        if not use_recovery:
            return await self.model.decide(prompt, projection, images, **inference_options)
        if inference_options:
            raise ValueError('Recovery executor routing is separate from request-option ablations')
        executor = self.recovery_executor_model
        # Keep one coherent executed-decision history. Existing rejection and
        # identity-recovery clears apply to the primary history before this copy.
        executor.history = deque(copy.deepcopy(list(self.model.history)), maxlen=self.model.history.maxlen)
        self.recovery_executor_remaining -= 1
        self.recovery_executor_requests += 1
        request_index = self.recovery_executor_requests
        request_id = executor.accounting['logical_requests'] + 1
        started = time.monotonic()
        outcome = 'failed'
        self.store.emit('executor_recovery.request_started', {
            'index': request_index, 'request_id': request_id, 'model': executor.model,
            'role': 'recovery_executor', 'decision_slot': self.steps,
            'action_seq': self.context.action_seq, 'observation_id': self.context.latest.id,
            'control_version': self.context.control_version,
            'contract_scope': list(self.recovery_executor_scope),
            'remaining_request_budget': self.recovery_executor_remaining})
        try:
            decision = await executor.decide(prompt, projection, images)
            self.model.history = deque(copy.deepcopy(list(executor.history)), maxlen=self.model.history.maxlen)
            outcome = 'completed'
            return decision
        except InferencePreempted:
            outcome = 'preempted'
            raise
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        finally:
            # Cancelled/failed requests consume the bounded routing allowance:
            # service work may be billed and no invisible retry is authorized.
            self.store.emit('executor_recovery.request_finished', {
                'index': request_index, 'request_id': request_id, 'outcome': outcome,
                'elapsed_s': time.monotonic() - started,
                'remaining_request_budget': self.recovery_executor_remaining})

    def completion_physical_scope(self):
        """Direct LIBERO clock, not a new rendering or contract revision."""
        m, o = self.context.milestone, self.context.latest
        if not m or not o or o.generation != m.generation or o.action_seq != self.context.action_seq:
            return None
        if type(o.sim_time) not in (int, float) or not math.isfinite(o.sim_time):
            return None
        return (o.generation, o.action_seq, o.sim_time, self.context.control_version)

    def note_completion_negative(self, evidence):
        m = self.context.milestone
        if not (self.completion_schema_guard and self.mode == 'dual_async'
                and self.async_completion_recovery and self.milestones
                and getattr(self.milestones, 'pipelined', False) and m and m.final_stage
                and not self.coordinator.paused and not self.coordinator.running
                and not self.coordinator.unknown and self.context.latest
                and evidence.get('status') == 'not_verified'
                and evidence.get('source') == 'official_libero_check_success'
                and evidence.get('observation_id') == self.context.latest.id):
            return
        scope = self.completion_physical_scope()
        if scope is not None:
            self.completion_excluded_scope = scope
            self.store.emit('completion.schema_exclusion_recorded', {
                'scope': list(scope), 'observation_id': self.context.latest.id,
                'official_status': evidence['status']})

    def use_completion_schema(self):
        m = self.context.milestone
        return bool(self.completion_schema_guard and self.mode == 'dual_async'
            and self.async_completion_recovery and self.milestones
            and getattr(self.milestones, 'pipelined', False) and m and m.final_stage
            and not self.coordinator.paused and not self.coordinator.running
            and not self.coordinator.unknown and not self.coordinator.contract_review_pending
            and not self.milestones.pending_reason
            and not (self.milestones.checkpoint_task and not self.milestones.checkpoint_task.done())
            and self.completion_excluded_scope is not None
            and self.completion_excluded_scope == self.completion_physical_scope())

    def current_recovery_scope(self):
        m = self.context.milestone
        return ((m.id, m.revision, m.generation, self.context.action_seq,
                 self.context.control_version) if m else None)

    def use_recovery_reasoning(self):
        m = self.context.milestone
        recovery = self.context.local_state.get("completion_recovery") or {}
        return bool(self.recovery_reasoning and self.mode == "dual_async"
            and self.async_completion_recovery and self.milestones
            and getattr(self.milestones, "pipelined", False)
            and m and m.final_stage and self.recovery_reasoning_calls < 4
            and recovery.get("status") == "claim_rejected"
            and recovery.get("after_action_seq") == self.context.action_seq
            and self.recovery_reasoning_scope == self.current_recovery_scope()
            and not (self.milestones.checkpoint_task
                     and not self.milestones.checkpoint_task.done()))

    def sync_contract(self):
        current = self.context.milestone
        if not self.milestones or current.id == self.active_contract_id:
            return
        if self.active_contract_id is not None and hasattr(self.model, "history"):
            self.model.history.clear()
            self.store.emit("fast.history_compacted", {"reason": "milestone_handoff", "physical_state_preserved": True})
        self.active_contract_id = current.id
        self.recovery_reasoning_scope = None
        self.readonly_signatures.clear()
        self.rejected_completion_seq = None
        self.rejected_completion_count = 0
        self.completion_review_ticket = None
        self.completion_waited_seq = None
        self.context.local_state["readonly_streak"] = 0

    async def wait_completion_review(self, deadline):
        """Join the requested review once; never cancel it or acquire its model lock."""
        import contextlib
        seq, version = self.context.action_seq, self.context.control_version
        ticket = self.completion_review_ticket
        if not self.slow or ticket is None or self.completion_waited_seq == seq:
            return
        if self.milestones and getattr(self.milestones, "remaining", 2) <= 1:
            self.store.emit("completion.review_wait_skipped", {"ticket": ticket,
                "action_seq": seq, "reason": "no_next_decision_budget"})
            return
        self.completion_waited_seq = seq
        started = time.monotonic()
        end = started + self.completion_review_timeout
        if deadline is not None:
            end = min(end, deadline)
        control = asyncio.create_task(self.context.wait_for_control(version))
        changed = None
        outcome = "timeout"
        self.store.emit("completion.review_wait_started", {"ticket": ticket, "action_seq": seq,
            "timeout_s": max(0, end-started), "policy": "join_existing_review_no_new_inference"})
        try:
            while True:
                if self.context.control_version != version or self.coordinator.paused or (
                        self.milestones and self.milestones.pending_reason):
                    outcome = "control_changed"
                    break
                if self.slow.delivered_ticket >= ticket:
                    outcome = "review_delivered"
                    break
                remaining = end - time.monotonic()
                if remaining <= 0:
                    if deadline is not None and time.monotonic() >= deadline:
                        outcome = "episode_deadline"
                    break
                self.slow.changed.clear()
                if self.slow.delivered_ticket >= ticket:
                    continue
                changed = asyncio.create_task(self.slow.changed.wait())
                done, _ = await asyncio.wait((changed, control), timeout=remaining,
                                             return_when=asyncio.FIRST_COMPLETED)
                if control in done:
                    outcome = "control_changed"
                    break
                if not done:
                    if deadline is not None and time.monotonic() >= deadline:
                        outcome = "episode_deadline"
                    break
                changed = None
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            for task in (changed, control):
                if task is not None:
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            result = {"ticket": ticket, "action_seq": seq, "outcome": outcome,
                      "elapsed_s": time.monotonic()-started,
                      "semantics": "review availability only; no task success or action authorization"}
            state = self.context.local_state.get("completion_recovery")
            if state is not None and state.get("after_action_seq") == seq:
                state["review_outcome"] = outcome
            self.store.emit("completion.review_wait_finished", result)

    def tool_arguments(self, decision):
        """Recover transport/envelope shape only; schema validation stays strict.

        Models occasionally serialize args as JSON text or put schema-named
        arguments at the envelope's top level. Never invent values, ignore an
        unknown field, or silently resolve conflicting representations.
        """
        name = decision.get("tool")
        schema = self.coordinator.adapter.schemas.get(name)
        if schema is None:
            raise ValueError("Unknown tool")
        properties = set((schema.get("parameters") or {}).get("properties") or {})
        envelope = {"kind", "tool", "args", "reason", "advice_response", "diagnostic_response"}
        flat = {key: value for key, value in decision.items() if key not in envelope}
        args = decision.get("args")
        recovery = None
        if isinstance(args, str):
            args = json.loads(args)
            recovery = "json_encoded_args"
        if args is None and flat and set(flat).issubset(properties):
            args = flat
            recovery = "schema_named_top_level_args"
        elif flat:
            raise ValueError("Ambiguous or unknown top-level tool fields; put only schema arguments inside args")
        if not isinstance(args, dict):
            raise ValueError('tool args must be object; example {"kind":"tool","tool":"select_candidate","args":{"candidate_id":"current ID","part":"top_rim"}}')
        if recovery:
            self.store.emit("fast.format_recovered", {"tool": name, "recovery": recovery, "args": args})
        return args

    def normalize_decision(self, decision):
        name = decision.get("kind")
        if name in self.coordinator.adapter.schemas:
            if "tool" in decision and decision["tool"] != name:
                raise ValueError("conflicting tool names in model decision")
            decision = {**decision, "kind": "tool", "tool": name}
            self.store.emit("fast.format_recovered", {"tool": name, "recovery": "tool_name_as_kind"})
        return decision

    async def identity_recovery_view(self):
        if hasattr(self.model, "history"):
            self.model.history.clear()
        self.store.emit("fast.history_reset", {"reason": "identity_mismatch", "structured_state_preserved": True})
        # The checker leaves a large crop of the rejected object on screen.
        # Restore the exploration view so a different object is actually visible
        # at readable resolution; this changes no physical state or target choice.
        await self.coordinator.observe()
        self.store.emit("identity.exploration_view_restored", {"action_seq": self.context.action_seq})

    async def recover_completion(self, deadline):
        """A failed claim is a new observation problem, not a reusable belief.

        Refresh sensing once per physical revision; never move the robot or
        choose a corrective target on the model's behalf.
        """
        seq = self.context.action_seq
        first = self.rejected_completion_seq != seq
        if first:
            self.rejected_completion_seq = seq
            self.rejected_completion_count = 0
            self.completion_review_ticket = None
            self.completion_waited_seq = None
        self.rejected_completion_count += 1
        repair_count = 0
        if (self.completion_contract_repair and self.context.milestone
                and self.context.milestone.final_stage):
            repair_key = (self.context.milestone.generation, seq)
            repair_count = self.completion_contract_claim_counts.get(repair_key, 0) + 1
            self.completion_contract_claim_counts[repair_key] = repair_count
        if self.recovery_reasoning and self.context.milestone and self.context.milestone.final_stage:
            self.recovery_reasoning_scope = self.current_recovery_scope()
        prior_review_outcome = (self.context.local_state.get("completion_recovery") or {}).get("review_outcome") if not first else None
        self.context.local_state["completion_recovery"] = {
            "status": "claim_rejected", "after_action_seq": seq,
            "claims_without_new_action": repair_count or self.rejected_completion_count,
            "next_step": "Inspect the CURRENT target, destination and gripper. Diagnose missing task outcome, then choose a corrective action. Repeating complete without a changed outcome cannot succeed.",
        }
        if self.completion_rendezvous:
            self.context.local_state["completion_recovery"].update(
                completion_eligible=False,
                policy="A rejected outcome at this physical revision needs correction. Inspect, act within the contract, or explicitly wait/escalate; repeated complete is not useful progress.")
            if prior_review_outcome is not None:
                self.context.local_state["completion_recovery"]["review_outcome"] = prior_review_outcome
        self.context.remember("claim.rejected", self.context.local_state["completion_recovery"])
        if hasattr(self.model, "history"):
            self.model.history.clear()
            self.store.emit("fast.history_reset", {"reason": "rejected_completion_belief", "structured_state_preserved": True})
        if first:
            await self.coordinator.observe()
            self.store.emit("recovery.observation_refreshed", {"reason": "completion_rejected", "action_seq": seq})
            if self.slow and self.mode == "dual_async" and repair_count != 2:
                reason = "completion_rejected: inspect actual final object/destination geometry; propose a concrete correction, not another completion claim"
                request = (self.slow.request_completion_review
                           if getattr(self.slow, "urgent_completion_review", False)
                           else self.slow.request_review)
                self.completion_review_ticket = request(reason)
        elif self.rejected_completion_count == 2:
            if self.completion_rendezvous:
                await self.wait_completion_review(deadline)
            elif not self.completion_contract_repair:
                await self.seek_help("completion_rejected_twice: current target is not in the requested final relation; diagnose and correct it", deadline)
        if repair_count == 2:
            return await self.repair_completion_contract(deadline)

    async def repair_completion_contract(self, deadline):
        """One bounded foreground outcome reassessment after repeated rejection.

        No executor action or success verdict is manufactured. The existing
        planner may continue with corrective advice or revise remaining outcomes.
        """
        m = self.context.milestone
        if not (self.mode == 'dual_async' and self.async_completion_recovery
                and self.milestones and getattr(self.milestones, 'pipelined', False)
                and self.slow and m and m.final_stage):
            return None
        key = (m.generation, self.context.action_seq)
        scope = {'action_seq':self.context.action_seq, 'milestone_id':m.id,
                 'revision':m.revision, 'generation':m.generation}
        if self.coordinator.paused:
            self.store.emit('completion.contract_repair_skipped', {
                **scope, 'reason':'higher_priority_pause'})
            return 'paused'
        if self.coordinator.running or self.coordinator.unknown:
            self.store.emit('completion.contract_repair_skipped', {
                **scope, 'reason':'no_confirmed_idle_boundary'})
            return 'state_unknown'
        if self.coordinator.contract_review_pending or self.milestones.pending_reason:
            self.store.emit('completion.contract_repair_skipped', {
                **scope, 'reason':'higher_priority_contract_review'})
            return None
        if key in self.completion_repaired_actions:
            self.store.emit('completion.contract_repair_skipped', {
                **scope, 'reason':'already_attempted_at_physical_revision'})
            return None
        if deadline is not None and time.monotonic() >= deadline:
            self.store.emit('completion.contract_repair_skipped', {
                **scope, 'reason':'episode_deadline'})
            return 'wall_time_budget'
        # record_decision already consumed the current slot. Reserve at least
        # two future slots so a correction need not stop immediately after grasp.
        if self.milestones.remaining <= 1:
            self.store.emit('completion.contract_repair_skipped', {
                **scope, 'reason':'fewer_than_two_executor_decisions_remain'})
            return None
        self.completion_repaired_actions.add(key)
        started = time.monotonic()
        outcome = 'cancelled'
        delivered = False
        self.store.emit('completion.contract_repair_started', {
            **scope, 'claims_without_new_action':self.completion_contract_claim_counts.get(key, 0),
            'blocking':True, 'planner_role':'slow',
            'episode_decisions_remaining':self.milestones.remaining,
            'recoveries_before':self.milestones.recoveries,
            'max_recoveries':self.milestones.max_replans})
        try:
            result = await self.milestones.recover(
                'completion_rejected_twice: the official verifier rejected this '
                'unchanged physical outcome twice. Reassess the remaining mission '
                'outcomes and constraints from current evidence; give a substantive '
                'correction within the contract or revise remaining outcomes. '
                'An apparent placement or another completion claim is not success.',
                allow_resume=False)
            delivered = self.milestones.last_recovery_valid
            outcome = result or ('planner_response_valid' if delivered
                                 else 'planner_response_unavailable')
            return result
        except Exception:
            outcome = 'error'
            raise
        finally:
            self.store.emit('completion.contract_repair_finished', {
                **scope, 'current_action_seq':self.context.action_seq,
                'elapsed_s':time.monotonic()-started, 'outcome':outcome,
                'planner_response_valid':delivered,
                'recoveries_after':self.milestones.recoveries})

    def signature(self, name, args):
        resolved=dict(args)
        for key in ("point_id","contact_point_id"):
            point=getattr(self.coordinator.adapter,"points",{}).get(resolved.get(key))
            if point:
                resolved[key]=[round(float(x),2) for x in point["xyz_m"]]
        return json.dumps([name,resolved],sort_keys=True)

    def record_recovery_feedback(self, feedback):
        if feedback is None:
            return
        self.context.local_state["recovery_feedback"] = feedback
        self.context.remember("recovery.physical_feedback", feedback)

    def refresh_recovery_pose(self, *, corrective=False, exclude_key=None):
        last = self.context.local_state.get("recovery_feedback") or {}
        if (last.get("failure_kind") == "unresolved_held_object"
                and self.context.local_state.get("possible_hold") is False):
            self.context.local_state.pop("recovery_feedback", None)
            self.store.emit("recovery.precondition_resolved", {"failure_kind": "unresolved_held_object"})
        for key in self.progress_recovery.observe(self.context.latest.state, corrective=corrective, exclude_key=exclude_key):
            self.blocked_proposals.pop(key, None)
            last = self.context.local_state.get("recovery_feedback") or {}
            if last.get("failure_key") == key:
                self.context.local_state["recovery_feedback"] = {
                    **self.progress_recovery.failures[key],
                    "instruction": "Measured robot reposition changed the approach. Reinspect the target using current evidence; a fresh attempt is allowed, but the old failure is not proof of success."}
            self.store.emit("recovery.measured_reposition", {"failure_key": key})

    async def blocked_attempt(self, feedback, deadline):
        key = feedback["failure_key"]
        count = self.blocked_proposals.get(key, 0)+1
        self.blocked_proposals[key] = count
        self.record_recovery_feedback({**feedback, "blocked_proposals": count})
        self.store.emit("recovery.equivalent_attempt_blocked", {**feedback, "blocked_proposals": count})
        # Remove repeated assistant claims while keeping current world evidence,
        # failure memory and the authorized milestone in the structured context.
        if hasattr(self.model, "history"):
            self.model.history.clear()
        if count == 2 and self.slow:
            await self.seek_help("No measured progress and repeated blocked physical target: " + feedback["instruction"], deadline, blocking=True)
        if count >= 3:
            self.store.emit("recovery.stalled", {"failure_key": key,
                "reason": "three_equivalent_blocked_proposals_without_progress"})
            return "recovery_exhausted"
        return None

    async def seek_help(self, reason, deadline, *, blocking=False):
        seq=self.context.action_seq
        if self.help_counts.get(seq,0)>=2:
            self.context.remember("recovery.exhausted",{"action_seq":seq,"reason":"Two reviews without a new physical attempt; change the approach using existing advice"})
            return
        self.help_counts[seq]=self.help_counts.get(seq,0)+1
        self.context.remember("recovery.requested",{"reason":reason})
        if self.milestones:
            if getattr(self.milestones, "pipelined", False) and not blocking and not (
                    self.milestones.pending_reason or self.coordinator.paused):
                self.milestones.request_local_review(reason)
                return
            await self.milestones.recover(reason)
            return
        if not self.slow:return
        if self.mode=="dual_sync":await self.slow.review()
        else:
            remaining=135 if deadline is None else min(135,max(0,deadline-time.monotonic()))
            start=time.monotonic()
            received=await self.slow.wait_for_review(reason,remaining)
            self.store.emit("recovery.rendezvous",{"reason":reason,"review_finished":received,"elapsed_s":time.monotonic()-start})
            self.context.local_state["last_help"]={"review_delivered":received,"reason":reason}

    async def run(self, max_decisions, deadline=None):
        try:
            return await self._run(max_decisions, deadline)
        finally:
            if hasattr(self.identity_checker, 'discard'):
                await self.identity_checker.discard('executor_finished')

    async def _run(self, max_decisions, deadline=None):
        for step in range(max_decisions):
            if deadline is not None and time.monotonic() >= deadline:
                return "wall_time_budget"
            self.steps = step + 1
            if self.coordinator.unknown:
                return "state_unknown"
            if self.milestones:
                outcome = await self.milestones.before_decision(max_decisions - step)
                if outcome:
                    return outcome
                self.sync_contract()
            if self.coordinator.paused:
                return "paused"
            if getattr(self.coordinator.adapter, "terminal", False):
                return "environment_terminated"
            local=self.context.local_state
            help_key=(self.context.action_seq,local["readonly_streak"]//4)
            if (local["consecutive_failed_actions"]>=2 or local["readonly_streak"]>=4) and help_key!=self.last_help_key:
                self.last_help_key=help_key
                await self.seek_help("no_progress: inspect failed actions/target identity and CHANGE the method",deadline)
                self.sync_contract()
            if self.slow and step > 0 and step % self.review_every == 0:
                if self.mode == "dual_sync":
                    await self.slow.review()
                else:
                    self.slow.request_review("periodic_progress")
            if deadline is not None and time.monotonic()>=deadline:return "wall_time_budget"
            if (hasattr(self.identity_checker, 'prefetch_current')
                    and not getattr(self.milestones, 'checkpoint_task', None)):
                await self.identity_checker.prefetch_current()
            projection = self.context.projection("fast")
            diagnostic_ids = {h["id"] for h in projection.get("historical_diagnostics", [])}
            if diagnostic_ids - self.seen_diagnostics:
                if hasattr(self.model, "history"):
                    self.model.history.clear()
                self.store.emit("diagnostic.exposed", {"ids": sorted(diagnostic_ids - self.seen_diagnostics),
                    "action_seq": self.context.action_seq, "semantics": "historical hypotheses, not action authority"})
                self.seen_diagnostics.update(diagnostic_ids)
            advice_ids={a["id"] for a in projection["advice"] if a.get("id")}
            new_advice=advice_ids-self.seen_advice
            if new_advice:
                self.store.emit("advice.seen",{"ids":sorted(new_advice),"action_seq":self.context.action_seq})
                self.seen_advice.update(new_advice)
            projection["new_advice_ids"]=sorted(new_advice)
            based_advice=set(advice_ids)
            based_m, based_o = self.context.milestone, self.context.latest
            projection["decision_budget_remaining"] = max_decisions - step
            try:
                attempted_name, attempted_args = None, None
                physical_executor_evidence = None
                base_prompt = execution_base_prompt(rim_grasp_guidance=self.rim_grasp_guidance,
                                                     view_recovery_guidance=self.view_recovery_guidance)
                rim_guidance_applied = self.rim_grasp_guidance
                view_recovery_applied = self.view_recovery_guidance
                prompt = base_prompt
                if self.destination_checker:
                    prompt += '\nPLACEMENT EVIDENCE: local_state.destination_check is an independent semantic judgment of the actual measured placement anchor, not retention or task success. If mismatch/uncertain, identify the requested support/container in the current whole scene and measure it again. A different point ID at the same rejected location is not a changed destination. Robot structure, the held source object, and nearby distractors are not interchangeable with the requested destination. Use current agentview for destination anchors; the exact latest marker will be checked before placement.\n'
                final_protocol_applied = False
                if self.milestones:
                    from .milestone_prompts import FAST_CONTRACT, FAST_FINAL_CONTRACT
                    if self.final_stage_prompt and self.context.milestone.final_stage:
                        final_protocol_applied = True
                        prompt = FAST_FINAL_CONTRACT + base_prompt.replace(
                            "Maintain identity and complete the mission.",
                            "Maintain identity and achieve the complete original mission under the current final contract.")
                        rejected = self.rejected_completion_seq == self.context.action_seq
                        projection["final_completion_protocol"] = {
                            "phase":"final_mission_outcome",
                            "authority":"official_environment_verifier",
                            "rejected_at_current_physical_revision":rejected,
                            "action_seq":self.context.action_seq,
                            "claims_without_new_action":self.rejected_completion_count if rejected else 0,
                            "next_step":("Diagnose an unmet condition and correct or inspect it; explicitly wait/escalate if no useful action is justified."
                                         if rejected else "Complete requires the whole mission outcome, not an intermediate stage claim.")}
                    else:
                        prompt = FAST_CONTRACT + base_prompt.replace(
                            "Maintain identity and complete the mission.",
                            "Maintain identity and achieve ONLY the current milestone; propose stage completion before later outcomes.")
                    self.milestones.record_decision()
                if self.progress_recovery:
                    from .milestone_prompts import PROGRESS_RECOVERY
                    prompt += PROGRESS_RECOVERY
                if self.completion_rendezvous and self.rejected_completion_seq == self.context.action_seq:
                    prompt += "\nREJECTED FINAL OUTCOME: The official check rejected this physical revision. Do not repeat complete on unchanged state. Use current evidence to choose a corrective step within the contract, inspect a missing condition, or explicitly wait/escalate. A delivered review or identity match is not task success.\n"
                if diagnostic_ids:
                    from .milestone_prompts import DIAGNOSTIC_EXECUTION
                    prompt += DIAGNOSTIC_EXECUTION
                if getattr(self.milestones, "pipelined", False):
                    prompt += "\nSUPERVISION PIPELINE: Automatic no-progress review runs in the background; continue useful work inside the current contract. If local_state.pipeline.phase is checkpoint_pending, choose ONE read-only observation/measurement for the upcoming goal or wait; no physical action is authorized by the preview. Explicit escalate waits for help when no useful local step remains.\n"
                    if (self.milestones.checkpoint_task and not self.milestones.checkpoint_task.done()
                            and not self.milestones.preparation_used):
                        from embodied_harness.runtime.pipeline import build_scout_prompt
                        prompt = build_scout_prompt(parity=self.scout_prompt_parity,
                                                    rim_grasp_guidance=self.rim_grasp_guidance,
                                                    view_recovery_guidance=self.view_recovery_guidance) + "\nCHECKPOINT PREPARATION: The current stage is already under Slow assessment. Do not propose complete again. Focus on preparation_goal: obtain a useful read-only view or measurement for that preview, or wait if nothing useful remains. The preview never authorizes physical execution.\n"
                        final_protocol_applied = False
                        rim_guidance_applied = self.scout_prompt_parity and self.rim_grasp_guidance
                        view_recovery_applied = self.scout_prompt_parity and self.view_recovery_guidance
                        self.store.emit("pipeline.scout_prompt_prepared", {
                            "phase": "checkpoint", "parity": self.scout_prompt_parity,
                            "rim_grasp_guidance": rim_guidance_applied,
                            "view_recovery_guidance": view_recovery_applied,
                            "decision_slot": self.steps})
                        projection.pop("final_completion_protocol", None)
                        projection["preparation_goal"] = (self.context.local_state.get("pipeline") or {}).get("next_goal_preview")
                        self.store.emit("pipeline.preparation_inference_started", {
                            "next_goal_preview": projection["preparation_goal"], "decision_slot": self.steps})
                if final_protocol_applied:
                    self.store.emit("fast.final_stage_prompt", {
                        "milestone_id":self.context.milestone.id,
                        "revision":self.context.milestone.revision,
                        "action_seq":self.context.action_seq,
                        "rejected_at_current_physical_revision":projection["final_completion_protocol"]["rejected_at_current_physical_revision"]})
                if rim_guidance_applied:
                    from .grasp_guidance import POLICY_ID
                    self.store.emit("fast.rim_guidance_prepared", {
                        "policy_id": POLICY_ID, "decision_slot": self.steps,
                        "action_seq": self.context.action_seq})
                if view_recovery_applied:
                    from .view_recovery import POLICY_ID
                    self.store.emit("fast.view_recovery_prepared", {
                        "policy_id": POLICY_ID, "decision_slot": self.steps,
                        "action_seq": self.context.action_seq})
                reasoning = self.use_recovery_reasoning()
                if reasoning:
                    self.recovery_reasoning_calls += 1
                    self.store.emit("fast.recovery_reasoning_requested", {
                        "episode_request_index": self.recovery_reasoning_calls, "episode_limit": 4,
                        "scope": list(self.recovery_reasoning_scope),
                        "trigger": "official_final_completion_rejected",
                        "model": self.model.model,
                        "option_override": {"chat_template_kwargs": {"enable_thinking": True}},
                        "output_budget_changed": False})
                inference_options = {"recovery_reasoning": True} if reasoning else {}
                if self.use_completion_schema():
                    inference_options['exclude_completion'] = True
                    self.store.emit('fast.completion_schema_prepared', {
                        'scope': list(self.completion_excluded_scope), 'decision_slot': self.steps,
                        'observation_id': self.context.latest.id,
                        'allowed_kinds': ['tool', 'wait', 'escalate']})
                decision = self.normalize_decision(await self.decide_executor(prompt, projection,
                    self.context.latest.images, inference_options))
                if deadline is not None and time.monotonic() >= deadline:
                    return "wall_time_budget"
                kind = decision.get("kind")
                if hasattr(self.identity_checker, 'discard'):
                    pending = self.identity_checker.pending
                    args_for_prefetch = decision.get('args')
                    if not (kind == 'tool' and decision.get('tool') == 'grasp_point'
                            and isinstance(args_for_prefetch, dict) and pending
                            and args_for_prefetch.get('point_id') == pending['point_id']):
                        await self.identity_checker.discard('actor_chose_other_operation')
                self.store.emit("fast.decision", decision)
                assessment = decision.get("diagnostic_response")
                if (isinstance(assessment, dict) and isinstance(assessment.get("id"), str)
                        and assessment["id"] in diagnostic_ids
                        and isinstance(assessment.get("status"), str)
                        and assessment["status"] in {"corroborated", "contradicted", "unresolved"}
                        and isinstance(assessment.get("evidence"), str)):
                    self.store.emit("diagnostic.assessed", {**assessment,
                        "action_seq": self.context.action_seq, "observation_id": self.context.latest.id,
                        "semantics": "executor model claim, not independent verification"})
                self.context.remember("fast.intent", {"kind": kind, "reason": str(decision.get("reason", ""))[:600],
                                                     "tool":decision.get("tool"),"args":decision.get("args"),
                                                     "advice_response":decision.get("advice_response"),
                                                     "diagnostic_response":decision.get("diagnostic_response"),
                                                     "provenance": "model_claim_not_world_fact"})
                if getattr(self.milestones, "pipelined", False):
                    allowed, terminal = await self.milestones.guard_proposal(kind, decision.get("tool"))
                    if terminal:
                        return terminal
                    if not allowed:
                        continue
                if self.milestones and (self.milestones.pending_reason or self.coordinator.paused):
                    self.store.emit("fast.decision_stale", {"reason": "supervisor_requested_contract_review"})
                    continue
                if (based_m != self.context.milestone or based_o.id != self.context.latest.id):
                    self.store.emit("fast.decision_stale", {"observation_id": based_o.id, "milestone_revision": based_m.revision})
                    continue
                current_advice={a["id"] for a in self.context.projection("fast")["advice"] if a.get("id")}
                if current_advice-based_advice and not getattr(self.context, "defer_routine_advice", False):
                    self.store.emit("fast.decision_stale",{"reason":"new_supervisor_advice_during_inference"})
                    continue
                if advice_ids:
                    self.store.emit("advice.considered", {"ids": sorted(advice_ids), "decision_kind": kind,
                        "tool": decision.get("tool"), "response": decision.get("advice_response"),
                        "semantics": "exposure before a fresh decision; not proof of adoption or benefit"})
                if kind == "tool":
                    name, args = decision.get("tool"), self.tool_arguments(decision)
                    attempted_name, attempted_args = name, args
                    # Validation itself can reject a known unreliable depth
                    # site. Check accumulated evidence before that rejection,
                    # otherwise the recovery gate is unreachable on repeats.
                    if self.progress_recovery and name not in {"observe", "mark_point", "locate_point", "select_candidate"}:
                        self.refresh_recovery_pose()
                        blocked = self.progress_recovery.blocked(name, args,
                            getattr(self.coordinator.adapter, "points", {}), self.context.milestone.max_retries)
                        if blocked:
                            stop = await self.blocked_attempt(blocked, deadline)
                            if stop:
                                return stop
                            continue
                    self.coordinator.adapter.validate_args(name, args)
                    if name in {"observe", "mark_point","locate_point","select_candidate"}:
                        key=json.dumps([self.context.action_seq,name,{k:v for k,v in args.items() if k!="source_packet_id"}],sort_keys=True)
                        count=self.readonly_signatures.get(key,0)+1
                        self.readonly_signatures[key]=count
                        if count>2:
                            if count==3 and hasattr(self.model,'history'):
                                self.model.history.clear()
                                self.store.emit('fast.history_reset',{'reason':'repeated_readonly_claims','structured_state_preserved':True})
                            self.context.local_state["readonly_streak"]+=1
                            self.context.remember("local.readonly_loop",{"tool":name,"args":args,"reason":"Same observation/measurement already returned twice without intervening action; use it or choose a different target"})
                            await self.seek_help("repeated_readonly_no_progress",deadline)
                            continue
                        async with self.coordinator.lock:
                            before = self.context.latest
                            before_points = set(getattr(self.coordinator.adapter, "points", {}))
                            obs = await self.coordinator.adapter.inspect(name, args, self.context.action_seq)
                            if getattr(self.milestones, "pipelined", False):
                                self.milestones.note_readonly(before, obs)
                                self.milestones.note_preparation_points(before_points)
                            self.context.ingest(obs)
                        self.context.local_state["readonly_streak"]+=1
                        if (name=='select_candidate' and args.get('part')=='top_surface' and self.identity_checker
                                and not getattr(self.identity_checker, 'pregrasp_only', False)
                                and not (getattr(self.milestones, "pipelined", False) and self.milestones.checkpoint_task)):
                            point=(obs.state.get('feedback') or {}).get('point')
                            if point and not await self.identity_checker.check(point['point_id']):
                                await self.identity_recovery_view()
                    else:
                        m, o = based_m, based_o
                        if name=="grasp_point" and self.identity_checker:
                            if not await self.identity_checker.check(args["point_id"]):
                                await self.identity_recovery_view()
                                if self.slow and getattr(self.identity_checker, 'pregrasp_only', False):
                                    self.slow.request_review("target_binding_rejected: independently reassess the original mission against current visual evidence; revise mistaken inferred target descriptions if needed, while retaining user constraints")
                                continue
                            if deadline is not None and time.monotonic()>=deadline:return 'wall_time_budget'
                            o=self.context.latest
                            if (m!=self.context.milestone or (not getattr(self.context, "defer_routine_advice", False) and {a["id"] for a in self.context.projection("fast")["advice"] if a.get("id")}-based_advice)
                                    or (self.milestones and (self.milestones.pending_reason or self.coordinator.paused))):
                                self.store.emit("fast.decision_stale",{"reason":"supervisor_changed_during_identity_check"});continue
                        if name == 'place_point' and self.destination_checker:
                            if not await self.destination_checker.check(args['point_id']):
                                if hasattr(self.model, 'history'):
                                    self.model.history.clear()
                                await self.coordinator.observe()
                                self.store.emit('destination_identity.exploration_view_restored', {
                                    'action_seq':self.context.action_seq, 'physical_action_dispatched':False})
                                if self.slow:
                                    self.slow.request_review('destination_binding_rejected: reassess the requested destination from current public images; retain the original mission and leave tools/parameters to Fast')
                                continue
                            if deadline is not None and time.monotonic() >= deadline:
                                return 'wall_time_budget'
                            o = self.context.latest
                            if (m != self.context.milestone or self.coordinator.paused
                                    or (self.milestones and self.milestones.pending_reason)
                                    or (not getattr(self.context,'defer_routine_advice',False)
                                        and {a['id'] for a in self.context.projection('fast')['advice'] if a.get('id')}-based_advice)):
                                self.store.emit('fast.decision_stale', {'reason':'supervisor_changed_during_destination_check'})
                                continue
                        signature = self.signature(name,args)
                        if not self.progress_recovery and self.failed_signatures.get(signature,0)>=self.context.milestone.max_retries:
                            self.context.remember("local.retry_limit", {"tool": name,
                                "reason": "Repeated identical actions exhausted local retry allowance; inspect and choose a different method."})
                            await self.seek_help("Repeated same measured target/action failed twice; change site, orientation or skill",deadline)
                            continue
                        request = ActionRequest(uid(), m.id, m.revision, m.generation, o.id, name, args,
                            control_version=projection.get("control_version") if self.milestones else None)
                        receipt = await self.coordinator.submit(request)
                        if receipt.status == "state_unknown":
                            return "state_unknown"
                        if self.progress_recovery:
                            points = getattr(self.coordinator.adapter, "points", {})
                            self.refresh_recovery_pose()
                            feedback = self.progress_recovery.record(name, args, points, receipt.detail)
                            self.record_recovery_feedback(feedback)
                            physical_executor_evidence = (request, receipt, feedback, o.sim_time)
                            completed = self.progress_recovery.execution_completed(receipt.detail)
                            if completed:
                                self.refresh_recovery_pose(corrective=True,
                                    exclude_key=self.progress_recovery.key(name, args, points))
                            if completed or (feedback and feedback.get("measured_progress")):
                                key = self.progress_recovery.key(name, args, points)
                                self.blocked_proposals.pop(key, None)
                                if completed and (self.context.local_state.get("recovery_feedback") or {}).get("failure_key") == key:
                                    self.context.local_state.pop("recovery_feedback", None)
                        if self.context.local_state["consecutive_failed_actions"]:
                            self.failed_signatures[signature]=self.failed_signatures.get(signature,0)+1
                            if self.milestones and self.slow:
                                self.slow.request_review("execution_failure: reassess the stage outcome and unmet evidence without prescribing tool arguments")
                        else:self.failed_signatures.pop(signature,None)
                    # Evaluator-only check. Reward never appears in the model projection.
                    evidence = await self.coordinator.verify()
                    if evidence["status"] == "verified":
                        if self.milestones:
                            self.milestones.note_terminal(evidence)
                        return "success"
                    if getattr(self.coordinator.adapter, "terminal", False):
                        return "environment_terminated"
                    if physical_executor_evidence is not None:
                        self.note_executor_physical_failure(*physical_executor_evidence, evidence)
                elif kind == "complete":
                    result = await self.coordinator.verify()
                    if result["status"] == "verified":
                        if self.milestones:
                            self.milestones.note_terminal(result)
                        return "success"
                    self.note_completion_negative(result)
                    self.arm_recovery_executor(result)
                    if self.milestones:
                        if (self.async_completion_recovery
                                and getattr(self.milestones, "pipelined", False)
                                and self.context.milestone.final_stage):
                            # Final success is already decided by the official
                            # verifier above. A delayed negative checkpoint used
                            # to bypass recover_completion entirely. Deliver the
                            # rejection now and let Slow diagnose in background;
                            # no stage or motion authority is granted here.
                            self.false_claims += 1
                            self.context.local_state["readonly_streak"] += 1
                            self.store.emit("pipeline.final_completion_rejected", {
                                "milestone_id": self.context.milestone.id,
                                "action_seq": self.context.action_seq,
                                "observation_id": self.context.latest.id,
                                "official_status": result["status"],
                                "recovery": "refresh_evidence_and_background_review"})
                            recovery_outcome = await self.recover_completion(deadline)
                            if recovery_outcome:
                                return recovery_outcome
                            continue
                        outcome = await self.milestones.checkpoint()
                        if outcome in {"wall_time_budget", "decision_budget", "checkpoint_unavailable"}:
                            return outcome
                        if outcome in {"advanced", "replan", "checkpoint_pending"}:
                            continue
                        self.false_claims += 1
                        self.context.local_state["readonly_streak"] += 1
                        if self.context.milestone.final_stage:
                            recovery_outcome = await self.recover_completion(deadline)
                            if recovery_outcome:
                                return recovery_outcome
                        continue
                    self.false_claims += 1
                    self.context.local_state["readonly_streak"]+=1
                    recovery_outcome = await self.recover_completion(deadline)
                    if recovery_outcome:
                        return recovery_outcome
                elif kind == "escalate":
                    self.context.local_state["readonly_streak"]+=1
                    self.context.remember("fast.escalated", {"problem": str(decision.get("problem", ""))})
                    if self.slow:await self.seek_help(str(decision.get("problem","fast.escalated")),deadline,blocking=True)
                    else:
                        self.context.remember("escalation.unavailable", {"reason": "single-agent baseline; locally recover"})
                elif kind == "wait":
                    self.context.local_state["readonly_streak"]+=1
                    if self.slow and self.context.review_state["in_flight"]:
                        await self.seek_help("Fast intentionally waiting for supervisor",deadline,blocking=True)
                    else:await asyncio.sleep(1)
                else:
                    raise ValueError("unknown decision kind")
            except InferencePreempted:
                self.store.emit("fast.decision_preempted", {"action_seq": self.context.action_seq,
                    "control_version": self.context.control_version})
                continue
            except Exception as exc:
                if isinstance(exc, GateError) and str(exc) in {
                        "superseded_control_context", "contract_review_pending", "paused_requires_explicit_resume",
                        "target_judge_evidence_or_authority_changed", "target_judge_requires_current_idle_authority",
                        "target_judge_observation_not_current"}:
                    self.store.emit("fast.decision_preempted", {"reason": str(exc), "action_seq": self.context.action_seq})
                    continue
                if isinstance(exc, GraspOrientationError) and attempted_name == "grasp_point":
                    self.context.remember("recovery.tool_argument_rejected", {
                        **exc.diagnostic(), "tool": attempted_name,
                        "action_seq": self.context.action_seq,
                        "observation_id": based_o.id,
                        "generation": based_m.generation, "milestone_id": based_m.id,
                        "milestone_revision": based_m.revision,
                        "control_version": self.context.control_version,
                        "recorded_at": time.monotonic(),
                        "decision": self.steps, "physical_action_dispatched": False,
                        "semantics": "Historical rejected proposal; no grasp primitive was dispatched. Not current object state, a bad grasp-site diagnosis, or permission to act. Reassess current evidence before retrying."})
                if self.progress_recovery and attempted_name and attempted_args is not None:
                    feedback = self.progress_recovery.record_error(attempted_name, attempted_args,
                        getattr(self.coordinator.adapter, "points", {}), str(exc))
                    self.record_recovery_feedback(feedback)
                    if feedback:
                        # A rejected physical precondition is useful feedback,
                        # not six independent transport/schema failures. It
                        # still consumes the unchanged episode decision budget.
                        self.context.local_state["readonly_streak"] += 1
                        self.context.remember("recovery.validation_rejected", {
                            "tool": attempted_name, "args": attempted_args,
                            "message": str(exc)[:600], "failure_kind": feedback["failure_kind"],
                            "instruction": feedback["instruction"]})
                        self.store.emit("recovery.validation_rejected", {
                            "tool": attempted_name, "failure_kind": feedback["failure_kind"],
                            "action_seq": self.context.action_seq, "decision": self.steps})
                        if hasattr(self.model, "history"):
                            self.model.history.clear()
                        if self.slow:
                            self.slow.request_review("physical_precondition_rejected: " + feedback["instruction"])
                        continue
                self.errors += 1
                self.context.local_state["readonly_streak"]+=1
                self.context.remember("fast.error", {"type": type(exc).__name__, "message": str(exc)[:600]})
                if isinstance(exc, ValueError) and any(marker in str(exc) for marker in (
                        "Measure destination first", "Unknown grasp point", "Remeasure destination", "Retired measurement reference")):
                    if hasattr(self.model, "history"):
                        self.model.history.clear()
                    self.store.emit("fast.history_reset", {"reason": "rejected_measurement_reference",
                        "structured_state_preserved": True})
                if self.errors >= 6:
                    return "model_or_tool_errors"
        return "decision_budget"
