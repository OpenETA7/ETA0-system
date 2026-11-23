from collections import deque
from dataclasses import asdict
import asyncio
import time


def compact(value, depth=0):
    """Bound the model view; the durable event retains the complete original payload."""
    if depth > 7:
        return "[details retained in event log]"
    if isinstance(value, dict):
        projected={k:compact(v,depth+1) for k,v in value.items()
                if k not in {"tool_schemas","images","substeps","request_id","created_at","fence"}}
        # Raw receipts use false for "not proven", not an observed failed grasp.
        # Preserve the raw event, but avoid presenting epistemic unknown as falsity.
        for name in ('retention','placement'):
            if projected.get(name+'_proven') is False:
                del projected[name+'_proven']
                projected[name+'_verification']='unverified; inspect contact continuity and images, not proof of failure'
        return projected
    if isinstance(value, (list,tuple)):
        return [compact(v,depth+1) for v in value[-24:]]
    if isinstance(value,str):return value[:1200]
    return value


def supervisor_evidence(value):
    """Keep outcomes/provenance, without exposing the executor's calling interface.

    Historical operation names and measured poses are evidence, not callable tools.
    Tool arguments, catalogs and tool-specific permission/guard configuration stay
    with Fast and the coordinator. This makes a detached copy of the shared state.
    """
    if isinstance(value, dict):
        return {key: supervisor_evidence(item) for key, item in value.items()
                if key not in {"tool_schemas", "capabilities", "allowed_tools", "mission_capabilities",
                               "args", "parameters", "inputSchema", "tools", "guards",
                               "fundamental_guards", "coordinate_rule", "repair"}}
    if isinstance(value, (list, tuple)):
        return [supervisor_evidence(item) for item in value]
    return value


class Context:
    def __init__(self, store, sensory_capacity=24, *, phase_aware_grasp_feedback=False):
        self.store = store
        self.phase_aware_grasp_feedback = phase_aware_grasp_feedback
        self.frames = deque(maxlen=sensory_capacity)
        self.history = deque(maxlen=12)
        self.advice = deque(maxlen=4)
        # Task-local message history, separate from executable/operational advice.
        self.strategic_advice = {}
        self.diagnostic_hypotheses = deque(maxlen=4)
        self.latest = None
        self.latest_visual = None
        self.milestone = None
        self.mission_plan = None
        self.action_seq = 0
        self.action_id = None
        self.action_phase = "idle"
        self.local_state = {"last_action":None,"consecutive_failed_actions":0,"readonly_streak":0}
        self.target_plan = {}
        self.review_state = {"pending":False,"in_flight":False,"completed":0}
        self.control_version = 0
        self.control_changed = asyncio.Event()
        # Coalesce each priority independently so routine traffic cannot evict a stop.
        self.control_notices = deque(maxlen=101)
        self.stage_evidence = {}
        self.stage_evidence_id = None

    def begin_stage_evidence(self, milestone_id):
        self.stage_evidence_id = milestone_id
        self.stage_evidence.clear()

    def record_stage_evidence(self, *, receipt=False):
        """Record public facts only. A fact's occurrence never proves current hold."""
        if not self.milestone or self.milestone.id != self.stage_evidence_id:
            return
        from .predicates import SENSOR_PREDICATES, evaluate
        predicates = ("grasp_lift_receipt",) if receipt else tuple(
            p for p in SENSOR_PREDICATES if p != "grasp_lift_receipt")
        for predicate in predicates:
            fact = evaluate(predicate, self)
            if fact["status"] == "satisfied":
                self.stage_evidence[predicate] = {**fact, "milestone_id": self.stage_evidence_id,
                    "semantics": "occurred_during_stage_not_a_current_state_claim"}

    def signal_control(self, kind, priority, source_id=None):
        self.expire_target_judgment()
        priority = max(0, min(100, int(priority)))
        self.control_version += 1
        notice = {"version": self.control_version, "kind": kind, "priority": priority,
                  "source_id": source_id, "received_monotonic": time.monotonic(),
                  "action_seq": self.action_seq}
        self.control_notices = deque((n for n in self.control_notices if n["priority"] != priority), maxlen=101)
        self.control_notices.append(notice)
        self.control_changed.set()
        self.store.emit("control.message_received", notice)
        return notice

    def expire_target_judgment(self):
        self.local_state.pop('target_identity_check', None)
        if (self.local_state.get('identity_candidates') or {}).get('judgment_role') == 'target_judge':
            self.local_state.pop('identity_candidates', None)

    async def wait_for_control(self, after_version):
        while self.control_version <= after_version:
            self.control_changed.clear()
            if self.control_version > after_version:
                break
            await self.control_changed.wait()
        return max((n for n in self.control_notices if n["version"] > after_version),
                   key=lambda n: (n["priority"], n["version"]))

    def ingest(self, observation):
        if self.latest and (observation.generation < self.latest.generation or
                           (observation.generation == self.latest.generation and observation.version <= self.latest.version)):
            return False
        if self.latest and (self.latest.generation != observation.generation
                            or self.latest.action_seq != observation.action_seq):
            self.expire_target_judgment()
        if self.latest and self.latest.state.get('candidate_scope') != observation.state.get('candidate_scope'):
            self.expire_target_judgment()
            self.local_state.pop('identity_candidates', None)
            self.local_state['visual_binding_note']='Observation/proposal version changed. Previous region IDs and pixels may be stale. Identify the target again in CURRENT cards and preserve its appearance/relations, not its old numeric ID.'
        self.latest = observation
        if observation.images:
            self.latest_visual = observation
        self.frames.append(observation)
        self.record_stage_evidence()
        # Frames stay in a bounded buffer/artifact files; durable log holds references, not image bytes.
        self.store.emit("sensor.observed", asdict(observation))
        return True

    def projection(self, role):
        now = time.monotonic()
        destination = self.local_state.get('destination_check')
        if destination:
            m = self.milestone
            current_scope = ({'milestone_id':m.id, 'revision':m.revision, 'generation':m.generation,
                'action_seq':self.action_seq, 'control_version':self.control_version,
                'visual_scope':self.latest.state.get('candidate_scope') if self.latest else None} if m else None)
            if destination.get('scope') != current_scope or current_scope is None:
                self.local_state.pop('destination_check', None)
                self.store.emit('destination_identity.expired', {'point_id':destination.get('point_id'),
                    'reason':'action_control_contract_or_visual_scope_changed'})
        self.diagnostic_hypotheses = deque((h for h in self.diagnostic_hypotheses
            if self.milestone and h["milestone_id"] == self.milestone.id
            and h["revision"] == self.milestone.revision
            and h["generation"] == self.milestone.generation
            and h["expires_at"] > now
            and 0 <= self.action_seq - h["observed_action_seq"] <= 4), maxlen=4)
        self.advice = deque((a for a in self.advice if a["expires_at"] > time.monotonic()
                            and self.milestone and a["milestone_id"] == self.milestone.id
                            and a["revision"] == self.milestone.revision
                            and (a.get("expected_visual_scope") is None or a["expected_visual_scope"] == (self.latest.state.get("candidate_scope") if self.latest else None))
                            and (a.get("expected_action_seq") is None or a["expected_action_seq"] == self.action_seq)
                            and (a.get("expected_action_phase") is None or
                                 (a["expected_action_phase"] == self.action_phase and a.get("expected_action_id") == self.action_id))), maxlen=4)
        plan=dict(self.target_plan)
        if plan.get('visual_scope') is not None and (not self.latest or plan['visual_scope']!=self.latest.state.get('candidate_scope')):
            plan={'strategy':'Prior frame-specific grounding expired; use current visual evidence. Original proposal is retained in the event log.','targets':[],'grounding_expired':True}
        projection = {
            "milestone": asdict(self.milestone) if self.milestone else None,
            "mission_plan": compact(self.mission_plan),
            "observation": compact(asdict(self.latest)) if self.latest else None,
            "recent_events": list(self.history)[-6 if role == "fast" else -12:],
            "advice": list(self.advice),
            "local_state":self.local_state,
            "target_plan":plan,
            "review_state":self.review_state,
            "active_action":{"id":self.action_id,"phase":self.action_phase,"sequence":self.action_seq},
            "control_version": self.control_version,
            "control_policy": "Pause and contract changes outrank local decisions. Advice is sorted by priority then recency; assess it against current evidence and explain any rejection.",
            "memory_note": "History is bounded; raw receipts and image references remain in SQLite/artifacts.",
            "stage_evidence": list(self.stage_evidence.values()),
        }
        if self.strategic_advice:
            from .advice_delivery import strategy_invalid_reason, strategy_projection
            # Contract/control changes affect relevance, not retention. The
            # episode/environment identity and provenance are still validated.
            for ident, message in list(self.strategic_advice.items()):
                reason = strategy_invalid_reason(message, self, now)
                if reason:
                    self.strategic_advice.pop(ident)
                    self.store.emit("advice.strategy_invalid", {"advice_id": ident, "reason": reason})
            # Stable sort preserves arrival order when publication times tie.
            messages = sorted(self.strategic_advice.values(),
                              key=lambda a: a["review_basis"]["published_at"])
            projection["advice"] += [strategy_projection(a, self, now) for a in messages]
            projection["control_policy"] = (
                "Current pause and contract govern execution. Operational advice is priority-sorted; "
                "Slow historical messages follow in publication order. Use their source stage and "
                "timestamps to judge relevance; old messages are not current facts or authority.")
        if self.diagnostic_hypotheses:
            projection["historical_diagnostics"] = [
                {**h, "age_actions": self.action_seq - h["observed_action_seq"],
                 "age_s": max(0, now - h["observed_at"])} for h in self.diagnostic_hypotheses]
        # Preserve the durable raw observation, but do not present a retired
        # next-goal measurement as an executable current binding to either model.
        invalidated = self.local_state.get("preparation_invalidated") or {}
        state = (projection.get("observation") or {}).get("state") or {}
        feedback = state.get("feedback") or {}
        point = feedback.get("point") or {}
        retired = set(self.local_state.get("retired_point_ids", []))
        retired.update(invalidated.get("point_ids", []))
        if point.get("point_id") in retired:
            feedback.pop("point", None)
            feedback["retired_measurement"] = {
                "point_id": point["point_id"], "reason": "unapproved_next_goal_preparation",
                "instruction": "Obtain a new measurement under the current contract before use."}
        return supervisor_evidence(projection) if role == "slow" else projection

    def remember(self, kind, detail):
        item = {"kind": kind, "detail": compact(detail)}
        if kind == "recovery.tool_argument_rejected":
            # Read-only observations must not erase a precise parameter error.
            # Keep one bounded historical record; it never authorizes execution.
            self.local_state["last_tool_argument_error"] = compact(detail)
        if kind=="action.accepted":
            self.expire_target_judgment()
            self.local_state.pop("completion_recovery", None)
            req=detail["request"]
            self.local_state["last_action"]={"tool":req["capability"],"args":req["args"],"status":"running"}
            self.local_state["readonly_streak"]=0
        elif kind=="action.receipt":
            result=detail.get("detail",{});feedback=result.get("feedback") or {}
            motion=feedback.get("motion") or {}
            status=result.get("macro_status",motion.get("execution_state") if isinstance(motion,dict) else None)
            last=self.local_state.get("last_action") or {}
            projected_feedback = compact(feedback)
            if self.phase_aware_grasp_feedback:
                from .phase_feedback import grasp_phase_evidence
                phase_evidence = grasp_phase_evidence(result)
                if phase_evidence is not None:
                    projected_feedback["grasp_phase_evidence"] = phase_evidence
                    item["detail"]["detail"]["feedback"]["grasp_phase_evidence"] = phase_evidence
                    self.store.emit("recovery.phase_evidence_projected", {
                        "action_seq": self.action_seq, "action_id": detail.get("action_id"),
                        "request_id": detail.get("request_id"),
                        "evidence": phase_evidence})
            last.update(status=status or detail["status"],feedback=projected_feedback,host_error=result.get("host_error"))
            self.local_state["last_action"]=last
            good=status in {"reached","completed","lift_completed_verify_retention_visually","released_verify_placement_visually","pull_completed_verify_visually"}
            self.local_state["consecutive_failed_actions"]=0 if good else self.local_state["consecutive_failed_actions"]+1
            if detail.get("status") == "completed" and result.get("host_error") is None:
                self.record_stage_evidence(receipt=True)
        self.history.append(item)
        self.store.emit(kind, detail)
