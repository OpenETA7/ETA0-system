import asyncio
import json
import time
from dataclasses import asdict, replace
from collections import deque
from embodied_harness.domain.models import Receipt, uid


class GateError(RuntimeError):
    pass


class Coordinator:
    """Single process simulator lease. Real robot controller fencing is NOT implemented."""
    def __init__(self, adapter, context, store):
        self.adapter, self.context, self.store = adapter, context, store
        self.lock = asyncio.Lock()
        self.receipts = {}
        self.fingerprints = {}
        self.fence = 0
        self.cancel = asyncio.Event()
        self.paused = False
        self.pause_epoch = 0
        self.unknown = False
        self.running = None
        self.seen_milestone_ids = set()
        self.mission_authority = None
        self.contract_review_pending = False
        self.contract_review_epoch = 0
        self.pause_requested_at = None
        # Never auto-replay actions after a crash. A new evaluation uses a new run directory.
        rows = store.since(0)
        if any(x["kind"] == "action.accepted" for x in rows):
            self.unknown = True

    def authorize_mission(self, mission, capabilities, constraints=()):
        """Called by the application from user input, never exposed as an agent tool."""
        if self.mission_authority is not None or self.context.milestone is not None:
            raise GateError("mission_authority_already_initialized")
        self.mission_authority = {"mission": mission, "capabilities": frozenset(capabilities),
                                  "resources": frozenset(("arm", "gripper")),
                                  "constraints": frozenset(constraints)}

    def delegate(self, milestone, expected_revision=None):
        if self.running:
            raise GateError("handoff_requires_action_boundary")
        current = self.context.milestone
        if milestone.id in self.seen_milestone_ids and (not current or milestone.id != current.id):
            raise GateError("retired_milestone_id")
        if current and expected_revision != current.revision:
            raise GateError("milestone_revision_conflict")
        expected_next = current.revision + 1 if current and current.id == milestone.id else 1
        if milestone.revision != expected_next:
            raise GateError("milestone_revision_must_advance")
        if self.mission_authority:
            authority = self.mission_authority
            if milestone.mission != authority["mission"]:
                raise GateError("mission_requires_user_authorization")
            if (not set(milestone.capabilities).issubset(authority["capabilities"])
                    or not set(milestone.resources).issubset(authority["resources"])
                    or not authority["constraints"].issubset(milestone.constraints)):
                raise GateError("milestone_exceeds_mission_authority")
        elif current:
            if milestone.mission != current.mission:
                raise GateError("mission_requires_user_authorization")
            if (not set(milestone.capabilities).issubset(current.capabilities)
                    or not set(milestone.resources).issubset(current.resources)
                    or not set(current.constraints).issubset(milestone.constraints)):
                raise GateError("milestone_cannot_expand_authority")
        if milestone.generation != self.adapter.generation:
            raise GateError("stale_environment")
        if milestone.success_predicate not in {"environment_success", "stage_conditions"}:
            raise GateError("unsupported_success_predicate")
        if milestone.success_predicate == "stage_conditions" and (not self.mission_authority or not milestone.success_conditions):
            raise GateError("stage_contract_requires_conditions_and_mission_authority")
        self.context.milestone = milestone
        self.seen_milestone_ids.add(milestone.id)
        self.context.remember("milestone.accepted", asdict(milestone))
        self.context.signal_control("contract", 80, milestone.id)

    async def observe(self):
        # Public sensing cannot race a physics mutation. Controller publishes intermediate samples itself.
        async with self.lock:
            obs = await self.adapter.observe(self.context.action_seq)
            self.context.ingest(obs)
            return obs

    def validate(self, req):
        m, o = self.context.milestone, self.context.latest
        if self.unknown:
            raise GateError("physical_state_unknown")
        if self.paused:
            raise GateError("paused_requires_explicit_resume")
        if self.contract_review_pending:
            raise GateError("contract_review_pending")
        if req.control_version is not None and req.control_version != self.context.control_version:
            raise GateError("superseded_control_context")
        if not m or req.milestone_id != m.id or req.revision != m.revision:
            raise GateError("inactive_or_stale_milestone")
        if req.generation != self.adapter.generation or req.generation != m.generation:
            raise GateError("stale_environment")
        if req.capability not in m.capabilities or not set(req.resources).issubset(m.resources):
            raise GateError("capability_or_resource_denied")
        if not {"arm", "gripper"}.issubset(req.resources):
            raise GateError("adapter_requires_both_resources")
        from .predicates import evaluate
        for guard in m.guards:
            if req.capability in guard["tools"] and evaluate(guard["predicate"], self.context)["status"] != "satisfied":
                raise GateError("milestone_guard_not_satisfied:" + guard["predicate"])
        if time.monotonic() > req.created_at + req.ttl:
            raise GateError("expired_request")
        if not o or o.id != req.observation_id or o.generation != req.generation or o.action_seq != self.context.action_seq:
            raise GateError("fresh_observation_required")

    async def submit(self, req):
        fingerprint = json.dumps(asdict(req), sort_keys=True)
        async with self.lock:
            if req.request_id in self.receipts:
                if self.fingerprints[req.request_id] != fingerprint:
                    raise GateError("request_id_collision")
                return self.receipts[req.request_id]
            self.validate(req)
            if hasattr(self.adapter, "validate_args"):
                try:
                    self.adapter.validate_args(req.capability, req.args)
                except Exception as exc:
                    raise GateError("invalid_tool_arguments") from exc
            self.fence += 1
            receipt = Receipt(uid(), req.request_id, "accepted", self.fence)
            self.receipts[req.request_id] = receipt
            self.fingerprints[req.request_id] = fingerprint
            self.running = receipt
            self.cancel.clear()
            self.context.remember("action.accepted", {"request": asdict(req), "receipt": asdict(receipt)})
            self.context.action_seq += 1
            receipt.status = "running"
            self.context.action_id = receipt.action_id
            self.context.action_phase = receipt.status
            try:
                detail = await self.adapter.execute(req.capability, req.args, self.fence,
                                                    self.cancel.is_set, self.context.ingest)
                receipt.detail = detail
                self.context.local_state["possible_hold"]=getattr(self.adapter,"possible_hold",None)
                if detail.get("state_unknown"):
                    raise RuntimeError("ETA returned uncertain physical execution state")
                receipt.status = "stop_confirmed" if detail.get("stop_confirmed") else "completed"
                # Some backends return a fresh post-action sensor packet with the command receipt.
                # Reuse only that exact published packet; otherwise explicitly refresh.
                obs = self.context.latest
                if not (obs and detail.get("observation_id") == obs.id and
                        obs.action_seq == self.context.action_seq and obs.generation == req.generation):
                    obs = await self.adapter.observe(self.context.action_seq)
                    self.context.ingest(obs)
                receipt.observation_id = obs.id
            except BaseException as exc:
                # Timeout/coroutine cancellation is not controller stop confirmation.
                receipt.status = "state_unknown"
                receipt.detail = {"error": type(exc).__name__, "message": str(exc)[:300]}
                self.unknown = True
                if isinstance(exc, asyncio.CancelledError):
                    raise
            finally:
                self.context.remember("action.receipt", asdict(receipt))
                if self.cancel.is_set() and self.pause_requested_at is not None:
                    observed = receipt.detail.get("controller_cancel_observed_at")
                    self.store.emit("control.action_returned_after_pause", {
                        "action_id": receipt.action_id, "status": receipt.status,
                        "stop_confirmed": receipt.status == "stop_confirmed",
                        "latency_s": time.monotonic() - self.pause_requested_at,
                        "controller_observed_latency_s": observed - self.pause_requested_at if observed is not None else None})
                self.running = None
                self.context.action_id = None
                self.context.action_phase = "idle" if not self.unknown else "state_unknown"
            return receipt

    def pause(self, reason):
        self.pause_epoch += 1
        if not self.paused:
            self.pause_requested_at = time.monotonic()
        self.paused = True
        self.cancel.set()
        self.context.signal_control("pause", 100)
        self.store.emit("control.pause_requested", {"reason": reason, "action_id": self.running.action_id if self.running else None})
        if self.running:
            self.running.status = "cancel_requested"
            self.context.action_phase = "cancel_requested"

    async def resume(self):
        if self.running or self.unknown:
            raise GateError("resume_requires_confirmed_boundary")
        epoch = self.pause_epoch
        await self.observe()
        if epoch != self.pause_epoch:
            raise GateError("newer_pause_received")
        self.paused = False
        self.pause_requested_at = None
        self.cancel.clear()
        self.context.advice = deque((a for a in self.context.advice if a["kind"] != "pause"), maxlen=4)
        self.store.emit("control.resumed", {})

    def apply_advice(self, advice):
        from embodied_harness.domain.models import StrategicAdvice
        if isinstance(advice, StrategicAdvice):
            from .advice_delivery import strategy_invalid_reason
            reason = strategy_invalid_reason(asdict(advice), self.context, time.monotonic())
            if advice.generation != self.adapter.generation:
                reason = "generation_changed"
            if reason or not getattr(self.context, "defer_routine_advice", False):
                self.store.emit("advice.strategy_rejected", {
                    "advice_id": advice.id, "reason": reason or "requires_pipeline_boundary_delivery"})
                return False
            advice = replace(advice, priority=min(50, max(0, advice.priority)))
            record = asdict(advice)
            existing = self.context.strategic_advice.get(advice.id)
            if existing is not None:
                if existing != record:
                    self.store.emit("advice.strategy_rejected", {
                        "advice_id": advice.id, "reason": "message_id_collision"})
                    return False
                return True
            self.context.strategic_advice[advice.id] = record
            self.store.emit("advice.accepted", record)
            self.store.emit("advice.strategy_delivered", {"advice_id": advice.id,
                "reviewed_action_seq": advice.review_basis["action_seq"],
                "current_action_seq": self.context.action_seq,
                "reviewed_observation_id": advice.review_basis["observation_id"],
                "ticket": advice.review_basis["ticket"],
                "delivery": "next_decision_and_subsequent_episode_history_without_interrupt"})
            return True
        m = self.context.milestone
        phase = self.running.status if self.running else "idle"
        action_id = self.running.action_id if self.running else None
        valid = (m and advice.milestone_id == m.id and advice.revision == m.revision
                 and advice.generation == self.adapter.generation and time.monotonic() < advice.expires_at
                 and (advice.expected_action_seq is None or advice.expected_action_seq == self.context.action_seq))
        if advice.expected_action_phase is not None:
            valid = valid and advice.expected_action_phase == phase and advice.expected_action_id == action_id
        if advice.expected_visual_scope is not None:
            valid = valid and self.context.latest and advice.expected_visual_scope == self.context.latest.state.get('candidate_scope')
        if not valid:
            self.store.emit("advice.stale", asdict(advice))
            return False
        advice = replace(advice, priority=100 if advice.kind == "pause" else min(50, max(0, advice.priority)))
        live = [a for a in self.context.advice if a["expires_at"] > time.monotonic()
                and a["milestone_id"] == m.id and a["revision"] == m.revision]
        self.context.advice = deque(sorted([*live, asdict(advice)],
            key=lambda a: (a.get("priority", 50), a["expires_at"]), reverse=True)[:4], maxlen=4)
        self.store.emit("advice.accepted", asdict(advice))
        if advice.kind == "pause":
            self.pause(advice.text)
        elif getattr(self.context, "defer_routine_advice", False):
            self.store.emit("advice.deferred_to_boundary", {"advice_id": advice.id,
                "scope": "remains subject to action and visual freshness; may expire before the next decision"})
        else:
            self.context.signal_control("advice", advice.priority, advice.id)
        return True

    async def verify(self):
        async with self.lock:
            if self.unknown or not self.context.latest:
                return {"status": "uncertain", "source": "host"}
            evidence = await self.adapter.verify(self.context.latest.id)
            self.store.emit("verification.result", evidence)
            return evidence
