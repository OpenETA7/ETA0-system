"""Actor-loop regressions for opt-in asynchronous final-claim recovery."""
import asyncio
import copy
import time
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


class ScriptedFast:
    def __init__(self, decisions):
        self.decisions = deque(decisions)
        self.history = deque([{"role": "assistant", "content": "unsupported completion belief"}])
        self.histories_at_decision = []
        self.projections = []

    async def decide(self, _prompt, projection, _images):
        self.histories_at_decision.append(copy.deepcopy(list(self.history)))
        self.projections.append(copy.deepcopy(projection))
        decision = self.decisions.popleft()
        self.history.append({"role": "assistant", "content": decision["kind"]})
        return decision


class Adapter:
    generation = 1
    terminal = False

    def __init__(self):
        self.points = {}
        self.schemas = {"move_to": {"parameters": {"properties": {"xyz_m": {"type": "array"}}}}}

    def validate_args(self, name, args):
        assert name == "move_to"
        assert isinstance(args.get("xyz_m"), list)


class Coordinator:
    def __init__(self, context, store, verifications):
        self.context, self.store = context, store
        self.adapter = Adapter()
        self.paused = self.unknown = False
        self.lock = asyncio.Lock()
        self.verifications = deque(verifications)
        self.verify_calls = 0
        self.observe_count = 0
        self.submitted = []

    async def observe(self):
        self.observe_count += 1
        observation = Observation(
            f"obs-{self.observe_count}", self.observe_count, 1, time.monotonic(), 0.0,
            self.context.action_seq, {"frame": self.observe_count},
        )
        self.context.ingest(observation)
        return observation

    async def verify(self):
        self.verify_calls += 1
        return {"status": self.verifications.popleft(), "evidence": {"official": True}}

    async def submit(self, request):
        self.submitted.append(request)
        self.context.remember("action.accepted", {"request": asdict(request), "receipt": {}})
        self.context.action_seq += 1
        await self.observe()
        self.context.remember("action.receipt", {
            "status": "completed", "detail": {"macro_status": "reached"},
        })
        return SimpleNamespace(status="completed", detail={"macro_status": "reached"})


class Slow:
    def __init__(self):
        self.review_requests = []

    def request_review(self, reason):
        self.review_requests.append(reason)


class FakePipelinedManager:
    """Keep the actor on the same contract while counting checkpoint requests."""
    pipelined = True
    pending_reason = None

    def __init__(self, context, slow, checkpoint_result="checkpoint_pending"):
        self.context = context
        self.slow = slow
        self.checkpoint_result = checkpoint_result
        self.checkpoint_task = None
        self.preparation_used = False
        self.checkpoint_calls = []
        self.decision_count = 0
        self.terminal_evidence = []
        self.guard_calls = []
        self.local_reviews = []

    async def before_decision(self, _remaining):
        return None

    async def guard_proposal(self, kind, tool=None):
        self.guard_calls.append((kind, tool))
        return True, None

    def record_decision(self):
        self.decision_count += 1

    async def checkpoint(self, trigger="fast_completion_proposal"):
        self.checkpoint_calls.append(trigger)
        return self.checkpoint_result

    def note_terminal(self, evidence):
        self.terminal_evidence.append(copy.deepcopy(evidence))

    def request_local_review(self, reason):
        self.local_reviews.append(reason)
        self.slow.request_review("local_recovery: " + str(reason))


async def _agent(tmp_path, decisions, verifications, *, enabled, final_stage=True):
    store = MemoryStore()
    context = Context(store)
    context.milestone = Milestone(
        "final-stage" if final_stage else "intermediate-stage", 1,
        "move the object", "move the object", capabilities=("move_to",),
        final_stage=final_stage,
    )
    model = ScriptedFast(decisions)
    coordinator = Coordinator(context, store, verifications)
    await coordinator.observe()
    slow = Slow()
    manager = FakePipelinedManager(context, slow)
    fast = FastAgent(
        model, context, coordinator, store, slow=slow, mode="dual_async",
        async_completion_recovery=enabled,
    )
    fast.milestones = manager
    return store, context, model, coordinator, slow, manager, fast


def test_enabled_final_rejection_refreshes_and_queues_background_review_before_corrective_action(tmp_path):
    async def run():
        store, context, model, coordinator, slow, manager, fast = await _agent(
            tmp_path,
            [{"kind": "complete", "reason": "looks finished"},
             {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0.2, 0.1, 0.8]},
              "reason": "correct the placement"}],
            ["not_verified", "verified"], enabled=True,
        )

        assert await fast.run(2) == "success"
        assert fast.false_claims == 1
        assert manager.checkpoint_calls == []
        assert len(coordinator.submitted) == 1
        assert slow.review_requests == [
            "completion_rejected: inspect actual final object/destination geometry; propose a concrete correction, not another completion claim"
        ]
        assert model.histories_at_decision[1] == []
        assert model.projections[1]["observation"]["id"] == "obs-2"
        assert model.projections[1]["local_state"]["readonly_streak"] == 1
        assert coordinator.observe_count == 3  # initial, recovery refresh, post-action observation
        assert context.local_state.get("completion_recovery") is None  # action acceptance clears stale recovery
        events = store.since(0)
        assert sum(event["kind"] == "pipeline.final_completion_rejected" for event in events) == 1
        assert sum(event["kind"] == "fast.history_reset" for event in events) == 1
        assert sum(event["kind"] == "recovery.observation_refreshed" for event in events) == 1

    asyncio.run(run())


def test_disabled_final_rejection_keeps_checkpoint_pending_path(tmp_path):
    async def run():
        _store, _context, model, coordinator, slow, manager, fast = await _agent(
            tmp_path,
            [{"kind": "complete", "reason": "looks finished"},
             {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0.2, 0.1, 0.8]},
              "reason": "correct the placement"}],
            ["not_verified", "verified"], enabled=False,
        )

        assert await fast.run(2) == "success"
        assert manager.checkpoint_calls == ["fast_completion_proposal"]
        assert fast.false_claims == 0
        assert slow.review_requests == []
        assert model.histories_at_decision[1]  # no rejected-claim history reset
        assert coordinator.observe_count == 2  # initial observation and corrective action

    asyncio.run(run())


def test_enabled_intermediate_completion_still_uses_checkpoint(tmp_path):
    async def run():
        _store, _context, _model, _coordinator, slow, manager, fast = await _agent(
            tmp_path,
            [{"kind": "complete", "reason": "intermediate outcome may be ready"}],
            ["not_verified"], enabled=True, final_stage=False,
        )

        assert await fast.run(1) == "decision_budget"
        assert manager.checkpoint_calls == ["fast_completion_proposal"]
        assert fast.false_claims == 0
        assert slow.review_requests == []

    asyncio.run(run())


def test_repeated_final_rejections_refresh_once_then_request_background_help(tmp_path):
    async def run():
        store, context, model, coordinator, slow, manager, fast = await _agent(
            tmp_path,
            [{"kind": "complete", "reason": f"repeat {i}"} for i in range(3)],
            ["not_verified"] * 3, enabled=True,
        )

        assert await fast.run(3) == "decision_budget"
        assert fast.false_claims == 3
        assert context.local_state["completion_recovery"]["claims_without_new_action"] == 3
        assert coordinator.observe_count == 2  # one initial observation and one refresh for this action_seq
        assert manager.checkpoint_calls == []
        assert manager.local_reviews == [
            "completion_rejected_twice: current target is not in the requested final relation; diagnose and correct it"
        ]
        assert len(slow.review_requests) == 2
        assert slow.review_requests[0].startswith("completion_rejected:")
        assert slow.review_requests[1].startswith("local_recovery: completion_rejected_twice:")
        assert model.histories_at_decision == [
            [{"role": "assistant", "content": "unsupported completion belief"}], [], []
        ]
        events = store.since(0)
        assert sum(event["kind"] == "claim.rejected" for event in events) == 3
        assert sum(event["kind"] == "recovery.observation_refreshed" for event in events) == 1

    asyncio.run(run())


def test_official_final_success_returns_without_recovery_or_checkpoint(tmp_path):
    async def run():
        _store, _context, model, coordinator, slow, manager, fast = await _agent(
            tmp_path,
            [{"kind": "complete", "reason": "officially complete"}],
            ["verified"], enabled=True,
        )

        assert await fast.run(1) == "success"
        assert fast.false_claims == 0
        assert coordinator.verify_calls == 1
        assert manager.checkpoint_calls == []
        assert manager.terminal_evidence == [{"status": "verified", "evidence": {"official": True}}]
        assert slow.review_requests == []
        assert model.histories_at_decision == [[
            {"role": "assistant", "content": "unsupported completion belief"}
        ]]

    asyncio.run(run())
