"""Offline regressions for rejected completion claims and recovery boundaries."""
import asyncio
import time
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.events import EventStore


class Model:
    def __init__(self, decisions):
        self.decisions = deque(decisions)
        self.history = deque([{"role": "assistant", "content": "unsupported completion belief"}])
        self.calls = 0

    async def decide(self, _system, _projection, _images):
        self.calls += 1
        return self.decisions.popleft()


class Adapter:
    generation = 1
    terminal = False

    def __init__(self):
        self.points = {}
        self.schemas = {"move_to": {}}
        self.validated = []

    def validate_args(self, capability, args):
        self.validated.append((capability, args))


class Coordinator:
    def __init__(self, context, store):
        self.context, self.store = context, store
        self.adapter = Adapter()
        self.paused = self.unknown = False
        self.submitted = []
        self.observe_count = 0
        self.verify_count = 0
        self.lock = asyncio.Lock()

    async def observe(self):
        self.observe_count += 1
        version = self.observe_count + 1
        observation = Observation(
            f"obs-{version}", version, 1, time.monotonic(), 0.0,
            self.context.action_seq, {"frame": version},
        )
        self.context.ingest(observation)
        return observation

    async def verify(self):
        self.verify_count += 1
        return {"status": "not_verified", "evidence": {}}

    async def submit(self, request):
        self.submitted.append(request)
        self.context.remember("action.accepted", {"request": asdict(request), "receipt": {}})
        self.context.action_seq += 1
        await self.observe()
        self.context.remember("action.receipt", {
            "status": "completed", "detail": {"macro_status": "reached"},
        })
        return SimpleNamespace(status="completed")


class Slow:
    def __init__(self):
        self.review_requests = []
        self.review_waits = []

    def request_review(self, reason):
        self.review_requests.append(reason)

    async def wait_for_review(self, reason, timeout):
        self.review_waits.append((reason, timeout))
        return True


def setup_agent(tmp_path, decisions, *, mode="single_fast", slow=None, capabilities=()):
    store = EventStore(tmp_path / "events.sqlite")
    context = Context(store)
    context.milestone = Milestone("mission-id", 1, "move the object", "move the object",
                                  capabilities=tuple(capabilities))
    context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 0.0, 0, {"frame": 1}))
    coordinator = Coordinator(context, store)
    model = Model(decisions)
    agent = FastAgent(model, context, coordinator, store, slow=slow, mode=mode)
    return store, context, coordinator, model, agent


def test_rejected_completion_clears_model_belief_but_preserves_durable_events(tmp_path):
    async def run():
        store, context, coordinator, model, agent = setup_agent(
            tmp_path, [{"kind": "complete", "reason": "looks finished"}])
        try:
            assert await agent.run(1) == "decision_budget"
            events = store.since(0)

            assert not model.history
            assert [event["kind"] for event in events].count("fast.intent") == 1
            assert [event["kind"] for event in events].count("claim.rejected") == 1
            assert [event["kind"] for event in events].count("fast.history_reset") == 1
            assert [event["kind"] for event in events].count("recovery.observation_refreshed") == 1
            assert context.local_state["completion_recovery"]["after_action_seq"] == 0
            assert coordinator.observe_count == 1
            assert coordinator.submitted == []
            assert not any(event["kind"].startswith("action.") for event in events)
        finally:
            store.close()

    asyncio.run(run())


def test_dual_async_first_rejection_queues_review_without_blocking_and_second_claim_seeks_help(tmp_path):
    async def run():
        slow = Slow()
        store, context, coordinator, model, agent = setup_agent(
            tmp_path,
            [{"kind": "complete", "reason": "first unsupported claim"},
             {"kind": "complete", "reason": "repeated unsupported claim"}],
            mode="dual_async", slow=slow,
        )
        try:
            assert await agent.run(2) == "decision_budget"

            # The next actor decision happened after request_review returned. Only
            # the second rejected claim takes the wait-for-review rendezvous path.
            assert model.calls == 2
            assert len(slow.review_requests) == 1
            assert slow.review_requests[0].startswith("completion_rejected:")
            assert len(slow.review_waits) == 1
            assert slow.review_waits[0][0].startswith("completion_rejected_twice:")
            assert context.local_state["completion_recovery"]["claims_without_new_action"] == 2
            assert coordinator.observe_count == 1
            assert coordinator.submitted == []
            assert sum(event["kind"] == "recovery.observation_refreshed"
                       for event in store.since(0)) == 1
        finally:
            store.close()

    asyncio.run(run())


def test_action_acceptance_clears_old_recovery_and_new_action_seq_allows_one_more_refresh(tmp_path):
    async def run():
        store, context, coordinator, _model, agent = setup_agent(
            tmp_path,
            [{"kind": "complete", "reason": "first rejected claim"},
             {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0.2, 0.1, 0.8]}},
             {"kind": "complete", "reason": "claim after corrective action"}],
            capabilities=("move_to",),
        )
        try:
            assert await agent.run(3) == "decision_budget"

            refreshed = [event["payload"] for event in store.since(0)
                         if event["kind"] == "recovery.observation_refreshed"]
            claims = [event["payload"] for event in store.since(0)
                      if event["kind"] == "claim.rejected"]
            assert [item["action_seq"] for item in refreshed] == [0, 1]
            assert [item["after_action_seq"] for item in claims] == [0, 1]
            assert context.local_state["completion_recovery"]["after_action_seq"] == 1
            assert context.local_state["completion_recovery"]["claims_without_new_action"] == 1
            assert len(coordinator.submitted) == 1
            assert coordinator.submitted[0].capability == "move_to"
            assert coordinator.observe_count == 3  # two recovery reads and the post-action packet
        finally:
            store.close()

    asyncio.run(run())

