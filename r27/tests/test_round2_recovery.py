"""Regression coverage for recovery, bounded context, and asynchronous review."""
import asyncio
import time

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.events import EventStore


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})

    def since(self, cursor):
        return self.events[cursor:]


class Robot:
    generation = 1
    terminal = False

    def __init__(self):
        self.version = 0
        self.executions = 0
        self.points = {}
        self.schemas = {"move_to": {"parameters": {
            "properties": {"xyz_m": {"type": "array", "items": {"type": "number"}}}
        }}}

    def validate_args(self, capability, args):
        pass

    async def observe(self, action_seq):
        self.version += 1
        return Observation(str(self.version), self.version, self.generation,
                           time.monotonic(), 0.0, action_seq, {})

    async def execute(self, capability, args, fence, cancelled, publish):
        self.executions += 1
        return {"macro_status": "reached"}

    async def verify(self, observation_id):
        return {"status": "not_verified", "evidence": {}}


def test_semantic_action_signature_ignores_point_id_when_xyz_matches():
    store = MemoryStore()
    context = Context(store)
    robot = Robot()
    robot.points = {
        "point-a": {"xyz_m": [0.31, -0.12, 0.44]},
        "point-b": {"xyz_m": [0.31, -0.12, 0.44]},
    }
    agent = FastAgent(object(), context, type("CoordinatorStub", (), {"adapter": robot})(), store)

    a = agent.signature("move_to", {"point_id": "point-a", "offset_m": [0, 0, 0.02]})
    b = agent.signature("move_to", {"point_id": "point-b", "offset_m": [0, 0, 0.02]})

    assert a == b


def test_compaction_keeps_raw_substeps_durable_and_hides_schemas_from_projection(tmp_path):
    store = EventStore(tmp_path / "events.sqlite")
    context = Context(store)
    raw_substeps = [
        {"index": index, "command": f"servo-{index}", "measured_xyz": [index / 100, 0, 0]}
        for index in range(11)
    ]
    raw_schemas = {"move_to": {"parameters": {"type": "object", "properties": {"x": {"type": "number"}}}}}
    observation = Observation(
        "observation-1", 1, 1, time.monotonic(), 0.0, 0,
        {"substeps": raw_substeps, "tool_schemas": raw_schemas, "visible": "table"},
    )

    try:
        assert context.ingest(observation)
        projected = context.projection("fast")["observation"]
        durable = store.since(0)[0]["payload"]

        assert "tool_schemas" not in projected["state"]
        assert "substeps" not in projected["state"]
        assert context.latest is observation
        assert context.latest.state["substeps"] == raw_substeps
        assert durable["state"]["substeps"] == raw_substeps
        assert durable["state"]["tool_schemas"] == raw_schemas
    finally:
        store.close()


def review_setup():
    store = MemoryStore()
    context = Context(store)
    robot = Robot()
    coordinator = Coordinator(robot, context, store)
    coordinator.delegate(Milestone("mission", 1, "mission", "goal", capabilities=("move_to",)))
    context.ingest(Observation("review-observation", 1, 1, time.monotonic(), 0.0,
                               context.action_seq, {}))
    return store, context, robot, coordinator


def test_wait_for_review_rejects_older_same_action_sequence_and_honors_timeout():
    async def run():
        store, context, _, coordinator = review_setup()
        entered_old_review = asyncio.Event()
        release_old_review = asyncio.Event()

        class Model:
            async def decide(self, system, projection, images):
                entered_old_review.set()
                await release_old_review.wait()
                return {"kind": "noop"}

        slow = SlowAgent(Model(), context, coordinator, store)
        old_action_seq = context.action_seq
        old_task = asyncio.create_task(slow.review())
        await entered_old_review.wait()

        timeout = 0.08
        waiter = asyncio.create_task(slow.wait_for_review("later recovery", timeout=timeout))
        await slow.pending.wait()
        assert slow.request_ticket == 1
        assert slow.delivered_ticket == 0

        # This review began before ticket 1 was requested. Its snapshot uses the same
        # action sequence, but it cannot deliver the later request.
        release_old_review.set()
        await old_task
        assert slow.last_review_action == old_action_seq
        assert slow.completed == 1
        assert slow.delivered_ticket == 0
        await asyncio.sleep(0.01)
        assert not waiter.done(), "same-sequence review started earlier must not satisfy the new ticket"

        started = time.monotonic()
        received = await waiter
        elapsed = time.monotonic() - started
        assert received is False
        assert elapsed >= timeout * 0.65, "the waiter must remain pending until its timeout"
        assert slow.context.review_state["last_outcome"] == "noop"
        assert sum(event["kind"] == "slow.review_requested" for event in store.events) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["error", "cancelled"])
def test_failed_or_cancelled_review_does_not_deliver_ticket(failure):
    async def run():
        store, context, _, coordinator = review_setup()
        model_entered = asyncio.Event()

        class Model:
            async def decide(self, system, projection, images):
                model_entered.set()
                if failure == "error":
                    raise RuntimeError("synthetic review failure")
                await asyncio.Event().wait()

        slow = SlowAgent(Model(), context, coordinator, store)
        worker = asyncio.create_task(slow.run())
        waiter = asyncio.create_task(slow.wait_for_review("recover", timeout=0.06))
        await model_entered.wait()
        if failure == "cancelled":
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        else:
            await asyncio.sleep(0)
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

        assert slow.request_ticket == 1
        assert slow.delivered_ticket == 0
        assert context.review_state["last_outcome"] == failure
        assert await waiter is False, "an error or cancellation must not satisfy a review rendezvous"

    asyncio.run(run())


def test_strategic_advice_without_targets_preserves_existing_target_plan():
    async def run():
        store, context, _, coordinator = review_setup()
        existing_targets = [{"label": "cup", "point_id": "p1"}]
        context.target_plan = {"strategy": "initial strategy", "targets": existing_targets,
                               "provenance": "slow_visual_hypothesis"}

        class Model:
            async def decide(self, system, projection, images):
                return {"kind": "advice", "text": "approach from the side", "action_sensitive": False}

        slow = SlowAgent(Model(), context, coordinator, store)
        await slow.review()

        assert context.target_plan["strategy"] == "approach from the side"
        assert context.target_plan["targets"] is existing_targets
        assert slow.delivered_ticket == 0, "an unsolicited review must not invent a request ticket"

    asyncio.run(run())


def test_fast_successful_actions_continue_while_slow_review_is_pending():
    async def run():
        store = MemoryStore()
        context = Context(store)
        robot = Robot()
        coordinator = Coordinator(robot, context, store)
        coordinator.delegate(Milestone("mission", 1, "mission", "goal", capabilities=("move_to",)))
        await coordinator.observe()

        review_entered = asyncio.Event()
        release_review = asyncio.Event()

        class SlowModel:
            async def decide(self, system, projection, images):
                review_entered.set()
                await release_review.wait()
                return {"kind": "noop"}

        class FastModel:
            calls = 0

            async def decide(self, system, projection, images):
                self.calls += 1
                if self.calls == 2:
                    await asyncio.wait_for(review_entered.wait(), timeout=0.5)
                    assert context.review_state["in_flight"]
                return {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0, 0, 0.1]}}

        slow = SlowAgent(SlowModel(), context, coordinator, store)
        fast = FastAgent(FastModel(), context, coordinator, store, slow=slow,
                         mode="dual_async", review_every=1)
        slow_task = asyncio.create_task(slow.run())
        fast_task = asyncio.create_task(fast.run(max_decisions=2))
        try:
            try:
                result = await asyncio.wait_for(asyncio.shield(fast_task), timeout=1.0)
            except asyncio.TimeoutError:
                release_review.set()
                await fast_task
                raise AssertionError("Fast blocked on a pending Slow review")

            assert result == "decision_budget"
            assert review_entered.is_set()
            assert not release_review.is_set(), "the Slow model should still be pending during Fast execution"
            assert robot.executions == 2
        finally:
            release_review.set()
            slow_task.cancel()
            try:
                await slow_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())
