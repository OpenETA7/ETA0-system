import asyncio
import time
from types import SimpleNamespace

from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, detail):
        self.events.append((kind, detail))

    def since(self, cursor):
        return []


def test_model_cannot_mark_old_holding_state_as_action_independent_advice():
    async def run():
        store = Store()
        context = Context(store)
        coordinator = Coordinator(SimpleNamespace(generation=1), context, store)
        coordinator.delegate(Milestone("m", 1, "put bowl on stove", "place the bowl"))
        context.ingest(Observation("o", 1, 1, time.monotonic(), 0, 0, {}))
        before = context.control_version
        entered, release = asyncio.Event(), asyncio.Event()

        class Model:
            async def decide(self, *args):
                entered.set()
                await release.wait()
                return {"kind": "advice", "action_sensitive": False,
                        "text": "The bowl is still held. Continue placing the held bowl.",
                        "semantic_advice": "The bowl is still held. Continue placing the held bowl."}

        slow = SlowAgent(Model(), context, coordinator, store)
        slow.milestones = SimpleNamespace(stamp=lambda: ("m", context.action_seq))
        task = asyncio.create_task(slow.review())
        await entered.wait()
        context.action_seq += 1  # Fast released the bowl while Slow was reasoning.
        release.set()
        await task
        assert not context.advice
        assert context.control_version == before
        assert any(kind == "advice.stale" for kind, _ in store.events)

    asyncio.run(run())


def test_fresh_same_boundary_advice_can_still_preempt_fast():
    async def run():
        store = Store()
        context = Context(store)
        coordinator = Coordinator(SimpleNamespace(generation=1), context, store)
        coordinator.delegate(Milestone("m", 1, "mission", "outcome"))
        context.ingest(Observation("o", 1, 1, time.monotonic(), 0, 0, {}))
        before = context.control_version

        class Model:
            async def decide(self, *args):
                return {"kind": "advice", "action_sensitive": False,
                        "text": "The current grasp has no contact; reassess the local geometry."}

        slow = SlowAgent(Model(), context, coordinator, store)
        slow.milestones = SimpleNamespace(stamp=lambda: ("m", context.action_seq))
        await slow.review()
        assert context.control_version == before + 1
        assert context.advice[0]["expected_action_seq"] == 0
        assert context.advice[0]["expected_action_phase"] == "idle"
        assert not context.target_plan  # No unscoped second copy of the same prose.
        context.action_seq += 1
        projection = context.projection("fast")
        assert projection["advice"] == []
        assert not projection["target_plan"]

    asyncio.run(run())


def test_checkpoint_correction_expires_after_a_new_physical_action():
    from embodied_harness.runtime.milestones import MilestoneManager
    store = Store()
    context = Context(store)
    coordinator = Coordinator(SimpleNamespace(generation=1), context, store)
    coordinator.delegate(Milestone("m", 1, "mission", "outcome"))
    context.ingest(Observation("o", 1, 1, time.monotonic(), 0, 0, {}))
    manager = object.__new__(MilestoneManager)
    manager.context, manager.coordinator = context, coordinator
    manager.advise("The bowl remains held; reassess the destination.")
    assert len(context.projection("fast")["advice"]) == 1
    context.action_seq += 1
    assert context.projection("fast")["advice"] == []


def test_identity_channel_cannot_persist_dynamic_prose_through_action_change():
    store = Store()
    context = Context(store)
    coordinator = Coordinator(SimpleNamespace(generation=1), context, store)
    milestone = Milestone("m", 1, "mission", "outcome")
    coordinator.delegate(milestone)
    context.ingest(Observation("o", 1, 1, time.monotonic(), 0, 0, {}))
    slow = SlowAgent(None, context, coordinator, store)
    slow.milestones = object()
    decision = {"semantic_advice": "The bowl is still held. Continue placing it."}
    assert not slow.publish_semantic_hypothesis(decision, milestone)
    assert slow.publish_semantic_hypothesis(decision, milestone, (0, None, "idle"))
    assert len(context.projection("fast")["advice"]) == 1
    context.action_seq += 1
    assert context.projection("fast")["advice"] == []
