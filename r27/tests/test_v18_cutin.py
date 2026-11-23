"""Offline regressions for control-message cut-in and stale-action fencing."""
import asyncio
import json
import time

import httpx
import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.model import InferencePreempted, ModelClient
from embodied_harness.domain.models import ActionRequest, Advice, Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator, GateError


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})

    def since(self, cursor):
        return self.events[cursor:]


class FakeAdapter:
    generation = 1
    schemas = {"move_to": {"description": "move", "parameters": {
        "type": "object", "properties": {"x": {"type": "number"}}}}}

    def __init__(self):
        self.executions = []
        self.version = 0

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, 1, time.monotonic(),
                           float(self.version), action_seq, {})

    def validate_args(self, capability, args):
        if capability != "move_to" or not isinstance(args, dict):
            raise ValueError("invalid action")

    async def execute(self, capability, args, fence, cancelled, publish):
        self.executions.append((capability, args))
        return {}

    async def verify(self, observation_id):
        return {"status": "not_verified", "observation_id": observation_id}


async def make_controlled_coordinator():
    store = MemoryStore()
    context = Context(store)
    adapter = FakeAdapter()
    coordinator = Coordinator(adapter, context, store)
    coordinator.authorize_mission("move the object", capabilities=("move_to",))
    await coordinator.observe()
    coordinator.delegate(Milestone("stage-1", 1, "move the object", "move the object",
                                    capabilities=("move_to",), generation=1))
    return store, context, adapter, coordinator


def completion(content='{"kind":"tool","tool":"move_to","args":{"x":0.1}}'):
    return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                   "finish_reason": "stop"}]})


async def install_transport(model, transport):
    await model.client.aclose()
    model.client = httpx.AsyncClient(
        base_url="http://mock.invalid/",
        transport=transport,
        headers={"Authorization": "Bearer synthetic-test-key"},
        trust_env=False,
    )


def test_accepted_advice_cuts_in_blocked_fast_inference_without_motion(tmp_path):
    async def run():
        store, context, adapter, coordinator = await make_controlled_coordinator()
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

            async def aclose(self):
                pass

        model = ModelClient("http://mock.invalid", "synthetic-test-key", "test-model", "fast", tmp_path)
        await install_transport(model, BlockingTransport())
        model.bind_tools(adapter.schemas)
        model.control_context = context
        fast = FastAgent(model, context, coordinator, store)
        task = asyncio.create_task(fast.run(1))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            advice = Advice("stage-1", 1, 1, "advice", "Recheck the target identity.",
                            time.monotonic() + 30, priority=50)
            assert coordinator.apply_advice(advice)

            # The server never releases its response. The accepted message must wake Fast locally.
            assert await asyncio.wait_for(task, timeout=0.5) == "decision_budget"
            assert cancelled.is_set()
            assert adapter.executions == []
            assert fast.errors == 0
            assert model.accounting["logical_preempted"] == 1
            assert model.accounting["logical_failed"] == 0
            assert model.accounting["cancelled_attempts"] == 1
            preempt = next(event for event in store.events
                           if event["kind"] == "control.inference_preempted")
            assert preempt["payload"]["notice"]["kind"] == "advice"
            assert preempt["payload"]["wake_latency_s"] >= 0
            assert preempt["payload"]["cancel_cleanup_s"] >= 0
            assert "may remain billable" in preempt["payload"]["provider_cost"]
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await model.close()

    asyncio.run(run())


def test_expired_advice_does_not_advance_control_version_or_wake_waiters():
    async def run():
        _store, context, _adapter, coordinator = await make_controlled_coordinator()
        before = context.control_version
        notices = len(context.control_notices)
        advice = Advice("stage-1", 1, 1, "advice", "Expired correction.",
                        time.monotonic() - 1, priority=50)

        assert not coordinator.apply_advice(advice)
        assert context.control_version == before
        assert len(context.control_notices) == notices

    asyncio.run(run())


def test_control_message_wins_when_message_and_model_finish_in_same_schedule(tmp_path):
    async def run():
        store, context, _adapter, _coordinator = await make_controlled_coordinator()
        entered, release = asyncio.Event(), asyncio.Event()

        class RacingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                entered.set()
                await release.wait()
                # No await between signaling and returning the response: both child tasks
                # become runnable in the same event-loop turn.
                context.signal_control("advice", 50, "race-advice")
                return completion()

            async def aclose(self):
                pass

        model = ModelClient("http://mock.invalid", "synthetic-test-key", "test-model", "fast", tmp_path)
        await install_transport(model, RacingTransport())
        model.control_context = context
        snapshot = context.projection("fast")
        task = asyncio.create_task(model.decide("system", snapshot))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            release.set()
            with pytest.raises(InferencePreempted):
                await asyncio.wait_for(task, timeout=1)
            assert model.accounting["logical_preempted"] == 1
            assert model.accounting["logical_completed"] == 0
            assert any(event["kind"] == "control.inference_preempted" for event in store.events)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await model.close()

    asyncio.run(run())


def test_ordinary_advice_while_submit_waits_invalidates_old_control_snapshot():
    async def run():
        _store, context, adapter, coordinator = await make_controlled_coordinator()
        milestone = context.milestone
        observation = context.latest
        old_version = context.control_version
        request = ActionRequest("queued-action", milestone.id, milestone.revision,
                                milestone.generation, observation.id, "move_to", {"x": 0.1},
                                control_version=old_version)

        await coordinator.lock.acquire()
        task = asyncio.create_task(coordinator.submit(request))
        await asyncio.sleep(0)
        assert coordinator.apply_advice(Advice(
            milestone.id, milestone.revision, milestone.generation, "advice",
            "Reassess before acting.", time.monotonic() + 30, priority=50,
        ))
        assert context.control_version > old_version
        coordinator.lock.release()

        with pytest.raises(GateError, match="superseded_control_context"):
            await task
        assert adapter.executions == []
        assert context.action_seq == 0

    asyncio.run(run())


def test_pause_notice_wins_over_later_lower_priority_advice():
    async def run():
        store = MemoryStore()
        context = Context(store)
        pause = context.signal_control("pause", 100, "urgent-stop")
        # A burst of routine corrections must not evict the still-unconsumed stop.
        for index in range(20):
            context.signal_control("advice", 50, f"local-correction-{index}")

        selected = await context.wait_for_control(after_version=0)
        assert selected["version"] == pause["version"]
        assert selected["kind"] == "pause"
        assert selected["priority"] == 100

    asyncio.run(run())


def test_control_notification_arriving_before_wait_is_not_lost():
    async def run():
        store = MemoryStore()
        context = Context(store)
        notice = context.signal_control("advice", 50, "early-message")

        selected = await context.wait_for_control(after_version=0)
        assert selected == notice

    asyncio.run(run())
