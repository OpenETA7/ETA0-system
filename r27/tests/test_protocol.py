"""In-memory regression tests for action and information boundaries."""
import asyncio
import time
from dataclasses import replace

from embodied_harness.agents.fast import FastAgent
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


class Robot:
    generation = 1

    def __init__(self):
        self.version = 0
        self.executions = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block_observe = False

    async def observe(self, action_seq):
        if self.block_observe:
            self.entered.set()
            await self.release.wait()
        self.version += 1
        return Observation(str(self.version), self.version, 1, time.monotonic(), 0, action_seq, {})

    def validate_args(self, capability, args):
        pass

    async def execute(self, capability, args, fence, cancelled, publish):
        self.executions += 1
        return {}

    async def verify(self, observation_id):
        return {"status": "not_verified", "evidence": {"hidden_success": False, "reward": 0.75}}


def setup():
    store = MemoryStore()
    context = Context(store)
    robot = Robot()
    coordinator = Coordinator(robot, context, store)
    milestone = Milestone("m", 1, "mission", "goal", capabilities=("move_to",))
    coordinator.delegate(milestone)
    return store, context, robot, coordinator, milestone


def test_duplicate_request_executes_once():
    async def run():
        _, context, robot, coordinator, milestone = setup()
        observation = await coordinator.observe()
        req = ActionRequest("request", milestone.id, 1, 1, observation.id, "move_to", {})
        a, b = await asyncio.gather(coordinator.submit(req), coordinator.submit(req))
        assert a.action_id == b.action_id
        assert robot.executions == 1
        try:
            await coordinator.submit(replace(req, args={"different": True}))
        except GateError as exc:
            assert str(exc) == "request_id_collision"
        else:
            assert False, "a reused id with different arguments must be rejected"
    asyncio.run(run())


def test_new_pause_during_resume_is_not_lost():
    async def run():
        _, _, robot, coordinator, _ = setup()
        coordinator.pause("first stop")
        robot.block_observe = True
        task = asyncio.create_task(coordinator.resume())
        await robot.entered.wait()
        coordinator.pause("newer stop")
        robot.release.set()
        try:
            await task
        except GateError as exc:
            assert str(exc) == "newer_pause_received"
        assert coordinator.paused, "a newer stop must not be overridden by an earlier resume"
    asyncio.run(run())


def test_evaluator_evidence_not_in_model_projection():
    async def run():
        _, context, _, coordinator, _ = setup()
        await coordinator.observe()
        result = await coordinator.verify()
        assert result["evidence"]["reward"] == 0.75
        for role in ("fast", "slow"):
            projection = repr(context.projection(role))
            assert "hidden_success" not in projection
            assert "reward" not in projection
    asyncio.run(run())


def test_model_result_cannot_cross_milestone_revision():
    async def run():
        store, context, robot, coordinator, milestone = setup()
        await coordinator.observe()
        entered, release = asyncio.Event(), asyncio.Event()

        class Model:
            async def decide(self, system, projection, images):
                entered.set()
                await release.wait()
                return {"kind": "tool", "tool": "move_to", "args": {}}

        agent = FastAgent(Model(), context, coordinator, store)
        task = asyncio.create_task(agent.run(1))
        await entered.wait()
        coordinator.delegate(replace(milestone, revision=2, goal="different goal"), expected_revision=1)
        release.set()
        await task
        assert robot.executions == 0, "a decision based on an old milestone must not execute under a new one"
    asyncio.run(run())


def test_eta_host_execution_error_blocks_further_actions(tmp_path):
    import json
    from types import SimpleNamespace
    from embodied_harness.adapters.openeta import OpenETAAdapter

    async def run():
        class Host:
            closed = False
            runner = SimpleNamespace(terminated=False, truncated=False)

            def call(self, name, args):
                body = {"robot": {"joint_state": []}}
                if name == "move_to":
                    self.closed = True
                    body["error"] = {"code": "host_execution_error", "message": "transport lost"}
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(body))])

        adapter = OpenETAAdapter(tmp_path, "unused", "unused", "unused")
        adapter.host = Host()
        adapter.schemas = {"move_to": {"parameters": {"type": "object"}}}
        store, context = MemoryStore(), Context(MemoryStore())
        context.store = store
        coordinator = Coordinator(adapter, context, store)
        milestone = Milestone("m", 1, "mission", "goal", capabilities=("move_to",))
        coordinator.delegate(milestone)
        obs = await coordinator.observe()
        receipt = await coordinator.submit(ActionRequest("r", "m", 1, 1, obs.id, "move_to", {}))
        assert receipt.status == "state_unknown"
        assert coordinator.unknown
        adapter.executor.shutdown(wait=True)
    asyncio.run(run())


def test_action_sensitive_pause_is_stale_after_action_finishes():
    async def run():
        _, context, robot, coordinator, milestone = setup()
        observation = await coordinator.observe()
        started, release = asyncio.Event(), asyncio.Event()

        async def execute(capability, args, fence, cancelled, publish):
            started.set()
            await release.wait()
            return {}

        robot.execute = execute
        req = ActionRequest("request", milestone.id, 1, 1, observation.id, "move_to", {})
        task = asyncio.create_task(coordinator.submit(req))
        await started.wait()
        advice = Advice("m", 1, 1, "pause", "stop this action", time.monotonic() + 30, context.action_seq, coordinator.running.action_id, coordinator.running.status)
        release.set()
        await task
        assert not coordinator.apply_advice(advice), "action-phase advice must expire when the action finishes"
        assert not coordinator.paused
    asyncio.run(run())


def test_expired_advice_is_not_in_projection():
    _, context, _, coordinator, _ = setup()
    advice = Advice("m", 1, 1, "advice", "old advice", time.monotonic() + 0.01)
    assert coordinator.apply_advice(advice)
    time.sleep(0.02)
    assert not context.projection("fast")["advice"]


def test_accepted_action_phase_advice_disappears_when_action_finishes():
    async def run():
        _, context, robot, coordinator, milestone = setup()
        obs = await coordinator.observe()
        started, release = asyncio.Event(), asyncio.Event()

        async def execute(capability, args, fence, cancelled, publish):
            started.set()
            await release.wait()
            return {}

        robot.execute = execute
        task = asyncio.create_task(coordinator.submit(ActionRequest("request", "m", 1, 1, obs.id, "move_to", {})))
        await started.wait()
        advice = Advice("m", 1, 1, "advice", "during this action", time.monotonic() + 30, context.action_seq, coordinator.running.action_id, coordinator.running.status)
        assert coordinator.apply_advice(advice)
        assert context.projection("fast")["advice"]
        release.set()
        await task
        assert not context.projection("fast")["advice"]
    asyncio.run(run())


def test_eta_preview_feedback_string_does_not_freeze(tmp_path):
    import json
    from types import SimpleNamespace
    from embodied_harness.adapters.openeta import OpenETAAdapter

    async def run():
        class Host:
            closed = False
            runner = SimpleNamespace(terminated=False, truncated=False)

            def call(self, name, args):
                body = {"robot": {}}
                if name == "move_to":
                    body["feedback"] = {"motion": "preview_only; no IK or physics executed"}
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(body))])

        adapter = OpenETAAdapter(tmp_path, "unused", "unused", "unused")
        adapter.host = Host()
        adapter.schemas = {"move_to": {"parameters": {"type": "object"}}}
        store = MemoryStore()
        context = Context(store)
        coordinator = Coordinator(adapter, context, store)
        coordinator.delegate(Milestone("m", 1, "mission", "goal", capabilities=("move_to",)))
        obs = await coordinator.observe()
        receipt = await coordinator.submit(ActionRequest("r", "m", 1, 1, obs.id, "move_to", {"preview": True}))
        assert receipt.status == "completed"
        assert not coordinator.unknown
        adapter.executor.shutdown(wait=True)
    asyncio.run(run())


def test_accepted_running_advice_disappears_on_cancel_request():
    async def run():
        _, context, robot, coordinator, _ = setup()
        obs = await coordinator.observe()
        started, release = asyncio.Event(), asyncio.Event()

        async def execute(capability, args, fence, cancelled, publish):
            started.set()
            await release.wait()
            return {"stop_confirmed": True}

        robot.execute = execute
        task = asyncio.create_task(coordinator.submit(ActionRequest("request", "m", 1, 1, obs.id, "move_to", {})))
        await started.wait()
        advice = Advice("m", 1, 1, "advice", "running phase", time.monotonic() + 30,
                        context.action_seq, coordinator.running.action_id, "running")
        assert coordinator.apply_advice(advice)
        coordinator.pause("stop")
        assert not context.projection("fast")["advice"]
        release.set()
        await task
    asyncio.run(run())


def test_action_authorization_and_freshness_gates():
    import pytest

    async def run():
        _, context, robot, coordinator, milestone = setup()
        obs = await coordinator.observe()
        req = ActionRequest("base", "m", 1, 1, obs.id, "move_to", {})
        invalid = [
            (replace(req, request_id="wrong-id", milestone_id="other"), "inactive_or_stale_milestone"),
            (replace(req, request_id="wrong-revision", revision=2), "inactive_or_stale_milestone"),
            (replace(req, request_id="wrong-capability", capability="gripper_control"), "capability_or_resource_denied"),
            (replace(req, request_id="extra-resource", resources=("arm", "gripper", "other")), "capability_or_resource_denied"),
            (replace(req, request_id="missing-resource", resources=("arm",)), "adapter_requires_both_resources"),
            (replace(req, request_id="expired", created_at=time.monotonic() - 10, ttl=1), "expired_request"),
        ]
        for request, reason in invalid:
            with pytest.raises(GateError, match=reason):
                await coordinator.submit(request)
        assert robot.executions == 0
        await coordinator.observe()
        with pytest.raises(GateError, match="fresh_observation_required"):
            await coordinator.submit(req)
        assert robot.executions == 0
        current = context.latest
        assert (await coordinator.submit(replace(req, observation_id=current.id))).status == "completed"
        with pytest.raises(GateError, match="fresh_observation_required"):
            await coordinator.submit(replace(req, request_id="after-motion", observation_id=current.id))
        assert robot.executions == 1
    asyncio.run(run())


def test_generation_change_invalidates_action_and_advice():
    import pytest

    async def run():
        _, context, robot, coordinator, _ = setup()
        obs = await coordinator.observe()
        req = ActionRequest("r", "m", 1, 1, obs.id, "move_to", {})
        advice = Advice("m", 1, 1, "advice", "old world", time.monotonic() + 30)
        robot.generation = 2
        with pytest.raises(GateError, match="stale_environment"):
            await coordinator.submit(req)
        assert not coordinator.apply_advice(advice)
        assert robot.executions == 0
    asyncio.run(run())


def test_distinct_requests_do_not_overlap_on_same_resources():
    import pytest

    async def run():
        _, context, robot, coordinator, _ = setup()
        obs = await coordinator.observe()
        started, release = asyncio.Event(), asyncio.Event()
        running, maximum, calls = 0, 0, 0

        async def execute(capability, args, fence, cancelled, publish):
            nonlocal running, maximum, calls
            calls += 1
            running += 1
            maximum = max(maximum, running)
            started.set()
            await release.wait()
            running -= 1
            return {}

        robot.execute = execute
        first = asyncio.create_task(coordinator.submit(ActionRequest("a", "m", 1, 1, obs.id, "move_to", {})))
        await started.wait()
        second = asyncio.create_task(coordinator.submit(ActionRequest("b", "m", 1, 1, obs.id, "move_to", {})))
        await asyncio.sleep(0)
        assert not second.done() and calls == 1
        release.set()
        assert (await first).status == "completed"
        with pytest.raises(GateError, match="fresh_observation_required"):
            await second
        assert maximum == 1 and calls == 1
    asyncio.run(run())


def test_cancel_request_keeps_resource_until_execution_returns():
    import pytest

    async def run():
        _, _, robot, coordinator, milestone = setup()
        obs = await coordinator.observe()
        started, release = asyncio.Event(), asyncio.Event()

        async def execute(capability, args, fence, cancelled, publish):
            started.set()
            await release.wait()
            assert cancelled()
            return {"stop_confirmed": True}

        robot.execute = execute
        first = asyncio.create_task(coordinator.submit(ActionRequest("a", "m", 1, 1, obs.id, "move_to", {})))
        await started.wait()
        coordinator.pause("stop")
        assert coordinator.running and coordinator.running.status == "cancel_requested"
        with pytest.raises(GateError, match="resume_requires_confirmed_boundary"):
            await coordinator.resume()
        with pytest.raises(GateError, match="handoff_requires_action_boundary"):
            coordinator.delegate(replace(milestone, revision=2), expected_revision=1)
        second = asyncio.create_task(coordinator.submit(ActionRequest("b", "m", 1, 1, obs.id, "move_to", {})))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        assert (await first).status == "stop_confirmed"
        with pytest.raises(GateError, match="paused_requires_explicit_resume"):
            await second
    asyncio.run(run())


def test_sqlite_event_deduplication_and_cursor_recovery(tmp_path):
    from embodied_harness.runtime.events import EventStore

    async def run():
        path = tmp_path / "events.sqlite"
        store = EventStore(path)
        store.emit("first", {"value": 1}, event_id="same")
        store.emit("duplicate", {"value": 2}, event_id="same")
        store.emit("second", {"value": 3}, event_id="other")
        assert [e["kind"] for e in store.since(0)] == ["first", "second"]
        consumer = store.stream("fast")
        assert (await anext(consumer))["kind"] == "first"
        assert (await anext(consumer))["kind"] == "second"
        await consumer.aclose()
        store.close()

        # Cursors commit only once processing returns to the generator: the last
        # yielded event is replayed after interruption, preserving at-least-once.
        reopened = EventStore(path)
        resumed = reopened.stream("fast")
        independent = reopened.stream("slow")
        assert (await anext(resumed))["kind"] == "second"
        assert (await anext(independent))["kind"] == "first"
        await resumed.aclose()
        await independent.aclose()
        reopened.close()
    asyncio.run(run())


def test_same_milestone_revision_must_advance():
    import pytest

    async def run():
        _, _, robot, coordinator, milestone = setup()
        obs = await coordinator.observe()
        old_request = ActionRequest("old", "m", 1, 1, obs.id, "move_to", {})
        with pytest.raises(GateError):
            coordinator.delegate(replace(milestone, goal="a different goal"), expected_revision=1)
        assert coordinator.context.milestone == milestone
        assert (await coordinator.submit(old_request)).status == "completed"
        assert robot.executions == 1
    asyncio.run(run())


def test_retired_milestone_id_cannot_be_reused():
    import pytest

    _, _, _, coordinator, milestone = setup()
    next_milestone = replace(milestone, id="other", revision=1)
    coordinator.delegate(next_milestone, expected_revision=1)
    with pytest.raises(GateError):
        coordinator.delegate(milestone, expected_revision=1)
    assert coordinator.context.milestone == next_milestone
