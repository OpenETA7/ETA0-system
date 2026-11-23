"""Offline actor-loop tests for joining an existing completion review ticket."""
import asyncio
import copy
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context


class MemoryStore:
    def __init__(self):
        self.events = []
        self.review_wait_started = asyncio.Event()

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})
        if kind == "completion.review_wait_started":
            self.review_wait_started.set()

    def since(self, cursor):
        return self.events[cursor:]


class ScriptedFast:
    def __init__(self, decisions):
        self.decisions = deque(decisions)
        self.history = deque([{"role": "assistant", "content": "unsupported completion belief"}])
        self.calls = []
        self.called = asyncio.Event()

    async def decide(self, prompt, projection, _images):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "history": copy.deepcopy(list(self.history))})
        decision = self.decisions.popleft()
        self.history.append({"role": "assistant", "content": decision["kind"]})
        self.called.set()
        return decision


class Adapter:
    generation = 1
    terminal = False

    def __init__(self):
        self.points = {}
        self.schemas = {}


class Coordinator:
    def __init__(self, context):
        self.context = context
        self.adapter = Adapter()
        self.paused = self.unknown = False
        self.lock = asyncio.Lock()
        self.verify_calls = 0
        self.observe_count = 0
        self.submitted = []
        self.running = None

    async def observe(self):
        self.observe_count += 1
        obs = Observation(
            f"obs-{self.observe_count + 1}", self.observe_count + 1, 1,
            time.monotonic(), 0.0, self.context.action_seq,
            {"frame": self.observe_count + 1},
        )
        self.context.ingest(obs)
        return obs

    async def verify(self):
        self.verify_calls += 1
        return {"status": "not_verified", "evidence": {"official": True}}


class TicketedSlow:
    def __init__(self, *, deliver_on_request=False):
        self.request_ticket = 0
        self.delivered_ticket = 0
        self.deliver_on_request = deliver_on_request
        self.changed = asyncio.Event()
        self.model_lock = asyncio.Lock()
        self.review_requests = []
        self.review_tasks = []
        self.release_review = asyncio.Event()

    async def _pending_review(self):
        await self.release_review.wait()

    def request_review(self, reason):
        self.request_ticket += 1
        ticket = self.request_ticket
        self.review_requests.append((ticket, reason))
        self.review_tasks.append(asyncio.create_task(self._pending_review()))
        if self.deliver_on_request:
            self.delivered_ticket = ticket
            self.changed.set()
        return ticket

    def deliver(self, ticket=None):
        self.delivered_ticket = max(self.delivered_ticket, ticket or self.request_ticket)
        self.changed.set()


class FakePipelinedManager:
    pipelined = True
    pending_reason = None

    def __init__(self, slow):
        self.slow = slow
        self.remaining = 2
        self.checkpoint_task = None
        self.preparation_used = False
        self.checkpoint_calls = []
        self.local_review_reasons = []

    async def before_decision(self, remaining):
        self.remaining = remaining
        return None

    async def guard_proposal(self, _kind, _tool=None):
        return True, None

    def record_decision(self):
        pass

    async def checkpoint(self, trigger="fast_completion_proposal"):
        self.checkpoint_calls.append(trigger)
        return "checkpoint_pending"

    def note_terminal(self, _evidence):
        pass

    def request_local_review(self, reason):
        self.local_review_reasons.append(reason)
        return self.slow.request_review("local_recovery: " + str(reason))


def make_agent(*, deliver_on_request=False, timeout=0.05, decisions=2, rendezvous=True):
    store = MemoryStore()
    context = Context(store)
    context.milestone = Milestone(
        "final-stage", 1, "put the cup inside the bowl", "put the cup inside the bowl",
        capabilities=("move_to",), final_stage=True,
    )
    context.ingest(Observation(
        "initial", 1, 1, time.monotonic(), 0.0, 0, {"frame": 1},
    ))
    coordinator = Coordinator(context)
    slow = TicketedSlow(deliver_on_request=deliver_on_request)
    model = ScriptedFast([{"kind": "complete", "reason": f"repeat claim {i}"}
                          for i in range(decisions)])
    agent = FastAgent(
        model, context, coordinator, store, slow=slow, mode="dual_async",
        async_completion_recovery=True, completion_rendezvous=rendezvous,
        completion_review_timeout=timeout,
    )
    manager = FakePipelinedManager(slow)
    agent.milestones = manager
    return store, context, coordinator, slow, model, manager, agent


async def _release_review_tasks(slow):
    slow.release_review.set()
    if slow.review_tasks:
        await asyncio.gather(*slow.review_tasks, return_exceptions=True)


def _finished_wait(store):
    return [event["payload"] for event in store.events
            if event["kind"] == "completion.review_wait_finished"]


def test_already_delivered_ticket_is_joined_without_new_request_or_checkpoint():
    async def run():
        store, _context, coordinator, slow, model, manager, agent = make_agent(
            deliver_on_request=True, decisions=3)
        try:
            assert await agent.run(3) == "decision_budget"
            assert len(model.calls) == agent.steps == 3
            assert coordinator.verify_calls == 3
            assert slow.request_ticket == slow.delivered_ticket == 1
            assert len(slow.review_requests) == 1
            assert manager.checkpoint_calls == []
            assert _finished_wait(store)[0]["outcome"] == "review_delivered"
            assert _finished_wait(store)[0]["ticket"] == 1
            assert agent.errors == 0
            assert coordinator.submitted == []
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_inflight_ticket_waits_without_locking_slow_model_or_spending_fast_slots():
    async def run():
        store, _context, _coordinator, slow, model, manager, agent = make_agent(decisions=3)
        await slow.model_lock.acquire()  # Slow may be doing useful work in the background.
        try:
            run_task = asyncio.create_task(agent.run(3))
            await asyncio.wait_for(store.review_wait_started.wait(), timeout=1)
            assert not run_task.done()
            assert slow.request_ticket == 1 and slow.delivered_ticket == 0
            assert slow.model_lock.locked()  # Fast joined the ticket without acquiring this lock.
            assert len(model.calls) == agent.steps == 2
            assert agent.errors == 0
            assert manager.checkpoint_calls == []

            slow.model_lock.release()
            slow.deliver(1)
            assert await asyncio.wait_for(run_task, timeout=1) == "decision_budget"
            assert len(model.calls) == agent.steps == 3
            assert _finished_wait(store)[0]["outcome"] == "review_delivered"
            assert slow.request_ticket == 1  # The rendezvous reused the first ticket.
        finally:
            if slow.model_lock.locked():
                slow.model_lock.release()
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_control_cut_in_wakes_wait_and_does_not_issue_another_review():
    async def run():
        store, context, _coordinator, slow, _model, manager, agent = make_agent(decisions=3)
        try:
            run_task = asyncio.create_task(agent.run(3))
            await asyncio.wait_for(store.review_wait_started.wait(), timeout=1)
            context.signal_control("pause", priority=90)
            assert await asyncio.wait_for(run_task, timeout=1) == "decision_budget"
            assert agent.steps == 3

            waits = _finished_wait(store)
            assert len(waits) == 1 and waits[0]["outcome"] == "control_changed"
            assert slow.request_ticket == 1
            assert len(slow.review_requests) == 1
            assert manager.local_review_reasons == []
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_timeout_leaves_shared_background_review_pending():
    async def run():
        store, _context, _coordinator, slow, _model, manager, agent = make_agent(
            timeout=0.02, decisions=3)
        try:
            assert await agent.run(3) == "decision_budget"
            waits = _finished_wait(store)
            assert len(waits) == 1 and waits[0]["outcome"] == "timeout"
            assert waits[0]["elapsed_s"] < 0.25
            assert slow.request_ticket == 1 and slow.delivered_ticket == 0
            assert len(slow.review_requests) == 1
            assert len(slow.review_tasks) == 1
            assert not slow.review_tasks[0].done() and not slow.review_tasks[0].cancelled()
            assert (agent.context.local_state["completion_recovery"]["review_outcome"] == "timeout")
            assert agent.steps == 3
            assert manager.checkpoint_calls == []
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_episode_deadline_caps_review_wait_and_returns_wall_time_budget():
    async def run():
        store, _context, _coordinator, slow, model, _manager, agent = make_agent(
            timeout=2.0, decisions=3)
        deadline = time.monotonic() + 0.04
        try:
            assert await agent.run(3, deadline=deadline) == "wall_time_budget"
            waits = _finished_wait(store)
            assert len(waits) == 1 and waits[0]["outcome"] == "episode_deadline"
            assert waits[0]["elapsed_s"] < 0.25
            assert len(model.calls) == agent.steps == 2
            assert slow.request_ticket == 1 and slow.delivered_ticket == 0
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_rejected_revision_prompt_disallows_repeating_complete_without_spending_extra_budget():
    async def run():
        store, _context, coordinator, slow, model, manager, agent = make_agent(
            deliver_on_request=True, decisions=3)
        try:
            assert await agent.run(3) == "decision_budget"
            assert len(model.calls) == agent.steps == 3
            prompt = model.calls[2]["prompt"].lower()
            assert "do not repeat complete on unchanged state" in prompt
            assert model.calls[2]["projection"]["local_state"]["completion_recovery"]["completion_eligible"] is False
            assert model.calls[2]["projection"]["local_state"]["completion_recovery"]["claims_without_new_action"] == 2
            assert slow.request_ticket == 1
            assert manager.checkpoint_calls == []
            assert coordinator.submitted == []  # Slow review never chose or dispatched an action.
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_disabled_rendezvous_keeps_existing_second_claim_review_path():
    async def run():
        store, _context, _coordinator, slow, _model, manager, agent = make_agent(
            decisions=2, rendezvous=False)
        try:
            assert await agent.run(2) == "decision_budget"
            assert slow.request_ticket == 2  # Second claim follows the old local-recovery route.
            assert len(slow.review_requests) == 2
            assert manager.local_review_reasons == [
                "completion_rejected_twice: current target is not in the requested final relation; diagnose and correct it"
            ]
            assert _finished_wait(store) == []
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_external_cancellation_cleans_only_waiters_and_propagates_cancelled_error():
    async def run():
        store, _context, _coordinator, slow, _model, _manager, agent = make_agent()
        context = agent.context
        await agent.recover_completion(deadline=None)  # Creates ticket 1 and leaves review in flight.

        control_waiter_ready = asyncio.Event()
        changed_waiter_ready = asyncio.Event()
        owned_tasks = {}
        original_control_wait = context.wait_for_control
        original_changed_wait = slow.changed.wait

        async def tracked_control_wait(version):
            owned_tasks["control"] = asyncio.current_task()
            control_waiter_ready.set()
            return await original_control_wait(version)

        async def tracked_changed_wait():
            owned_tasks["changed"] = asyncio.current_task()
            changed_waiter_ready.set()
            return await original_changed_wait()

        context.wait_for_control = tracked_control_wait
        slow.changed.wait = tracked_changed_wait
        wait_task = asyncio.create_task(agent.wait_completion_review(deadline=None))
        try:
            await asyncio.wait_for(control_waiter_ready.wait(), timeout=1)
            await asyncio.wait_for(changed_waiter_ready.wait(), timeout=1)
            wait_task.cancel()
            try:
                await wait_task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("external cancellation must propagate")

            assert _finished_wait(store)[0]["outcome"] == "cancelled"
            assert owned_tasks["control"].done() and owned_tasks["control"].cancelled()
            assert owned_tasks["changed"].done() and owned_tasks["changed"].cancelled()
            assert slow.request_ticket == 1
            assert not slow.review_tasks[0].done() and not slow.review_tasks[0].cancelled()
            assert agent.errors == 0
        finally:
            if not wait_task.done():
                wait_task.cancel()
            await _release_review_tasks(slow)

    asyncio.run(run())


def test_single_remaining_decision_skips_rendezvous_without_wait_or_new_review():
    async def run():
        store, _context, _coordinator, slow, model, manager, agent = make_agent(decisions=2)
        try:
            assert await agent.run(2) == "decision_budget"
            assert len(model.calls) == agent.steps == 2
            assert slow.request_ticket == 1  # First rejection only.
            assert len(slow.review_requests) == 1
            assert manager.remaining == 1
            skipped = [event["payload"] for event in store.events
                       if event["kind"] == "completion.review_wait_skipped"]
            assert len(skipped) == 1
            assert skipped[0]["reason"] == "no_next_decision_budget"
            assert skipped[0]["ticket"] == 1 and skipped[0]["action_seq"] == 0
            assert store.review_wait_started.is_set() is False
            assert _finished_wait(store) == []
            assert manager.checkpoint_calls == []
            assert agent.errors == 0
        finally:
            await _release_review_tasks(slow)

    asyncio.run(run())


@pytest.mark.parametrize(("mode", "pipeline", "final_recovery"), [
    ("single_fast", True, True),
    ("dual_async", False, True),
    ("dual_async", True, False),
])
def test_app_requires_async_pipelined_final_recovery_for_rendezvous(
        tmp_path, mode, pipeline, final_recovery):
    from embodied_harness import app

    args = SimpleNamespace(
        completion_rendezvous=True, mode=mode, backend="direct_libero",
        pipeline_supervision=pipeline, async_completion_recovery=final_recovery,
    )
    with pytest.raises(ValueError, match="completion_rendezvous requires dual_async pipelined final completion recovery"):
        asyncio.run(app.episode(args))


def test_round22_runner_dispatches_and_validates_fixed_review_timeout():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner
    from test_round22_runner import fixture_config, load_fixture_catalog, valid_result_for

    config = fixture_config()
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True, completion_rendezvous=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    selected = next(case for case in config["cases"]
                    if case["variant"] == "patched22" and case["mode"] == "dual_async"
                    and case["suite"] != "libero_10")
    config["cases"] = [copy.deepcopy(selected)]
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]

    assert job["feature_flags"]["completion_rendezvous"] is True
    command_args = runner.app_arguments(job, Path("/tmp/unused-memory-manifest.json"))
    assert "--completion-rendezvous" in command_args
    result = valid_result_for(job)
    result["completion_review_timeout_s"] = 30.0
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []
    result["completion_review_timeout_s"] = 29.0
    assert "mismatch_completion_review_timeout" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})
