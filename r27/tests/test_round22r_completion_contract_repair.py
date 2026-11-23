"""Offline tests for bounded completion-driven milestone repair."""
from __future__ import annotations

import asyncio
import copy
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator as RealCoordinator
from embodied_harness.runtime.pipeline import PipelinedMilestoneManager


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


class Adapter:
    generation = 1
    terminal = False
    schemas = {"observe": {}, "move_to": {}}

    def __init__(self):
        self.version = 1
        self.executed = []

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, 1, time.monotonic(), 0.0,
            action_seq, {"candidate_scope": f"frame-{self.version}", "views": []}, ())


class Coordinator:
    def __init__(self, context, adapter, store, verify_statuses=None):
        self.context, self.adapter, self.store = context, adapter, store
        self.paused = self.unknown = self.contract_review_pending = False
        self.running = None
        self.verify_statuses = deque(verify_statuses or ["not_verified"])
        self.verify_calls = self.observe_calls = 0
        self.action_seq = 0
        self.pause_on_observe = False
        self.pause_on_verify_call = None

    async def observe(self):
        self.observe_calls += 1
        observation = await self.adapter.observe(self.context.action_seq)
        self.context.ingest(observation)
        if self.pause_on_observe:
            self.paused = True
        return observation

    async def verify(self):
        self.verify_calls += 1
        status = self.verify_statuses.popleft() if len(self.verify_statuses) > 1 else self.verify_statuses[0]
        if self.verify_calls == self.pause_on_verify_call:
            self.paused = True
        return {"status": status, "evidence": []}


class Slow:
    urgent_completion_review = False

    def __init__(self):
        self.review_requests = []
        self.local_reviews = []

    def request_review(self, reason):
        self.review_requests.append(reason)
        return len(self.review_requests)


class ScriptedFast:
    def __init__(self, decisions, *, delay_call=None, delay_s=0):
        self.decisions = deque(decisions)
        self.history = deque([{"role": "assistant", "content": "stale claim"}])
        self.calls = []
        self.delay_call, self.delay_s = delay_call, delay_s

    async def decide(self, prompt, projection, images):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images)})
        if len(self.calls) == self.delay_call:
            await asyncio.sleep(self.delay_s)
        return copy.deepcopy(self.decisions.popleft())


class Pipelined:
    pipelined = True
    pending_reason = None
    checkpoint_task = None
    preparation_used = False
    max_replans = 3

    def __init__(self, context, *, remaining=5, repair_result="wall_time_budget",
                 started=None, release=None, handoff=False):
        self.context = context
        self.remaining = remaining
        self.repair_result = repair_result
        self.recoveries = 0
        self.last_recovery_valid = False
        self.reasons = []
        self.allow_resume_values = []
        self.started, self.release = started, release
        self.handoff = handoff

    async def before_decision(self, remaining):
        self.remaining = remaining
        return None

    def record_decision(self):
        self.remaining -= 1

    async def guard_proposal(self, _kind, _tool=None):
        return True, None

    def request_local_review(self, reason):
        self.reasons.append(reason)
        return len(self.reasons)

    async def recover(self, reason, *, allow_resume=True):
        self.recoveries += 1
        self.reasons.append(reason)
        self.allow_resume_values.append(allow_resume)
        if self.started:
            self.started.set()
        if self.release:
            await self.release.wait()
        if self.handoff:
            current = self.context.milestone
            self.context.milestone = Milestone(current.id + "-revised", 1, current.mission,
                current.goal, capabilities=current.capabilities, final_stage=True)
            self.last_recovery_valid = True
            return None
        self.last_recovery_valid = self.repair_result is None
        return self.repair_result

    def note_terminal(self, _evidence):
        pass


def make_agent(*, decisions=None, repair=True, repair_result="wall_time_budget", remaining=5,
               coordinator=None, delay_call=None, delay_s=0, recover_started=None,
               recover_release=None, handoff=False):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("final", 1, "put the cup in the bowl", "place the cup",
        capabilities=("observe", "move_to"), final_stage=True)
    context.ingest(Observation("initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "frame-1", "views": []}, ()))
    adapter = Adapter()
    coordinator = coordinator or Coordinator(context, adapter, store, ["not_verified"] * 4)
    slow = Slow()
    model = ScriptedFast(decisions or [
        {"kind": "complete", "reason": "first failed claim"},
        {"kind": "complete", "reason": "second failed claim"},
        {"kind": "complete", "reason": "after repair"},
    ], delay_call=delay_call, delay_s=delay_s)
    fast = FastAgent(model, context, coordinator, store, slow=slow, mode="dual_async",
        async_completion_recovery=True, completion_contract_repair=repair)
    manager = Pipelined(context, remaining=remaining, repair_result=repair_result,
        started=recover_started, release=recover_release, handoff=handoff)
    fast.milestones = manager
    return store, context, coordinator, adapter, slow, model, manager, fast


def run_async(test):
    @wraps(test)
    def wrapped(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return wrapped


def events(store, kind):
    return [row["payload"] for row in store.events if row["kind"] == kind]


def valid_plan():
    return {"kind": "plan", "fundamental_constraints": [], "milestones": [{
        "goal": "place the cup in the bowl",
        "success_conditions": [{"predicate": "visual", "description": "the cup is inside the bowl"}],
        "constraints": [], "preconditions": [], "max_retries": 2, "decision_budget": 4,
    }]}


class PlanModel:
    def __init__(self, *, block=False):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block
        self.calls = 0

    async def decide(self, _prompt, _projection, _images):
        self.calls += 1
        self.started.set()
        if self.block:
            await self.release.wait()
        return valid_plan()


def make_real_pipeline(*, block=False):
    store = Store()
    context = Context(store)
    context.ingest(Observation("initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "frame-1", "views": []}, ()))
    adapter = Adapter()
    coordinator = RealCoordinator(adapter, context, store)
    mission = "put the cup in the bowl"
    coordinator.authorize_mission(mission, ("observe", "move_to"))
    context.milestone = Milestone("current", 1, mission, "place the cup",
        capabilities=("observe", "move_to"), final_stage=True)
    model = PlanModel(block=block)
    slow = SlowAgent(model, context, coordinator, store, priority_foreground=True)
    manager = PipelinedMilestoneManager(slow, context, coordinator, store, 8, time.monotonic() + 5)
    manager.remaining = 5
    return store, context, coordinator, model, slow, manager


@run_async
async def test_real_planner_recovery_does_not_resume_if_pause_arrives_during_model_call():
    store, _context, coordinator, model, _slow, manager = make_real_pipeline(block=True)
    recovery = asyncio.create_task(manager.recover("optional completion repair", allow_resume=False))
    await asyncio.wait_for(model.started.wait(), 1)
    coordinator.pause("user pause during repair inference")
    model.release.set()
    result = await asyncio.wait_for(recovery, 1)
    assert result == "paused"
    assert coordinator.paused is True
    assert not events(store, "control.resumed")
    assert not events(store, "milestone.plan_accepted")
    assert manager.last_recovery_valid is False


@run_async
async def test_real_planner_recovery_does_not_accept_or_resume_after_pause_while_waiting_on_contract_lock():
    store, _context, coordinator, model, _slow, manager = make_real_pipeline()

    class ObservableLock(asyncio.Lock):
        def __init__(self):
            super().__init__()
            self.waiting = asyncio.Event()

        async def acquire(self):
            if self.locked():
                self.waiting.set()
            return await super().acquire()

    lock = ObservableLock()
    coordinator.lock = lock
    await lock.acquire()
    recovery = asyncio.create_task(manager.recover("optional completion repair", allow_resume=False))
    try:
        await asyncio.wait_for(model.started.wait(), 1)
        await asyncio.wait_for(lock.waiting.wait(), 1)
        coordinator.pause("user pause during contract handoff")
        lock.release()
        assert await asyncio.wait_for(recovery, 1) == "paused"
        assert coordinator.paused is True
        assert not events(store, "control.resumed")
        assert not events(store, "milestone.plan_accepted")
        assert manager.last_recovery_valid is False
    finally:
        if lock.locked():
            lock.release()
        if not recovery.done():
            recovery.cancel()
            await asyncio.gather(recovery, return_exceptions=True)


@run_async
async def test_second_official_negative_uses_bounded_foreground_repair_without_motion_or_fast_slot():
    store, context, coordinator, adapter, slow, model, manager, fast = make_agent(
        repair_result="wall_time_budget")
    result = await fast.run(4)
    assert result == "wall_time_budget"
    assert len(model.calls) == fast.steps == 2
    assert coordinator.verify_calls == 2
    assert coordinator.observe_calls == 1
    assert len(slow.review_requests) == 1  # First negative keeps the existing background review.
    assert manager.recoveries == 1
    assert manager.allow_resume_values == [False]
    assert "completion_rejected_twice" in manager.reasons[-1]
    assert adapter.executed == [] and context.action_seq == 0
    started, finished = events(store, "completion.contract_repair_started"), events(store, "completion.contract_repair_finished")
    assert len(started) == len(finished) == 1
    assert started[0]["blocking"] is True and started[0]["episode_decisions_remaining"] == 2
    assert started[0]["claims_without_new_action"] == 2
    assert finished[0]["outcome"] == "wall_time_budget"


@pytest.mark.parametrize("outcome", ["wall_time_budget", "decision_budget", "paused", "state_unknown", "environment_terminated"])
@run_async
async def test_planner_repair_outcome_propagates_without_an_extra_fast_decision(outcome):
    store, _context, coordinator, adapter, _slow, model, manager, fast = make_agent(repair_result=outcome)
    assert await fast.run(4) == outcome
    assert len(model.calls) == fast.steps == coordinator.verify_calls == 2
    assert manager.recoveries == 1 and adapter.executed == []


@run_async
async def test_official_success_and_initial_pause_unknown_terminal_gates_propagate():
    store, context, _coord, _adapter, _slow, _model, manager, fast = make_agent(
        decisions=[{"kind": "complete"}], repair_result="wall_time_budget")
    fast.coordinator.verify_statuses = deque(["verified"])
    assert await fast.run(2) == "success"
    assert manager.recoveries == 0 and not events(store, "completion.contract_repair_started")

    for field, expected in (("unknown", "state_unknown"), ("paused", "paused")):
        row = make_agent(decisions=[{"kind": "complete"}])
        setattr(row[2], field, True)
        assert await row[7].run(2) == expected
        assert row[5].calls == [] and row[6].recoveries == 0

    row = make_agent(decisions=[{"kind": "complete"}])
    row[3].terminal = True
    assert await row[7].run(2) == "environment_terminated"
    assert row[5].calls == [] and row[6].recoveries == 0


@run_async
async def test_pause_during_negative_refresh_or_verification_prevents_contract_repair():
    observe_case = make_agent()
    observe_case[2].pause_on_observe = True
    assert await observe_case[7].run(4) == "paused"
    assert len(observe_case[5].calls) == observe_case[2].verify_calls == 1
    assert observe_case[6].recoveries == 0
    assert not events(observe_case[0], "completion.contract_repair_started")

    verify_case = make_agent()
    verify_case[2].pause_on_verify_call = 2
    assert await verify_case[7].run(4) == "paused"
    assert len(verify_case[5].calls) == verify_case[2].verify_calls == 2
    assert verify_case[6].recoveries == 0
    skipped = events(verify_case[0], "completion.contract_repair_skipped")
    assert len(skipped) == 1 and skipped[0]["reason"] == "higher_priority_pause"
    assert not verify_case[7].completion_repaired_actions


@pytest.mark.parametrize(("blocker", "expected", "reason"), [
    ("paused", "paused", "higher_priority_pause"),
    ("running", "state_unknown", "no_confirmed_idle_boundary"),
    ("unknown", "state_unknown", "no_confirmed_idle_boundary"),
    ("contract", None, "higher_priority_contract_review"),
])
@run_async
async def test_repair_defers_to_pause_unknown_motion_and_contract_review_without_marking(blocker, expected, reason):
    store, _context, coordinator, _adapter, _slow, _model, manager, fast = make_agent(repair_result=None)
    fast.rejected_completion_count = 2
    if blocker == "paused":
        coordinator.paused = True
    elif blocker == "running":
        coordinator.running = SimpleNamespace(action_id="in-flight")
    elif blocker == "unknown":
        coordinator.unknown = True
    else:
        coordinator.contract_review_pending = True
    assert await fast.repair_completion_contract(deadline=None) == expected
    skipped = events(store, "completion.contract_repair_skipped")
    assert len(skipped) == 1 and skipped[0]["reason"] == reason
    assert not fast.completion_repaired_actions and manager.recoveries == 0
    if blocker == "contract":
        coordinator.contract_review_pending = False
        assert await fast.repair_completion_contract(deadline=None) is None
        assert manager.recoveries == 1


@pytest.mark.parametrize("remaining", [1])
@run_async
async def test_fewer_than_two_future_decisions_skips_repair(remaining):
    store, _context, coordinator, adapter, slow, model, manager, fast = make_agent(
        repair_result="wall_time_budget", remaining=remaining)
    result = await fast.run(2)
    assert result == "decision_budget"
    assert "fewer_than_two_executor_decisions_remain" in [event["reason"] for event in events(store, "completion.contract_repair_skipped")]
    assert manager.recoveries == 0
    assert len(model.calls) == coordinator.verify_calls == 2
    assert len(slow.review_requests) == 1 and adapter.executed == []


@run_async
async def test_expired_deadline_skips_repair_without_marking_revision():
    store, _context, _coordinator, _adapter, _slow, _model, manager, fast = make_agent(repair_result=None)
    fast.rejected_completion_count = 2
    assert await fast.repair_completion_contract(time.monotonic() - 1) == "wall_time_budget"
    skipped = events(store, "completion.contract_repair_skipped")
    assert len(skipped) == 1 and skipped[0]["reason"] == "episode_deadline"
    assert not fast.completion_repaired_actions and manager.recoveries == 0


@run_async
async def test_same_physical_revision_contract_handoff_does_not_reset_one_repair_limit():
    decisions = [{"kind": "complete", "reason": f"claim-{i}"} for i in range(4)]
    store, context, coordinator, adapter, _slow, model, manager, fast = make_agent(
        decisions=decisions, repair_result=None, handoff=True)
    assert await fast.run(4) == "decision_budget"
    assert len(model.calls) == fast.steps == coordinator.verify_calls == 4
    assert manager.recoveries == 1
    assert fast.completion_contract_claim_counts[(1, 0)] == 4
    assert not events(store, "completion.contract_repair_skipped")
    assert context.milestone.id == "final-revised"
    assert context.action_seq == 0 and adapter.executed == []


@run_async
async def test_default_off_keeps_existing_nonblocking_local_review_path():
    store, _context, coordinator, adapter, slow, model, manager, fast = make_agent(repair=False)
    assert fast.completion_contract_repair is False
    assert await fast.run(2) == "decision_budget"
    assert len(model.calls) == fast.steps == coordinator.verify_calls == 2
    assert manager.recoveries == 0
    assert len(slow.review_requests) == 1
    assert len(manager.reasons) == 1 and "completion_rejected_twice" in manager.reasons[0]
    assert not events(store, "completion.contract_repair_started")
    assert adapter.executed == []


@run_async
async def test_later_physical_revision_allows_new_repair_but_repair_exception_and_shutdown_propagate():
    store, context, _coordinator, _adapter, _slow, _model, manager, fast = make_agent(repair_result=None)
    await fast.recover_completion(deadline=None)
    first = await fast.recover_completion(deadline=None)
    assert first is None and manager.recoveries == 1
    assert len(fast.completion_repaired_actions) == 1
    context.action_seq += 1
    await fast.recover_completion(deadline=None)
    second = await fast.recover_completion(deadline=None)
    assert second is None and manager.recoveries == 2
    assert len(fast.completion_repaired_actions) == 2
    assert fast.completion_contract_claim_counts[(1, 0)] == 2
    assert fast.completion_contract_claim_counts[(1, 1)] == 2

    store2, _ctx2, _coord2, _adapter2, _slow2, _model2, manager2, fast2 = make_agent(repair_result=None)
    async def broken(_reason, *, allow_resume=True):
        raise RuntimeError("fixture planner fault")
    manager2.recover = broken
    fast2.rejected_completion_count = 2
    with pytest.raises(RuntimeError, match="fixture planner fault"):
        await fast2.repair_completion_contract(deadline=None)
    failed = events(store2, "completion.contract_repair_finished")[-1]
    assert failed["outcome"] == "error" and failed["planner_response_valid"] is False

    release = asyncio.Event()
    started = asyncio.Event()
    store3, _ctx3, _coord3, _adapter3, _slow3, _model3, manager3, fast3 = make_agent(
        repair_result=None, recover_started=started, recover_release=release)
    fast3.rejected_completion_count = 2
    task = asyncio.create_task(fast3.repair_completion_contract(deadline=None))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    cancelled = events(store3, "completion.contract_repair_finished")[-1]
    assert cancelled["outcome"] == "cancelled" and cancelled["planner_response_valid"] is False


@pytest.mark.parametrize("route", [
    {"mode": "single_fast", "backend": "direct_libero", "pipeline_supervision": True,
     "priority_foreground": True, "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "eta", "pipeline_supervision": True,
     "priority_foreground": True, "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "direct_libero", "pipeline_supervision": False,
     "priority_foreground": True, "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "direct_libero", "pipeline_supervision": True,
     "priority_foreground": False, "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "direct_libero", "pipeline_supervision": True,
     "priority_foreground": True, "async_completion_recovery": False},
])
def test_app_rejects_invalid_repair_route_before_environment(monkeypatch, tmp_path, route):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args

    args = _episode_args(tmp_path, route["mode"])
    args.backend = route["backend"]
    args.pipeline_supervision = route["pipeline_supervision"]
    args.priority_foreground = route["priority_foreground"]
    args.async_completion_recovery = route["async_completion_recovery"]
    args.completion_contract_repair = True
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("route validation must run first"))
    with pytest.raises(ValueError, match="completion_contract_repair requires direct dual_async pipeline, foreground priority and completion recovery"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("other", ["completion_rendezvous", "urgent_completion_review"])
def test_app_rejects_other_completion_scheduling_ablations(monkeypatch, tmp_path, other):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args

    args = _episode_args(tmp_path, "dual_async")
    args.backend = "direct_libero"
    args.pipeline_supervision = args.priority_foreground = args.async_completion_recovery = True
    args.completion_contract_repair = True
    setattr(args, other, True)
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("separation validation must run first"))
    with pytest.raises(ValueError, match="completion_contract_repair is a separate ablation"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_exposes_default_off_completion_repair(monkeypatch, tmp_path, enabled):
    from embodied_harness import app

    captured = {}

    async def fake_episode(args):
        captured.update(mode=args.mode, repair=args.completion_contract_repair)

    monkeypatch.setattr(app, "episode", fake_episode)
    argv = ["app", "--mission", "place the cup", "--output", str(tmp_path / "cli"),
        "--backend", "direct_libero", "--mode", "dual_async"]
    if enabled:
        argv.append("--completion-contract-repair")
    monkeypatch.setattr(sys, "argv", argv)
    app.main()
    assert captured == {"mode": "dual_async", "repair": enabled}


def test_app_wires_repair_flag_to_fast_and_records_result(monkeypatch, tmp_path):
    from embodied_harness import app
    from embodied_harness.domain.models import Milestone as AppMilestone
    from test_round20c_model_role_routing import _episode_args, _install_offline_app
    from embodied_harness.runtime import milestones as milestone_module
    from embodied_harness.runtime import pipeline as pipeline_module

    _install_offline_app(monkeypatch, {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"})
    captured = []

    class CapturingSlow:
        def __init__(self, *_args, **_kwargs):
            self.reviews = 0

        def request_review(self, _reason):
            return 1

        async def run(self):
            await asyncio.Future()

        async def watch_execution(self):
            await asyncio.Future()

    class FakePipeline:
        pipelined = True

        def __init__(self, _slow, context, _coordinator, *_args):
            self.context = context
            self.plan_id = "plan-fixture"
            self.replans = self.checkpoints = self.recoveries = self.index = 0
            self.completed = []
            self.official_terminal = False
            self.remaining = 2

        async def initialize_with_scout(self, _model):
            self.context.milestone = AppMilestone("stage", 1, "place the cup", "place the cup",
                capabilities=("observe",), final_stage=True)

        async def close(self):
            pass

    class CapturingFast:
        def __init__(self, *_args, **kwargs):
            self.repair = kwargs.get("completion_contract_repair")
            self.steps = self.errors = self.false_claims = 0
            self.identity_checker = None
            self.destination_checker = None
            self.milestones = None
            captured.append(self)

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "SlowAgent", CapturingSlow)
    monkeypatch.setattr(app, "FastAgent", CapturingFast)
    monkeypatch.setattr(pipeline_module, "PipelinedMilestoneManager", FakePipeline)
    monkeypatch.setattr(milestone_module, "MilestoneManager", FakePipeline)
    args = _episode_args(tmp_path, "dual_async")
    args.hierarchy = "milestones"
    args.pipeline_supervision = True
    args.priority_foreground = True
    args.async_completion_recovery = True
    args.completion_contract_repair = True
    result = asyncio.run(app.episode(args))
    assert len(captured) == 1 and captured[0].repair is True
    assert result["completion_contract_repair"] is True
    assert result["pipeline_supervision"] is True


def test_runner_routes_audits_repair_flag_and_rejects_pure_or_combined_modes(tmp_path):
    from test_round22_runner import fixture_config, load_fixture_catalog, valid_result_for
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    selected = next(case for case in config["cases"] if case["variant"] == "patched22"
                    and case["mode"] == "dual_async" and case["suite"] == "libero_spatial")
    config["cases"] = [copy.deepcopy(selected)]
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True, completion_contract_repair=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert "--completion-contract-repair" in runner.app_arguments(job, Path("/tmp/unused-memory.json"))
    result = valid_result_for(job)
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []
    result["completion_contract_repair"] = False
    assert "mismatch_completion_contract_repair" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})

    pure = copy.deepcopy(config)
    pure["variants"]["patched22"]["modes"] = ["single_fast"]
    pure["variants"]["patched22"]["feature_flags_by_mode"] = copy.deepcopy(runner.R21_FLAGS)
    pure["variants"]["patched22"]["feature_flags_by_mode"]["single_fast"]["completion_contract_repair"] = True
    pure["cases"] = [{**selected, "mode": "single_fast"}]
    with pytest.raises(ValueError, match="completion_contract_repair requires dual_async pipeline, foreground priority and completion recovery"):
        runner.validate_config(pure)

    for incompatible in ("completion_rendezvous", "urgent_completion_review"):
        combined = copy.deepcopy(config)
        combined["variants"]["patched22"]["feature_flags_by_mode"]["dual_async"][incompatible] = True
        with pytest.raises(ValueError, match="completion_contract_repair must be tested separately"):
            runner.validate_config(combined)
