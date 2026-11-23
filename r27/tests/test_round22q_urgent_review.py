"""Offline scheduling regressions for urgent, scope-bound completion reviews."""
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
from embodied_harness.runtime.coordinator import Coordinator


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
    schemas = {}

    def __init__(self):
        self.version = 1

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, 1, time.monotonic(), 0.0,
            action_seq, {"candidate_scope": f"frame-{self.version}", "views": []}, ())


class Pipeline:
    pipelined = True
    pending_reason = None

    def stamp(self):
        return ("stage", 1, 1)


class SlowModel:
    """Block each model call on a barrier so cancellation and replacement order are observable."""
    def __init__(self):
        self.calls = []
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()

    async def decide(self, prompt, projection, images):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images)})
        if len(self.calls) == 1:
            self.first_started.set()
            await asyncio.Event().wait()
        self.second_started.set()
        await self.release_second.wait()
        return {"kind": "noop"}


class FastModel:
    def __init__(self):
        self.history = deque([{"role": "assistant", "content": "old completion"}])


def make_slow(*, urgent=True, priority=True, model=None):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("final", 1, "place the object", "place the object",
        capabilities=("observe",), final_stage=True)
    context.ingest(Observation("initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "frame-1", "views": []}, ()))
    adapter = Adapter()
    coordinator = Coordinator(adapter, context, store)
    slow = SlowAgent(model or SlowModel(), context, coordinator, store,
        priority_foreground=priority, urgent_completion_review=urgent)
    slow.milestones = Pipeline()
    return store, context, coordinator, slow


def events(store, kind):
    return [row["payload"] for row in store.events if row["kind"] == kind]


def run_async(test):
    @wraps(test)
    def wrapped(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return wrapped


@run_async
async def test_fast_official_negative_preempts_older_scope_and_latest_ticket_runs_without_fast_wait():
    store, context, coordinator, slow = make_slow()
    reviewer = asyncio.create_task(slow.run())
    slow.request_review("older background review")
    try:
        await asyncio.wait_for(slow.model.first_started.wait(), 1)
        old_scope = slow._review_scope
        assert old_scope is not None and old_scope[1:] == (0, 1)

        # Fast calls this after the official verifier has rejected the current result.
        context.action_seq = 1
        fast = FastAgent(FastModel(), context, coordinator, store, slow=slow, mode="dual_async")
        fast.milestones = slow.milestones
        await asyncio.wait_for(fast.recover_completion(deadline=None), 0.5)

        # Replacement inference is still blocked, but the foreground Fast method already returned.
        await asyncio.wait_for(slow.model.second_started.wait(), 1)
        request = events(store, "slow.urgent_completion_preempt_requested")
        assert len(request) == 1
        assert request[0]["interrupted_review_ticket"] == 1
        assert request[0]["ticket"] == 2
        assert request[0]["reviewed_action_seq"] == 0
        assert request[0]["current_action_seq"] == 1
        assert request[0]["wait_for_review"] is False
        assert len(events(store, "slow.urgent_completion_preempted")) == 1
        started = events(store, "slow.review_started")
        assert [(row["ticket"], row["action_seq"]) for row in started] == [(1, 0), (2, 1)]
        assert slow.model.calls[-1]["projection"]["active_action"]["sequence"] == 1
        assert slow._review_scope is not None and slow._review_scope[1:] == (1, 2)
        assert slow.delivered_ticket == 0
        assert not events(store, "completion.review_wait_started")

        slow.model.release_second.set()
        for _ in range(100):
            if slow.delivered_ticket == 2:
                break
            await asyncio.sleep(0.005)
        assert slow.delivered_ticket == 2
        assert slow._review_scope is None
    finally:
        reviewer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reviewer, 1)
    assert slow._review_scope is None


@pytest.mark.parametrize("case", ["same_action", "unstarted_task", "owner_mismatch"])
@run_async
async def test_same_action_and_unbound_or_mismatched_tasks_are_not_preempted(case):
    store, context, _coordinator, slow = make_slow()
    gate = asyncio.Event()
    current = asyncio.create_task(gate.wait())
    other = None
    if case != "unstarted_task":
        await asyncio.sleep(0)
    if case == "owner_mismatch":
        other = asyncio.create_task(gate.wait())
        await asyncio.sleep(0)
    slow._review_task = current
    if case == "same_action":
        context.action_seq = 0
        slow._review_scope = (current, 0, 1)
    elif case == "owner_mismatch":
        context.action_seq = 1
        slow._review_scope = (other, 0, 1)
    else:
        context.action_seq = 1
        slow._review_scope = None
    try:
        slow.request_completion_review("completion rejected")
        assert current.cancelling() == 0
        assert not events(store, "slow.urgent_completion_preempt_requested")
        queued = events(store, "slow.urgent_completion_queued")
        assert len(queued) == 1 and queued[0]["preempted"] is False
    finally:
        current.cancel()
        if other is not None:
            other.cancel()
        await asyncio.gather(*(t for t in (current, other) if t is not None), return_exceptions=True)


@pytest.mark.parametrize("blocker", [
    "foreground_pending", "paused", "unknown", "running", "contract_review_pending",
    "pipeline_pending", "foreground_priority_disabled",
])
@run_async
async def test_safety_and_foreground_blockers_keep_current_review_running(blocker):
    store, context, coordinator, slow = make_slow(priority=blocker != "foreground_priority_disabled")
    context.action_seq = 1
    gate = asyncio.Event()
    active = asyncio.create_task(gate.wait())
    await asyncio.sleep(0)
    slow._review_task = active
    slow._review_scope = (active, 0, 1)
    if blocker == "foreground_pending":
        slow.foreground_pending = 1
    elif blocker == "paused":
        coordinator.paused = True
    elif blocker == "unknown":
        coordinator.unknown = True
    elif blocker == "running":
        coordinator.running = SimpleNamespace(action_id="in-flight")
    elif blocker == "contract_review_pending":
        coordinator.contract_review_pending = True
    elif blocker == "pipeline_pending":
        slow.milestones.pending_reason = "contract review pending"
    try:
        slow.request_completion_review("completion rejected")
        assert active.cancelling() == 0
        assert not events(store, "slow.urgent_completion_preempt_requested")
        queued = events(store, "slow.urgent_completion_queued")
        assert len(queued) == 1 and queued[0]["preempted"] is False
    finally:
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)


@run_async
async def test_duplicate_completion_requests_cancel_old_task_once_and_default_off_never_preempts():
    store, context, _coordinator, slow = make_slow()
    context.action_seq = 1
    gate = asyncio.Event()
    active = asyncio.create_task(gate.wait())
    await asyncio.sleep(0)
    slow._review_task, slow._review_scope = active, (active, 0, 1)
    try:
        slow.request_completion_review("first negative")
        slow.request_completion_review("duplicate notification")
        assert active.cancelling() == 1
        assert len(events(store, "slow.urgent_completion_preempt_requested")) == 1
        assert len(events(store, "slow.urgent_completion_queued")) == 1
    finally:
        await asyncio.gather(active, return_exceptions=True)

    off_store, off_context, _off_coord, off = make_slow(urgent=False)
    off_context.action_seq = 1
    idle = asyncio.Event()
    current = asyncio.create_task(idle.wait())
    await asyncio.sleep(0)
    off._review_task, off._review_scope = current, (current, 0, 1)
    try:
        off.request_completion_review("negative with flag off")
        assert current.cancelling() == 0
        assert not events(off_store, "slow.urgent_completion_preempt_requested")
        assert not events(off_store, "slow.urgent_completion_queued")
    finally:
        current.cancel()
        await asyncio.gather(current, return_exceptions=True)


@run_async
async def test_default_off_fast_keeps_existing_review_active_and_uses_ordinary_request():
    store, context, coordinator, slow = make_slow(urgent=False)
    reviewer = asyncio.create_task(slow.run())
    slow.request_review("older background review")
    try:
        await asyncio.wait_for(slow.model.first_started.wait(), 1)
        context.action_seq = 1
        fast = FastAgent(FastModel(), context, coordinator, store, slow=slow, mode="dual_async")
        fast.milestones = slow.milestones
        await asyncio.wait_for(fast.recover_completion(deadline=None), 0.5)
        assert slow.request_ticket == 2
        assert len(events(store, "slow.review_started")) == 1
        assert not events(store, "slow.urgent_completion_preempt_requested")
        assert slow._review_task is not None and slow._review_task.cancelling() == 0
        assert slow.pending.is_set()
        assert not events(store, "completion.review_wait_started")
    finally:
        reviewer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reviewer, 1)
    assert slow._review_scope is None


@run_async
async def test_external_shutdown_cancellation_propagates_and_clears_task_owned_scope():
    store, _context, _coordinator, slow = make_slow()
    review = asyncio.create_task(slow.review())
    try:
        await asyncio.wait_for(slow.model.first_started.wait(), 1)
        assert slow._review_scope is not None
        review.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(review, 1)
        assert slow._review_scope is None
        assert slow._review_task is None
        assert events(store, "slow.review_finished")[-1]["outcome"] == "cancelled"
    finally:
        if not review.done():
            review.cancel()
            await asyncio.gather(review, return_exceptions=True)


@pytest.mark.parametrize("invalid", [
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
def test_app_rejects_urgent_review_without_full_route_before_environment(monkeypatch, tmp_path, invalid):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args

    args = _episode_args(tmp_path, invalid["mode"])
    args.backend = invalid["backend"]
    args.pipeline_supervision = invalid["pipeline_supervision"]
    args.priority_foreground = invalid["priority_foreground"]
    args.async_completion_recovery = invalid["async_completion_recovery"]
    args.urgent_completion_review = True
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("route validation must run first"))
    with pytest.raises(ValueError, match="urgent_completion_review requires direct dual_async pipeline, foreground priority and completion recovery"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


def test_app_rejects_urgent_review_combined_with_no_wait_incompatible_rendezvous(monkeypatch, tmp_path):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args

    args = _episode_args(tmp_path, "dual_async")
    args.backend = "direct_libero"
    args.pipeline_supervision = args.priority_foreground = args.async_completion_recovery = True
    args.urgent_completion_review = args.completion_rendezvous = True
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("combo validation must run first"))
    with pytest.raises(ValueError, match="urgent_completion_review is a no-wait ablation and cannot be combined with completion_rendezvous"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_exposes_default_off_flag(monkeypatch, tmp_path, enabled):
    from embodied_harness import app

    received = {}

    async def fake_episode(args):
        received.update(mode=args.mode, urgent=args.urgent_completion_review)

    monkeypatch.setattr(app, "episode", fake_episode)
    argv = ["app", "--mission", "place the cup", "--output", str(tmp_path / "cli"),
            "--backend", "direct_libero", "--mode", "dual_async"]
    if enabled:
        argv.append("--urgent-completion-review")
    monkeypatch.setattr(sys, "argv", argv)
    app.main()
    assert received == {"mode": "dual_async", "urgent": enabled}


def test_app_routes_flag_to_slow_and_records_result(monkeypatch, tmp_path):
    from embodied_harness import app
    from embodied_harness.domain.models import Milestone
    from test_round20c_model_role_routing import _episode_args, _install_offline_app
    from embodied_harness.runtime import milestones as milestone_module
    from embodied_harness.runtime import pipeline as pipeline_module

    _install_offline_app(monkeypatch, {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"})
    slow_agents = []

    class CapturingSlow:
        def __init__(self, *_args, **kwargs):
            self.urgent = kwargs.get("urgent_completion_review", False)
            self.reviews = 0
            slow_agents.append(self)

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
            self.context.milestone = Milestone("stage", 1, "place the cup", "place the cup",
                capabilities=("observe",), final_stage=True)

        async def close(self):
            pass

    class CapturingFast:
        def __init__(self, *_args, **_kwargs):
            self.steps = self.errors = self.false_claims = 0
            self.identity_checker = None
            self.destination_checker = None
            self.milestones = None

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
    args.urgent_completion_review = True
    result = asyncio.run(app.episode(args))
    assert len(slow_agents) == 1 and slow_agents[0].urgent is True
    assert result["urgent_completion_review"] is True
    assert result["pipeline_supervision"] is True


def test_runner_routes_audits_urgent_flag_and_rejects_pure_mode_or_rendezvous(monkeypatch, tmp_path):
    from test_round22_runner import fixture_config, load_fixture_catalog, valid_result_for
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    selected = next(case for case in config["cases"] if case["variant"] == "patched22"
                    and case["mode"] == "dual_async" and case["suite"] == "libero_spatial")
    config["cases"] = [copy.deepcopy(selected)]
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True, urgent_completion_review=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert job["feature_flags"]["urgent_completion_review"] is True
    assert "--urgent-completion-review" in runner.app_arguments(job, Path("/tmp/unused-memory.json"))
    result = valid_result_for(job)
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []
    result["urgent_completion_review"] = False
    assert "mismatch_urgent_completion_review" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})

    pure = copy.deepcopy(config)
    pure["variants"]["patched22"]["modes"] = ["single_fast"]
    pure["variants"]["patched22"]["feature_flags_by_mode"] = copy.deepcopy(runner.R21_FLAGS)
    pure["variants"]["patched22"]["feature_flags_by_mode"]["single_fast"]["urgent_completion_review"] = True
    pure["cases"] = [{**selected, "mode": "single_fast"}]
    with pytest.raises(ValueError, match="urgent_completion_review requires dual_async pipeline, foreground priority and completion recovery"):
        runner.validate_config(pure)

    combined = copy.deepcopy(config)
    combined["variants"]["patched22"]["feature_flags_by_mode"]["dual_async"]["completion_rendezvous"] = True
    with pytest.raises(ValueError, match="urgent_completion_review cannot combine with completion_rendezvous"):
        runner.validate_config(combined)
