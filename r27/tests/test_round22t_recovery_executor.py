"""Offline coverage for the bounded Strong recovery executor route."""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.model import InferencePreempted, ModelClient, ModelError
from embodied_harness.domain.models import Advice, Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class Store:
    def __init__(self):
        self.rows = []

    def emit(self, kind, payload, event_id=None):
        self.rows.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.rows[cursor:]

    def of(self, kind):
        return [row["payload"] for row in self.rows if row["kind"] == kind]


class Adapter:
    generation = 1
    terminal = False
    cancellation = "safe_boundary_only"
    schemas = {
        "observe": {"parameters": {"properties": {"camera": {"type": "string"}}}},
        "move_to": {"parameters": {"properties": {"delta_m": {"type": "array"}}}},
    }

    def __init__(self):
        self.version = 1
        self.executed = []
        self.inspected = []

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, self.generation,
            time.monotonic(), 4.0, action_seq, {"candidate_scope": "fixture", "views": []}, ())

    async def inspect(self, tool, args, action_seq):
        self.inspected.append((tool, copy.deepcopy(args), action_seq))
        return await self.observe(action_seq)

    def validate_args(self, tool, args):
        assert tool in self.schemas and isinstance(args, dict)

    async def execute(self, *args, **kwargs):
        self.executed.append((args, kwargs))
        raise AssertionError("these tests do not execute physical actions")

    async def verify(self, observation_id):
        return {"status": "not_verified", "source": "official_libero_check_success",
                "observation_id": observation_id}


class Pipeline:
    pipelined = True
    pending_reason = None
    checkpoint_task = None
    preparation_used = False

    def __init__(self, remaining=8):
        self.remaining = remaining

    async def before_decision(self, remaining):
        self.remaining = remaining
        return None

    def record_decision(self):
        self.remaining -= 1

    async def guard_proposal(self, _kind, _tool=None):
        return True, None

    def note_readonly(self, *_args):
        pass

    def note_preparation_points(self, *_args):
        pass

    def note_terminal(self, _evidence):
        pass


class Slow:
    def __init__(self):
        self.requests = []

    def request_review(self, reason):
        self.requests.append(reason)
        return len(self.requests)


class Actor:
    model = "fast-fixture"
    role = "fast"

    def __init__(self, decisions=()):
        self.decisions = deque(copy.deepcopy(list(decisions)))
        self.history = deque(maxlen=6)
        self.calls = []

    async def decide(self, prompt, projection, images=(), **kwargs):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images), "kwargs": kwargs})
        return copy.deepcopy(self.decisions.popleft())


class Executor:
    model = "strong-fixture"
    role = "recovery_executor"

    def __init__(self, decisions=(), error=None):
        self.decisions = deque(copy.deepcopy(list(decisions)))
        self.error = error
        self.history = deque(maxlen=6)
        self.accounting = {"logical_requests": 0}
        self.calls = []

    async def decide(self, prompt, projection, images=()):
        self.accounting["logical_requests"] += 1
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images), "history": copy.deepcopy(list(self.history))})
        self.history.append({"role": "assistant", "content": "strong-returned-decision"})
        if self.error:
            raise self.error
        return copy.deepcopy(self.decisions.popleft())


def build(*, remaining=8, actor_decisions=(), executor_decisions=(), executor_error=None,
          enabled=True):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("final", 1, "put the cup in the bowl", "place the cup",
        capabilities=("observe", "move_to"), generation=1, final_stage=True)
    context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 4.0, 0,
        {"candidate_scope": "fixture", "views": []}, ()))
    adapter = Adapter()
    coordinator = Coordinator(adapter, context, store)
    actor = Actor(actor_decisions)
    executor = Executor(executor_decisions, executor_error)
    slow = Slow()
    fast = FastAgent(actor, context, coordinator, store, slow=slow, mode="dual_async",
        async_completion_recovery=True,
        recovery_executor_model=executor if enabled else None)
    fast.milestones = Pipeline(remaining)
    return store, context, adapter, coordinator, slow, actor, executor, fast


def official_negative(context):
    return {"status": "not_verified", "source": "official_libero_check_success",
            "observation_id": context.latest.id}


def arm(fast, context):
    fast.arm_recovery_executor(official_negative(context))


def test_first_current_official_negative_arms_only_with_two_future_slots_and_only_once():
    store, context, _adapter, _coordinator, _slow, _actor, _executor, fast = build(remaining=3)
    arm(fast, context)
    assert fast.recovery_executor_armed
    assert fast.recovery_executor_remaining == 3
    assert store.of("executor_recovery.armed")[0]["episode_decisions_remaining"] == 3

    arm(fast, context)
    assert len(store.of("executor_recovery.armed")) == 1
    assert fast.recovery_executor_remaining == 3

    no_budget = build(remaining=1)
    arm(no_budget[-1], no_budget[1])
    assert not no_budget[-1].recovery_executor_armed
    assert no_budget[-1].recovery_executor_remaining == 0
    assert no_budget[0].of("executor_recovery.skipped")[0]["reason"] == "fewer_than_two_future_decisions"


@pytest.mark.parametrize("invalid", [
    "uncertain", "wrong_source", "stale_observation", "intermediate", "paused",
    "running", "unknown", "action_mismatch", "generation_mismatch",
])
def test_arm_gate_requires_official_negative_current_final_idle_scope(invalid):
    store, context, _adapter, coordinator, _slow, _actor, _executor, fast = build()
    evidence = official_negative(context)
    if invalid == "uncertain":
        evidence["status"] = "uncertain"
    elif invalid == "wrong_source":
        evidence["source"] = "actor_claim"
    elif invalid == "stale_observation":
        evidence["observation_id"] = "old-frame"
    elif invalid == "intermediate":
        context.milestone = replace(context.milestone, final_stage=False)
    elif invalid == "paused":
        coordinator.paused = True
    elif invalid == "running":
        coordinator.running = SimpleNamespace(action_id="in-flight")
    elif invalid == "unknown":
        coordinator.unknown = True
    elif invalid == "action_mismatch":
        context.action_seq = 1
    else:
        context.milestone = replace(context.milestone, generation=2)
    fast.arm_recovery_executor(evidence)
    assert not fast.recovery_executor_armed
    assert not store.of("executor_recovery.armed")


def test_fast_replaces_next_actor_calls_inside_original_decision_budget():
    decisions = [{"kind": "complete", "reason": "claim"}]
    strong = [
        {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}},
        {"kind": "tool", "tool": "observe", "args": {"camera": "wrist"}},
        {"kind": "tool", "tool": "observe", "args": {"camera": "overhead"}},
    ]
    store, context, adapter, coordinator, slow, actor, executor, fast = build(
        remaining=8, actor_decisions=decisions, executor_decisions=strong)

    async def run():
        return await fast.run(4)

    assert asyncio.run(run()) == "decision_budget"
    assert fast.steps == 4
    assert len(actor.calls) == 1
    assert len(executor.calls) == 3
    assert fast.recovery_executor_requests == 3
    assert fast.recovery_executor_remaining == 0
    assert len(adapter.executed) == 0 and len(adapter.inspected) == 3
    assert context.action_seq == 0
    assert len(store.of("fast.decision")) == 4
    assert len(store.of("executor_recovery.request_started")) == 3
    assert [e["outcome"] for e in store.of("executor_recovery.request_finished")] == ["completed"] * 3
    # The rejected completion clears the primary stale history as usual; the
    # Strong calls are new actor slots, not auxiliary action-selection calls.
    assert not actor.calls[0]["kwargs"]
    assert len(slow.requests) >= 1


def test_new_ordinary_advice_is_in_the_next_executor_projection_and_not_action_authority():
    store, context, adapter, coordinator, slow, actor, executor, fast = build(
        actor_decisions=[{"kind": "complete"}],
        executor_decisions=[
            {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}},
            {"kind": "tool", "tool": "observe", "args": {"camera": "wrist"}},
        ])
    original_request_review = slow.request_review

    def publish_advice(reason):
        result = original_request_review(reason)
        advice = Advice(context.milestone.id, context.milestone.revision, context.milestone.generation,
            "inspect", "recheck the current destination", time.monotonic() + 30)
        assert coordinator.apply_advice(advice)
        return result

    slow.request_review = publish_advice
    assert asyncio.run(fast.run(3)) == "decision_budget"
    assert len(actor.calls) == 1 and len(executor.calls) == 2
    projection = executor.calls[0]["projection"]
    assert projection["control_version"] == context.control_version
    assert [row["text"] for row in projection["advice"]] == ["recheck the current destination"]
    assert len(adapter.executed) == 0 and len(adapter.inspected) == 2
    assert len(store.of("advice.accepted")) == 1


def test_routing_copies_actual_history_and_commits_only_successful_response():
    _store, context, _adapter, _coordinator, _slow, actor, executor, fast = build(executor_decisions=[
        {"kind": "wait"}, {"kind": "wait"}])
    actor.history.extend([
        {"role": "user", "content": "executed tool result"},
        {"role": "assistant", "content": "executed decision"},
    ])
    arm(fast, context)
    starting = copy.deepcopy(list(actor.history))
    asyncio.run(fast.decide_executor("system", {"observation": {"id": context.latest.id}}, (), {}))
    assert executor.calls[0]["history"] == starting
    assert list(actor.history) == [*starting, {"role": "assistant", "content": "strong-returned-decision"}]

    # Existing identity recovery clears primary history; the next routed call
    # copies that cleared state instead of restoring the stale prefix.
    actor.history.clear()
    asyncio.run(fast.decide_executor("system", {"observation": {"id": context.latest.id}}, (), {}))
    assert executor.calls[1]["history"] == []
    assert len(actor.history) == 1


@pytest.mark.parametrize("failure", [ModelError("parse failure"), asyncio.CancelledError()])
def test_failed_or_cancelled_routed_request_consumes_allowance_without_committing_history(failure):
    _store, context, _adapter, _coordinator, _slow, actor, executor, fast = build(executor_error=failure)
    actor.history.append({"role": "assistant", "content": "stable-primary-history"})
    original = copy.deepcopy(list(actor.history))
    arm(fast, context)
    with pytest.raises(type(failure)):
        asyncio.run(fast.decide_executor("system", {"observation": {"id": context.latest.id}}, (), {}))
    assert fast.recovery_executor_remaining == 3
    assert list(actor.history) == original
    assert executor.history != deque(original, maxlen=6)
    assert _store.of("executor_recovery.request_finished")[-1]["outcome"] in {"failed", "cancelled"}


@pytest.mark.parametrize("changed", ["id", "revision", "generation", "mission"])
def test_contract_change_expires_unused_calls_and_never_rearms(changed):
    _store, context, _adapter, _coordinator, _slow, actor, executor, fast = build(
        actor_decisions=[{"kind": "wait"}])
    arm(fast, context)
    old = context.milestone
    updates = {"id": old.id, "revision": old.revision, "generation": old.generation,
               "mission": old.mission}
    if changed == "id":
        updates["id"] = "new-contract"
    elif changed == "revision":
        updates["revision"] += 1
    elif changed == "generation":
        updates["generation"] += 1
    else:
        updates["mission"] = "different mission"
    context.milestone = replace(old, **updates)
    asyncio.run(fast.decide_executor("system", {"observation": {"id": context.latest.id}}, (), {}))
    assert len(actor.calls) == 1 and not executor.calls
    assert fast.recovery_executor_remaining == 0
    assert _store.of("executor_recovery.expired")[-1]["reason"] == "contract_changed"
    arm(fast, context)
    assert fast.recovery_executor_remaining == 0
    assert len(_store.of("executor_recovery.armed")) == 1


def test_default_off_and_inference_option_ablations_keep_primary_route():
    store, context, _adapter, _coordinator, _slow, actor, executor, fast = build(
        enabled=False, actor_decisions=[{"kind": "wait"}])
    arm(fast, context)
    result = asyncio.run(fast.decide_executor("system", {"observation": {"id": context.latest.id}}, (), {}))
    assert result["kind"] == "wait"
    assert len(actor.calls) == 1 and not executor.calls
    assert not store.of("executor_recovery.armed")

    enabled = build()
    arm(enabled[-1], enabled[1])
    with pytest.raises(ValueError, match="separate from request-option ablations"):
        asyncio.run(enabled[-1].decide_executor("system", {}, (), {"recovery_reasoning": True}))
    assert not enabled[6].calls


class StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content):
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class StubHTTP:
    def __init__(self, content=None, *, block=False):
        self.content = content or '{"kind":"tool","tool":"observe","args":{}}'
        self.block = block
        self.started = asyncio.Event()

    async def post(self, path, *, json):
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        return StubResponse(self.content)


def real_executor(tmp_path, context, *, content=None, block=False):
    model = ModelClient("http://offline.invalid/v1", "fake-key", "strong-profile",
        "recovery_executor", tmp_path)
    model.client = StubHTTP(content, block=block)
    model.control_context = context
    return model


def test_actual_control_cut_in_during_strong_request_cancels_without_stale_history(tmp_path):
    store, context, _adapter, coordinator, _slow, actor, _executor, fast = build()
    actor.history.append({"role": "assistant", "content": "last executed decision"})
    original = copy.deepcopy(list(actor.history))
    strong = real_executor(tmp_path, context, block=True)
    fast.recovery_executor_model = strong
    arm(fast, context)

    async def cut_in():
        task = asyncio.create_task(fast.decide_executor("system", {
            "observation": {"id": context.latest.id}, "control_version": context.control_version}, (), {}))
        await asyncio.wait_for(strong.client.started.wait(), timeout=1)
        coordinator.pause("test pause during inference")
        with pytest.raises(InferencePreempted):
            await asyncio.wait_for(task, timeout=1)

    asyncio.run(cut_in())
    assert coordinator.paused
    assert list(actor.history) == original
    assert fast.recovery_executor_remaining == 3
    assert strong.calls == strong.accounting["attempts_started"] == 1
    assert strong.accounting["cancelled_attempts"] == 1
    assert store.of("control.inference_preempted")
    assert store.of("executor_recovery.request_finished")[-1]["outcome"] == "preempted"


def test_real_strong_parse_failure_consumes_slot_without_history_commit(tmp_path):
    _store, context, _adapter, _coordinator, _slow, actor, _executor, fast = build()
    actor.history.append({"role": "assistant", "content": "executed-only"})
    original = copy.deepcopy(list(actor.history))
    strong = real_executor(tmp_path, context, content="not-json")
    fast.recovery_executor_model = strong
    arm(fast, context)
    with pytest.raises(ModelError):
        asyncio.run(fast.decide_executor("system", {"observation": {"id": context.latest.id}}, (), {}))
    assert list(actor.history) == original
    assert fast.recovery_executor_remaining == 3
    assert strong.calls == strong.accounting["attempts_started"] == 1
    assert not strong.history or list(strong.history) == original


def test_app_cli_and_runner_route_flag_and_reject_conflicting_completion_modes(tmp_path, monkeypatch):
    from embodied_harness import app

    parsed = []

    async def fake_episode(args):
        parsed.append(args.strong_recovery_executor)

    monkeypatch.setattr(app, "episode", fake_episode)
    monkeypatch.setattr(sys, "argv", ["embodied-eval", "--mission", "fixture", "--output", "/tmp/offline"])
    app.main()
    monkeypatch.setattr(sys, "argv", ["embodied-eval", "--mission", "fixture", "--output", "/tmp/offline",
                                       "--strong-recovery-executor"])
    app.main()
    assert parsed == [False, True]

    import copy as _copy
    from test_round22_runner import fixture_config, load_fixture_catalog, valid_result_for
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    flags = _copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True, strong_recovery_executor=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = next(row for row in runner.make_jobs(config, catalog, Path(config["output_root"]))
               if row["variant"] == "patched22" and row["mode"] == "dual_async")
    assert job["expected_active_models"]["recovery_executor"] == job["slow_profile"]["model"]
    assert "--strong-recovery-executor" in runner.app_arguments(job, tmp_path / "memory.json")
    result = valid_result_for(job)
    result["models"].append({"role": "recovery_executor", "model": job["slow_profile"]["model"],
                             "request_accounting": {"logical_requests": 0}})
    result["recovery_executor_policy"] = {
        "model": job["slow_profile"]["model"], "options": job["slow_profile"]["options"],
        "role": "recovery_executor", "max_requests": 4,
        "trigger": "first_official_negative_final_claim_with_two_future_decisions",
        "scheduling": "replace_actor_calls_within_existing_decision_budget",
        "scope": "current_final_contract_id_revision_generation_mission",
        "history": "copy_primary_executed_history_and_commit_only_returned_decisions",
        "cost": "separate_strong_executor_accounting_no_planner_tool_access"}
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []
    result["models"][-1]["request_accounting"]["logical_requests"] = 5
    assert "recovery_executor_request_budget_exceeded_or_missing" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})
    result["models"][-1]["request_accounting"]["logical_requests"] = 0
    result["models"].append(copy.deepcopy(result["models"][-1]))
    assert "invalid_recovery_executor_role_count" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})

    for conflict in ("completion_schema_guard", "completion_contract_repair", "urgent_completion_review",
                     "completion_rendezvous", "recovery_reasoning", "final_stage_prompt"):
        invalid = fixture_config()
        bad = _copy.deepcopy(runner.R21_FLAGS)
        bad["dual_async"].update(async_completion_recovery=True, strong_recovery_executor=True,
                                 **{conflict: True})
        invalid["variants"]["patched22"]["feature_flags_by_mode"] = bad
        with pytest.raises(ValueError, match="strong_recovery_executor"):
            runner.validate_config(invalid)


def test_app_binds_distinct_tool_enabled_recovery_executor_but_tool_free_slow(tmp_path, monkeypatch):
    from embodied_harness import app
    import embodied_harness.runtime.memory as memory_module
    import embodied_harness.runtime.pipeline as pipeline_module

    memory = [{"image": "fixture.png", "image_sha256": "abc", "manifest_sha256": "def",
               "caption": "matched fixture reference"}]
    monkeypatch.setattr(memory_module, "retrieve", lambda *_args, **_kwargs: copy.deepcopy(memory))
    for name, value in {
        "FAST_MODEL": "fast-model", "SLOW_MODEL": "strong-model",
        "FAST_API_KEY": "offline-fast-key", "SLOW_API_KEY": "offline-strong-key",
        "MODEL_BASE_URL": "http://offline.invalid/v1", "FAST_MODEL_OPTIONS": "{}",
        "SLOW_MODEL_OPTIONS": '{"temperature":0.1}',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(app, "load_env", lambda _path: None)

    class FakeStore(Store):
        def __init__(self, _path):
            super().__init__()

        def close(self):
            pass

    class FakeAdapter(Adapter):
        def __init__(self, *_args):
            super().__init__()
            self.host = None

        async def start(self):
            pass

        async def close(self):
            pass

    class FakeModel:
        instances = []

        def __init__(self, base_url, key, model, role, output, options=None):
            self.base_url, self.key, self.model, self.role = base_url, key, model, role
            self.options = options or {}
            self.calls = 0
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self.accounting = {"responses_with_usage": 0}
            self.latencies = []
            self.history = deque(maxlen=6)
            self.control_context = None
            self.tool_schemas = None
            self.memory = None
            self.mission_reference = None
            self.closed = False
            self.instances.append(self)

        def bind_tools(self, schemas):
            self.tool_schemas = copy.deepcopy(schemas)

        def bind_memory(self, entries):
            self.memory = copy.deepcopy(entries)

        def bind_mission_reference(self, observation):
            self.mission_reference = observation

        async def decide(self, *_args, **_kwargs):
            raise AssertionError("the offline app wiring test must not make model requests")

        async def close(self):
            self.closed = True

    class FakeSlowAgent:
        instances = []

        def __init__(self, model, *_args, **_kwargs):
            self.model = model
            self.reviews = 0
            self.urgent_completion_review = False
            self.instances.append(self)

        def request_review(self, _reason):
            self.reviews += 1
            return self.reviews

        async def run(self):
            await asyncio.Event().wait()

        async def watch_execution(self):
            await asyncio.Event().wait()

    class FakeManager(Pipeline):
        def __init__(self, _slow, context, coordinator, _store, _budget, _deadline):
            super().__init__(remaining=4)
            self.context, self.coordinator = context, coordinator
            self.scout_decisions = 0
            self.plan_id = "fixture-plan"
            self.replans = 0
            self.completed = []
            self.checkpoints = 0
            self.official_terminal = False
            self.recoveries = 0
            self.index = 0

        async def initialize_with_scout(self, _main):
            authority = self.coordinator.mission_authority
            self.coordinator.delegate(Milestone("final", 1, authority["mission"], authority["mission"],
                capabilities=tuple(authority["capabilities"]), generation=1, final_stage=True))

        async def close(self):
            pass

    class FakeFast:
        instances = []

        def __init__(self, model, context, coordinator, store, slow, mode, review_every, **_kwargs):
            self.model, self.context, self.coordinator, self.store = model, context, coordinator, store
            self.slow, self.mode, self.review_every = slow, mode, review_every
            self.steps = self.false_claims = self.errors = 0
            self.recovery_executor_model = None
            self.milestones = None
            self.instances.append(self)

        async def run(self, _budget, deadline=None):
            return "decision_budget"

    fake_libero = ModuleType("embodied_harness.adapters.libero_direct")
    fake_libero.DirectLiberoAdapter = FakeAdapter
    monkeypatch.setitem(sys.modules, "embodied_harness.adapters.libero_direct", fake_libero)
    monkeypatch.setattr(app, "EventStore", FakeStore)
    monkeypatch.setattr(app, "ModelClient", FakeModel)
    monkeypatch.setattr(app, "SlowAgent", FakeSlowAgent)
    monkeypatch.setattr(app, "FastAgent", FakeFast)
    monkeypatch.setattr(pipeline_module, "PipelinedMilestoneManager", FakeManager)

    args = SimpleNamespace(
        mode="dual_async", backend="direct_libero", pipeline_supervision=True,
        async_completion_recovery=True, strong_recovery_executor=True,
        output=str(tmp_path / "offline-episode"), model_env="unused.env", mission="fixture mission",
        sim_url="http://offline.invalid/sse", env_id="fixture/env", seed=0,
        timeout=30, max_decisions=4, review_every=3, hierarchy="milestones", identity_check=False,
        priority_foreground=False, review_economy=False, diagnostic_memory=False,
        memory="fixture-manifest.json", appearance_reference_memory=False,
    )
    result = asyncio.run(app.episode(args))
    by_role = {model.role: model for model in FakeModel.instances}
    primary, slow, executor = by_role["fast"], by_role["slow"], by_role["recovery_executor"]
    assert executor is not slow and executor.model == slow.model == "strong-model"
    assert executor.base_url == slow.base_url == "http://offline.invalid/v1"
    assert primary.tool_schemas == executor.tool_schemas == FakeAdapter.schemas
    assert slow.tool_schemas is None
    assert primary.memory == slow.memory == executor.memory == memory
    assert primary.mission_reference is slow.mission_reference is executor.mission_reference
    assert executor.control_context is FakeFast.instances[0].context
    assert slow.control_context is None
    assert FakeFast.instances[0].recovery_executor_model is executor
    assert result["recovery_executor_policy"]["max_requests"] == 4
    assert {row["role"] for row in result["models"]} == {"fast", "slow", "recovery_executor"}
    assert all(model.closed for model in FakeModel.instances)


@pytest.mark.parametrize("bad_args", [
    {"mode": "single_fast", "backend": "direct_libero", "pipeline_supervision": True,
     "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "eta", "pipeline_supervision": True,
     "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "direct_libero", "pipeline_supervision": False,
     "async_completion_recovery": True},
    {"mode": "dual_async", "backend": "direct_libero", "pipeline_supervision": True,
     "async_completion_recovery": False},
])
def test_app_rejects_invalid_strong_executor_route_before_env_loading(tmp_path, monkeypatch, bad_args):
    from embodied_harness import app

    args = SimpleNamespace(strong_recovery_executor=True, output=str(tmp_path / "no-run"), **bad_args)
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("environment should not be read"))
    with pytest.raises(ValueError, match="strong_recovery_executor requires"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()

