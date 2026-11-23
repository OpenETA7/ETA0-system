"""Offline tests for the default-off scoped nonterminal output schema."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.model import InferencePreempted, ModelClient, ModelError
from embodied_harness.agents.prompts import FAST
from embodied_harness.agents.milestone_prompts import FAST_CONTRACT
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.domain.models import Advice
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
    schemas = {"observe": {"parameters": {"properties": {"camera": {"type": "string"}}}},
               "move_to": {"parameters": {"properties": {"delta_m": {"type": "array"}}}}}

    def __init__(self):
        self.version = 1
        self.inspect_calls = []
        self.execute_calls = []
        self.verify_calls = []

    def validate_args(self, tool, args):
        assert tool in self.schemas and isinstance(args, dict)

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"render-{self.version}", self.version, self.generation,
            time.monotonic(), 12.5, action_seq,
            {"candidate_scope": "frame-1", "views": []}, ())

    async def inspect(self, tool, args, action_seq):
        self.inspect_calls.append((tool, copy.deepcopy(args), action_seq))
        return await self.observe(action_seq)

    async def execute(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        raise AssertionError("schema-guard tests must not execute physical actions")

    async def verify(self, observation_id):
        self.verify_calls.append(observation_id)
        return {"status": "not_verified", "source": "official_libero_check_success",
                "observation_id": observation_id}


class Pipeline:
    pipelined = True
    pending_reason = None
    checkpoint_task = None
    preparation_used = False

    def __init__(self):
        self.remaining = 2
        self.calls = 0

    async def before_decision(self, remaining):
        self.remaining = remaining
        return None

    def record_decision(self):
        self.remaining -= 1

    async def guard_proposal(self, _kind, _tool=None):
        return True, None

    def note_terminal(self, _evidence):
        pass

    def request_local_review(self, _reason):
        self.calls += 1


class Slow:
    def __init__(self):
        self.requests = []

    def request_review(self, reason):
        self.requests.append(reason)
        return len(self.requests)


class Actor:
    def __init__(self, decisions):
        self.decisions = deque(copy.deepcopy(decisions))
        self.calls = []
        self.history = deque()

    async def decide(self, system, projection, images, **kwargs):
        self.calls.append({"system": system, "projection": copy.deepcopy(projection),
                           "images": tuple(images), "kwargs": kwargs})
        return copy.deepcopy(self.decisions.popleft())


def build_fast(*, enabled=False, decisions=None, final=True):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("final", 1, "put the cup in the bowl", "place the cup",
        capabilities=("observe", "move_to"), generation=1, final_stage=final)
    context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 12.5, 0,
        {"candidate_scope": "frame-1", "views": []}, ()))
    adapter = Adapter()
    coordinator = Coordinator(adapter, context, store)
    slow = Slow()
    actor = Actor(decisions or [
        {"kind": "complete", "reason": "official negative"},
        {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}},
    ])
    fast = FastAgent(actor, context, coordinator, store, slow=slow, mode="dual_async",
        async_completion_recovery=True, completion_schema_guard=enabled)
    fast.milestones = Pipeline()
    return store, context, adapter, coordinator, slow, actor, fast


def run_async(test):
    @wraps(test)
    def wrapped(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return wrapped


def events(store, kind):
    return [row["payload"] for row in store.events if row["kind"] == kind]


def negative(observation_id):
    return {"status": "not_verified", "source": "official_libero_check_success",
            "observation_id": observation_id}


def test_only_current_official_negative_final_idle_scope_is_recorded():
    store, context, _adapter, coordinator, _slow, _actor, fast = build_fast(enabled=True)
    evidence = negative(context.latest.id)
    fast.note_completion_negative(evidence)
    assert fast.completion_excluded_scope == (1, 0, 12.5, context.control_version)
    assert len(events(store, "completion.schema_exclusion_recorded")) == 1
    assert fast.use_completion_schema()

    invalid = [
        {**evidence, "status": "uncertain"},
        {**evidence, "source": "agent_self_report"},
        {**evidence, "observation_id": "stale-observation"},
    ]
    for row in invalid:
        store2, context2, _adapter2, _coordinator2, _slow2, _actor2, agent2 = build_fast(enabled=True)
        agent2.note_completion_negative(row)
        assert agent2.completion_excluded_scope is None
        assert not events(store2, "completion.schema_exclusion_recorded")

    for attribute, value in (("paused", True), ("unknown", True), ("running", object())):
        store3, context3, _adapter3, coordinator3, _slow3, _actor3, agent3 = build_fast(enabled=True)
        setattr(coordinator3, attribute, value)
        agent3.note_completion_negative(negative(context3.latest.id))
        assert agent3.completion_excluded_scope is None

    intermediate = build_fast(enabled=True, final=False)
    intermediate[-1].note_completion_negative(negative(intermediate[1].latest.id))
    assert intermediate[-1].completion_excluded_scope is None


@pytest.mark.parametrize("sim_time", [float("nan"), float("inf"), float("-inf"), True, "12.5"])
def test_nonfinite_or_non_numeric_sim_clock_fails_closed(sim_time):
    _store, context, _adapter, _coordinator, _slow, _actor, fast = build_fast(enabled=True)
    old = context.latest
    context.ingest(Observation("bad-clock", old.version + 1, old.generation, time.monotonic(),
        sim_time, old.action_seq, copy.deepcopy(old.state), ()))
    fast.note_completion_negative(negative(context.latest.id))
    assert fast.completion_excluded_scope is None
    assert not fast.use_completion_schema()


def test_new_render_preserves_same_physical_exclusion():
    _store, context, _adapter, _coordinator, _slow, _actor, fast = build_fast(enabled=True)
    fast.note_completion_negative(negative(context.latest.id))
    scope = fast.completion_excluded_scope
    old = context.latest
    context.ingest(Observation("new-render", old.version + 1, old.generation, time.monotonic(),
        old.sim_time, old.action_seq, copy.deepcopy(old.state), ()))
    assert context.latest.id != old.id and fast.completion_physical_scope() == scope
    assert fast.use_completion_schema()


@pytest.mark.parametrize("advance", ["clock", "action", "control", "generation"])
def test_physical_clock_action_control_or_generation_change_invalidates_exclusion(advance):
    _store, context, _adapter, _coordinator, _slow, _actor, fast = build_fast(enabled=True)
    fast.note_completion_negative(negative(context.latest.id))
    assert fast.use_completion_schema()
    old = context.latest
    if advance == "clock":
        context.ingest(Observation("later-time", old.version + 1, old.generation, time.monotonic(),
            old.sim_time + 0.01, old.action_seq, copy.deepcopy(old.state), ()))
    elif advance == "action":
        context.action_seq += 1
    elif advance == "control":
        context.signal_control("advice", 40)
    else:
        context.milestone = replace(context.milestone, generation=2)
        context.ingest(Observation("new-generation", old.version + 1, 2, time.monotonic(),
            old.sim_time, old.action_seq, copy.deepcopy(old.state), ()))
    assert not fast.use_completion_schema()


@pytest.mark.parametrize("gate", ["paused", "unknown", "running", "pending_contract", "pending_reason", "checkpoint", "intermediate"])
def test_schema_not_activated_under_pause_unknown_action_contract_or_checkpoint(gate):
    _store, context, _adapter, coordinator, _slow, _actor, fast = build_fast(enabled=True)
    fast.note_completion_negative(negative(context.latest.id))
    if gate == "paused":
        coordinator.paused = True
    elif gate == "unknown":
        coordinator.unknown = True
    elif gate == "running":
        coordinator.running = SimpleNamespace(action_id="in-flight")
    elif gate == "pending_contract":
        coordinator.contract_review_pending = True
    elif gate == "pending_reason":
        fast.milestones.pending_reason = "replan"
    elif gate == "checkpoint":
        fast.milestones.checkpoint_task = SimpleNamespace(done=lambda: False)
    else:
        context.milestone = replace(context.milestone, final_stage=False)
    assert not fast.use_completion_schema()


@pytest.mark.parametrize("enabled", [False, True])
@run_async
async def test_fast_requests_schema_only_after_current_official_negative_without_more_slots_or_motion(enabled):
    store, context, adapter, coordinator, slow, actor, fast = build_fast(enabled=enabled)
    assert fast.completion_schema_guard is enabled
    assert await fast.run(2) == "decision_budget"
    assert len(actor.calls) == fast.steps == 2
    expected_prompt = FAST_CONTRACT + FAST.replace(
        "Maintain identity and complete the mission.",
        "Maintain identity and achieve ONLY the current milestone; propose stage completion before later outcomes.")
    expected_prompt += ("\nSUPERVISION PIPELINE: Automatic no-progress review runs in the background; continue useful work inside the current contract. "
        "If local_state.pipeline.phase is checkpoint_pending, choose ONE read-only observation/measurement for the upcoming goal or wait; "
        "no physical action is authorized by the preview. Explicit escalate waits for help when no useful local step remains.\n")
    assert actor.calls[0]["system"] == actor.calls[1]["system"] == expected_prompt
    assert "exclude_completion" not in actor.calls[0]["kwargs"]
    assert actor.calls[1]["kwargs"].get("exclude_completion", False) is enabled
    assert adapter.verify_calls == [actor.calls[0]["projection"]["observation"]["id"]]
    assert len(slow.requests) == 1
    assert len(adapter.inspect_calls) == 1
    assert adapter.execute_calls == [] and context.action_seq == 0
    assert fast.milestones.remaining == 0
    prepared = events(store, "fast.completion_schema_prepared")
    assert len(prepared) == int(enabled)
    if enabled:
        assert prepared[0]["observation_id"] == actor.calls[1]["projection"]["observation"]["id"]


@run_async
async def test_fast_without_an_official_negative_never_requests_exclusion():
    decisions = [
        {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}},
        {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}},
    ]
    store, _context, _adapter, coordinator, _slow, actor, fast = build_fast(enabled=True, decisions=decisions)
    assert await fast.run(2) == "decision_budget"
    assert all("exclude_completion" not in call["kwargs"] for call in actor.calls)
    assert not events(store, "fast.completion_schema_prepared")
    assert not coordinator.adapter.verify_calls


def test_actual_delegate_and_accepted_advice_invalidate_control_scope():
    _store, context, adapter, coordinator, _slow, _actor, fast = build_fast(enabled=True)
    fast.note_completion_negative(negative(context.latest.id))
    assert fast.use_completion_schema()

    # A real contract handoff is control-bearing even though schema scope
    # itself has no contract ID.
    old = context.milestone
    coordinator.delegate(Milestone(old.id, old.revision + 1, old.mission, old.goal,
        capabilities=old.capabilities, generation=old.generation, final_stage=True),
        expected_revision=old.revision)
    assert not fast.use_completion_schema()

    # A newly recorded negative can be invalidated by accepted ordinary advice.
    fast.note_completion_negative(negative(context.latest.id))
    assert fast.use_completion_schema()
    advice = Advice(context.milestone.id, context.milestone.revision, adapter.generation,
        "inspect", "look at the current scene", time.monotonic() + 30)
    assert coordinator.apply_advice(advice)
    assert not fast.use_completion_schema()


class StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content):
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class StubHTTP:
    def __init__(self, content='{"kind":"tool","tool":"observe","args":{}}'):
        self.content = content
        self.requests = []

    async def post(self, path, *, json):
        self.requests.append((path, copy.deepcopy(json)))
        return StubResponse(self.content)


def model_client(tmp_path, *, role="fast", options=None, content=None):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    model = ModelClient("http://offline.invalid/v1", "fixture-key", "fixture-model",
                        role, tmp_path, options=options)
    model.client = StubHTTP(content) if content is not None else StubHTTP()
    return model


@run_async
async def test_modelclient_actual_request_has_minimal_schema_only_when_requested(tmp_path):
    enabled = model_client(tmp_path)
    result = await enabled.decide("fast system", {"observation": {"id": "obs-9"}},
                                  exclude_completion=True)
    assert result["kind"] == "tool"
    assert len(enabled.client.requests) == enabled.calls == 1
    path, body = enabled.client.requests[0]
    assert path == "chat/completions"
    assert body["response_format"] == enabled.effective_options(exclude_completion=True)["response_format"]
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["kind"]
    assert schema["properties"]["kind"]["enum"] == ["tool", "wait", "escalate"]
    assert schema["additionalProperties"] is True
    assert enabled.history and len(enabled.history) == 2
    assert enabled.accounting["logical_requests"] == enabled.accounting["attempts_started"] == 1
    assert enabled.accounting["retries"] == 0

    disabled = model_client(tmp_path / "default-off")
    await disabled.decide("fast system", {"observation": {"id": "obs-9"}})
    _, default_body = disabled.client.requests[0]
    assert "response_format" not in default_body
    assert disabled.effective_options() == {"temperature": 0.1, "max_tokens": 4096}


@pytest.mark.parametrize("content", [
    '{"kind":"complete"}', '{"kind":"completed"}', '{"kind":"success"}',
])
@run_async
async def test_schema_violations_are_rejected_before_history_without_retry(tmp_path, content):
    client = model_client(tmp_path, content=content)
    with pytest.raises(ModelError, match="violates requested nonterminal"):
        await client.decide("fast system", {"observation": {"id": "obs-10"}},
                            exclude_completion=True)
    assert len(client.client.requests) == client.calls == 1
    assert not client.history
    assert client.accounting["logical_requests"] == client.accounting["attempts_started"] == 1
    assert client.accounting["retries"] == 0
    trace = [json.loads(row) for row in client.request_trace.read_text().splitlines()]
    assert [row["kind"] for row in trace].count("response.constraint_violation") == 1
    assert trace[0]["completion_excluded"] is True


@run_async
async def test_cut_in_cancellation_does_not_retry_or_append_history(tmp_path):
    client = model_client(tmp_path)

    class Controls:
        control_version = 0

        def __init__(self):
            self.store = Store()
            self.started = asyncio.Event()
            self.changed = asyncio.Event()

        async def wait_for_control(self, after):
            await self.changed.wait()
            return {"version": self.control_version, "received_monotonic": time.monotonic()}

    controls = Controls()
    client.control_context = controls
    original_post = client.client.post

    async def blocked_post(path, *, json):
        controls.started.set()
        await asyncio.Event().wait()

    client.client.post = blocked_post
    task = asyncio.create_task(client.decide("fast system", {
        "observation": {"id": "obs-11"}, "control_version": 0}, exclude_completion=True))
    await asyncio.wait_for(controls.started.wait(), timeout=1)
    controls.control_version = 1
    controls.changed.set()
    with pytest.raises(InferencePreempted):
        await asyncio.wait_for(task, timeout=1)
    assert client.calls == client.accounting["attempts_started"] == 1
    assert client.accounting["cancelled_attempts"] == 1
    assert client.accounting["logical_preempted"] == client.accounting["logical_cancelled"] == 1
    assert not client.history
    assert not getattr(client.client, "requests", [])
    assert original_post is not None


def test_completion_schema_option_rejects_wrong_role_or_competing_format_before_dispatch(tmp_path):
    slow = model_client(tmp_path, role="slow")
    with pytest.raises(ValueError, match="requires Fast"):
        slow.effective_options(exclude_completion=True)
    configured = model_client(tmp_path, options={"response_format": {"type": "json_object"}})
    with pytest.raises(ValueError, match="another response format"):
        configured.effective_options(exclude_completion=True)
    with pytest.raises(ValueError, match="boolean"):
        configured.effective_options(exclude_completion=1)
    assert not slow.client.requests and not configured.client.requests
    assert slow.accounting["logical_requests"] == configured.accounting["logical_requests"] == 0


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
def test_app_rejects_invalid_completion_schema_routes_before_env_loading(tmp_path, monkeypatch, bad_args):
    from embodied_harness import app

    args = SimpleNamespace(completion_schema_guard=True, output=str(tmp_path / "must-not-exist"),
        **bad_args)
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("environment should not be read"))
    with pytest.raises(ValueError, match="completion_schema_guard requires"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("conflict", [
    "completion_contract_repair", "urgent_completion_review", "completion_rendezvous",
    "recovery_reasoning", "final_stage_prompt",
])
def test_app_rejects_other_completion_ablations_before_env_loading(tmp_path, monkeypatch, conflict):
    from embodied_harness import app

    args = SimpleNamespace(mode="dual_async", backend="direct_libero", pipeline_supervision=True,
        async_completion_recovery=True, completion_schema_guard=True, output=str(tmp_path / conflict),
        **{conflict: True})
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("environment should not be read"))
    with pytest.raises(ValueError, match="separate completion ablation"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


def test_cli_flag_defaults_off_and_is_forwarded_when_selected(monkeypatch):
    from embodied_harness import app

    observed = []

    async def fake_episode(args):
        observed.append(args.completion_schema_guard)

    monkeypatch.setattr(app, "episode", fake_episode)
    monkeypatch.setattr(sys, "argv", ["embodied-eval", "--mission", "fixture", "--output", "/tmp/offline"])
    app.main()
    monkeypatch.setattr(sys, "argv", ["embodied-eval", "--mission", "fixture", "--output", "/tmp/offline",
                                       "--completion-schema-guard"])
    app.main()
    assert observed == [False, True]


def test_runner_routes_completion_schema_flag_only_in_declared_compatible_variant():
    import copy as _copy
    from test_round22_runner import fixture_config, load_fixture_catalog, valid_result_for
    import run_round22_targeted as runner

    config = fixture_config()
    flags = _copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True,
        completion_schema_guard=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = next(row for row in runner.make_jobs(config, catalog, Path(config["output_root"]))
               if row["variant"] == "patched22" and row["mode"] == "dual_async")
    command = runner.app_arguments(job, Path("/tmp/offline-memory.json"))
    assert "--completion-schema-guard" in command
    result = valid_result_for(job)
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []
    result["completion_schema_guard"] = False
    assert runner.check_result(result, job, {"package_sha256": "b" * 64})

    for conflict in ("completion_contract_repair", "urgent_completion_review",
                     "completion_rendezvous", "recovery_reasoning", "final_stage_prompt"):
        invalid = fixture_config()
        bad_flags = _copy.deepcopy(runner.R21_FLAGS)
        bad_flags["dual_async"].update(async_completion_recovery=True,
            completion_schema_guard=True, **{conflict: True})
        invalid["variants"]["patched22"]["feature_flags_by_mode"] = bad_flags
        with pytest.raises(ValueError, match="completion_schema_guard"):
            runner.validate_config(invalid)
