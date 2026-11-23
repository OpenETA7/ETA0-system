"""Offline regressions for opt-in, prompt-only measured bowl-rim guidance."""
import asyncio
import copy
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.grasp_guidance import POLICY_ID, rim_guided_prompt
from embodied_harness.agents.milestone_prompts import FAST_FINAL_CONTRACT
from embodied_harness.agents.model import ModelClient
from embodied_harness.agents.prompts import FAST
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.pipeline import SCOUT


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
    schemas = {
        "observe": {"parameters": {"properties": {"camera": {"type": "string"}}}},
        "grasp_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
    }

    def __init__(self, *, reject_grasp=False):
        self.version = 1
        self.reject_grasp = reject_grasp
        self.validation_calls, self.inspect_calls, self.execute_calls = [], [], []

    def validate_args(self, tool, args):
        self.validation_calls.append((tool, copy.deepcopy(args)))
        if tool == "grasp_point" and self.reject_grasp:
            raise ValueError("fixture validation gate")

    async def inspect(self, tool, args, action_seq):
        self.inspect_calls.append((tool, copy.deepcopy(args), action_seq))
        self.version += 1
        return Observation(f"read-{self.version}", self.version, 1, time.monotonic(), 0.0,
            action_seq, {"candidate_scope": "frame-1/objects", "views": []}, ())

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"observe-{self.version}", self.version, 1, time.monotonic(), 0.0,
            action_seq, {"candidate_scope": "frame-1/objects", "views": []}, ())

    async def execute(self, capability, args, *_args):
        self.execute_calls.append((capability, copy.deepcopy(args)))
        return {"macro_status": "reached"}

    async def verify(self, _observation_id):
        return {"status": "not_verified"}


class Actor:
    def __init__(self, decision=None):
        self.history = deque(maxlen=6)
        self.decision = decision or {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}}
        self.calls = []

    async def decide(self, system, projection, images, **kwargs):
        self.calls.append({"system": system, "projection": copy.deepcopy(projection),
                           "images": tuple(images), "kwargs": kwargs})
        return copy.deepcopy(self.decision)


class Milestones:
    def __init__(self, *, pipelined=False, checkpoint_task=None):
        self.pipelined = pipelined
        self.checkpoint_task = checkpoint_task
        self.preparation_used = False
        self.pending_reason = None
        self.remaining = 2

    async def before_decision(self, _remaining):
        return None

    def record_decision(self):
        pass

    async def guard_proposal(self, _kind, _tool=None):
        return True, None

    def note_readonly(self, _before, _after):
        pass

    def note_preparation_points(self, _prior_ids):
        pass


def build_agent(*, enabled=False, decision=None, milestones=None, final_stage=False,
                paused=False, reject_grasp=False):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("stage", 1, "pick up the bowl", "move the object",
        capabilities=("observe", "grasp_point"), final_stage=final_stage)
    context.ingest(Observation("initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "frame-1/objects", "views": []}, ()))
    adapter = Adapter(reject_grasp=reject_grasp)
    coordinator = Coordinator(adapter, context, store)
    coordinator.paused = paused
    actor = Actor(decision)
    fast = FastAgent(actor, context, coordinator, store, mode="dual_async",
                     final_stage_prompt=final_stage, rim_grasp_guidance=enabled)
    fast.milestones = milestones
    return fast, actor, adapter, context, coordinator, store


def test_rim_prompt_replaces_only_contradictory_instructions_and_keeps_fast_immutable():
    original = FAST
    guided = rim_guided_prompt(FAST)
    assert guided != FAST
    assert FAST is original
    assert "Choose the contact geometry BEFORE grasping" in guided
    assert "top_rim, bottom_rim, left_rim or right_rim" in guided
    assert "suggested_jaw_world" in guided
    assert "When present, OMIT BOTH jaw_world and" in guided
    assert "If the point has no suggested_jaw_world, omission preserves the current wrist orientation" in guided
    assert "it does NOT infer a bowl-wall pinch" in " ".join(guided.split())
    assert "Set jaw_world so closing spans the local rim wall" not in guided
    assert "approach_world=[0,0,-1]" not in guided
    assert rim_guided_prompt(FAST) == guided


def test_prompt_drift_fails_closed():
    with pytest.raises(ValueError, match="Rim guidance no longer matches the executor prompt"):
        rim_guided_prompt(FAST + FAST)


@pytest.mark.parametrize("enabled", [False, True])
def test_normal_decision_uses_exact_default_or_guided_prompt_without_extra_inference(enabled):
    async def run():
        fast, actor, _adapter, _context, _coordinator, store = build_agent(enabled=enabled)
        assert await fast.run(1) == "decision_budget"
        assert len(actor.calls) == 1
        expected = rim_guided_prompt(FAST) if enabled else FAST
        assert actor.calls[0]["system"] == expected
        prepared = [row for row in store.events if row["kind"] == "fast.rim_guidance_prepared"]
        assert bool(prepared) is enabled
        if enabled:
            assert prepared[0]["payload"]["policy_id"] == POLICY_ID

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
def test_final_contract_uses_the_same_selected_prompt_variant(enabled):
    async def run():
        fast, actor, _adapter, _context, _coordinator, store = build_agent(
            enabled=enabled, final_stage=True, milestones=Milestones())
        assert await fast.run(1) == "decision_budget"
        prompt = actor.calls[0]["system"]
        base = rim_guided_prompt(FAST) if enabled else FAST
        expected = FAST_FINAL_CONTRACT + base.replace(
            "Maintain identity and complete the mission.",
            "Maintain identity and achieve the complete original mission under the current final contract.")
        assert prompt == expected
        assert len(actor.calls) == 1
        assert bool([row for row in store.events if row["kind"] == "fast.rim_guidance_prepared"]) is enabled

    asyncio.run(run())


def test_checkpoint_scout_prompt_suppresses_rim_guidance_and_prepared_event():
    async def run():
        checkpoint = asyncio.create_task(asyncio.Event().wait())
        fast, actor, _adapter, _context, _coordinator, store = build_agent(
            enabled=True, milestones=Milestones(pipelined=True, checkpoint_task=checkpoint))
        try:
            assert await fast.run(1) == "decision_budget"
            prompt = actor.calls[0]["system"]
            assert prompt.startswith(SCOUT)
            assert "CHECKPOINT PREPARATION" in prompt
            assert "suggested_jaw_world" not in prompt
            assert len(actor.calls) == 1
            assert not [row for row in store.events if row["kind"] == "fast.rim_guidance_prepared"]
        finally:
            checkpoint.cancel()
            with pytest.raises(asyncio.CancelledError):
                await checkpoint

    asyncio.run(run())


def test_pause_and_tool_validation_gates_are_unchanged_by_prompt_flag():
    async def run():
        outcomes = []
        for enabled in (False, True):
            fast, actor, adapter, _context, coordinator, store = build_agent(enabled=enabled, paused=True)
            status = await fast.run(1)
            outcomes.append((status, len(actor.calls), len(adapter.validation_calls), fast.errors,
                            coordinator.action_seq if hasattr(coordinator, "action_seq") else fast.context.action_seq,
                            len(adapter.execute_calls), len(store.events)))
        assert outcomes[0][:6] == outcomes[1][:6] == ("paused", 0, 0, 0, 0, 0)

        failed = []
        for enabled in (False, True):
            decision = {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "p1"}}
            fast, actor, adapter, context, coordinator, store = build_agent(
                enabled=enabled, decision=decision, reject_grasp=True)
            status = await fast.run(1)
            failed.append((status, len(actor.calls), len(adapter.validation_calls), fast.errors,
                           context.action_seq, coordinator.running,
                           sum(row["kind"] == "action.accepted" for row in store.events),
                           len(adapter.execute_calls)))
        assert failed[0] == failed[1]
        assert failed[0] == ("decision_budget", 1, 1, 1, 0, None, 0, 0)

    asyncio.run(run())


def test_app_direct_backend_gate_and_result_flag_propagation(monkeypatch, tmp_path):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args, _install_offline_app

    invalid = SimpleNamespace(rim_grasp_guidance=True, backend="eta")
    with pytest.raises(ValueError, match="rim_grasp_guidance describes direct_libero measured candidate geometry"):
        asyncio.run(app.episode(invalid))

    endpoints = {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"}
    constructed = _install_offline_app(monkeypatch, endpoints)
    agents = []

    class CapturingFast:
        def __init__(self, *_args, **kwargs):
            self.steps = self.false_claims = self.errors = 0
            self.identity_checker = None
            self.guidance = kwargs.get("rim_grasp_guidance")
            agents.append(self)

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "FastAgent", CapturingFast)
    args = _episode_args(tmp_path, "dual_async")
    args.rim_grasp_guidance = True
    result = asyncio.run(app.episode(args))
    assert agents[0].guidance is True
    assert result["rim_grasp_guidance"] is True
    assert [model.role for model in constructed] == ["fast", "slow", "grounding"]


def test_cli_and_runner_expose_and_audit_default_off_flag(monkeypatch, tmp_path):
    import argparse
    from embodied_harness import app
    from test_round22_runner import fixture_config, load_fixture_catalog

    captured = {}

    async def fake_episode(args):
        captured["flag"] = args.rim_grasp_guidance

    monkeypatch.setattr(app, "episode", fake_episode)
    monkeypatch.setattr(sys, "argv", ["app", "--mission", "pick up a bowl", "--output",
        str(tmp_path / "cli"), "--backend", "direct_libero", "--rim-grasp-guidance"])
    app.main()
    assert captured["flag"] is True

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner
    config = fixture_config()
    config["cases"] = [{"suite": "libero_spatial", "task_index": 6, "seed": 36,
                        "variant": "patched22", "mode": "dual_async"}]
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert not job["feature_flags"].get("rim_grasp_guidance", False)
    assert "--rim-grasp-guidance" not in runner.app_arguments(job, Path("/tmp/memory.json"))

    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"]["rim_grasp_guidance"] = True
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert "--rim-grasp-guidance" in runner.app_arguments(job, Path("/tmp/memory.json"))
    result = {"backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "evidence": {"status": "not_verified"},
        "max_decisions": job["budget"]["max_decisions"], "timeout": job["budget"]["timeout_s"],
        "review_every": 3, "identity_check": job["identity_check"],
        "fast_model": runner.FAST_MODEL, "fast_model_options": runner.FAST_OPTIONS,
        "slow_model": runner.SLOW_MODEL, "slow_model_options": runner.SLOW_OPTIONS,
        "models": [{"role": "fast", "model": runner.FAST_MODEL},
                   {"role": "slow", "model": runner.SLOW_MODEL}],
        "source_sha256": "b" * 64, **job["feature_flags"]}
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []
    result.pop("rim_grasp_guidance")
    assert "missing_rim_grasp_guidance" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})
    result["rim_grasp_guidance"] = False
    assert "mismatch_rim_grasp_guidance" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})


@pytest.mark.parametrize("role", ["fast", "slow", "grounding", "target_judge", "destination_judge"])
def test_system_prompt_hash_metadata_matches_captured_http_body_for_every_role(role, tmp_path):
    class CaptureHTTP:
        def __init__(self):
            self.request = None

        async def post(self, _path, *, json):
            self.request = copy.deepcopy(json)
            return SimpleNamespace(status_code=200, text="", json=lambda: {
                "choices": [{"message": {"content": '{"kind":"wait"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

    async def run():
        output = tmp_path / role
        output.mkdir()
        model = ModelClient("http://offline.invalid/v1", "fixture-key", "fixture-model", role, output)
        capture = CaptureHTTP()
        model.client = capture
        # A nonempty, distinctive tools contract catches omissions and ordering changes.
        model.tools_prefix = "\nAvailable tools (stable schema):\n{\"grasp_point\":{\"point_id\":\"string\"}}"
        system = rim_guided_prompt(FAST)
        await model.decide(system, {"milestone": "offline fixture"})

        transmitted = capture.request["messages"][0]["content"]
        expected = {
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "tools_prefix_sha256": hashlib.sha256(model.tools_prefix.encode("utf-8")).hexdigest(),
            "transmitted_system_sha256": hashlib.sha256(transmitted.encode("utf-8")).hexdigest(),
        }
        assert transmitted == system + model.tools_prefix
        assert "suggested_jaw_world" in transmitted
        attempt = next(json.loads(line) for line in model.request_trace.read_text().splitlines()
                       if json.loads(line)["kind"] == "attempt.started")
        trace = json.loads(model.trace.read_text().splitlines()[-1])
        assert {key: attempt[key] for key in expected} == expected
        assert {key: trace[key] for key in expected} == expected
        assert not any(key in attempt for key in ("system", "prompt", "messages", "tools_prefix"))
        assert "Choose the contact geometry BEFORE grasping" not in model.request_trace.read_text()

    asyncio.run(run())
