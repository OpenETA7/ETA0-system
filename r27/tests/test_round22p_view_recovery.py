"""Offline regressions for default-off occluded-view recovery guidance."""
from __future__ import annotations

import asyncio
import copy
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.grasp_guidance import rim_guided_prompt
from embodied_harness.agents.milestone_prompts import FAST_CONTRACT, FAST_FINAL_CONTRACT
from embodied_harness.agents.prompts import FAST
from embodied_harness.agents.view_recovery import POLICY_ID, VIEW_RECOVERY, view_recovery_prompt
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
    schemas = {"observe": {"parameters": {"properties": {"camera": {"type": "string"}}}}}

    def __init__(self):
        self.version = 1
        self.validated = []
        self.inspected = []
        self.executed = []

    def validate_args(self, tool, args):
        self.validated.append((tool, copy.deepcopy(args)))

    async def inspect(self, tool, args, action_seq):
        self.inspected.append((tool, copy.deepcopy(args), action_seq))
        self.version += 1
        return Observation(f"read-{self.version}", self.version, 1, time.monotonic(), 0.0,
            action_seq, {"candidate_scope": "frame-1/objects", "views": []}, ())

    async def verify(self, _observation_id):
        return {"status": "not_verified"}


class Actor:
    def __init__(self):
        self.history = deque(maxlen=6)
        self.calls = []

    async def decide(self, system, projection, images, **kwargs):
        self.calls.append({"system": system, "projection": copy.deepcopy(projection),
                           "images": tuple(images), "kwargs": kwargs})
        return {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}}


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


def build_fast(*, enabled=False, rim=False, mode="dual_async", milestones=None,
               final=False, decision=None):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("stage", 1, "pick up the object", "move the object",
        capabilities=("observe",), final_stage=final)
    context.ingest(Observation("initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "frame-1/objects", "views": []}, ()))
    adapter = Adapter()
    coordinator = Coordinator(adapter, context, store)
    actor = Actor()
    if decision is not None:
        actor.decide = decision
    fast = FastAgent(actor, context, coordinator, store, mode=mode,
                     rim_grasp_guidance=rim, view_recovery_guidance=enabled,
                     final_stage_prompt=final)
    fast.milestones = milestones
    return fast, actor, adapter, context, coordinator, store


def _base(*, rim=False, enabled=False):
    result = rim_guided_prompt(FAST) if rim else FAST
    return view_recovery_prompt(result) if enabled else result


@pytest.mark.parametrize("mode", ["single_fast", "dual_async"])
@pytest.mark.parametrize("enabled", [False, True])
def test_normal_pure_and_default_off_prompt_are_exact(mode, enabled):
    async def run():
        fast, actor, adapter, context, _coordinator, store = build_fast(
            enabled=enabled, mode=mode)
        status = await fast.run(1)
        assert status == "decision_budget"
        assert len(actor.calls) == 1
        assert actor.calls[0]["system"] == _base(enabled=enabled)
        assert adapter.inspected == [("observe", {"camera": "agentview"}, 0)]
        assert context.action_seq == 0
        prepared = [e for e in store.events if e["kind"] == "fast.view_recovery_prepared"]
        assert bool(prepared) is enabled
        if enabled:
            assert prepared[0]["payload"]["policy_id"] == POLICY_ID

    asyncio.run(run())


@pytest.mark.parametrize("final", [False, True])
def test_milestone_and_final_contracts_wrap_the_selected_view_guidance(final):
    async def run():
        fast, actor, _adapter, _context, _coordinator, store = build_fast(
            enabled=True, milestones=Milestones(), final=final)
        assert await fast.run(1) == "decision_budget"
        base = _base(enabled=True)
        if final:
            expected = FAST_FINAL_CONTRACT + base.replace(
                "Maintain identity and complete the mission.",
                "Maintain identity and achieve the complete original mission under the current final contract.")
        else:
            expected = FAST_CONTRACT + base.replace(
                "Maintain identity and complete the mission.",
                "Maintain identity and achieve ONLY the current milestone; propose stage completion before later outcomes.")
        assert actor.calls[0]["system"] == expected
        assert actor.calls[0]["system"].endswith(VIEW_RECOVERY)
        assert sum(e["kind"] == "fast.view_recovery_prepared" for e in store.events) == 1

    asyncio.run(run())


def test_guidance_is_conditional_prompt_only_and_combines_with_rim_guidance():
    guided = view_recovery_prompt(FAST)
    assert guided == FAST + VIEW_RECOVERY
    assert "possible_hold is resolved" in guided
    assert "measurably open" in guided and "empty hand" in guided
    assert "opposite the CURRENT robot.approach_world" in guided
    assert "at most 0.08 m" in guided
    assert "short reverse-approach path is visibly clear" in guided
    assert "use its fresh returned scene to reidentify" in guided
    assert "Do not reuse prior candidate/point IDs" in guided
    assert view_recovery_prompt(rim_guided_prompt(FAST)) == rim_guided_prompt(FAST) + VIEW_RECOVERY

    async def run():
        fast, actor, _adapter, _context, _coordinator, store = build_fast(
            enabled=True, rim=True, mode="single_fast")
        assert await fast.run(1) == "decision_budget"
        assert actor.calls[0]["system"] == _base(enabled=True, rim=True)
        assert "suggested_jaw_world" in actor.calls[0]["system"]
        kinds = [e["kind"] for e in store.events]
        assert kinds.count("fast.view_recovery_prepared") == 1
        assert kinds.count("fast.rim_guidance_prepared") == 1

    asyncio.run(run())


def test_checkpoint_scout_discards_guidance_and_emits_no_prepared_marker():
    async def run():
        checkpoint = asyncio.create_task(asyncio.Event().wait())
        fast, actor, _adapter, _context, _coordinator, store = build_fast(
            enabled=True, rim=True, milestones=Milestones(pipelined=True, checkpoint_task=checkpoint))
        try:
            assert await fast.run(1) == "decision_budget"
            expected = SCOUT + "\nCHECKPOINT PREPARATION: The current stage is already under Slow assessment. Do not propose complete again. Focus on preparation_goal: obtain a useful read-only view or measurement for that preview, or wait if nothing useful remains. The preview never authorizes physical execution.\n"
            assert actor.calls[0]["system"] == expected
            assert "VIEW RECOVERY AFTER A FAILED GRASP" not in expected
            assert "suggested_jaw_world" not in expected
            assert not [e for e in store.events if e["kind"] in {
                "fast.view_recovery_prepared", "fast.rim_guidance_prepared"}]
        finally:
            checkpoint.cancel()
            with pytest.raises(asyncio.CancelledError):
                await checkpoint

    asyncio.run(run())


def test_app_backend_gate_precedes_environment_and_result_records_flag(monkeypatch, tmp_path):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args, _install_offline_app

    invalid = _episode_args(tmp_path, "single_fast")
    invalid.backend = "eta"
    invalid.view_recovery_guidance = True
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("must reject before environment loading"))
    with pytest.raises(ValueError, match="view_recovery_guidance describes direct_libero robot state and relative motion"):
        asyncio.run(app.episode(invalid))
    assert not Path(invalid.output).exists()

    _install_offline_app(monkeypatch, {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"})
    captured = []

    class CapturingFast:
        def __init__(self, *_args, **kwargs):
            self.guidance = kwargs.get("view_recovery_guidance")
            self.steps = self.errors = self.false_claims = 0
            self.identity_checker = None
            captured.append(self)

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "FastAgent", CapturingFast)
    args = _episode_args(tmp_path, "single_fast")
    args.view_recovery_guidance = True
    result = asyncio.run(app.episode(args))
    assert captured[0].guidance is True
    assert result["view_recovery_guidance"] is True


@pytest.mark.parametrize("mode", ["single_fast", "single_strong", "dual_sync", "dual_async"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_routes_optional_flag_for_each_app_mode(monkeypatch, tmp_path, mode, enabled):
    from embodied_harness import app

    captured = {}

    async def fake_episode(args):
        captured.update(mode=args.mode, enabled=args.view_recovery_guidance)

    monkeypatch.setattr(app, "episode", fake_episode)
    argv = ["app", "--mission", "clear the view", "--output", str(tmp_path / mode),
            "--backend", "direct_libero", "--mode", mode]
    if enabled:
        argv.append("--view-recovery-guidance")
    monkeypatch.setattr(sys, "argv", argv)
    app.main()
    assert captured == {"mode": mode, "enabled": enabled}


def test_runner_routes_and_audits_flag_for_every_mode_default_off(monkeypatch, tmp_path):
    from test_round22_runner import fixture_config, load_fixture_catalog

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    modes = ["single_fast", "single_strong", "dual_sync", "dual_async"]
    variant = config["variants"]["patched22"]
    variant["modes"] = modes
    flags = {mode: {**runner.R21_FLAGS.get(mode, {
        "progress_recovery": True, "priority_foreground": False,
        "review_economy": False, "pipeline_supervision": False}),
        "view_recovery_guidance": True} for mode in modes}
    variant["feature_flags_by_mode"] = flags
    config["cases"] = [{"suite": "libero_spatial", "task_index": 6, "seed": 36,
                        "variant": "patched22", "mode": mode} for mode in modes]
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    jobs = runner.make_jobs(config, catalog, Path(config["output_root"]))
    assert {job["mode"] for job in jobs} == set(modes)
    for job in jobs:
        assert job["feature_flags"]["view_recovery_guidance"] is True
        assert "--view-recovery-guidance" in runner.app_arguments(job, Path("/tmp/memory.json"))
        expected = job["expected_recorded_model_options"]
        result = {"backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
            "mode": job["mode"], "status": "decision_budget", "success": False,
            "evidence": {"status": "not_verified"}, "max_decisions": job["budget"]["max_decisions"],
            "timeout": job["budget"]["timeout_s"], "review_every": 3,
            "identity_check": job["identity_check"], "fast_model": expected["fast_model"],
            "fast_model_options": expected["fast_model_options"], "slow_model": expected["slow_model"],
            "slow_model_options": expected["slow_model_options"],
            "models": [{"role": role, "model": model} for role, model in
                       job["expected_active_models"].items() if model is not None],
            "source_sha256": "a" * 64, **job["feature_flags"]}
        assert runner.check_result(result, job, {"package_sha256": "a" * 64}) == []
        result["view_recovery_guidance"] = False
        assert "mismatch_view_recovery_guidance" in runner.check_result(
            result, job, {"package_sha256": "a" * 64})

    default_config = fixture_config()
    default_config["cases"] = [{"suite": "libero_spatial", "task_index": 6, "seed": 36,
                                 "variant": "patched22", "mode": "dual_async"}]
    default_catalog, _, _ = load_fixture_catalog(default_config)
    default_job = runner.make_jobs(default_config, default_catalog, Path(default_config["output_root"]))[0]
    assert default_job["feature_flags"].get("view_recovery_guidance", False) is False
    assert "--view-recovery-guidance" not in runner.app_arguments(default_job, Path("/tmp/memory.json"))
