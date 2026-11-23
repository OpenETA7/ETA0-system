"""Offline prompt-parity tests for initial and checkpoint read-only scouts."""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from embodied_harness import app  # noqa: E402
from embodied_harness.agents.prompts import FAST, execution_base_prompt  # noqa: E402
from embodied_harness.runtime.pipeline import (  # noqa: E402
    PREPARATION_SCOPE,
    READ_ONLY,
    SCOUT,
    build_scout_prompt,
)
import run_round22_targeted as runner  # noqa: E402
from test_round20_pipeline import ScriptedSlow, make_pipeline, plan  # noqa: E402
from test_round22p_view_recovery import Milestones as FastMilestones, build_fast  # noqa: E402


CHECKPOINT_SCOPE = (
    "\nCHECKPOINT PREPARATION: The current stage is already under Slow assessment. "
    "Do not propose complete again. Focus on preparation_goal: obtain a useful read-only "
    "view or measurement for that preview, or wait if nothing useful remains. "
    "The preview never authorizes physical execution.\n"
)


@pytest.mark.parametrize(
    "rim,view",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_scout_uses_shared_fast_base_only_when_parity_is_enabled(rim, view):
    normal = execution_base_prompt(rim_grasp_guidance=rim, view_recovery_guidance=view)
    assert build_scout_prompt(parity=True, rim_grasp_guidance=rim,
                              view_recovery_guidance=view) == normal + PREPARATION_SCOPE
    # Preserve the historical scout byte-for-byte when disabled, even when
    # the actor guidance flags are otherwise on.
    assert build_scout_prompt(parity=False, rim_grasp_guidance=rim,
                              view_recovery_guidance=view) == SCOUT
    assert SCOUT == FAST + PREPARATION_SCOPE


@pytest.mark.parametrize("rim,view", [(False, False), (True, False), (False, True), (True, True)])
def test_normal_fast_inference_uses_the_same_shared_base_builder(rim, view):
    async def run():
        fast, actor, _adapter, _context, _coordinator, _store = build_fast(
            enabled=view, rim=rim)
        assert await fast.run(1) == "decision_budget"
        assert len(actor.calls) == 1
        assert actor.calls[0]["system"] == execution_base_prompt(
            rim_grasp_guidance=rim, view_recovery_guidance=view)

    asyncio.run(run())


@pytest.mark.parametrize(
    "parity,rim,view",
    [(False, True, True), (True, True, True)],
    ids=["default-off-reports-guidance-not-applied", "parity-reports-guidance-applied"],
)
def test_initial_pipeline_scout_uses_exact_selected_prompt_and_telemetry(parity, rim, view):
    async def run():
        plan_entered, release_plan, scout_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def delayed_plan(_prompt, _projection, _images):
            plan_entered.set()
            await release_plan.wait()
            return plan()

        class Scout:
            def __init__(self):
                self.calls = []

            async def decide(self, prompt, projection, images):
                self.calls.append((prompt, copy.deepcopy(projection), tuple(images)))
                scout_done.set()
                return {"kind": "wait", "reason": "preparation fixture"}

        store, _context, _adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([delayed_plan]))
        manager.scout_prompt_parity = parity
        manager.rim_grasp_guidance = rim
        manager.view_recovery_guidance = view
        scout = Scout()
        task = asyncio.create_task(manager.initialize_with_scout(scout))
        try:
            await asyncio.wait_for(plan_entered.wait(), timeout=1)
            await asyncio.wait_for(scout_done.wait(), timeout=1)
        finally:
            release_plan.set()
        assert await asyncio.wait_for(task, timeout=1) == "planned"
        assert len(scout.calls) == 1
        expected_base = execution_base_prompt(rim_grasp_guidance=rim,
                                              view_recovery_guidance=view) if parity else FAST
        assert scout.calls[0][0] == expected_base + PREPARATION_SCOPE
        rows = [e["payload"] for e in store.events if e["kind"] == "pipeline.scout_prompt_prepared"]
        assert rows == [{"phase": "initial", "parity": parity,
                         "rim_grasp_guidance": parity and rim,
                         "view_recovery_guidance": parity and view}]

    asyncio.run(run())


@pytest.mark.parametrize("parity", [False, True], ids=["legacy-default", "parity"])
def test_checkpoint_scout_prompt_and_telemetry_follow_selected_base(parity):
    async def run():
        checkpoint = asyncio.create_task(asyncio.Event().wait())
        fast, actor, _adapter, _context, _coordinator, store = build_fast(
            enabled=True, rim=True,
            milestones=FastMilestones(pipelined=True, checkpoint_task=checkpoint),
        )
        fast.scout_prompt_parity = parity
        try:
            assert await fast.run(1) == "decision_budget"
        finally:
            checkpoint.cancel()
            with pytest.raises(asyncio.CancelledError):
                await checkpoint

        expected_base = (execution_base_prompt(rim_grasp_guidance=True,
                                               view_recovery_guidance=True)
                         + PREPARATION_SCOPE if parity else SCOUT)
        assert len(actor.calls) == 1
        assert actor.calls[0]["system"] == expected_base + CHECKPOINT_SCOPE
        rows = [e["payload"] for e in store.events if e["kind"] == "pipeline.scout_prompt_prepared"]
        assert rows == [{"phase": "checkpoint", "parity": parity,
                         "rim_grasp_guidance": parity,
                         "view_recovery_guidance": parity,
                         "decision_slot": 1}]
        guided_events = [e["kind"] for e in store.events
                         if e["kind"] in {"fast.rim_guidance_prepared", "fast.view_recovery_prepared"}]
        assert len(guided_events) == (2 if parity else 0)

    asyncio.run(run())


def test_preparation_whitelist_remains_read_only_and_unchanged():
    assert READ_ONLY == frozenset({"observe", "mark_point", "locate_point", "select_candidate"})
    assert "move_to" not in READ_ONLY
    assert "grasp_point" not in READ_ONLY


@pytest.mark.parametrize(
    "mode,pipeline,two_role",
    [
        ("dual_sync", True, True),
        ("dual_async", False, True),
        ("dual_async", True, False),
    ],
)
def test_app_rejects_parity_outside_two_role_dual_async_pipeline_before_setup(
    monkeypatch, tmp_path, mode, pipeline, two_role
):
    touched = []

    def forbidden(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("invalid parity options must fail before setup")

    monkeypatch.setattr(app, "load_env", forbidden)
    monkeypatch.setattr(app, "OpenETAAdapter", forbidden)
    args = SimpleNamespace(
        scout_prompt_parity=True, mode=mode, pipeline_supervision=pipeline,
        two_role_runtime=two_role, identity_check=False,
    )
    with pytest.raises(ValueError, match="requires two-role dual_async pipelined preparation"):
        asyncio.run(app.episode(args))
    assert touched == []
    assert not (tmp_path / "episode").exists()


def _runner_config_with_scout_parity():
    from test_round22_runner import fixture_config

    config = fixture_config()
    config["cases"] = [{"suite": "libero_spatial", "task_index": 6, "seed": 36,
                        "variant": "patched22", "mode": "dual_async"}]
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(two_role_runtime=True, scout_prompt_parity=True,
                               pipeline_supervision=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    return config


def test_runner_opt_in_dispatch_and_result_audit_for_parity_flag():
    from test_round22_runner import fixture_config, load_fixture_catalog

    config = _runner_config_with_scout_parity()
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert job["feature_flags"]["scout_prompt_parity"] is True
    assert "--scout-prompt-parity" in runner.app_arguments(job, Path("/tmp/memory.json"))
    assert job["identity_check"] is False
    expected = job["expected_recorded_model_options"]
    result = {
        "backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "evidence": {"status": "not_verified"},
        "max_decisions": job["budget"]["max_decisions"],
        "timeout": job["budget"]["timeout_s"], "review_every": 3,
        "identity_check": False, "fast_model": expected["fast_model"],
        "fast_model_options": expected["fast_model_options"],
        "slow_model": expected["slow_model"], "slow_model_options": expected["slow_model_options"],
        "models": [{"role": "fast", "model": runner.FAST_MODEL},
                   {"role": "slow", "model": runner.SLOW_MODEL}],
        "source_sha256": "c" * 64, "agent_topology": {
            "name": "slow_fast_only", "roles": ["fast", "slow"],
            "independent_checkers": False, "strong_executor_handoff": False,
            "slow_execution_tools": False,
        },
        **job["feature_flags"],
    }
    source = {"package_sha256": "c" * 64}
    assert runner.check_result(result, job, source) == []
    result["scout_prompt_parity"] = False
    assert "mismatch_scout_prompt_parity" in runner.check_result(result, job, source)

    default_config = fixture_config()
    default_config["cases"] = [{"suite": "libero_spatial", "task_index": 6, "seed": 36,
                                 "variant": "patched22", "mode": "dual_async"}]
    default_catalog, _, _ = load_fixture_catalog(default_config)
    default_job = runner.make_jobs(default_config, default_catalog,
                                   Path(default_config["output_root"]))[0]
    assert default_job["feature_flags"].get("scout_prompt_parity", False) is False
    assert "--scout-prompt-parity" not in runner.app_arguments(
        default_job, Path("/tmp/memory.json"))


@pytest.mark.parametrize(
    "mode,pipeline,two_role",
    [
        ("dual_sync", True, True),
        ("dual_async", False, True),
        ("dual_async", True, False),
    ],
)
def test_runner_rejects_invalid_parity_feature_flag_dependencies(mode, pipeline, two_role):
    from test_round22_runner import fixture_config

    config = fixture_config()
    flags = copy.deepcopy(runner.R21_FLAGS)
    if mode == "dual_sync":
        config["variants"]["patched22"]["modes"].append("dual_sync")
        flags["dual_sync"] = copy.deepcopy(runner.R21_FLAGS["dual_async"])
    flags[mode].update(scout_prompt_parity=True, pipeline_supervision=pipeline,
                       two_role_runtime=two_role)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    with pytest.raises(ValueError, match="scout_prompt_parity requires"):
        runner.validate_config(config)
