"""Offline regressions for preserving user referents in initial plans."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_v18_milestone_protocol import (  # noqa: E402
    FUNDAMENTAL,
    MISSION,
    checkpoint_decision,
    harness,
    two_stage_plan,
)
from embodied_harness.agents.milestone_prompts import (  # noqa: E402
    CHECKPOINT,
    ECONOMICAL_PLAN,
    FAITHFUL_MISSION_REFERENTS,
    PLAN,
)
from embodied_harness.runtime.milestones import PLAN_SCHEMA  # noqa: E402


def emitted(store, event_type):
    return [event["payload"] for event in store.events if event["kind"] == event_type]


def test_feature_off_keeps_initial_plan_prompt_byte_for_byte_unchanged():
    async def run():
        store, _context, _adapter, _coordinator, slow, manager = await harness([two_stage_plan()])
        assert manager.faithful_initial_referents is False

        await manager.initialize()

        assert slow.model.calls[0]["prompt"] == PLAN
        assert emitted(store, "milestone.referent_prompt_prepared") == []

    import asyncio
    asyncio.run(run())


@pytest.mark.parametrize("review_economy", [False, True])
def test_enabled_faithful_rule_appends_only_to_initial_plan_after_economy_choice(review_economy):
    async def run():
        plan = two_stage_plan()
        store, context, _adapter, _coordinator, slow, manager = await harness([plan])
        manager.faithful_initial_referents = True
        slow.review_economy = review_economy

        await manager.initialize()

        prompt = slow.model.calls[0]["prompt"]
        assert prompt.endswith(FAITHFUL_MISSION_REFERENTS)
        assert "original object names and distinguishing spatial relations" in " ".join(prompt.split())
        assert "Do not replace or narrow those referents with an inferred color" in " ".join(prompt.split())
        if review_economy:
            assert ECONOMICAL_PLAN in prompt
            assert prompt.index(ECONOMICAL_PLAN) < prompt.rindex(FAITHFUL_MISSION_REFERENTS)
        else:
            assert prompt.startswith(PLAN)
        assert slow.model.calls[0]["projection"]["mission"] == MISSION
        assert slow.model.calls[0]["projection"]["plan_schema"] == PLAN_SCHEMA
        assert manager.fundamental == plan["fundamental_constraints"] == [FUNDAMENTAL]
        assert manager.stages == plan["milestones"]
        assert context.milestone.capabilities == ("grasp_point", "move_to", "place_point")
        assert emitted(store, "milestone.referent_prompt_prepared") == [{
            "initial": True, "attempt": 1,
            "policy": "user_referents_without_inferred_appearance",
        }]

    import asyncio
    asyncio.run(run())


def test_replan_and_checkpoint_prompts_do_not_receive_initial_referent_appendix():
    async def run():
        store, _context, _adapter, _coordinator, slow, manager = await harness([two_stage_plan()])
        manager.faithful_initial_referents = True
        await manager.initialize()

        slow.model.responses.append(two_stage_plan())
        assert await manager.plan("new evidence changed the remaining route") == "planned"
        assert slow.model.calls[1]["prompt"] == PLAN

        slow.model.responses.append(checkpoint_decision(manager))
        assert await manager.checkpoint() == "advanced"
        checkpoint_prompt = slow.model.calls[2]["prompt"]
        assert checkpoint_prompt == CHECKPOINT
        assert FAITHFUL_MISSION_REFERENTS not in slow.model.calls[1]["prompt"]
        assert FAITHFUL_MISSION_REFERENTS not in checkpoint_prompt
        assert len(emitted(store, "milestone.referent_prompt_prepared")) == 1

    import asyncio
    asyncio.run(run())


@pytest.mark.parametrize("mode,pipeline", [("single_fast", True), ("dual_async", False)])
def test_app_requires_dual_async_pipeline_for_faithful_initial_referents(mode, pipeline):
    import asyncio
    from embodied_harness.app import episode

    args = SimpleNamespace(faithful_initial_referents=True, mode=mode,
                           pipeline_supervision=pipeline)
    with pytest.raises(ValueError, match="faithful_initial_referents requires"):
        asyncio.run(episode(args))


def _runner_config():
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_round22_targeted as runner

    flags = copy.deepcopy(runner.R21_FLAGS["dual_async"])
    flags["faithful_initial_referents"] = True
    config = {
        "schema_version": 1,
        "protocol_id": "round22-targeted-development",
        "status": "prepared_not_launched",
        "output_root": "/tmp/round22i-faithful-referents-tests",
        "catalog": {"path": str(REPO / "configs" / "round21-task-catalog.json")},
        "variants": {"patched22": {"kind": "patched22", "repository_root": str(REPO),
                                     "expected_commit": "f" * 40, "modes": ["dual_async"],
                                     "feature_flags_by_mode": {"dual_async": flags}}},
        "models": {"fast": {"model": runner.FAST_MODEL, "options": runner.FAST_OPTIONS},
                   "slow": {"profile_id": runner.SLOW_PROFILE_ID, "model": runner.SLOW_MODEL,
                            "options": runner.SLOW_OPTIONS}},
        "transport": {"host": runner.TRANSPORT_HOST, "address": runner.TRANSPORT_ADDRESS},
        "scheduler": {"max_concurrent_episodes": 2, "memory_reserve_gib": 8,
                      "memory_per_episode_gib": 2, "stop_sentinel": "STOP_NEW"},
        "no_automatic_replay": True,
        "retain_all_attempts": True,
        "cases": [{"suite": "libero_spatial", "task_index": 0, "seed": 36,
                   "variant": "patched22", "mode": "dual_async"}],
    }
    return runner, config


def test_runner_validates_flag_and_serializes_it_in_cli_arguments():
    runner, config = _runner_config()
    runner.validate_config(config)
    catalog, _catalog_path, _catalog_hash = runner.load_catalog(
        config, REPO / "configs" / "round21-benchmark.json")
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]

    assert job["feature_flags"]["faithful_initial_referents"] is True
    assert "--faithful-initial-referents" in runner.app_arguments(job, Path("/tmp/memory.json"))

    invalid, invalid_config = _runner_config()
    invalid_config["variants"]["patched22"]["feature_flags_by_mode"]["dual_async"]["pipeline_supervision"] = False
    with pytest.raises(ValueError, match="faithful_initial_referents requires"):
        invalid.validate_config(invalid_config)
