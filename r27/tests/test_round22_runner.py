"""Offline Round22 runner preflight fixtures. These tests never launch a simulator or model."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_round22_targeted as runner  # noqa: E402


def fixture_config() -> dict:
    cells = [
        ("libero_10", 5),
        ("libero_spatial", 6),
        ("libero_spatial", 0),
        ("libero_goal", 9),
        ("libero_object", 4),
        ("libero_goal", 8),
    ]
    variants = {
        "baseline_r21": {"kind": "baseline_r21", "repository_root": "/tmp/r21",
                         "expected_commit": runner.BASELINE_R21_COMMIT,
                         "modes": ["dual_async"]},
        "patched22": {"kind": "patched22", "repository_root": "/tmp/r22",
                      "expected_commit": "f" * 40,
                      "modes": ["dual_async", "single_fast", "single_strong"]},
    }
    cases = []
    for suite, task_index in cells:
        cases.append({"suite": suite, "task_index": task_index, "seed": 36,
                      "variant": "baseline_r21", "mode": "dual_async"})
        for mode in variants["patched22"]["modes"]:
            cases.append({"suite": suite, "task_index": task_index, "seed": 36,
                          "variant": "patched22", "mode": mode})
    return {
        "schema_version": 1,
        "protocol_id": "round22-targeted-development",
        "status": "prepared_not_launched",
        "output_root": "/tmp/round22-dev01",
        "catalog": {"path": str(REPO / "configs" / "round21-task-catalog.json")},
        "variants": variants,
        "models": {
            "fast": {"model": runner.FAST_MODEL, "options": runner.FAST_OPTIONS},
            "slow": {"profile_id": runner.SLOW_PROFILE_ID, "model": runner.SLOW_MODEL,
                     "options": runner.SLOW_OPTIONS},
        },
        "transport": {"host": runner.TRANSPORT_HOST, "address": runner.TRANSPORT_ADDRESS},
        "scheduler": {"max_concurrent_episodes": 24, "memory_reserve_gib": 8,
                       "memory_per_episode_gib": 2, "stop_sentinel": "STOP_NEW"},
        "no_automatic_replay": True,
        "retain_all_attempts": True,
        "cases": cases,
    }


def load_fixture_catalog(config: dict) -> tuple[dict, Path, str]:
    config_path = REPO / "configs" / "round21-benchmark.json"
    return runner.load_catalog(config, config_path)


@pytest.mark.parametrize("mode,pipelined", [("single_fast", True), ("dual_async", False)])
def test_final_recovery_rejects_invalid_routing_before_dispatch(mode, pipelined):
    config = fixture_config()
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags[mode].update(async_completion_recovery=True, pipeline_supervision=pipelined)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    with pytest.raises(ValueError, match="async_completion_recovery requires"):
        runner.validate_config(config)


def test_targeted_24_case_plan_uses_pinned_initial_state_hashes() -> None:
    config = fixture_config()
    runner.validate_config(config)
    catalog, _, catalog_sha = load_fixture_catalog(config)
    jobs = runner.make_jobs(config, catalog, Path(config["output_root"]))
    assert len(jobs) == 24
    assert len({job["job_id"] for job in jobs}) == 24
    assert all(job["seed"] == 36 and len(job["initial_state_sha256"]) == 64 for job in jobs)
    assert sum(job["stratum"] == "long" for job in jobs) == 4
    assert sum(job["stratum"] == "standard" for job in jobs) == 20
    assert catalog_sha == runner.sha256_file(REPO / "configs" / "round21-task-catalog.json")


def test_memory_gate_caps_at_requested_count_and_hard_limit() -> None:
    config = fixture_config()
    resources = runner.concurrency_for_memory(config["scheduler"], 128 * 1024**3)
    assert resources["effective_concurrency"] == 24
    config["scheduler"]["max_concurrent_episodes"] = 60
    resources = runner.concurrency_for_memory(config["scheduler"], 128 * 1024**3)
    assert resources["effective_concurrency"] == 60
    resources = runner.concurrency_for_memory(config["scheduler"], 26 * 1024**3)
    assert resources["effective_concurrency"] == 9
    with pytest.raises(ValueError, match="fewer than one"):
        runner.concurrency_for_memory(config["scheduler"], 9 * 1024**3)


def test_round21_baseline_contract_cannot_be_relaxed() -> None:
    config = fixture_config()
    config["variants"]["baseline_r21"]["expected_commit"] = "a" * 40
    with pytest.raises(ValueError, match="baseline commit"):
        runner.validate_config(config)
    config = fixture_config()
    config["variants"]["baseline_r21"]["modes"] = ["dual_sync"]
    with pytest.raises(ValueError, match="baseline modes"):
        runner.validate_config(config)


def test_missing_seed_hash_and_duplicate_attempt_are_rejected() -> None:
    config = fixture_config()
    config["cases"][0]["seed"] = 37
    catalog, _, _ = load_fixture_catalog(config)
    with pytest.raises(ValueError, match="no pinned catalog hash"):
        runner.make_jobs(config, catalog, Path(config["output_root"]))
    config = fixture_config()
    config["cases"].append(copy.deepcopy(config["cases"][0]))
    with pytest.raises(ValueError, match="Duplicate case"):
        runner.validate_config(config)


def test_generated_and_explicit_output_roots_are_one_shot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = fixture_config()
    config["output_root"] = str(tmp_path / "round22-dev01")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    root1, _ = runner.campaign_path(config, config_path)
    root2, _ = runner.campaign_path(config, config_path)
    assert root1 == root2 == Path(config["output_root"])
    root1.mkdir()
    with pytest.raises(ValueError, match="no overwrite"):
        runner.output_root_preflight(root1)


def test_source_preflight_requires_clean_pinned_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    (repo / "src" / "embodied_harness").mkdir(parents=True)
    (repo / "src" / "embodied_harness" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "memory").mkdir()
    (repo / "memory" / "verified-experience.json").write_text("{}\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    for name in ("run_with_slow_profile.py", "with_resolved_host.py", "env.sh"):
        (repo / "scripts" / name).write_text("# fixture\n", encoding="utf-8")
    python_path = repo / ".venv311" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_path.chmod(0o755)
    subprocess.run(["git", "init", "-q", "-b", "fixture"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    variant = {"kind": "patched22", "repository_root": str(repo), "expected_commit": commit,
               "modes": ["dual_async"]}
    source = runner.preflight_source({"protocol_id": "unused"}, "patched22", variant)
    assert source["commit"] == commit
    assert len(source["package_sha256"]) == 64
    (repo / "src" / "embodied_harness" / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree is dirty"):
        runner.preflight_source({}, "patched22", variant)


def test_child_command_uses_frozen_wrappers_and_contains_no_credentials(tmp_path: Path) -> None:
    config = fixture_config()
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    snapshot = tmp_path / "source-snapshots" / job["variant"]
    memory = tmp_path / "source-snapshots" / "memory" / job["variant"] / "verified-experience.json"
    command = runner.app_command(config, job, snapshot, memory)
    assert command[1] == str(snapshot / "scripts" / "run_with_slow_profile.py")
    assert "--profile-id" in command and "kimi_new" in command
    assert str(snapshot / "scripts" / "with_resolved_host.py") in command
    assert "fixture-secret" not in " ".join(command)
    assert "--max-decisions" in command and "36" in command


def valid_result_for(job: dict) -> dict:
    expected = job["expected_recorded_model_options"]
    return {
        "backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "evidence": {"status": "not_verified"},
        "max_decisions": job["budget"]["max_decisions"], "timeout": job["budget"]["timeout_s"],
        "review_every": 3, "identity_check": job["identity_check"],
        "fast_model": expected["fast_model"], "fast_model_options": expected["fast_model_options"],
        "slow_model": expected["slow_model"], "slow_model_options": expected["slow_model_options"],
        "models": [{"role": "fast", "model": job["expected_active_models"]["fast"]},
                   {"role": "slow", "model": job["expected_active_models"]["slow"]}],
        "source_sha256": "b" * 64, **job["feature_flags"],
    }


def test_grounding_route_is_required_only_for_observed_identity_memory() -> None:
    config = fixture_config()
    catalog, _, _ = load_fixture_catalog(config)
    jobs = runner.make_jobs(config, catalog, Path(config["output_root"]))
    source = {"package_sha256": "b" * 64}
    long_job = next(job for job in jobs if job["stratum"] == "long")
    assert long_job["identity_check"] is False
    assert "missing_grounding_route_for_memory" not in runner.check_result(valid_result_for(long_job), long_job, source)
    standard_job = next(job for job in jobs if job["stratum"] == "standard")
    result = valid_result_for(standard_job)
    result["appearance_memory"] = [{"object": "fixture"}]
    assert "missing_grounding_route_for_memory" in runner.check_result(result, standard_job, source)


def test_success_requires_official_libero_verification_evidence() -> None:
    config = fixture_config()
    catalog, _, _ = load_fixture_catalog(config)
    jobs = runner.make_jobs(config, catalog, Path(config["output_root"]))
    long_job = next(job for job in jobs if job["stratum"] == "long")
    source = {"package_sha256": "b" * 64}
    result = valid_result_for(long_job)
    result["success"] = True
    result["status"] = "success"
    result["evidence"] = {
        "status": "verified", "source": "official_libero_check_success",
        "evidence": [{"source": "official_libero_check_success", "sim_time": 26.49}],
    }
    assert "success_without_official_verification" not in runner.check_result(result, long_job, source)
    result["evidence"]["source"] = "agent_self_report"
    assert "success_without_official_verification" in runner.check_result(result, long_job, source)


def test_catalog_rejects_suite_task_env_id_mismatch(tmp_path: Path) -> None:
    original = json.loads((REPO / "configs" / "round21-task-catalog.json").read_text(encoding="utf-8"))
    original["tasks"][0]["env_id"] = "openeta/libero_libero_spatial_task9-v0"
    broken_path = tmp_path / "bad-catalog.json"
    broken_path.write_text(json.dumps(original), encoding="utf-8")
    config = fixture_config()
    config["catalog"]["path"] = str(broken_path)
    with pytest.raises(ValueError, match="env_id does not match"):
        runner.load_catalog(config, REPO / "configs" / "round21-benchmark.json")


def test_optional_ablation_flags_are_explicitly_dispatched_and_audited():
    config = fixture_config()
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(diagnostic_memory=True, reliable_surfaces=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = next(j for j in runner.make_jobs(config, catalog, Path(config["output_root"]))
               if j["variant"] == "patched22" and j["mode"] == "dual_async")
    command = runner.app_arguments(job, Path("/tmp/frozen-memory.json"))
    assert "--diagnostic-memory" in command and "--reliable-surfaces" in command
    result = valid_result_for(job)
    source = {"package_sha256": "b" * 64}
    assert not runner.check_result(result, job, source)
    del result["diagnostic_memory"]
    assert "missing_diagnostic_memory" in runner.check_result(result, job, source)


def test_frozen_r21_cannot_silently_enable_new_ablation():
    config = fixture_config()
    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"]["diagnostic_memory"] = True
    config["variants"]["baseline_r21"]["feature_flags_by_mode"] = flags
    with pytest.raises(ValueError, match="baseline feature flags"):
        runner.validate_config(config)
