"""Offline runner checks for allowlisted per-variant Slow profiles."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import run_round22_targeted as runner  # noqa: E402
from test_round22_runner import fixture_config, load_fixture_catalog  # noqa: E402


def config_for(profile=None, *, mode="dual_async", source_judges=False):
    config = fixture_config()
    case = {"suite": "libero_spatial", "task_index": 6, "seed": 36,
            "variant": "patched22", "mode": mode}
    config["cases"] = [case]
    variant = config["variants"]["patched22"]
    if profile is not None:
        variant["slow_profile"] = profile
    if source_judges:
        flags = copy.deepcopy(runner.R21_FLAGS)
        flags["dual_async"].update(strong_target_judge=True, strong_destination_judge=True)
        variant["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    return config


def one_job(config, tmp_path):
    catalog, catalog_path, catalog_sha = load_fixture_catalog(config)
    jobs = runner.make_jobs(config, catalog, Path(config["output_root"]))
    return jobs[0], catalog, catalog_path, catalog_sha


def valid_result(job):
    expected = job["expected_recorded_model_options"]
    result = {
        "backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "evidence": {"status": "not_verified"},
        "max_decisions": job["budget"]["max_decisions"], "timeout": job["budget"]["timeout_s"],
        "review_every": 3, "identity_check": job["identity_check"],
        "fast_model": expected["fast_model"], "fast_model_options": expected["fast_model_options"],
        "slow_model": expected["slow_model"], "slow_model_options": expected["slow_model_options"],
        "models": [{"role": role, "model": model}
                   for role, model in job["expected_active_models"].items() if model is not None],
        "source_sha256": "b" * 64, **job["feature_flags"],
    }
    profile = runner.slow_profile_for_job(job)
    if job["feature_flags"].get("strong_target_judge"):
        result.update(target_judge_model=profile["model"], target_judge_options=profile["options"],
                      target_judge_trigger="every_pregrasp_with_same_scope_cache")
    if job["feature_flags"].get("strong_destination_judge"):
        result.update(destination_judge_model=profile["model"], destination_judge_options=profile["options"],
            destination_judge_trigger="every_place_with_exact_point_and_scope_cache",
            destination_judge_memory="original_mission_task_start_and_current_images_only")
    return result


def test_default_omission_preserves_kimi_profile_and_legacy_command(tmp_path):
    config = config_for()
    job, *_ = one_job(config, tmp_path)
    expected = {"configuration": "kimi_new", **copy.deepcopy(runner.SLOW_PROFILES["kimi_new"])}
    assert job["slow_profile"] == expected
    assert job["expected_active_models"]["slow"] == runner.SLOW_MODEL
    assert job["expected_recorded_model_options"]["slow_model_options"] == runner.SLOW_OPTIONS

    snapshot = tmp_path / "snapshot"
    memory = tmp_path / "memory.json"
    command = runner.app_command(config, job, snapshot, memory)
    assert command[1] == str(snapshot / "scripts" / "run_with_slow_profile.py")
    assert command[command.index("--profile-id") + 1] == "kimi_new"
    assert command[command.index("--expected-options-json") + 1] == json.dumps(
        runner.SLOW_OPTIONS, sort_keys=True, separators=(",", ":"))
    assert command[command.index("--") + 1] != "env"
    assert "SLOW_MODEL_OPTIONS=" not in " ".join(command)


@pytest.mark.parametrize("selector", [None, 17, {}, "not-allowlisted"])
def test_malformed_or_unknown_slow_profile_selectors_fail_closed(selector):
    variant = {"slow_profile": selector}
    with pytest.raises(ValueError, match="slow_profile must name a declared public Slow configuration"):
        runner.slow_profile_for_variant(variant)

    config = config_for()
    config["variants"]["patched22"]["slow_profile"] = selector
    with pytest.raises(ValueError, match="slow_profile must name a declared public Slow configuration"):
        runner.validate_config(config)


def test_frozen_baseline_rejects_deepseek_override_and_global_model_tampering():
    config = fixture_config()
    config["variants"]["baseline_r21"]["slow_profile"] = "deepseek_new_low"
    with pytest.raises(ValueError, match="Frozen Round21 baseline Slow profile cannot be changed"):
        runner.validate_config(config)

    config = fixture_config()
    config["models"]["slow"] = {"profile_id": "deepseek_new", "model": "deepseek-v4.1-flash-1017",
                                 "options": {"temperature": 0.1}}
    with pytest.raises(ValueError, match="Slow profile/model/options must match the frozen kimi_new profile"):
        runner.validate_config(config)


def test_deepseek_jobs_route_each_mode_and_optional_judges_to_the_selected_model(tmp_path):
    config = config_for("deepseek_new_low", mode="single_strong")
    strong_job, *_ = one_job(config, tmp_path)
    deepseek = runner.SLOW_PROFILES["deepseek_new_low"]["model"]
    assert strong_job["expected_active_models"] == {
        "fast": deepseek, "slow": None, "grounding": deepseek,
    }
    assert strong_job["expected_recorded_model_options"] == {
        "fast_model": runner.FAST_MODEL, "fast_model_options": runner.FAST_OPTIONS,
        "slow_model": deepseek,
        "slow_model_options": {"temperature": 0.1, "reasoning_effort": "low"},
    }

    dual_config = config_for("deepseek_new_low", source_judges=True)
    dual_job, *_ = one_job(dual_config, tmp_path)
    assert dual_job["expected_active_models"] == {
        "fast": runner.FAST_MODEL, "slow": deepseek, "grounding": None,
        "target_judge": deepseek, "destination_judge": deepseek,
    }


def test_deepseek_command_validates_private_profile_options_then_overrides_only_declared_low_effort(tmp_path):
    config = config_for("deepseek_new_low", source_judges=True)
    job, *_ = one_job(config, tmp_path)
    snapshot = tmp_path / "snapshot"
    command = runner.app_command(config, job, snapshot, tmp_path / "memory.json")

    assert command[command.index("--profile-id") + 1] == "deepseek_new"
    assert command[command.index("--expected-options-json") + 1] == '{"temperature":0.1}'
    separator = command.index("--")
    child = command[separator + 1:]
    assert child[0] == "env"
    assert child[1] == 'SLOW_MODEL_OPTIONS={"reasoning_effort":"low","temperature":0.1}'
    assert child[2].endswith("/.venv311/bin/python")
    assert child[3] == str(snapshot / "scripts" / "with_resolved_host.py")
    assert "--strong-target-judge" in child
    assert "--strong-destination-judge" in child
    rendered = " ".join(command)
    assert "api_key" not in rendered and "secret" not in rendered.lower()
    assert '"reasoning_effort":"high"' not in rendered


def test_tampered_materialized_jobs_and_variant_mismatches_are_rejected(tmp_path):
    config = config_for("deepseek_new_low")
    job, *_ = one_job(config, tmp_path)
    tampered = copy.deepcopy(job)
    tampered["slow_profile"]["options"]["reasoning_effort"] = "high"
    with pytest.raises(ValueError, match="Materialized Slow profile differs from its public declaration"):
        runner.slow_profile_for_job(tampered)

    snapshot = tmp_path / "snapshot"
    good_job, *_ = one_job(config_for(), tmp_path / "default")
    with pytest.raises(ValueError, match="Job Slow profile differs from the frozen variant"):
        runner.app_command(config, good_job, snapshot, tmp_path / "memory.json")


def test_selected_result_model_route_and_request_options_are_audited(tmp_path):
    config = config_for("deepseek_new_low", source_judges=True)
    job, *_ = one_job(config, tmp_path)
    source = {"package_sha256": "b" * 64}
    result = valid_result(job)
    assert runner.check_result(result, job, source) == []

    wrong_model = copy.deepcopy(result)
    wrong_model["slow_model"] = runner.SLOW_MODEL
    wrong_model["models"] = [dict(row, model=runner.SLOW_MODEL) if row["role"] in
        {"slow", "target_judge", "destination_judge"} else row for row in wrong_model["models"]]
    issues = runner.check_result(wrong_model, job, source)
    assert "mismatch_slow_model" in issues
    assert "mismatch_slow_route" in issues
    assert "mismatch_target_judge_route" in issues
    assert "mismatch_destination_judge_route" in issues

    wrong_options = copy.deepcopy(result)
    wrong_options["slow_model_options"] = {"temperature": 0.1, "reasoning_effort": "high"}
    assert "mismatch_slow_model_options" in runner.check_result(wrong_options, job, source)


def test_profiles_and_jobs_use_detached_options_without_cross_job_contamination(tmp_path):
    original = copy.deepcopy(runner.SLOW_PROFILES["deepseek_new_low"])
    profile = runner.slow_profile_for_variant({"slow_profile": "deepseek_new_low"})
    profile["options"]["reasoning_effort"] = "high"
    profile["profile_options"]["temperature"] = 0.9
    assert runner.SLOW_PROFILES["deepseek_new_low"] == original

    config = config_for("deepseek_new_low")
    config["cases"].append({**config["cases"][0], "task_index": 0, "attempt_id": "attempt-02"})
    catalog, _, _ = load_fixture_catalog(config)
    jobs = runner.make_jobs(config, catalog, Path(config["output_root"]))
    jobs[0]["slow_profile"]["options"]["reasoning_effort"] = "high"
    jobs[0]["expected_recorded_model_options"]["slow_model_options"]["temperature"] = 0.9
    assert jobs[1]["slow_profile"] == {"configuration": "deepseek_new_low", **original}
    assert runner.slow_profile_for_job(jobs[1])["options"] == original["options"]
    assert runner.SLOW_PROFILES["deepseek_new_low"] == original


def test_dry_run_discloses_selected_public_profile_without_reading_private_profiles(tmp_path, capsys):
    config = config_for("deepseek_new_low", source_judges=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    job, _catalog, _catalog_path, catalog_sha = one_job(config, tmp_path)
    resources = {"effective_concurrency": 1, "available_memory_gib": 32.0}
    assert runner.dry_run(config, config_path, catalog_sha, [job], resources) == 0
    output = capsys.readouterr().out
    disclosure = json.loads(output)
    assert disclosure["private_profile_read"] is False
    assert disclosure["simulator_or_model_started"] is False
    assert disclosure["job_preview"][0]["slow_profile"] == job["slow_profile"]
    assert disclosure["job_preview"][0]["expected_recorded_model_options"]["slow_model"] == \
        runner.SLOW_PROFILES["deepseek_new_low"]["model"]
    assert "api_key" not in output.lower() and "fixture-secret" not in output
