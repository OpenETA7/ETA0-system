#!/usr/bin/env python3
"""One-shot targeted Round22 LIBERO runner. Dry-run is the default.

The JSON configuration is deliberately explicit. Each ``cases`` row names a
suite, task index, official initial-state index, variant, and mode. Variants
declare their own repository root, frozen commit, and allowed modes. The
runner never reads model credentials itself; execution delegates Slow profile
loading to the existing ``run_with_slow_profile.py`` child wrapper.

Example invocation::

    python scripts/run_round22_targeted.py --config /path/to/round22.json
    python scripts/run_round22_targeted.py --config /path/to/round22.json --execute

See ``--help`` for the command-line surface. Config files are external so the
same runner can plan different development selections without editing code.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "round22-targeted.json"
BASELINE_R21_COMMIT = "630921df26c3fdf75edfc4b7158ce78a49b9bf80"
ALLOWED_MODES = {"single_fast", "single_strong", "dual_async", "dual_sync"}
BASELINE_MODES = {"single_fast", "single_strong", "dual_async"}
FEATURE_FLAGS = ("progress_recovery", "priority_foreground", "review_economy", "pipeline_supervision")
OPTIONAL_FEATURE_FLAGS = ("diagnostic_memory", "reliable_surfaces", "async_completion_recovery", "strong_target_judge", "completion_rendezvous", "final_stage_prompt", "recovery_reasoning", "faithful_initial_referents", "appearance_reference_memory", "target_judge_overlap", "strong_destination_judge", "phase_aware_grasp_feedback", "rim_grasp_guidance", "view_recovery_guidance", "urgent_completion_review", "completion_contract_repair", "completion_schema_guard", "strong_recovery_executor", "physical_failure_executor", "two_role_runtime", "scout_prompt_parity", "milestone_advice_delivery")
R21_FLAGS = {
    "single_fast": {"progress_recovery": True, "priority_foreground": False,
                    "review_economy": False, "pipeline_supervision": False},
    "single_strong": {"progress_recovery": True, "priority_foreground": False,
                      "review_economy": False, "pipeline_supervision": False},
    "dual_async": {"progress_recovery": True, "priority_foreground": True,
                   "review_economy": True, "pipeline_supervision": True},
}
FAST_MODEL = "Qwen3.6-35B-A3B"
FAST_OPTIONS = {"chat_template_kwargs": {"enable_thinking": False}}
SLOW_PROFILE_ID = "kimi_new"
SLOW_MODEL = "kimi-k3.1-pro"
SLOW_OPTIONS = {"temperature": 1, "reasoning_effort": "low"}
# Public request settings only. Credentials continue to come exclusively from
# the existing child profile wrapper. The DeepSeek profile is validated against
# its stored temperature-only options before applying the declared low-effort
# request override in that child process; the private profile is never edited.
SLOW_PROFILES = {
    "kimi_new": {"profile_id": SLOW_PROFILE_ID, "model": SLOW_MODEL,
                 "profile_options": SLOW_OPTIONS, "options": SLOW_OPTIONS},
    "deepseek_new_low": {
        "profile_id": "deepseek_new", "model": "deepseek-v4.1-flash-1017",
        "profile_options": {"temperature": 0.1},
        "options": {"temperature": 0.1, "reasoning_effort": "low"}},
}
TRANSPORT_HOST = "apicz.boyuerichdata.com"
TRANSPORT_ADDRESS = "35.241.73.161"
FATAL_STATES = {"launcher_error", "failed_no_result", "invalid_result", "integrity_error",
                "completed_nonzero_exit"}


def slow_profile_for_variant(variant: dict[str, Any]) -> dict[str, Any]:
    name = variant.get("slow_profile", SLOW_PROFILE_ID)
    if not isinstance(name, str) or name not in SLOW_PROFILES:
        raise ValueError("slow_profile must name a declared public Slow configuration")
    if variant.get("kind") == "baseline_r21" and name != SLOW_PROFILE_ID:
        raise ValueError("Frozen Round21 baseline Slow profile cannot be changed")
    return {"configuration": name, **copy.deepcopy(SLOW_PROFILES[name])}


def slow_profile_for_job(job: dict[str, Any]) -> dict[str, Any]:
    # Omitted fields in legacy offline fixtures retain the original Kimi route.
    profile = job.get("slow_profile")
    if profile is None:
        return slow_profile_for_variant({})
    if not isinstance(profile, dict):
        raise ValueError("Malformed materialized Slow profile")
    expected = slow_profile_for_variant({"slow_profile": profile.get("configuration")})
    if profile != expected:
        raise ValueError("Materialized Slow profile differs from its public declaration")
    return expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"Required source directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})
    if not files:
        raise ValueError(f"Required source directory is empty: {root}")
    for path in files:
        if path.is_symlink():
            raise ValueError("Source tree contains a symlink")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_source_sha256(package_root: Path) -> str:
    files = sorted(package_root.rglob("*.py"))
    if not files:
        raise ValueError("Source package contains no Python files")
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise ValueError("Source package contains a symlink")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("Cannot inspect a configured source repository") from None


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.resolve().parent / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("protocol_id") != "round22-targeted-development":
        raise ValueError("Expected Round22 targeted-development schema version 1")
    if config.get("status") not in {"prepared_not_launched", "prepared"}:
        raise ValueError("Configuration must be marked prepared_not_launched")
    if config.get("no_automatic_replay") is not True or config.get("retain_all_attempts") is not True:
        raise ValueError("Configuration must retain every attempt and prohibit automatic replay")
    output_base = config.get("output_base")
    fixed_output_root = config.get("output_root")
    if fixed_output_root is not None:
        if not isinstance(fixed_output_root, str) or not Path(fixed_output_root).is_absolute():
            raise ValueError("output_root must be an absolute path")
    elif not isinstance(output_base, str) or not Path(output_base).is_absolute():
        raise ValueError("output_base must be an absolute path when output_root is omitted")
    catalog = config.get("catalog", {})
    if not isinstance(catalog.get("path"), str) or not catalog["path"]:
        raise ValueError("catalog.path is required")
    expected_catalog_hash = catalog.get("sha256")
    if expected_catalog_hash is not None and (not isinstance(expected_catalog_hash, str)
                                                or len(expected_catalog_hash) != 64):
        raise ValueError("catalog.sha256 must be a 64-character SHA256 value")
    variants = config.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("At least one source variant is required")
    for name, variant in variants.items():
        if not isinstance(name, str) or not name or not isinstance(variant, dict):
            raise ValueError("Malformed variant declaration")
        if variant.get("kind") not in {"baseline_r21", "patched22"}:
            raise ValueError(f"Unsupported source variant kind: {name}")
        slow_profile_for_variant(variant)
        if not isinstance(variant.get("repository_root"), str) or not Path(variant["repository_root"]).is_absolute():
            raise ValueError(f"{name}.repository_root must be absolute")
        commit = variant.get("expected_commit")
        if not isinstance(commit, str) or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            raise ValueError(f"{name}.expected_commit must pin a full 40-character commit")
        if variant["kind"] == "baseline_r21" and commit != BASELINE_R21_COMMIT:
            raise ValueError("Frozen Round21 baseline commit must remain 630921df26c3fdf75edfc4b7158ce78a49b9bf80")
        modes = variant.get("modes")
        if not isinstance(modes, list) or not modes or any(mode not in ALLOWED_MODES for mode in modes):
            raise ValueError(f"{name}.modes must list supported app modes")
        if len(set(modes)) != len(modes):
            raise ValueError(f"{name}.modes contains duplicates")
        if variant["kind"] == "baseline_r21" and not set(modes).issubset(BASELINE_MODES):
            raise ValueError("Round21 baseline modes must preserve the frozen R21 mode set")
        expected_branch = variant.get("expected_branch")
        if expected_branch is not None and (not isinstance(expected_branch, str) or not expected_branch):
            raise ValueError(f"{name}.expected_branch must be a non-empty string")
        memory = variant.get("memory", {})
        if not isinstance(memory, dict):
            raise ValueError(f"{name}.memory must be an object")
        if memory.get("path", "memory") != "memory":
            raise ValueError("Round22 runner snapshots the repository memory/ directory")
        if memory.get("manifest_file", "verified-experience.json") != "verified-experience.json":
            raise ValueError("Memory manifest must use the frozen verified-experience.json name")
        flags = variant.get("feature_flags_by_mode", R21_FLAGS)
        if not isinstance(flags, dict):
            raise ValueError(f"{name}.feature_flags_by_mode must be an object")
        for mode in modes:
            row = flags.get(mode)
            if (not isinstance(row, dict) or not set(FEATURE_FLAGS).issubset(row)
                    or not set(row).issubset((*FEATURE_FLAGS, *OPTIONAL_FEATURE_FLAGS))
                    or any(type(v) is not bool for v in row.values())):
                raise ValueError(f"{name} must specify all boolean feature flags for {mode}")
            if variant["kind"] == "baseline_r21" and row != R21_FLAGS[mode]:
                raise ValueError("Round21 baseline feature flags differ from its frozen protocol")
            if row.get('milestone_advice_delivery') and (mode != 'dual_async'
                    or not row['pipeline_supervision'] or not row.get('two_role_runtime')):
                raise ValueError('milestone_advice_delivery requires two-role dual_async pipeline')
            if row.get('scout_prompt_parity') and (mode != 'dual_async'
                    or not row['pipeline_supervision'] or not row.get('two_role_runtime')):
                raise ValueError('scout_prompt_parity requires two-role dual_async pipeline')
            if row.get('two_role_runtime') and any(row.get(flag) for flag in (
                    'strong_target_judge', 'target_judge_overlap', 'strong_destination_judge',
                    'strong_recovery_executor', 'physical_failure_executor')):
                raise ValueError('two_role_runtime forbids independent model routes')
            if row.get("async_completion_recovery") and (
                    mode != "dual_async" or not row["pipeline_supervision"]):
                raise ValueError("async_completion_recovery requires dual_async pipelined supervision")
            if row.get("strong_target_judge") and mode != "dual_async":
                raise ValueError("strong_target_judge currently requires dual_async")
            if row.get('target_judge_overlap') and (mode != 'dual_async'
                    or not row['pipeline_supervision'] or not row.get('strong_target_judge')):
                raise ValueError('target_judge_overlap requires dual_async pipeline and Strong target judge')
            if row.get('strong_destination_judge') and (mode != 'dual_async'
                    or not row['pipeline_supervision'] or not row.get('strong_target_judge')):
                raise ValueError('strong_destination_judge requires dual_async pipeline and Strong target judge')
            if row.get("completion_rendezvous") and (mode != "dual_async"
                    or not row["pipeline_supervision"] or not row.get("async_completion_recovery")):
                raise ValueError("completion_rendezvous requires dual_async pipelined final completion recovery")
            if row.get("urgent_completion_review") and (mode != "dual_async"
                    or not row["pipeline_supervision"] or not row["priority_foreground"]
                    or not row.get("async_completion_recovery")):
                raise ValueError("urgent_completion_review requires dual_async pipeline, foreground priority and completion recovery")
            if row.get("urgent_completion_review") and row.get("completion_rendezvous"):
                raise ValueError("urgent_completion_review cannot combine with completion_rendezvous in this no-wait ablation")
            if row.get("completion_contract_repair") and (mode != "dual_async"
                    or not row.get("pipeline_supervision") or not row.get("priority_foreground")
                    or not row.get("async_completion_recovery")):
                raise ValueError("completion_contract_repair requires dual_async pipeline, foreground priority and completion recovery")
            if row.get("completion_contract_repair") and (row.get("completion_rendezvous")
                    or row.get("urgent_completion_review")):
                raise ValueError("completion_contract_repair must be tested separately from other completion scheduling ablations")
            if row.get('completion_schema_guard') and (mode != 'dual_async'
                    or not row.get('pipeline_supervision') or not row.get('async_completion_recovery')):
                raise ValueError('completion_schema_guard requires dual_async pipeline and completion recovery')
            if row.get('completion_schema_guard') and any(row.get(name) for name in (
                    'completion_contract_repair', 'urgent_completion_review', 'completion_rendezvous',
                    'recovery_reasoning', 'final_stage_prompt')):
                raise ValueError('completion_schema_guard must be a separate completion ablation')
            if row.get('physical_failure_executor') and (not row.get('strong_recovery_executor')
                    or not row.get('progress_recovery')):
                raise ValueError('physical_failure_executor requires Strong routing and physical progress feedback')
            if row.get('strong_recovery_executor') and (mode != 'dual_async'
                    or not row.get('pipeline_supervision') or not row.get('async_completion_recovery')):
                raise ValueError('strong_recovery_executor requires dual_async pipeline and completion recovery')
            if row.get('strong_recovery_executor') and any(row.get(name) for name in (
                    'completion_schema_guard', 'completion_contract_repair', 'urgent_completion_review',
                    'completion_rendezvous', 'recovery_reasoning', 'final_stage_prompt')):
                raise ValueError('strong_recovery_executor must be a separate completion ablation')
            if row.get("faithful_initial_referents") and (mode != "dual_async" or not row["pipeline_supervision"]):
                raise ValueError("faithful_initial_referents requires dual_async pipelined milestones")
            if row.get("recovery_reasoning") and (mode != "dual_async"
                    or not row["pipeline_supervision"] or not row.get("async_completion_recovery")):
                raise ValueError("recovery_reasoning requires dual_async pipelined final completion recovery")
            if row.get("final_stage_prompt") and (mode != "dual_async"
                    or not row["pipeline_supervision"] or not row.get("async_completion_recovery")):
                raise ValueError("final_stage_prompt requires dual_async pipelined final completion recovery")
    models = config.get("models", {})
    if models.get("fast") != {"model": FAST_MODEL, "options": FAST_OPTIONS}:
        raise ValueError("Fast model/options must match the frozen Qwen3.6-35B-A3B profile")
    if models.get("slow") != {"profile_id": SLOW_PROFILE_ID, "model": SLOW_MODEL, "options": SLOW_OPTIONS}:
        raise ValueError("Slow profile/model/options must match the frozen kimi_new profile")
    transport = config.get("transport", {})
    if transport != {"host": TRANSPORT_HOST, "address": TRANSPORT_ADDRESS}:
        raise ValueError("Transport must use the existing R21 DNS wrapper endpoint")
    scheduler = config.get("scheduler", {})
    cap = scheduler.get("max_concurrent_episodes", 60)
    if type(cap) is not int or cap < 1 or cap > 60:
        raise ValueError("Requested concurrency must be between 1 and the authorized cap of 60")
    for key, default in (("memory_reserve_gib", 8.0), ("memory_per_episode_gib", 2.0)):
        value = scheduler.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"scheduler.{key} must be a non-negative number")
    if scheduler.get("memory_per_episode_gib", 2.0) == 0:
        raise ValueError("scheduler.memory_per_episode_gib must be greater than zero")
    if scheduler.get("stop_sentinel", "STOP_NEW") != "STOP_NEW":
        raise ValueError("The dispatch stop sentinel is fixed at STOP_NEW")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must contain the explicitly selected experiment rows")
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        if not isinstance(case.get("suite"), str) or not isinstance(case.get("task_index"), int):
            raise ValueError(f"cases[{index}] needs suite and integer task_index")
        if case["suite"] not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError(f"cases[{index}] has an unsupported suite")
        if type(case.get("seed")) is not int or case["seed"] < 0:
            raise ValueError(f"cases[{index}].seed must be a non-negative integer")
        variant_name = case.get("variant")
        if variant_name not in variants:
            raise ValueError(f"cases[{index}] names an undeclared variant")
        if case.get("mode") not in variants[variant_name]["modes"]:
            raise ValueError(f"cases[{index}] mode is not enabled for its variant")
        attempt_id = case.get("attempt_id", "attempt-01")
        if (not isinstance(attempt_id, str) or not attempt_id or len(attempt_id) > 48
                or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in attempt_id)):
            raise ValueError(f"cases[{index}].attempt_id contains unsupported characters")
        case_key = (case["suite"], case["task_index"], case["seed"], variant_name, case["mode"], attempt_id)
        if case_key in seen_ids:
            raise ValueError("Duplicate case and attempt_id; assign a distinct attempt_id to an intentional repeat")
        seen_ids.add(case_key)


def load_catalog(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], Path, str]:
    catalog_path = resolve_config_path(config["catalog"]["path"], config_path)
    catalog_bytes = catalog_path.read_bytes()
    catalog_hash = hashlib.sha256(catalog_bytes).hexdigest()
    expected_hash = config["catalog"].get("sha256")
    if expected_hash and expected_hash != catalog_hash:
        raise ValueError("Task catalog SHA256 differs from the pinned config")
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, dict) or catalog.get("task_order_index") != 0:
        raise ValueError("Catalog must be an official task_order_index 0 catalog")
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 40:
        raise ValueError("Catalog must contain the 40 official evaluation tasks")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    env_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Malformed task catalog row")
        key = (task.get("suite"), task.get("task_index"))
        env_id, mission = task.get("env_id"), task.get("mission")
        hashes = task.get("state_hashes")
        if (key in by_key or key[0] not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
                or type(key[1]) is not int or not isinstance(env_id, str) or env_id in env_ids
                or not isinstance(mission, str) or not isinstance(hashes, dict)):
            raise ValueError("Malformed or duplicate task catalog entry")
        expected_env_id = f"openeta/libero_{key[0]}_task{key[1]}-v0"
        if env_id != expected_env_id:
            raise ValueError("Catalog env_id does not match its suite and task_index")
        if type(task.get("official_state_count")) is not int or task["official_state_count"] != 50:
            raise ValueError("Task catalog does not advertise 50 official initial states")
        for state, digest in hashes.items():
            if not state.isdigit() or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Catalog contains an invalid initial-state hash")
        by_key[key] = task
        env_ids.add(env_id)
    counts: dict[str, int] = {}
    for suite, _ in by_key:
        counts[suite] = counts.get(suite, 0) + 1
    if counts != {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10}:
        raise ValueError("Catalog suite counts differ from the frozen 40-task contract")
    catalog["_by_key"] = by_key
    return catalog, catalog_path, catalog_hash


def budget_for_suite(suite: str) -> dict[str, Any]:
    if suite == "libero_10":
        return {"stratum": "long", "max_decisions": 36, "timeout_s": 1800, "identity_check": False}
    return {"stratum": "standard", "max_decisions": 24, "timeout_s": 1200, "identity_check": True}


def expected_models(mode: str, slow_profile: dict[str, Any] | None = None) -> dict[str, str | None]:
    slow_model = (slow_profile or slow_profile_for_variant({}))["model"]
    if mode == "single_fast":
        return {"fast": FAST_MODEL, "slow": None, "grounding": FAST_MODEL}
    if mode == "single_strong":
        return {"fast": slow_model, "slow": None, "grounding": slow_model}
    return {"fast": FAST_MODEL, "slow": slow_model, "grounding": FAST_MODEL}


def expected_recorded_options(slow_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = slow_profile or slow_profile_for_variant({})
    return {"fast_model": FAST_MODEL, "fast_model_options": copy.deepcopy(FAST_OPTIONS),
            "slow_model": profile["model"], "slow_model_options": copy.deepcopy(profile["options"])}


def make_jobs(config: dict[str, Any], catalog: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    jobs = []
    for dispatch_order, requested in enumerate(config["cases"], 1):
        task = catalog["_by_key"].get((requested["suite"], requested["task_index"]))
        if task is None:
            raise ValueError("Selected task is absent from the official catalog")
        seed = requested["seed"]
        state_hash = task["state_hashes"].get(str(seed))
        if seed >= task["official_state_count"] or not isinstance(state_hash, str) or len(state_hash) != 64:
            raise ValueError("Selected official initial-state index has no pinned catalog hash")
        variant_name = requested["variant"]
        mode = requested["mode"]
        attempt_id = requested.get("attempt_id", "attempt-01")
        budget = budget_for_suite(task["suite"])
        basename = task["env_id"].rsplit("/", 1)[-1]
        job_id = f"{variant_name}__{task['suite']}_task{task['task_index']}_s{seed}__{mode}__{attempt_id}"
        job_path = output_root / "attempts" / variant_name / task["suite"] / f"task{task['task_index']}-s{seed}" / mode / attempt_id
        variant = config["variants"][variant_name]
        slow_profile = slow_profile_for_variant(variant)
        flags = variant.get("feature_flags_by_mode", R21_FLAGS)[mode]
        routes = expected_models(mode, slow_profile)
        if flags.get('two_role_runtime'):
            # Removing independent semantic checks is an explicit protocol change;
            # decision/time budgets and physical evidence gates are unchanged.
            budget = {**budget, 'identity_check': False}
            routes.update(grounding=None, target_judge=None, destination_judge=None, recovery_executor=None)
        if flags.get("strong_target_judge"):
            if not budget["identity_check"]:
                raise ValueError("strong_target_judge requires identity_check enabled in the frozen task budget")
            routes.update(grounding=None, target_judge=slow_profile["model"])
        if flags.get('strong_destination_judge'):
            routes['destination_judge'] = slow_profile["model"]
        if flags.get('strong_recovery_executor'):
            routes['recovery_executor'] = slow_profile['model']
        jobs.append({
            "job_id": job_id, "dispatch_order": dispatch_order, "suite": task["suite"],
            "task_index": int(task["task_index"]), "env_id": task["env_id"], "mission": task["mission"],
            "seed": seed, "initial_state_sha256": state_hash, "variant": variant_name,
            "variant_kind": variant["kind"], "mode": mode, "attempt_id": attempt_id,
            "budget": budget, "identity_check": budget["identity_check"], "feature_flags": dict(flags),
            "expected_active_models": routes,
            "slow_profile": slow_profile,
            "expected_recorded_model_options": expected_recorded_options(slow_profile), "path": str(job_path),
            "stratum": budget["stratum"], "task_basename": basename,
        })
    return jobs


def available_memory_bytes() -> int:
    """Return the conservative available-memory limit, including container caps."""
    candidates: list[int] = []
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            fields = {}
            for line in meminfo.read_text(encoding="ascii").splitlines():
                name, value = line.split(":", 1)
                fields[name] = int(value.strip().split()[0]) * 1024
            if "MemAvailable" in fields:
                candidates.append(fields["MemAvailable"])
        except (OSError, ValueError, IndexError):
            pass
    # Linux MemAvailable includes reclaimable page cache. AVPHYS_PAGES is only
    # free pages, so taking their minimum incorrectly treats cache as committed
    # process memory. Use the latter solely as an unavailable-meminfo fallback.
    if not candidates:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            candidates.append(page_size * available_pages)
        except (OSError, ValueError, AttributeError):
            pass
    # Resolve the process membership through mountinfo. For a delegated mount
    # with a hidden hierarchy root we cannot inspect ancestor constraints; refuse
    # dispatch instead of silently falling back to the host's available memory.
    try:
        membership = Path("/proc/self/cgroup").read_text(encoding="ascii")
    except FileNotFoundError as exc:
        if sys.platform.startswith("linux"):
            raise ValueError("Cannot inspect process cgroup membership") from exc
        membership = ""
    except OSError as exc:
        raise ValueError("Cannot inspect process cgroup membership") from exc
    if sys.platform.startswith("linux") and not membership.strip():
        raise ValueError("Empty process cgroup membership; refusing dispatch")
    for line in membership.splitlines():
        try:
            hierarchy, controllers, relative = line.split(":", 2)
        except ValueError as exc:
            raise ValueError("Malformed process cgroup membership") from exc
        version = 2 if hierarchy == "0" and not controllers else (
            1 if "memory" in controllers.split(",") else 0)
        if not version:
            continue
        if not relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("Unsafe process cgroup path")
        try:
            mounts = Path("/proc/self/mountinfo").read_text(encoding="ascii")
        except OSError as exc:
            raise ValueError("Cannot resolve process cgroup mount") from exc
        roots = []
        for mount in mounts.splitlines():
            try:
                left, right = mount.split(" - ", 1)
                fields, options = left.split(), right.split()
                matches = (version == 2 and options[0] == "cgroup2") or (
                    version == 1 and options[0] == "cgroup" and
                    "memory" in options[2].split(","))
                if matches and fields[3] == "/":
                    # Kernel escapes spaces, tabs and backslashes in mountinfo.
                    point = fields[4]
                    for escaped, literal in ((r"\040", " "), (r"\011", "\t"),
                                             (r"\012", "\n"), (r"\134", "\\")):
                        point = point.replace(escaped, literal)
                    if Path(point).is_absolute() and ".." not in Path(point).parts:
                        roots.append(Path(point))
            except (ValueError, IndexError):
                continue
        if not roots:
            raise ValueError("Cannot inspect full process cgroup hierarchy; refusing dispatch")
        root = roots[0]
        names = ("memory.max", "memory.current") if version == 2 else (
            "memory.limit_in_bytes", "memory.usage_in_bytes")
        directory = root / relative.lstrip("/")
        while True:
            limit_path, current_path = directory / names[0], directory / names[1]
            try:
                limit_text = limit_path.read_text(encoding="ascii").strip()
            except FileNotFoundError as exc:
                # The global v2 root has no memory controller files by design.
                if directory == root and version == 2:
                    break
                raise ValueError("Cannot inspect process cgroup memory limit") from exc
            except OSError as exc:
                raise ValueError("Cannot inspect process cgroup memory limit") from exc
            if limit_text != "max":
                try:
                    limit = int(limit_text)
                    if limit < 0:
                        raise ValueError("Negative cgroup memory limit")
                    if limit < 1 << 60:
                        current = int(current_path.read_text(encoding="ascii").strip())
                        if current < 0:
                            raise ValueError("Negative cgroup memory usage")
                        candidates.append(max(0, limit - current))
                except (OSError, ValueError) as exc:
                    raise ValueError("Cannot determine finite cgroup memory headroom") from exc
            if directory == root:
                break
            directory = directory.parent
    if not candidates and sys.platform == "darwin":
        try:
            report = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages: dict[str, int] = {}
            for line in report.splitlines()[1:]:
                if ":" not in line:
                    continue
                name, value = line.split(":", 1)
                digits = "".join(character for character in value if character.isdigit())
                if digits:
                    pages[name.strip()] = int(digits)
            available_pages = sum(pages.get(name, 0) for name in ("Pages free", "Pages inactive", "Pages speculative"))
            if available_pages:
                candidates.append(available_pages * page_size)
        except (OSError, subprocess.CalledProcessError, ValueError, AttributeError):
            pass
    if not candidates:
        raise ValueError("Could not determine available system or container memory")
    return min(candidates)


def concurrency_for_memory(scheduler: dict[str, Any], available_bytes: int) -> dict[str, Any]:
    gib = 1024 ** 3
    requested = min(int(scheduler.get("max_concurrent_episodes", 60)), 60)
    reserve = float(scheduler.get("memory_reserve_gib", 8.0))
    per_episode = float(scheduler.get("memory_per_episode_gib", 2.0))
    capacity = max(0, int((available_bytes / gib - reserve) // per_episode))
    effective = min(requested, capacity)
    if effective < 1:
        raise ValueError("Memory preflight leaves capacity for fewer than one episode")
    return {"requested_cap": requested, "authorized_hard_cap": 60, "available_bytes": available_bytes,
            "memory_measurement_policy": "host_available_min_resolved_finite_cgroup_headroom",
            "reserve_gib": reserve, "estimated_gib_per_episode": per_episode,
            "memory_derived_cap": capacity, "effective_concurrency": effective}


def preflight_source(config: dict[str, Any], name: str, variant: dict[str, Any]) -> dict[str, Any]:
    repo = Path(variant["repository_root"]).resolve()
    package = repo / "src" / "embodied_harness"
    if not package.is_dir():
        raise ValueError(f"Configured source package is missing for {name}")
    commit = git_value(repo, "rev-parse", "HEAD")
    if commit != variant["expected_commit"]:
        raise ValueError(f"{name} source commit is not the pinned expected commit")
    if variant.get("expected_branch") and git_value(repo, "branch", "--show-current") != variant["expected_branch"]:
        raise ValueError(f"{name} source branch differs from the pinned branch")
    if git_value(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"{name} source worktree is dirty; freeze a clean commit before execution")
    for required in (repo / "scripts" / "run_with_slow_profile.py", repo / "scripts" / "with_resolved_host.py",
                     repo / "scripts" / "env.sh", repo / ".venv311" / "bin" / "python"):
        if not required.is_file():
            raise ValueError(f"Required runtime file is missing for {name}: {required.name}")
    memory = repo / "memory"
    manifest_file = variant.get("memory", {}).get("manifest_file", "verified-experience.json")
    memory_manifest = memory / manifest_file
    if not memory_manifest.is_file():
        raise ValueError(f"Frozen memory manifest is missing for {name}")
    return {
        "name": name, "kind": variant["kind"], "repository": str(repo),
        "branch": git_value(repo, "branch", "--show-current"), "commit": commit,
        "expected_commit": variant["expected_commit"],
        "package_sha256": package_source_sha256(package), "package_tree_sha256": tree_sha256(package),
        "memory_tree_sha256": tree_sha256(memory), "memory_manifest_file": manifest_file,
        "memory_manifest_sha256": sha256_file(memory_manifest),
        "run_with_slow_profile_sha256": sha256_file(repo / "scripts" / "run_with_slow_profile.py"),
        "with_resolved_host_sha256": sha256_file(repo / "scripts" / "with_resolved_host.py"),
        "env_script_sha256": sha256_file(repo / "scripts" / "env.sh"),
    }


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ValueError("Snapshot contains a symlink")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def snapshot_source(root: Path, variant: dict[str, Any], source: dict[str, Any]) -> Path:
    repo = Path(source["repository"])
    name = source["name"]
    snapshot = root / "source-snapshots" / name
    package_target = snapshot / "src" / "embodied_harness"
    package_target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(repo / "src" / "embodied_harness", package_target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    scripts_target = snapshot / "scripts"
    scripts_target.mkdir(parents=True, exist_ok=False)
    for script in ("run_with_slow_profile.py", "with_resolved_host.py", "env.sh"):
        shutil.copy2(repo / "scripts" / script, scripts_target / script)
    memory_target = root / "source-snapshots" / "memory" / name
    shutil.copytree(repo / "memory", memory_target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    if package_source_sha256(package_target) != source["package_sha256"]:
        raise ValueError(f"Frozen package snapshot hash mismatch for {name}")
    if tree_sha256(package_target) != source["package_tree_sha256"]:
        raise ValueError(f"Frozen package tree hash mismatch for {name}")
    if tree_sha256(memory_target) != source["memory_tree_sha256"]:
        raise ValueError(f"Frozen memory snapshot hash mismatch for {name}")
    if sha256_file(memory_target / source["memory_manifest_file"]) != source["memory_manifest_sha256"]:
        raise ValueError(f"Frozen memory manifest hash mismatch for {name}")
    snapshot_scripts = snapshot / "scripts"
    for filename, key in (("run_with_slow_profile.py", "run_with_slow_profile_sha256"),
                          ("with_resolved_host.py", "with_resolved_host_sha256"),
                          ("env.sh", "env_script_sha256")):
        if sha256_file(snapshot_scripts / filename) != source[key]:
            raise ValueError(f"Frozen runtime script hash mismatch for {name}")
    make_read_only(snapshot)
    make_read_only(memory_target)
    return snapshot


def app_arguments(job: dict[str, Any], memory_path: Path, review_every: int = 3) -> list[str]:
    args = ["--backend", "direct_libero", "--env-id", job["env_id"], "--mission", job["mission"],
            "--seed", str(job["seed"]), "--mode", job["mode"],
            "--max-decisions", str(job["budget"]["max_decisions"]),
            "--timeout", str(job["budget"]["timeout_s"]), "--review-every", str(review_every),
            "--hierarchy", "milestones", "--output", job["path"], "--memory", str(memory_path),
            "--identity-check" if job["identity_check"] else "--no-identity-check"]
    for flag in (*FEATURE_FLAGS, *OPTIONAL_FEATURE_FLAGS):
        if job["feature_flags"].get(flag):
            args.append("--" + flag.replace("_", "-"))
    return args


def app_command(config: dict[str, Any], job: dict[str, Any], snapshot_root: Path,
                memory_path: Path) -> list[str]:
    repo = Path(config["variants"][job["variant"]]["repository_root"]).resolve()
    python = str(repo / ".venv311" / "bin" / "python")
    profile = slow_profile_for_job(job)
    if profile != slow_profile_for_variant(config["variants"][job["variant"]]):
        raise ValueError("Job Slow profile differs from the frozen variant")
    slow_options = json.dumps(profile["profile_options"], sort_keys=True, separators=(",", ":"))
    app_args = app_arguments(job, memory_path)
    dns_command = [python, str(snapshot_root / "scripts" / "with_resolved_host.py"),
                   "--host", TRANSPORT_HOST, "--address", TRANSPORT_ADDRESS,
                   "--module", "embodied_harness.app", "--", *app_args]
    if profile["options"] != profile["profile_options"]:
        # argv, not shell interpolation: only the allowlisted public options
        # change after the wrapper has validated and loaded its private profile.
        dns_command = ["env", "SLOW_MODEL_OPTIONS=" + json.dumps(
            profile["options"], sort_keys=True, separators=(",", ":")), *dns_command]
    return [python, str(snapshot_root / "scripts" / "run_with_slow_profile.py"),
            "--profile-id", profile["profile_id"], "--expected-options-json", slow_options,
            "--", *dns_command]


def shell_command(command: list[str], repo: Path, snapshot_root: Path) -> str:
    import_root = snapshot_root / "src"
    return ("set -e\nsource scripts/env.sh\n"
            f"export PYTHONPATH={shlex.quote(str(import_root))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
            "export PYTHONDONTWRITEBYTECODE=1\nexec " + shlex.join(command))


def check_result(result: dict[str, Any], job: dict[str, Any], source: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    profile = slow_profile_for_job(job)
    required = ("backend", "env_id", "seed", "mode", "status", "success", "max_decisions", "timeout",
                "review_every", "identity_check", "fast_model", "fast_model_options", "slow_model",
                "slow_model_options", "models", "source_sha256", *job["feature_flags"])
    issues.extend(f"missing_{key}" for key in required if key not in result)
    exact = {"backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
             "mode": job["mode"], "max_decisions": job["budget"]["max_decisions"],
             "timeout": job["budget"]["timeout_s"], "review_every": 3,
             "identity_check": job["identity_check"], "source_sha256": source["package_sha256"],
             **job["feature_flags"]}
    for key, expected in exact.items():
        if key in result and result[key] != expected:
            issues.append(f"mismatch_{key}")
    if not isinstance(result.get("status"), str) or not result.get("status"):
        issues.append("invalid_status")
    if type(result.get("success")) is not bool:
        issues.append("invalid_success")
    if result.get("success") is True:
        evidence = result.get("evidence")
        official_evidence = evidence.get("evidence") if isinstance(evidence, dict) else None
        if (not isinstance(evidence, dict) or evidence.get("status") != "verified"
                or evidence.get("source") != "official_libero_check_success"
                or not isinstance(official_evidence, list) or not official_evidence
                or any(not isinstance(item, dict)
                       or item.get("source") != "official_libero_check_success" for item in official_evidence)):
            issues.append("success_without_official_verification")
    for key, expected in job["expected_recorded_model_options"].items():
        if result.get(key) != expected:
            issues.append(f"mismatch_{key}")
    models = result.get("models")
    if not isinstance(models, list):
        issues.append("invalid_models")
        models = []
    routes: dict[str, list[str]] = {}
    for model in models:
        if isinstance(model, dict) and isinstance(model.get("role"), str) and isinstance(model.get("model"), str):
            routes.setdefault(model["role"], []).append(model["model"])
        else:
            issues.append("invalid_model_route_record")
    for role, expected in job["expected_active_models"].items():
        actual = routes.get(role, [])
        if expected is None and actual:
            issues.append(f"unexpected_{role}_route")
        elif expected is not None and actual and any(model != expected for model in actual):
            issues.append(f"mismatch_{role}_route")
        elif role in {"fast", "slow", "target_judge", "destination_judge", "recovery_executor"} and expected is not None and not actual:
            issues.append(f"missing_{role}_route")
    if not job['feature_flags'].get('strong_recovery_executor') and routes.get('recovery_executor'):
        issues.append('unexpected_recovery_executor_route')
    if job['feature_flags'].get('two_role_runtime'):
        allowed = {'fast', 'slow'} if job['mode'].startswith('dual') else {'fast'}
        if set(routes) != allowed or any(len(names) != 1 for names in routes.values()):
            issues.append('invalid_two_role_topology')
        policy = result.get('agent_topology')
        if (not isinstance(policy, dict) or policy.get('name') != 'slow_fast_only'
                or policy.get('roles') != sorted(allowed)
                or policy.get('independent_checkers') is not False
                or policy.get('strong_executor_handoff') is not False
                or policy.get('slow_execution_tools') is not False):
            issues.append('mismatch_two_role_policy')
        if result.get('identity_check') is not False:
            issues.append('two_role_legacy_identity_enabled')
    appearance_memory = result.get("appearance_memory")
    if (job["identity_check"] and not job["feature_flags"].get("strong_target_judge")
            and isinstance(appearance_memory, list) and appearance_memory
            and not routes.get("grounding")):
        issues.append("missing_grounding_route_for_memory")
    if job["feature_flags"].get("strong_target_judge"):
        if result.get("target_judge_model") != profile["model"] or result.get("target_judge_options") != profile["options"]:
            issues.append("mismatch_target_judge_model_options")
        if result.get("target_judge_trigger") != "every_pregrasp_with_same_scope_cache":
            issues.append("mismatch_target_judge_trigger")
        if (job["feature_flags"].get("appearance_reference_memory") and result.get('target_judge_memory')
                != "verified_prior_appearance_if_matched_plus_original_mission_and_public_images"):
            issues.append('mismatch_target_judge_memory_policy')
    if (job['feature_flags'].get('target_judge_overlap') and result.get('target_judge_scheduling')
            != 'one_speculative_same_snapshot_check_overlapped_with_actor'):
        issues.append('mismatch_target_judge_scheduling')
    if job['feature_flags'].get('strong_destination_judge'):
        for field, expected in {'destination_judge_model':profile["model"], 'destination_judge_options':profile["options"],
                'destination_judge_trigger':'every_place_with_exact_point_and_scope_cache',
                'destination_judge_memory':'original_mission_task_start_and_current_images_only'}.items():
            if result.get(field) != expected:
                issues.append('mismatch_' + field)
    if job['feature_flags'].get('strong_recovery_executor'):
        expected_policy = {'model':profile['model'], 'options':profile['options'],
            'role':'recovery_executor', 'max_requests':4,
            'trigger':('first_official_negative_final_claim_or_two_same_key_physical_failures'
                if job['feature_flags'].get('physical_failure_executor')
                else 'first_official_negative_final_claim_with_two_future_decisions'),
            'scheduling':'replace_actor_calls_within_existing_decision_budget',
            'scope':'current_final_contract_id_revision_generation_mission',
            'history':'copy_primary_executed_history_and_commit_only_returned_decisions',
            'cost':'separate_strong_executor_accounting_no_planner_tool_access'}
        if result.get('recovery_executor_policy') != expected_policy:
            issues.append('mismatch_recovery_executor_policy')
        recovery_models = [m for m in models if isinstance(m, dict) and m.get('role') == 'recovery_executor']
        if len(recovery_models) != 1:
            issues.append('invalid_recovery_executor_role_count')
        else:
            count = (recovery_models[0].get('request_accounting') or {}).get('logical_requests')
            if type(count) is not int or not 0 <= count <= 4:
                issues.append('recovery_executor_request_budget_exceeded_or_missing')
    if job["feature_flags"].get("completion_rendezvous") and result.get("completion_review_timeout_s") != 30.0:
        issues.append("mismatch_completion_review_timeout")
    if job["feature_flags"].get("recovery_reasoning"):
        policy = result.get("recovery_reasoning_policy") or {}
        expected_policy = {
            "trigger": "official_final_completion_rejected_at_current_scope",
            "max_requests_per_episode": 4, "model": FAST_MODEL,
            "option_override": {"chat_template_kwargs": {"enable_thinking": True}},
            "changes_output_budget": False}
        if any(policy.get(key) != value for key, value in expected_policy.items()):
            issues.append("mismatch_recovery_reasoning_policy")
    return issues


def read_init_record(output: Path, job: dict[str, Any]) -> tuple[str | None, list[str]]:
    path = output / "direct-initialization.json"
    if not path.is_file():
        return None, ["initialization_record_missing"]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ["initialization_record_invalid"]
    expected = {"suite": job["suite"], "task_index": job["task_index"],
                "initial_state_index": job["seed"], "initial_state_sha256": job["initial_state_sha256"]}
    issues = [f"initialization_mismatch_{key}" for key, value in expected.items() if record.get(key) != value]
    return (record.get("initial_state_sha256") if not issues else None), issues


def count_request_events(output: Path) -> dict[str, Any]:
    totals = {"http_401_attempts": 0, "http_403_attempts": 0, "http_429_attempts": 0}
    errors: list[str] = []
    per_trace: dict[str, dict[str, int]] = {}
    for path in sorted(output.glob("*-requests.jsonl")):
        counts = dict.fromkeys(totals, 0)
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"{path.name}:{line_number}:invalid_json")
                    continue
                if not isinstance(event, dict) or event.get("kind") != "attempt.finished":
                    continue
                status = event.get("status_code")
                if type(status) is int and status in (401, 403, 429):
                    counts[f"http_{status}_attempts"] += 1
        except OSError:
            errors.append(f"{path.name}:unreadable")
        per_trace[path.name] = counts
        for key in totals:
            totals[key] += counts[key]
    return {**totals, "per_trace_counts": per_trace, "trace_errors": errors}


def active_auth_error(jobs: list[dict[str, Any]]) -> bool:
    for job in jobs:
        for path in Path(job["path"]).glob("*-requests.jsonl"):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (isinstance(event, dict) and event.get("kind") == "attempt.finished"
                        and type(event.get("status_code")) is int and event["status_code"] in (401, 403)):
                    return True
    return False


def run_job(config: dict[str, Any], job: dict[str, Any], source: dict[str, Any], root: Path,
            snapshots: dict[str, Path], progress: dict[str, Any], progress_lock: threading.Lock,
            stop_event: threading.Event) -> dict[str, Any]:
    output = Path(job["path"])
    if stop_event.is_set():
        record = {"job_id": job["job_id"], "state": "skipped_stop_new", "path": str(output)}
        with progress_lock:
            progress["jobs"][job["job_id"]].update(record)
            atomic_json(root / "status.json", progress)
        return record
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        record = {"job_id": job["job_id"], "state": "interrupted_not_replayed", "path": str(output),
                  "integrity_issues": ["attempt_directory_exists"]}
        with progress_lock:
            progress["jobs"][job["job_id"]].update(record)
            atomic_json(root / "status.json", progress)
        return record
    started = time.time()
    with progress_lock:
        progress["jobs"][job["job_id"]].update({"state": "running", "started_at": started})
        atomic_json(root / "status.json", progress)
    snapshot = snapshots[job["variant"]]
    memory_path = root / "source-snapshots" / "memory" / job["variant"] / "verified-experience.json"
    command = app_command(config, job, snapshot, memory_path)
    repo = Path(source["repository"])
    shell = shell_command(command, repo, snapshot)
    atomic_json(output / "attempt.json", {
        "job": job, "source": {key: source[key] for key in ("name", "kind", "branch", "commit", "package_sha256",
                                                                  "package_tree_sha256", "memory_tree_sha256")},
        "started_at": started,
    })
    process_error: str | None = None
    exit_code: int | None = None
    try:
        with (output / "console.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(["bash", "-c", shell], cwd=repo, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False, start_new_session=True)
            exit_code = int(completed.returncode)
    except Exception as exc:
        process_error = type(exc).__name__
    result = None
    result_issues: list[str] = []
    result_path = output / "result.json"
    if result_path.is_file():
        try:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(candidate, dict):
                result_issues = ["result_json_not_object"]
            else:
                result = candidate
                result_issues = check_result(result, job, source)
        except (OSError, json.JSONDecodeError):
            result_issues = ["result_json_invalid"]
    init_hash, init_issues = read_init_record(output, job)
    request_audit = count_request_events(output)
    issues = result_issues + init_issues + ["request_trace_" + item for item in request_audit["trace_errors"]]
    if process_error:
        state = "launcher_error"
    elif result is None:
        state = "invalid_result" if result_issues else "failed_no_result"
    elif issues:
        state = "integrity_error"
    elif exit_code != 0:
        state = "completed_nonzero_exit"
    else:
        state = "completed"
    record = {"job_id": job["job_id"], "path": str(output), "state": state, "exit_code": exit_code,
              "wall_s": round(time.time() - started, 6), "result_status": result.get("status") if result else None,
              "success": result.get("success") if result else None,
              "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
              "initial_state_sha256": init_hash, "request_audit": request_audit,
              "integrity_issues": issues}
    if process_error:
        record["process_error_type"] = process_error
    with progress_lock:
        progress["jobs"][job["job_id"]].update(record)
        atomic_json(root / "status.json", progress)
    print(json.dumps({"job_id": job["job_id"], "state": state, "exit_code": exit_code,
                      "result_status": record["result_status"], "success": record["success"]}), flush=True)
    return record


def campaign_path(config: dict[str, Any], config_path: Path) -> tuple[Path, str]:
    config_sha = sha256_file(config_path)
    if config.get("output_root"):
        return Path(config["output_root"]), config_sha
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"round22-targeted-{now}-{config_sha[:8]}-{uuid.uuid4().hex[:8]}"
    return Path(config["output_base"]) / run_id, config_sha


def dry_run(config: dict[str, Any], config_path: Path, catalog_sha: str,
            jobs: list[dict[str, Any]], resources: dict[str, Any]) -> int:
    preview = [{key: job[key] for key in ("dispatch_order", "job_id", "suite", "task_index", "seed",
                                          "initial_state_sha256", "variant", "mode", "stratum", "budget",
                                          "feature_flags", "slow_profile", "expected_active_models",
                                          "expected_recorded_model_options")} for job in jobs]
    summary = {
        "dry_run": True, "protocol_id": config["protocol_id"], "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path), "catalog_sha256": catalog_sha,
        "simulator_or_model_started": False, "private_profile_read": False,
        "source_snapshots_written": False, "planned_attempts": len(jobs),
        "output_root": str(jobs[0]["path"]).split("/attempts/", 1)[0], "resources": resources,
        "models": config["models"], "models_note": "Global models are legacy defaults; each job materializes its declared Slow profile.",
        "variants": {name: {"kind": v["kind"], "repository_root": v["repository_root"],
                           "expected_commit": v["expected_commit"], "modes": v["modes"],
                           "slow_profile": slow_profile_for_variant(v)}
                                                       for name, v in config["variants"].items()},
        "job_preview": preview,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def output_root_preflight(root: Path) -> None:
    if root.exists():
        raise ValueError("Generated campaign output root already exists; no overwrite or replay is allowed")
    root.mkdir(parents=True, exist_ok=False)


def execute(config: dict[str, Any], config_path: Path, catalog_path: Path, catalog_sha: str,
            jobs: list[dict[str, Any]], resources: dict[str, Any], root: Path, config_sha: str) -> int:
    sources = {name: preflight_source(config, name, variant) for name, variant in config["variants"].items()}
    output_root_preflight(root)
    lock = root / "launcher.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": time.time()}, stream)
        stream.write("\n")
    try:
        snapshots = {name: snapshot_source(root, config["variants"][name], source)
                     for name, source in sources.items()}
        shutil.copy2(config_path, root / "protocol.json")
        shutil.copy2(catalog_path, root / "task-catalog.json")
        public_sources = {name: {key: value for key, value in source.items() if key != "repository"}
                          | {"repository": source["repository"]} for name, source in sources.items()}
        manifest = {
            "schema_version": 1, "protocol_id": config["protocol_id"], "config_sha256": config_sha,
            "catalog_sha256": catalog_sha, "created_at": time.time(), "planned_attempts": len(jobs),
            "source_variants": public_sources, "models": config["models"],
            "transport": config["transport"], "budgets": {
                "standard": {"max_decisions": 24, "timeout_s": 1200, "identity_check": True},
                "long": {"max_decisions": 36, "timeout_s": 1800, "identity_check": False}},
            "scheduler": resources | {"stop_sentinel": config["scheduler"].get("stop_sentinel", "STOP_NEW"),
                                      "no_automatic_replay": True},
            "no_automatic_replay": True, "retain_all_attempts": True,
            "cases": jobs,
        }
        atomic_json(root / "manifest.json", manifest)
        progress = {"protocol_id": config["protocol_id"], "config_sha256": config_sha,
                    "catalog_sha256": catalog_sha, "campaign_status": "running", "created_at": time.time(),
                    "effective_concurrency": resources["effective_concurrency"], "stop_reason": None,
                    "jobs": {job["job_id"]: {"state": "planned", "path": job["path"],
                                              "dispatch_order": job["dispatch_order"]} for job in jobs}}
        progress_lock = threading.Lock()
        atomic_json(root / "status.json", progress)
        stop_event = threading.Event()
        stop_reason: str | None = None
        signal_number: int | None = None

        def signal_handler(signum: int, _frame: Any) -> None:
            nonlocal signal_number
            signal_number = signum

        old_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        for signum in old_handlers:
            signal.signal(signum, signal_handler)
        records: dict[str, dict[str, Any]] = {}
        next_index = 0
        active: dict[concurrent.futures.Future, dict[str, Any]] = {}
        max_workers = resources["effective_concurrency"]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                while active or next_index < len(jobs):
                    if signal_number is not None and stop_reason is None:
                        stop_reason = f"signal_{signal.Signals(signal_number).name.lower()}"
                        stop_event.set()
                    if stop_reason is None and active_auth_error(list(active.values())):
                        stop_reason = "http_401_or_403"
                        stop_event.set()
                    if (root / "STOP_NEW").exists() and stop_reason is None:
                        stop_reason = "stop_new_sentinel"
                        stop_event.set()
                    while not stop_event.is_set() and next_index < len(jobs) and len(active) < max_workers:
                        if active_auth_error(list(active.values())):
                            stop_reason = "http_401_or_403"
                            stop_event.set()
                            break
                        if (root / "STOP_NEW").exists():
                            stop_reason = "stop_new_sentinel"
                            stop_event.set()
                            break
                        job = jobs[next_index]
                        future = pool.submit(run_job, config, job, sources[job["variant"]], root,
                                             snapshots, progress, progress_lock, stop_event)
                        active[future] = job
                        next_index += 1
                    if not active:
                        break
                    done, _ = concurrent.futures.wait(active, timeout=0.5,
                                                      return_when=concurrent.futures.FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        job = active.pop(future)
                        try:
                            record = future.result()
                        except Exception as exc:
                            record = {"job_id": job["job_id"], "state": "launcher_error",
                                      "path": job["path"], "error_type": type(exc).__name__}
                            with progress_lock:
                                progress["jobs"][job["job_id"]].update(record)
                                atomic_json(root / "status.json", progress)
                        records[job["job_id"]] = record
                        if record.get("state") in FATAL_STATES or record.get("state") == "interrupted_not_replayed":
                            if stop_reason is None:
                                stop_reason = "launcher_integrity_failure"
                                stop_event.set()
                        audit = record.get("request_audit", {})
                        if audit.get("http_401_attempts", 0) or audit.get("http_403_attempts", 0):
                            if stop_reason is None:
                                stop_reason = "http_401_or_403"
                                stop_event.set()
                        threshold = int(config["scheduler"].get("http_429_attempts_per_job_threshold", 3))
                        if audit.get("http_429_attempts", 0) >= threshold and stop_reason is None:
                            stop_reason = "http_429_circuit"
                            stop_event.set()
                        if audit.get("trace_errors") and stop_reason is None:
                            stop_reason = "launcher_integrity_failure"
                            stop_event.set()
                for job in jobs[next_index:]:
                    record = {"job_id": job["job_id"], "state": "not_dispatched", "path": job["path"]}
                    records[job["job_id"]] = record
                    with progress_lock:
                        progress["jobs"][job["job_id"]].update(record)
                        atomic_json(root / "status.json", progress)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        ordered = [records.get(job["job_id"], {"job_id": job["job_id"], "state": "not_dispatched",
                                               "path": job["path"]}) for job in jobs]
        fatal = any(row.get("state") in FATAL_STATES or row.get("state") == "interrupted_not_replayed" for row in ordered)
        if fatal:
            campaign_status = "finished_with_errors"
        elif stop_reason or any(row.get("state") != "completed" for row in ordered):
            campaign_status = "partial_stopped"
        else:
            campaign_status = "complete"
        completed_at = time.time()
        progress.update({"campaign_status": campaign_status, "stop_reason": stop_reason,
                         "completed_at": completed_at})
        atomic_json(root / "status.json", progress)
        atomic_json(root / "results.json", {
            "schema_version": 1, "protocol_id": config["protocol_id"], "config_sha256": config_sha,
            "catalog_sha256": catalog_sha, "campaign_status": campaign_status,
            "stop_reason": stop_reason, "completed_at": completed_at,
            "planned_attempts": len(jobs), "dispatched_attempts": sum(row.get("state") not in {
                "not_dispatched", "skipped_stop_new"} for row in ordered), "jobs": ordered,
        })
        return 0 if campaign_status == "complete" else 1
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="External prepared Round22 JSON config")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Execute only after source and resource preflight")
    action.add_argument("--dry-run", action="store_true", help="Print a plan; this is the default")
    parser.add_argument("--show-jobs", action="store_true", help="Retained for CLI consistency; all cases are shown")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    try:
        config = read_json(config_path, "Round22 config")
        validate_config(config)
        catalog, catalog_path, catalog_sha = load_catalog(config, config_path)
        root, config_sha = campaign_path(config, config_path)
        jobs = make_jobs(config, catalog, root)
        memory = concurrency_for_memory(config["scheduler"], available_memory_bytes())
        if args.execute:
            sources = {name: preflight_source(config, name, variant) for name, variant in config["variants"].items()}
            summary_sources = {name: {"commit": source["commit"], "package_sha256": source["package_sha256"],
                                      "memory_tree_sha256": source["memory_tree_sha256"]}
                               for name, source in sources.items()}
            memory["sources_preflight"] = summary_sources
            return execute(config, config_path, catalog_path, catalog_sha, jobs, memory, root, config_sha)
        return dry_run(config, config_path, catalog_sha, jobs, memory)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Round22 runner refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
