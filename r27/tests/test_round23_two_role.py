"""Offline admission and audit tests for the two-role runtime."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from embodied_harness import app  # noqa: E402
from embodied_harness.domain.models import Observation  # noqa: E402
from embodied_harness.runtime.topology import ModelTopology, allowed_roles  # noqa: E402
import run_round22_targeted as runner  # noqa: E402


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("single_fast", ("fast",)),
        ("single_strong", ("fast",)),
        ("dual_async", ("fast", "slow")),
        ("dual_sync", ("fast", "slow")),
    ],
)
def test_allowed_roles_are_exact_for_each_mode(mode, expected):
    assert allowed_roles(mode) == expected


@pytest.mark.parametrize("mode", ["single_fast", "single_strong", "dual_async", "dual_sync"])
def test_extra_and_duplicate_roles_are_rejected_before_client_construction(mode):
    created = []

    def factory(*args, **kwargs):
        created.append((args, kwargs))
        return object()

    topology = ModelTopology(mode, enabled=True, store=SimpleNamespace(emit=lambda *args: None))
    with pytest.raises(ValueError, match="role forbidden"):
        topology.create(factory, "url", "key", "model", "extra", Path("."))
    assert created == []

    topology.create(factory, "url", "key", "model", "fast", Path("."))
    assert len(created) == 1
    with pytest.raises(ValueError, match="Duplicate model client"):
        topology.create(factory, "url", "key", "model", "fast", Path("."))
    assert len(created) == 1


@pytest.mark.parametrize("mode", ["single_fast", "single_strong", "dual_async", "dual_sync"])
def test_two_role_runtime_requires_identity_check_to_be_explicitly_disabled(mode):
    defaults = SimpleNamespace(two_role_runtime=True, mode=mode)
    with pytest.raises(ValueError, match="requires --no-identity-check"):
        app.validate_two_role_options(defaults)

    explicit = SimpleNamespace(two_role_runtime=True, mode=mode, identity_check=False)
    app.validate_two_role_options(explicit)


CONFLICTING_ROUTES = (
    "strong_target_judge",
    "target_judge_overlap",
    "strong_destination_judge",
    "strong_recovery_executor",
    "physical_failure_executor",
)


@pytest.mark.parametrize("flag", CONFLICTING_ROUTES)
def test_conflicting_route_preflight_happens_before_env_or_simulator(monkeypatch, tmp_path, flag):
    touched = []

    def forbidden(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("preflight must run before environment or simulator setup")

    monkeypatch.setattr(app, "load_env", forbidden)
    monkeypatch.setattr(app, "OpenETAAdapter", forbidden)
    args = SimpleNamespace(
        two_role_runtime=True,
        identity_check=False,
        mode="dual_async",
        **{name: name == flag for name in CONFLICTING_ROUTES},
    )

    with pytest.raises(ValueError, match="forbids independent model routes"):
        asyncio.run(app.episode(args))
    assert touched == []
    assert not (tmp_path / "episode").exists()


class _OfflineAdapter:
    generation = 1
    cancellation = "test_boundary"
    schemas = {"observe": {}, "move_to": {}}

    def __init__(self, *args):
        self.host = None

    async def start(self):
        self.host = SimpleNamespace(execution_accounting={})

    async def observe(self, action_seq):
        return Observation("observation", 1, 1, time.monotonic(), 0, action_seq, {})

    async def verify(self, observation_id):
        return {"status": "not_verified"}

    async def close(self):
        pass


class _RecordingModel:
    constructed = []
    tool_roles = []

    def __init__(self, base_url, key, model, role, output, options=None):
        self.base_url, self.key, self.model, self.role = base_url, key, model, role
        self.options = options
        self.calls, self.usage, self.latencies = 0, {}, []
        self.accounting = {"responses_with_usage": 0}
        type(self).constructed.append(self)

    def bind_tools(self, schemas):
        type(self).tool_roles.append(self.role)

    def bind_memory(self, memory):
        pass

    def bind_mission_reference(self, observation):
        pass

    async def decide(self, prompt, projection, images):
        return {"kind": "delegate", "goal": "execute the authorized mission"}

    async def close(self):
        pass


class _OfflineFast:
    def __init__(self, model, context, coordinator, store, *args, **kwargs):
        self.steps = self.errors = self.false_claims = 0

    async def run(self, max_decisions, deadline):
        return "decision_budget"


class _OfflineSlow:
    def __init__(self, model, context, coordinator, store, **kwargs):
        self.model = model
        self.reviews = 0

    async def delegate(self, mission):
        return None

    def request_review(self, reason):
        pass

    async def run(self):
        await asyncio.Event().wait()

    async def watch_execution(self):
        await asyncio.Event().wait()


def _write_memory_manifest(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "reference.bin").write_bytes(b"offline reference image")
    manifest = folder / "verified-experience.json"
    manifest.write_text(json.dumps({"entries": [{
        "match_text": "cup",
        "image": "reference.bin",
        "provenance": {"official_success": True},
    }]}), encoding="utf-8")
    return manifest


@pytest.mark.parametrize(
    "mode,actor_model,actor_key,expected_roles",
    [
        ("single_fast", "FAST_MODEL", "FAST_API_KEY", ["fast"]),
        ("single_strong", "SLOW_MODEL", "SLOW_API_KEY", ["fast"]),
        ("dual_async", "FAST_MODEL", "FAST_API_KEY", ["fast", "slow"]),
        ("dual_sync", "FAST_MODEL", "FAST_API_KEY", ["fast", "slow"]),
    ],
)
def test_memory_does_not_add_grounding_client_slow_is_tool_free_and_actor_profile_is_preserved(
    monkeypatch, tmp_path, mode, actor_model, actor_key, expected_roles
):
    _RecordingModel.constructed = []
    _RecordingModel.tool_roles = []
    monkeypatch.setattr(app, "ModelClient", _RecordingModel)
    monkeypatch.setattr(app, "FastAgent", _OfflineFast)
    monkeypatch.setattr(app, "SlowAgent", _OfflineSlow)
    # Stub the optional simulator adapter module before app.episode imports it;
    # these unit tests do not need scipy, MuJoCo, or any simulator dependency.
    monkeypatch.setitem(
        sys.modules,
        "embodied_harness.adapters.libero_direct",
        SimpleNamespace(DirectLiberoAdapter=_OfflineAdapter),
    )
    monkeypatch.setattr(app, "load_env", lambda _: None)
    values = {
        "MODEL_BASE_URL": "https://fast.invalid/v1",
        "FAST_MODEL_BASE_URL": "https://fast.invalid/v1",
        "SLOW_MODEL_BASE_URL": "https://slow.invalid/v1",
        "FAST_API_KEY": "fast-key",
        "SLOW_API_KEY": "slow-key",
        "FAST_MODEL": "fast-profile-model",
        "SLOW_MODEL": "slow-profile-model",
        "FAST_MODEL_OPTIONS": json.dumps(runner.FAST_OPTIONS),
        "SLOW_MODEL_OPTIONS": json.dumps(runner.SLOW_OPTIONS),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    memory = _write_memory_manifest(tmp_path / "memory")
    args = SimpleNamespace(
        output=str(tmp_path / mode), model_env="unused", memory=str(memory), mission="move the cup",
        backend="direct_libero", sim_url="unused", env_id="unused", seed=0, timeout=5,
        max_decisions=8, mode=mode, hierarchy="legacy_review", review_every=3,
        identity_check=False, two_role_runtime=True,
    )
    result = asyncio.run(app.episode(args))

    assert result["error"] is None
    assert [model.role for model in _RecordingModel.constructed] == expected_roles
    assert _RecordingModel.tool_roles == ["fast"]
    actor = next(model for model in _RecordingModel.constructed if model.role == "fast")
    assert actor.model == values[actor_model]
    assert actor.key == values[actor_key]
    assert actor.options == ({"temperature": 1, "reasoning_effort": "low"}
                             if mode == "single_strong" else {"chat_template_kwargs": {"enable_thinking": False}})
    if mode.startswith("dual"):
        planner = next(model for model in _RecordingModel.constructed if model.role == "slow")
        assert planner.model == values["SLOW_MODEL"]
        assert planner.role not in _RecordingModel.tool_roles
    assert result["agent_topology"]["roles"] == list(expected_roles)


def test_legacy_runtime_keeps_default_identity_grounding_route(monkeypatch, tmp_path):
    _RecordingModel.constructed = []
    _RecordingModel.tool_roles = []
    monkeypatch.setattr(app, "ModelClient", _RecordingModel)
    monkeypatch.setattr(app, "FastAgent", _OfflineFast)
    monkeypatch.setitem(
        sys.modules,
        "embodied_harness.adapters.libero_direct",
        SimpleNamespace(DirectLiberoAdapter=_OfflineAdapter),
    )
    monkeypatch.setattr(app, "load_env", lambda _: None)
    for key, value in {
        "MODEL_BASE_URL": "https://fast.invalid/v1",
        "FAST_MODEL_BASE_URL": "https://fast.invalid/v1",
        "FAST_API_KEY": "fast-key",
        "FAST_MODEL": "fast-profile-model",
        "SLOW_MODEL": "slow-profile-model",
        "FAST_MODEL_OPTIONS": json.dumps(runner.FAST_OPTIONS),
        "SLOW_MODEL_OPTIONS": json.dumps(runner.SLOW_OPTIONS),
    }.items():
        monkeypatch.setenv(key, value)

    memory = _write_memory_manifest(tmp_path / "legacy-memory")
    # identity_check is deliberately omitted: the legacy default remains True.
    args = SimpleNamespace(
        output=str(tmp_path / "legacy"), model_env="unused", memory=str(memory), mission="move the cup",
        backend="direct_libero", sim_url="unused", env_id="unused", seed=0, timeout=5,
        max_decisions=8, mode="single_fast", hierarchy="legacy_review", review_every=3,
        two_role_runtime=False,
    )
    result = asyncio.run(app.episode(args))

    assert result["error"] is None
    assert [model.role for model in _RecordingModel.constructed] == ["fast", "grounding"]
    assert _RecordingModel.tool_roles == ["fast"]


def _two_role_runner_config() -> dict:
    flags = {
        mode: {
            "progress_recovery": False,
            "priority_foreground": False,
            "review_economy": False,
            "pipeline_supervision": False,
            "two_role_runtime": True,
        }
        for mode in ("single_fast", "single_strong", "dual_async", "dual_sync")
    }
    return {
        "schema_version": 1,
        "protocol_id": "round22-targeted-development",
        "status": "prepared_not_launched",
        "output_root": "/tmp/round23-two-role-test",
        "catalog": {"path": str(REPO / "configs" / "round21-task-catalog.json")},
        "variants": {
            "patched23": {
                "kind": "patched22",
                "repository_root": "/tmp/patched23",
                "expected_commit": "f" * 40,
                "modes": list(flags),
                "feature_flags_by_mode": flags,
            }
        },
        "models": {
            "fast": {"model": runner.FAST_MODEL, "options": runner.FAST_OPTIONS},
            "slow": {"profile_id": runner.SLOW_PROFILE_ID, "model": runner.SLOW_MODEL,
                     "options": runner.SLOW_OPTIONS},
        },
        "transport": {"host": runner.TRANSPORT_HOST, "address": runner.TRANSPORT_ADDRESS},
        "scheduler": {"max_concurrent_episodes": 2, "memory_reserve_gib": 1,
                      "memory_per_episode_gib": 1, "stop_sentinel": "STOP_NEW"},
        "no_automatic_replay": True,
        "retain_all_attempts": True,
        "cases": [
            {"suite": "libero_spatial", "task_index": 0, "seed": 36,
             "variant": "patched23", "mode": "dual_async", "attempt_id": "standard"},
            {"suite": "libero_10", "task_index": 0, "seed": 36,
             "variant": "patched23", "mode": "dual_async", "attempt_id": "long"},
        ],
    }


def _runner_jobs():
    config = _two_role_runner_config()
    runner.validate_config(config)
    state_hashes = {"36": "a" * 64}
    catalog = {"_by_key": {
        ("libero_spatial", 0): {
            "suite": "libero_spatial", "task_index": 0,
            "env_id": "openeta/libero_libero_spatial_task0-v0",
            "mission": "move the cup", "official_state_count": 50,
            "state_hashes": state_hashes,
        },
        ("libero_10", 0): {
            "suite": "libero_10", "task_index": 0,
            "env_id": "openeta/libero_libero_10_task0-v0",
            "mission": "move the cup", "official_state_count": 50,
            "state_hashes": state_hashes,
        },
    }}
    return config, runner.make_jobs(config, catalog, Path(config["output_root"]))


def test_runner_materializes_two_role_identity_and_preserves_decision_and_time_budgets():
    config, jobs = _runner_jobs()
    standard, long = jobs
    assert standard["identity_check"] is False
    assert long["identity_check"] is False
    assert (standard["budget"]["max_decisions"], standard["budget"]["timeout_s"]) == (24, 1200)
    assert (long["budget"]["max_decisions"], long["budget"]["timeout_s"]) == (36, 1800)
    assert standard["expected_active_models"] == {
        "fast": runner.FAST_MODEL, "slow": runner.SLOW_MODEL, "grounding": None,
        "target_judge": None, "destination_judge": None, "recovery_executor": None,
    }

    arguments = runner.app_arguments(standard, Path("/tmp/memory.json"))
    assert "--two-role-runtime" in arguments
    assert "--no-identity-check" in arguments
    assert arguments[arguments.index("--max-decisions") + 1] == "24"
    assert arguments[arguments.index("--timeout") + 1] == "1200"


def _valid_two_role_result(job: dict) -> dict:
    models = [{"role": role, "model": model}
              for role, model in job["expected_active_models"].items()
              if role in {"fast", "slow"} and model is not None]
    expected = job["expected_recorded_model_options"]
    allowed = ["fast", "slow"] if job["mode"].startswith("dual") else ["fast"]
    return {
        "backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "evidence": {"status": "not_verified"},
        "max_decisions": job["budget"]["max_decisions"], "timeout": job["budget"]["timeout_s"],
        "review_every": 3, "identity_check": False,
        "fast_model": expected["fast_model"], "fast_model_options": expected["fast_model_options"],
        "slow_model": expected["slow_model"], "slow_model_options": expected["slow_model_options"],
        "source_sha256": "b" * 64,
        "models": models,
        "agent_topology": {"name": "slow_fast_only", "roles": allowed,
                           "independent_checkers": False, "strong_executor_handoff": False,
                           "slow_execution_tools": False},
        **job["feature_flags"],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.append({"role": "unapproved", "model": "extra-model"}),
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows.pop(),
    ],
    ids=["extra-role", "duplicate-role", "incomplete-roles"],
)
def test_runner_rejects_extra_duplicate_and_incomplete_model_role_records(mutate):
    _, jobs = _runner_jobs()
    job = jobs[0]
    source = {"package_sha256": "b" * 64}
    result = _valid_two_role_result(job)
    mutate(result["models"])
    assert "invalid_two_role_topology" in runner.check_result(result, job, source)

