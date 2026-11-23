"""Offline regressions for role-specific model endpoints and scout policy."""

import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

from embodied_harness import app


def test_role_base_url_overrides_are_independent_and_fall_back(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "http://default.invalid/v1/")
    monkeypatch.delenv("FAST_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("SLOW_MODEL_BASE_URL", raising=False)
    assert app.role_base_url("fast") == "http://default.invalid/v1/"
    assert app.role_base_url("slow") == "http://default.invalid/v1/"

    monkeypatch.setenv("FAST_MODEL_BASE_URL", "http://fast.invalid/v1/")
    assert app.role_base_url("fast") == "http://fast.invalid/v1/"
    assert app.role_base_url("slow") == "http://default.invalid/v1/"

    monkeypatch.setenv("SLOW_MODEL_BASE_URL", "http://slow.invalid/v1/")
    assert app.role_base_url("fast") == "http://fast.invalid/v1/"
    assert app.role_base_url("slow") == "http://slow.invalid/v1/"


def _install_offline_app(monkeypatch, endpoints):
    from embodied_harness.runtime import memory

    constructed = []
    mission_start_observation = SimpleNamespace(
        id="mission-start", images=("mission-start-agentview.png",),
        state={"views": [{"camera_frame_id": "agentview"}]},
    )
    older_frame = SimpleNamespace(id="older-frame", images=("older.png",), state={"views": []})

    class Store:
        def __init__(self, _path):
            pass

        def since(self, _cursor):
            return []

        def close(self):
            pass

        def emit(self, *_args, **_kwargs):
            pass

    class Context:
        def __init__(self, _store):
            self.action_seq = 0
            # The frame buffer's first entry is deliberately stale. App wiring
            # must bind the exact current mission-start observation.
            self.frames = [older_frame, mission_start_observation]
            self.latest = mission_start_observation

    class Adapter:
        cancellation = "synthetic_boundary"

        def __init__(self, *_args):
            self.schemas = {"grasp_point": {}, "observe": {}}
            self.host = None

        async def start(self):
            self.host = SimpleNamespace(execution_accounting={})

        async def close(self):
            pass

    class Coordinator:
        def __init__(self, *_args):
            pass

        async def observe(self):
            pass

        def delegate(self, _milestone):
            pass

        def authorize_mission(self, *_args):
            pass

        async def verify(self):
            return {"status": "not_verified", "evidence": []}

    class Model:
        def __init__(self, base_url, key, model, role, output, options=None):
            self.base_url, self.key, self.model, self.role = base_url, key, model, role
            self.options = options or {}
            self.calls, self.usage, self.latencies = 0, {}, []
            self.accounting = {"logical_requests": 0, "responses_with_usage": 0}
            self.history = []
            self.mission_references = []
            constructed.append(self)

        def bind_tools(self, _schemas):
            pass

        def bind_memory(self, _memory):
            pass

        def bind_mission_reference(self, observation):
            assert observation is mission_start_observation
            self.mission_references.append(observation)

        async def close(self):
            pass

    class Slow:
        def __init__(self, *_args, **_kwargs):
            self.reviews = 0

        def request_review(self, *_args):
            pass

        async def run(self):
            await asyncio.Future()

        async def watch_execution(self):
            await asyncio.Future()

    class Fast:
        def __init__(self, *_args, **_kwargs):
            self.steps = self.false_claims = self.errors = 0
            self.identity_checker = None

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "EventStore", Store)
    monkeypatch.setattr(app, "Context", Context)
    monkeypatch.setattr(app, "Coordinator", Coordinator)
    monkeypatch.setattr(app, "ModelClient", Model)
    monkeypatch.setattr(app, "FastAgent", Fast)
    monkeypatch.setattr(app, "SlowAgent", Slow)
    monkeypatch.setattr(app, "load_env", lambda _path: None)
    monkeypatch.setattr(memory, "retrieve", lambda *_args: [{"image": "unused.png", "caption": "blue jar"}])
    # Avoid importing the simulator adapter (and its heavyweight runtime) in
    # this offline app-wiring test.
    monkeypatch.setitem(sys.modules, "embodied_harness.adapters.libero_direct",
                        SimpleNamespace(DirectLiberoAdapter=Adapter))

    environment = {
        "MODEL_BASE_URL": "http://default.invalid/v1/",
        "FAST_MODEL_BASE_URL": endpoints["fast"],
        "SLOW_MODEL_BASE_URL": endpoints["slow"],
        "FAST_API_KEY": "synthetic-fast-key",
        "SLOW_API_KEY": "synthetic-slow-key",
        "FAST_MODEL": "fast-test",
        "SLOW_MODEL": "slow-test",
        "FAST_MODEL_OPTIONS": '{"temperature":0.7}',
        "SLOW_MODEL_OPTIONS": '{"temperature":0.1}',
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return constructed


def _episode_args(tmp_path, mode):
    return SimpleNamespace(
        output=str(tmp_path / mode), model_env="unused.env", memory=None,
        mission="pick up the blue jar", backend="direct_libero", sim_url="unused",
        env_id="unused", seed=0, timeout=1.0, max_decisions=1,
        mode=mode, review_every=1, identity_check=True,
        hierarchy="legacy_review",
    )


def test_single_strong_actor_and_identity_checker_use_slow_endpoint_and_results_only_fingerprint_it(
        monkeypatch, tmp_path):
    endpoints = {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"}
    constructed = _install_offline_app(monkeypatch, endpoints)

    result = asyncio.run(app.episode(_episode_args(tmp_path, "single_strong")))

    assert [(m.role, m.base_url, m.model, m.key) for m in constructed] == [
        ("fast", endpoints["slow"], "slow-test", "synthetic-slow-key"),
        ("grounding", endpoints["slow"], "slow-test", "synthetic-slow-key"),
    ]
    expected_fingerprints = {
        role: hashlib.sha256(url.rstrip("/").encode()).hexdigest()
        for role, url in endpoints.items()
    }
    assert result["model_service_fingerprints"] == expected_fingerprints
    assert [model.role for model in constructed] == ["fast", "grounding"]
    assert all(model.mission_references == [constructed[0].mission_references[0]]
               for model in constructed)
    assert constructed[0].mission_references[0].id == "mission-start"

    persisted = (tmp_path / "single_strong" / "result.json").read_text()
    config = (tmp_path / "single_strong" / "config.json").read_text()
    for secret in (*endpoints.values(), "synthetic-fast-key", "synthetic-slow-key"):
        assert secret not in persisted
        assert secret not in config


def test_default_dual_mode_keeps_actor_grounding_fast_and_reviewer_slow(monkeypatch, tmp_path):
    endpoints = {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"}
    constructed = _install_offline_app(monkeypatch, endpoints)

    result = asyncio.run(app.episode(_episode_args(tmp_path, "dual_async")))

    assert [(m.role, m.base_url, m.model, m.key, m.options) for m in constructed] == [
        ("fast", endpoints["fast"], "fast-test", "synthetic-fast-key", {"temperature": 0.7}),
        ("slow", endpoints["slow"], "slow-test", "synthetic-slow-key", {"temperature": 0.1}),
        ("grounding", endpoints["fast"], "fast-test", "synthetic-fast-key",
         {"temperature": 0.7, "max_tokens": 1024}),
    ]
    assert result["error"] is None
    assert len(constructed) == 3
    assert all(model.mission_references == [constructed[0].mission_references[0]]
               for model in constructed)
