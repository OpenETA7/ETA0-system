"""Offline accounting regressions for model requests and single-strong routing."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from embodied_harness.agents import model as model_module
from embodied_harness.agents.model import ModelClient, ModelError


def completion(content='{"kind":"noop"}', usage=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


async def use_mock_transport(model, handler):
    """Replace the real AsyncClient before any request can be sent."""
    await model.client.aclose()
    model.client = httpx.AsyncClient(
        base_url="http://mock.invalid/",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer synthetic-test-key"},
        trust_env=False,
    )


def make_model(tmp_path):
    return ModelClient("http://mock.invalid", "synthetic-test-key", "test-model", "fast", tmp_path)


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_successful_response_accounts_logical_request_and_provider_usage(tmp_path):
    async def run():
        async def handler(request):
            assert request.url.path == "/chat/completions"
            assert request.headers["authorization"] == "Bearer synthetic-test-key"
            return completion(usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})

        model = make_model(tmp_path)
        await use_mock_transport(model, handler)
        try:
            assert await model.decide("system", {"observation": {"id": "obs-1", "state": {}}}) == {"kind": "noop"}
        finally:
            await model.close()

        accounting = model.accounting
        assert accounting["logical_requests"] == accounting["logical_completed"] == 1
        assert accounting["logical_failed"] == accounting["logical_cancelled"] == 0
        assert accounting["attempts_started"] == accounting["responses"] == 1
        assert accounting["retries"] == accounting["transport_errors"] == accounting["http_errors"] == 0
        assert accounting["responses_with_usage"] == 1
        assert accounting["responses_without_usage"] == 0
        assert len(accounting["attempt_latencies_s"]) == len(accounting["logical_latencies_s"]) == 1
        assert model.usage == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
        assert model.calls == 1
        events = read_events(tmp_path / "fast-requests.jsonl")
        assert [event["kind"] for event in events] == [
            "request.started", "attempt.started", "attempt.finished", "request.finished"]
        assert events[-1]["outcome"] == "completed"
        assert all("synthetic-test-key" not in json.dumps(event) for event in events)

    asyncio.run(run())


def test_timeout_then_success_is_one_logical_request_and_two_attempts(monkeypatch, tmp_path):
    async def run():
        async def no_sleep(_):
            return None

        monkeypatch.setattr(model_module.asyncio, "sleep", no_sleep)
        results = iter([
            httpx.ReadTimeout("synthetic timeout"),
            completion(usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}),
        ])

        async def handler(request):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        model = make_model(tmp_path)
        await use_mock_transport(model, handler)
        try:
            assert await model.decide("system", {}) == {"kind": "noop"}
        finally:
            await model.close()

        accounting = model.accounting
        assert accounting["logical_requests"] == accounting["logical_completed"] == 1
        assert accounting["attempts_started"] == 2
        assert accounting["retries"] == accounting["transport_errors"] == 1
        assert accounting["responses"] == accounting["responses_with_usage"] == 1
        assert accounting["responses_without_usage"] == accounting["http_errors"] == 0
        assert len(accounting["attempt_latencies_s"]) == 2
        events = read_events(tmp_path / "fast-requests.jsonl")
        attempts = [event for event in events if event["kind"] == "attempt.finished"]
        assert [event["outcome"] for event in attempts] == ["transport_error", "response"]
        assert attempts[0]["attempt"] == 1 and attempts[1]["attempt"] == 2
        assert events[-1]["outcome"] == "completed"

    asyncio.run(run())


def test_cancellation_propagates_and_is_recorded_at_attempt_and_logical_levels(tmp_path):
    async def run():
        entered = asyncio.Event()

        class BlockingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                entered.set()
                await asyncio.Future()

            async def aclose(self):
                pass

        model = make_model(tmp_path)
        await model.client.aclose()
        model.client = httpx.AsyncClient(
            base_url="http://mock.invalid/", transport=BlockingTransport(),
            headers={"Authorization": "Bearer synthetic-test-key"}, trust_env=False,
        )
        task = asyncio.create_task(model.decide("system", {}))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await model.close()

        accounting = model.accounting
        assert accounting["logical_requests"] == accounting["logical_cancelled"] == 1
        assert accounting["logical_completed"] == accounting["logical_failed"] == 0
        assert accounting["attempts_started"] == accounting["cancelled_attempts"] == 1
        assert accounting["responses"] == accounting["transport_errors"] == 0
        assert len(accounting["attempt_latencies_s"]) == len(accounting["logical_latencies_s"]) == 1
        events = read_events(tmp_path / "fast-requests.jsonl")
        assert [event["kind"] for event in events] == [
            "request.started", "attempt.started", "attempt.finished", "request.finished"]
        assert events[-2]["outcome"] == events[-1]["outcome"] == "cancelled"

    asyncio.run(run())


def test_logical_timeout_bounds_inflight_attempt_and_is_distinct_from_external_cancellation(tmp_path):
    async def run():
        entered = asyncio.Event()

        class BlockingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                entered.set()
                await asyncio.Future()

            async def aclose(self):
                pass

        model = ModelClient("http://mock.invalid", "synthetic-test-key", "test-model", "fast",
                            tmp_path, timeout=0.01)
        await model.client.aclose()
        model.client = httpx.AsyncClient(
            base_url="http://mock.invalid/", transport=BlockingTransport(),
            headers={"Authorization": "Bearer synthetic-test-key"}, trust_env=False,
        )
        try:
            with pytest.raises(ModelError, match="logical inference deadline"):
                await model.decide("system", {})
        finally:
            await model.close()

        assert entered.is_set()
        accounting = model.accounting
        assert accounting["logical_requests"] == accounting["logical_failed"] == 1
        assert accounting["logical_timeouts"] == 1
        assert accounting["logical_cancelled"] == 0
        assert accounting["attempts_started"] == accounting["cancelled_attempts"] == 1
        assert accounting["responses"] == 0
        events = read_events(tmp_path / "fast-requests.jsonl")
        assert events[-1]["outcome"] == "failed"
        assert events[-1]["failure_type"] == "logical_timeout"
        assert events[-2]["outcome"] == "cancelled"

    asyncio.run(run())


def test_success_without_usage_is_distinguished_from_an_unobserved_call(tmp_path):
    async def run():
        async def handler(_request):
            return completion()

        model = make_model(tmp_path)
        await use_mock_transport(model, handler)
        try:
            assert await model.decide("system", {}) == {"kind": "noop"}
        finally:
            await model.close()

        accounting = model.accounting
        assert accounting["logical_completed"] == 1
        assert accounting["responses"] == 1
        assert accounting["responses_with_usage"] == 0
        assert accounting["responses_without_usage"] == 1
        assert model.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    asyncio.run(run())


def test_http_error_response_increments_http_and_logical_failure_counters(tmp_path):
    async def run():
        async def handler(_request):
            return httpx.Response(400, text="bad request included synthetic-test-key")

        model = make_model(tmp_path)
        await use_mock_transport(model, handler)
        try:
            with pytest.raises(ModelError) as error:
                await model.decide("system", {})
        finally:
            await model.close()

        assert "synthetic-test-key" not in str(error.value)
        accounting = model.accounting
        assert accounting["logical_failed"] == 1
        assert accounting["responses"] == accounting["http_errors"] == 1
        assert accounting["responses_with_usage"] == accounting["responses_without_usage"] == 0
        events = read_events(tmp_path / "fast-requests.jsonl")
        assert events[-2]["outcome"] == "response"
        assert events[-2]["status_code"] == 400
        assert events[-1]["failure_type"] == "ModelError"

    asyncio.run(run())


def test_malformed_model_json_fails_logical_request_after_accounting_http_response(tmp_path):
    async def run():
        async def handler(_request):
            return completion(content="not json", usage={"prompt_tokens": 1, "completion_tokens": 1,
                                                          "total_tokens": 2})

        model = make_model(tmp_path)
        await use_mock_transport(model, handler)
        try:
            with pytest.raises(ModelError, match="one JSON object"):
                await model.decide("system", {})
        finally:
            await model.close()

        accounting = model.accounting
        assert accounting["logical_requests"] == accounting["logical_failed"] == 1
        assert accounting["logical_completed"] == accounting["logical_cancelled"] == 0
        assert accounting["attempts_started"] == accounting["responses"] == 1
        assert accounting["responses_with_usage"] == 1
        assert accounting["http_errors"] == accounting["transport_errors"] == 0
        assert len(accounting["logical_latencies_s"]) == 1

    asyncio.run(run())


def test_single_strong_wires_both_actor_and_identity_judge_to_strong_model(monkeypatch, tmp_path):
    """Exercise baseline role routing with a fake simulator and zero model traffic."""
    from embodied_harness import app
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
    monkeypatch.setattr(app, "load_env", lambda _path: None)
    monkeypatch.setattr(memory, "retrieve", lambda *_args: [{"image": "unused.png", "caption": "blue jar"}])
    import embodied_harness.adapters.libero_direct as direct
    monkeypatch.setattr(direct, "DirectLiberoAdapter", Adapter)
    for key, value in {
        "MODEL_BASE_URL": "http://unused.invalid",
        "FAST_API_KEY": "fast-secret-placeholder",
        "SLOW_API_KEY": "slow-secret-placeholder",
        "FAST_MODEL": "fast-test",
        "SLOW_MODEL": "slow-test",
        "FAST_MODEL_OPTIONS": '{"temperature":0.7}',
        "SLOW_MODEL_OPTIONS": '{"temperature":0.1}',
    }.items():
        monkeypatch.setenv(key, value)

    args = SimpleNamespace(
        output=str(tmp_path / "single-strong"), model_env="unused.env", memory=None,
        mission="pick up the blue jar", backend="direct_libero", sim_url="unused",
        env_id="unused", seed=0, timeout=1.0, max_decisions=1,
        mode="single_strong", review_every=1, identity_check=True,
    )
    result = asyncio.run(app.episode(args))

    assert [(model.role, model.model, model.key, model.options) for model in constructed] == [
        ("fast", "slow-test", "slow-secret-placeholder", {"temperature": 0.1}),
        ("grounding", "slow-test", "slow-secret-placeholder", {"temperature": 0.1, "max_tokens": 1024}),
    ]
    assert all(model.mission_references == [mission_start_observation] for model in constructed)
    assert [entry["role"] for entry in result["models"]] == ["fast", "grounding"]
    assert all(entry["request_accounting"]["logical_requests"] == 0 for entry in result["models"])

