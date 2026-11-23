"""Offline tests for bounded, scope-bound recovery reasoning."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.model import InferencePreempted, ModelClient, ModelError
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})


class ScriptedModel:
    model = "Qwen3.6-35B-A3B"

    def __init__(self, decisions, *, preempt_calls=(), on_decide=None):
        self.decisions = deque(decisions)
        self.preempt_calls = set(preempt_calls)
        self.on_decide = on_decide
        self.calls = []
        self.history = deque([{"role": "assistant", "content": "prior claim"}])

    async def decide(self, prompt, projection, images=(), *, recovery_reasoning=False):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "recovery_reasoning": recovery_reasoning})
        if self.on_decide:
            self.on_decide(len(self.calls), prompt, projection)
        if len(self.calls) in self.preempt_calls:
            raise InferencePreempted("synthetic accepted-control cut-in")
        value = self.decisions.popleft()
        self.history.append({"role": "assistant", "content": value["kind"]})
        return value


class Adapter:
    generation = 1
    terminal = False

    def __init__(self):
        self.points = {}
        self.schemas = {
            "move_to": {"parameters": {"properties": {"xyz_m": {"type": "array"}}}},
            "observe": {"parameters": {"properties": {}}},
        }
        self.coordinator = None

    def validate_args(self, name, args):
        if name == "move_to":
            assert isinstance(args.get("xyz_m"), list)
        else:
            assert name == "observe" and args == {}

    async def inspect(self, name, args, _action_seq):
        assert name == "observe" and args == {}
        return await self.coordinator.observe()


class Coordinator:
    def __init__(self, context, statuses):
        self.context = context
        self.adapter = Adapter()
        self.adapter.coordinator = self
        self.paused = self.unknown = False
        self.lock = asyncio.Lock()
        self.statuses = deque(statuses)
        self.verify_calls = 0
        self.observe_count = 0
        self.submitted = []
        self.after_verify = None

    async def observe(self):
        self.observe_count += 1
        observation = Observation(f"obs-{self.observe_count}", self.observe_count, 1,
                                  time.monotonic(), 0.0, self.context.action_seq,
                                  {"frame": self.observe_count})
        self.context.ingest(observation)
        return observation

    async def verify(self):
        self.verify_calls += 1
        result = {"status": self.statuses.popleft(), "evidence": {"official": True}}
        if self.after_verify:
            self.after_verify(self.verify_calls, result)
        return result

    async def submit(self, request):
        self.submitted.append(request)
        self.context.remember("action.accepted", {"request": asdict(request), "receipt": {}})
        self.context.action_seq += 1
        await self.observe()
        self.context.remember("action.receipt", {
            "status": "completed", "detail": {"macro_status": "reached"},
        })
        return SimpleNamespace(status="completed", detail={"macro_status": "reached"})


class Slow:
    def __init__(self):
        self.review_requests = []

    def request_review(self, reason):
        self.review_requests.append(reason)


class FakePipeline:
    pipelined = True
    pending_reason = None

    def __init__(self, context, slow):
        self.context = context
        self.slow = slow
        self.checkpoint_task = None
        self.preparation_used = False
        self.checkpoint_calls = []
        self.local_reviews = []

    async def before_decision(self, _remaining):
        return None

    async def guard_proposal(self, kind, tool=None):
        if (self.checkpoint_task and not self.checkpoint_task.done()
                and not self.preparation_used and kind == "tool" and tool == "observe"):
            self.preparation_used = True
        return True, None

    def record_decision(self):
        pass

    async def checkpoint(self, trigger="fast_completion_proposal"):
        self.checkpoint_calls.append(trigger)
        return "checkpoint_pending"

    def note_readonly(self, _before, _after):
        pass

    def note_preparation_points(self, _prior_ids):
        pass

    def note_terminal(self, _evidence):
        pass

    def request_local_review(self, reason):
        self.local_reviews.append(reason)
        self.slow.request_review("local_recovery: " + str(reason))


async def make_agent(decisions, statuses, *, enabled=True, final_stage=True, preempt_calls=(), on_decide=None):
    store = MemoryStore()
    context = Context(store)
    context.milestone = Milestone(
        "final-stage" if final_stage else "intermediate-stage", 1,
        "move the object to the requested destination", "place the object",
        capabilities=("move_to",), final_stage=final_stage,
    )
    model = ScriptedModel(decisions, preempt_calls=preempt_calls, on_decide=on_decide)
    coordinator = Coordinator(context, statuses)
    await coordinator.observe()
    slow = Slow()
    pipeline = FakePipeline(context, slow)
    fast = FastAgent(model, context, coordinator, store, slow=slow, mode="dual_async",
                     async_completion_recovery=True, recovery_reasoning=enabled)
    fast.milestones = pipeline
    return store, context, model, coordinator, slow, pipeline, fast


def event_rows(store, event_type):
    return [event["payload"] for event in store.events if event["kind"] == event_type]


def test_default_disabled_keeps_legacy_model_signature_and_official_success_path():
    async def run():
        store = MemoryStore()
        context = Context(store)
        context.milestone = Milestone("final", 1, "mission", "mission", final_stage=True)
        coordinator = Coordinator(context, ["verified"])
        await coordinator.observe()

        class LegacyModel:
            model = "legacy"
            async def decide(self, _prompt, _projection, _images=()):
                return {"kind": "complete", "reason": "success"}

        fast = FastAgent(LegacyModel(), context, coordinator, store, mode="dual_async",
                         async_completion_recovery=True)
        fast.milestones = FakePipeline(context, Slow())
        assert await fast.run(1) == "success"
        assert fast.recovery_reasoning is False
        assert fast.recovery_reasoning_calls == 0
        assert event_rows(store, "fast.recovery_reasoning_requested") == []

    asyncio.run(run())


def test_enabled_option_does_not_stamp_scope_without_official_rejection():
    async def run():
        store, _context, model, _coordinator, _slow, _pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "finished"}], ["verified"], enabled=True)
        assert await fast.run(1) == "success"
        assert model.calls[0]["recovery_reasoning"] is False
        assert fast.recovery_reasoning_scope is None
        assert fast.recovery_reasoning_calls == 0
        assert event_rows(store, "fast.recovery_reasoning_requested") == []

    asyncio.run(run())


def test_episode_reasoning_cap_counts_preempted_attempts_and_preserves_decision_budget():
    async def run():
        decisions = [{"kind": "complete", "reason": f"false claim {index}"} for index in range(8)]
        store, _context, model, coordinator, _slow, pipeline, fast = await make_agent(
            decisions, ["not_verified"] * 8, enabled=True, preempt_calls={2})

        assert await fast.run(8) == "decision_budget"
        assert [call["recovery_reasoning"] for call in model.calls] == [
            False, True, True, True, True, False, False, False,
        ]
        assert fast.recovery_reasoning_calls == 4
        assert fast.steps == len(model.calls) == coordinator.verify_calls + 1
        assert len(event_rows(store, "fast.recovery_reasoning_requested")) == 4
        assert pipeline.checkpoint_calls == []
        assert fast.errors == 0

    asyncio.run(run())


def test_physical_revision_invalidates_scope_then_new_official_rejection_restamps_it():
    async def run():
        decisions = [
            {"kind": "complete", "reason": "claim"},
            {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0.1, 0.2, 0.3]}},
            {"kind": "complete", "reason": "claim after action"},
            {"kind": "tool", "tool": "observe", "args": {}},
        ]
        _store, context, model, coordinator, _slow, _pipeline, fast = await make_agent(
            decisions, ["not_verified"] * 3, enabled=True)

        assert await fast.run(4) == "decision_budget"
        assert [call["recovery_reasoning"] for call in model.calls] == [False, True, False, True]
        assert fast.recovery_reasoning_calls == 2
        assert context.action_seq == 1
        assert len(coordinator.submitted) == 1

    asyncio.run(run())


def test_scope_binds_control_contract_and_generation_while_contract_handoff_preserves_cap():
    async def run():
        _store, context, _model, _coordinator, _slow, _pipeline, fast = await make_agent(
            [], [], enabled=True)
        fast.sync_contract()
        await fast.recover_completion(None)  # The actor calls this only after an official rejection.
        expected = ("final-stage", 1, 1, 0, context.control_version)
        assert fast.recovery_reasoning_scope == expected
        assert fast.use_recovery_reasoning() is True

        context.milestone = Milestone("final-stage", 2, "mission", "revised", final_stage=True)
        assert fast.use_recovery_reasoning() is False
        context.milestone = Milestone("final-stage", 2, "mission", "revised", generation=2, final_stage=True)
        assert fast.use_recovery_reasoning() is False
        context.milestone = Milestone("final-stage", 1, "mission", "mission", final_stage=True)
        context.signal_control("advice", 50, "scope-change")
        assert fast.use_recovery_reasoning() is False
        context.control_changed.clear()
        context.action_seq += 1
        assert fast.use_recovery_reasoning() is False
        context.milestone = Milestone("next-final", 2, "mission", "next", final_stage=True)
        fast.recovery_reasoning_calls = 4
        fast.sync_contract()
        assert fast.recovery_reasoning_scope is None
        assert fast.recovery_reasoning_calls == 4
        await fast.recover_completion(None)
        assert fast.recovery_reasoning_scope == ("next-final", 2, 1, 1, context.control_version)
        assert fast.use_recovery_reasoning() is False  # The episode cap survives the handoff.

    asyncio.run(run())


def test_nonfinal_rejection_does_not_open_recovery_reasoning_scope():
    async def run():
        _store, _context, _model, _coordinator, _slow, _pipeline, fast = await make_agent(
            [], [], enabled=True, final_stage=False)
        await fast.recover_completion(None)
        assert fast.recovery_reasoning_scope is None
        assert fast.use_recovery_reasoning() is False

    asyncio.run(run())


def test_pending_scout_uses_normal_request_even_after_final_rejection():
    async def run():
        _store, _context, model, coordinator, _slow, pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "official check rejects"},
             {"kind": "tool", "tool": "observe", "args": {}, "reason": "inspect preview"}],
            ["not_verified"], enabled=True)
        pending = asyncio.Event()

        def start_checkpoint_after_rejection(_index, _result):
            pipeline.checkpoint_task = asyncio.create_task(pending.wait())

        coordinator.after_verify = start_checkpoint_after_rejection
        try:
            assert await fast.run(2) == "decision_budget"
            assert [call["recovery_reasoning"] for call in model.calls] == [False, False]
            assert "PREPARATION-ONLY SCOPE OVERRIDE" in model.calls[1]["prompt"]
            assert "final_completion_protocol" not in model.calls[1]["projection"]
            assert fast.recovery_reasoning_calls == 0
            assert event_rows(_store, "fast.recovery_reasoning_requested") == []
            assert pipeline.preparation_used is True
        finally:
            if pipeline.checkpoint_task and not pipeline.checkpoint_task.done():
                pipeline.checkpoint_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pipeline.checkpoint_task

    asyncio.run(run())


async def install_transport(model, transport):
    await model.client.aclose()
    model.client = httpx.AsyncClient(base_url="http://mock.invalid/", transport=transport,
                                     trust_env=False)


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, reasoning="private synthetic reasoning", content='{"kind":"wait"}'):
        self.requests = []
        self.reasoning = reasoning
        self.content = content

    async def handle_async_request(self, request):
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": self.content,
                                      "reasoning_content": self.reasoning},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    async def aclose(self):
        pass


def test_model_reasoning_override_is_request_local_audited_and_keeps_base_options(tmp_path):
    async def run():
        base_options = {"chat_template_kwargs": {"enable_thinking": False, "preserve": "yes"},
                        "temperature": 0.35, "max_tokens": 4096, "top_p": 0.9}
        original = copy.deepcopy(base_options)
        model = ModelClient("http://mock.invalid", "synthetic-test-key", "Qwen3.6-35B-A3B",
                            "fast", tmp_path, options=base_options)
        transport = RecordingTransport()
        await install_transport(model, transport)
        try:
            assert await model.decide("system", {}) == {"kind": "wait"}
            assert await model.decide("system", {}, recovery_reasoning=True) == {"kind": "wait"}

            ordinary, recovery = transport.requests
            assert ordinary["chat_template_kwargs"] == {"enable_thinking": False, "preserve": "yes"}
            assert recovery["chat_template_kwargs"] == {"enable_thinking": True, "preserve": "yes"}
            for request in (ordinary, recovery):
                assert request["temperature"] == 0.35
                assert request["max_tokens"] == 4096
                assert request["top_p"] == 0.9
            assert model.options == base_options == original

            traces = [json.loads(line) for line in model.trace.read_text().splitlines()]
            assert [row["recovery_reasoning"] for row in traces] == [False, True]
            assert traces[0]["effective_options"]["chat_template_kwargs"]["enable_thinking"] is False
            assert traces[1]["effective_options"]["chat_template_kwargs"]["enable_thinking"] is True
            assert all(row["reasoning_content_chars"] == len(transport.reasoning) for row in traces)

            request_events = [json.loads(line) for line in model.request_trace.read_text().splitlines()]
            started = [row for row in request_events if row["kind"] == "request.started"]
            finished = [row for row in request_events if row["kind"] == "request.finished"]
            assert [row["recovery_reasoning"] for row in started] == [False, True]
            assert [row["recovery_reasoning"] for row in finished] == [False, True]
            metadata = [row for row in request_events if row["kind"] == "response.reasoning_metadata"]
            assert len(metadata) == 1
            assert metadata[0]["recovery_reasoning"] is True
            assert metadata[0]["reasoning_content_chars"] == len(transport.reasoning)
            assert "private synthetic reasoning" not in model.trace.read_text()
            assert "private synthetic reasoning" not in model.request_trace.read_text()
        finally:
            await model.close()

    asyncio.run(run())


@pytest.mark.parametrize("role,name,template", [
    ("slow", "Qwen3.6-35B-A3B", {"enable_thinking": False}),
    ("fast", "other-model", {"enable_thinking": False}),
    ("fast", "Qwen3.6-35B-A3B", {"enable_thinking": True}),
    ("fast", "Qwen3.6-35B-A3B", {}),
])
def test_reasoning_override_rejects_unsupported_role_model_or_base_options(tmp_path, role, name, template):
    async def run():
        model = ModelClient("http://mock.invalid", "synthetic-test-key", name, role, tmp_path,
                            options={"chat_template_kwargs": copy.deepcopy(template)})
        try:
            with pytest.raises(ValueError, match="recovery reasoning"):
                await model.decide("system", {}, recovery_reasoning=True)
            assert model.accounting["logical_requests"] == 0
        finally:
            await model.close()

    asyncio.run(run())


def test_recovery_reasoning_parameter_requires_literal_boolean(tmp_path):
    async def run():
        model = ModelClient("http://mock.invalid", "synthetic-test-key", "Qwen3.6-35B-A3B", "fast",
                            tmp_path, options={"chat_template_kwargs": {"enable_thinking": False}})
        try:
            with pytest.raises(ValueError, match="must be a boolean"):
                await model.decide("system", {}, recovery_reasoning=1)
            assert model.accounting["logical_requests"] == 0
        finally:
            await model.close()

    asyncio.run(run())


def test_recovery_reasoning_does_not_relax_json_decision_schema(tmp_path):
    async def run():
        model = ModelClient("http://mock.invalid", "synthetic-test-key", "Qwen3.6-35B-A3B", "fast",
                            tmp_path, options={"chat_template_kwargs": {"enable_thinking": False}})
        transport = RecordingTransport(content='{"kind":"wait"}\n{"kind":"complete"}')
        await install_transport(model, transport)
        try:
            with pytest.raises(ModelError, match="ambiguous"):
                await model.decide("system", {}, recovery_reasoning=True)
            assert transport.requests[0]["chat_template_kwargs"]["enable_thinking"] is True
            assert model.accounting["logical_failed"] == 1
        finally:
            await model.close()

    asyncio.run(run())


def test_recovery_reasoning_keeps_logical_timeout_and_attempt_accounting(tmp_path):
    async def run():
        entered = asyncio.Event()

        class NeverRespond(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                entered.set()
                await asyncio.Future()

            async def aclose(self):
                pass

        model = ModelClient("http://mock.invalid", "synthetic-test-key", "Qwen3.6-35B-A3B", "fast",
                            tmp_path, timeout=120,
                            options={"chat_template_kwargs": {"enable_thinking": False}})
        await install_transport(model, NeverRespond())
        model.timeout_s = 0.03
        try:
            with pytest.raises(ModelError, match="logical inference deadline"):
                await model.decide("system", {}, recovery_reasoning=True)
            assert entered.is_set()
            assert model.accounting["logical_timeouts"] == 1
            assert model.accounting["logical_requests"] == 1
            assert model.accounting["cancelled_attempts"] == 1
            assert model.accounting["attempts_started"] == 1
            request_events = [json.loads(line) for line in model.request_trace.read_text().splitlines()]
            started = next(row for row in request_events if row["kind"] == "request.started")
            finished = next(row for row in request_events if row["kind"] == "request.finished")
            assert started["recovery_reasoning"] is True
            assert finished["recovery_reasoning"] is True
        finally:
            await model.close()

    asyncio.run(run())


def test_recovery_reasoning_request_remains_preemptible_by_control_cut_in(tmp_path):
    async def run():
        store = MemoryStore()
        context = Context(store)
        entered = asyncio.Event()

        class NeverRespond(httpx.AsyncBaseTransport):
            async def handle_async_request(self, _request):
                entered.set()
                await asyncio.Future()

            async def aclose(self):
                pass

        model = ModelClient("http://mock.invalid", "synthetic-test-key", "Qwen3.6-35B-A3B", "fast",
                            tmp_path, options={"chat_template_kwargs": {"enable_thinking": False}})
        await install_transport(model, NeverRespond())
        model.control_context = context
        snapshot = {"control_version": context.control_version}
        task = asyncio.create_task(model.decide("system", snapshot, recovery_reasoning=True))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            context.signal_control("pause", 100, "urgent-pause")
            with pytest.raises(InferencePreempted):
                await asyncio.wait_for(task, timeout=1)
            assert model.accounting["logical_preempted"] == 1
            assert model.accounting["cancelled_attempts"] == 1
            assert any(event["kind"] == "control.inference_preempted" for event in store.events)
            request_events = [json.loads(line) for line in model.request_trace.read_text().splitlines()]
            started = next(row for row in request_events if row["kind"] == "request.started")
            finished = next(row for row in request_events if row["kind"] == "request.finished")
            assert started["recovery_reasoning"] is True
            assert finished["recovery_reasoning"] is True
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await model.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode,pipeline,recovery", [
    ("single_fast", True, True),
    ("dual_async", False, True),
    ("dual_async", True, False),
])
def test_app_rejects_recovery_reasoning_without_required_route(mode, pipeline, recovery):
    from embodied_harness.app import episode

    args = SimpleNamespace(recovery_reasoning=True, mode=mode, pipeline_supervision=pipeline,
                           async_completion_recovery=recovery)
    with pytest.raises(ValueError, match="recovery_reasoning requires"):
        asyncio.run(episode(args))


def runner_fixture(repo):
    scripts = repo / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_round22_targeted as runner

    flags = copy.deepcopy(runner.R21_FLAGS["dual_async"])
    flags.update(async_completion_recovery=True, recovery_reasoning=True)
    config = {
        "schema_version": 1,
        "protocol_id": "round22-targeted-development",
        "status": "prepared_not_launched",
        "output_root": "/tmp/round22h-recovery-reasoning-tests",
        "catalog": {"path": str(repo / "configs" / "round21-task-catalog.json")},
        "variants": {"patched22": {"kind": "patched22", "repository_root": str(repo),
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


def test_runner_accepts_and_serializes_recovery_reasoning_feature_flag():
    repo = Path(__file__).resolve().parents[1]
    runner, config = runner_fixture(repo)
    runner.validate_config(config)
    catalog, _catalog_path, _catalog_hash = runner.load_catalog(
        config, repo / "configs" / "round21-benchmark.json")
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert job["feature_flags"]["recovery_reasoning"] is True
    assert "--recovery-reasoning" in runner.app_arguments(job, Path("/tmp/memory.json"))

    for mode, pipeline, completion_recovery in [
        ("single_fast", False, False), ("dual_async", False, False), ("dual_async", True, False),
    ]:
        invalid, invalid_config = runner_fixture(repo)
        invalid_config["variants"]["patched22"]["modes"] = [mode]
        row = invalid_config["variants"]["patched22"]["feature_flags_by_mode"].pop("dual_async")
        invalid_config["variants"]["patched22"]["feature_flags_by_mode"][mode] = {
            **row, "pipeline_supervision": pipeline,
            "async_completion_recovery": completion_recovery,
        }
        invalid_config["cases"][0]["mode"] = mode
        with pytest.raises(ValueError, match="recovery_reasoning requires"):
            invalid.validate_config(invalid_config)


@pytest.mark.parametrize("limit", [2048, 8192, True])
def test_recovery_reasoning_cannot_change_output_budget(tmp_path, limit):
    async def run():
        model = ModelClient("http://mock.invalid", "synthetic-test-key", "Qwen3.6-35B-A3B", "fast", tmp_path,
                            options={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": limit})
        try:
            with pytest.raises(ValueError, match="4096-token"):
                await model.decide("system", {}, recovery_reasoning=True)
            assert model.accounting["logical_requests"] == 0
            assert model.accounting["attempts_started"] == 0
        finally:
            await model.close()
    asyncio.run(run())
