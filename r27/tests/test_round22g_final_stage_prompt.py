"""Offline regressions for the opt-in final-stage completion prompt."""

from __future__ import annotations

import asyncio
import copy
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})


class ScriptedModel:
    def __init__(self, decisions, on_decide=None):
        self.decisions = deque(decisions)
        self.history = deque([{"role": "assistant", "content": "prior belief"}])
        self.calls = []
        self.on_decide = on_decide

    async def decide(self, prompt, projection, _images):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "history": copy.deepcopy(list(self.history))})
        if self.on_decide:
            self.on_decide(len(self.calls), prompt, projection)
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

    async def observe(self):
        self.observe_count += 1
        obs = Observation(f"obs-{self.observe_count}", self.observe_count, 1,
                          time.monotonic(), 0.0, self.context.action_seq,
                          {"frame": self.observe_count})
        self.context.ingest(obs)
        return obs

    async def verify(self):
        self.verify_calls += 1
        return {"status": self.statuses.popleft(), "evidence": {"official": True}}

    async def submit(self, request):
        self.submitted.append(request)
        self.context.remember("action.accepted", {"request": asdict(request), "receipt": {}})
        self.context.action_seq += 1
        await self.observe()
        self.context.remember("action.receipt", {"status": "completed", "detail": {"macro_status": "reached"}})
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
        self.recorded_decisions = 0
        self.guard_calls = []
        self.local_reviews = []
        self.handoff_on_decision = None

    async def before_decision(self, _remaining):
        if self.handoff_on_decision is not None and self.recorded_decisions >= self.handoff_on_decision[0]:
            self.context.milestone = self.handoff_on_decision[1]
            self.handoff_on_decision = None
        return None

    async def guard_proposal(self, kind, tool=None):
        self.guard_calls.append((kind, tool))
        if (self.checkpoint_task is not None and not self.checkpoint_task.done()
                and not self.preparation_used and kind == "tool" and tool == "observe"):
            self.preparation_used = True
        return True, None

    def note_readonly(self, _before, _after):
        pass

    def note_preparation_points(self, _prior_ids):
        pass

    def record_decision(self):
        self.recorded_decisions += 1

    async def checkpoint(self, trigger="fast_completion_proposal"):
        self.checkpoint_calls.append(trigger)
        return "checkpoint_pending"

    def note_terminal(self, evidence):
        pass

    def request_local_review(self, reason):
        self.local_reviews.append(reason)
        self.slow.request_review("local_recovery: " + str(reason))


async def make_agent(decisions, statuses, *, enabled=True, final_stage=True, on_decide=None):
    store = MemoryStore()
    context = Context(store)
    context.milestone = Milestone(
        "final-stage" if final_stage else "intermediate-stage", 1,
        "move the object to the requested destination", "place the object",
        capabilities=("move_to",), final_stage=final_stage,
    )
    model = ScriptedModel(decisions, on_decide=on_decide)
    coordinator = Coordinator(context, statuses)
    await coordinator.observe()
    slow = Slow()
    pipeline = FakePipeline(context, slow)
    fast = FastAgent(model, context, coordinator, store, slow=slow, mode="dual_async",
                     async_completion_recovery=True, final_stage_prompt=enabled)
    fast.milestones = pipeline
    return store, context, model, coordinator, slow, pipeline, fast


def events(store, name):
    return [row["payload"] for row in store.events if row["kind"] == name]


def test_final_prompt_projects_verifier_authority_and_rejection_resets_after_action():
    async def run():
        decisions = [
            {"kind": "complete", "reason": "looks complete"},
            {"kind": "complete", "reason": "repeat unchanged claim"},
            {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0.2, 0.1, 0.8]},
             "reason": "correct the placement"},
            {"kind": "complete", "reason": "check corrected outcome"},
        ]
        async def run_case(enabled):
            store, context, model, coordinator, slow, pipeline, fast = await make_agent(
                copy.deepcopy(decisions), ["not_verified"] * 4, enabled=enabled)
            status = await fast.run(4)
            runtime = {
                "status": status,
                "reviews": list(slow.review_requests),
                "checkpoints": list(pipeline.checkpoint_calls),
                "verify_calls": coordinator.verify_calls,
                "submitted": len(coordinator.submitted),
                "false_claims": fast.false_claims,
                "steps": fast.steps,
            }
            return runtime, store, context, model, coordinator, slow, pipeline, fast

        baseline = await run_case(False)
        enabled = await run_case(True)
        baseline_runtime = baseline[0]
        _runtime, store, context, model, coordinator, _slow, pipeline, fast = enabled

        assert _runtime == baseline_runtime
        assert _runtime["status"] == "decision_budget"
        assert len(model.calls) == 4
        assert "FINAL MISSION OUTCOME CONTRACT" in model.calls[0]["prompt"]
        assert "official environment verifier determines whether the episode succeeds" in model.calls[0]["prompt"]
        protocols = [call["projection"].get("final_completion_protocol") for call in model.calls]
        assert protocols[0]["authority"] == "official_environment_verifier"
        assert protocols[0]["rejected_at_current_physical_revision"] is False
        assert protocols[0]["claims_without_new_action"] == 0
        assert protocols[1]["rejected_at_current_physical_revision"] is True
        assert protocols[1]["action_seq"] == 0
        assert protocols[1]["claims_without_new_action"] == 1
        assert protocols[2]["rejected_at_current_physical_revision"] is True
        assert protocols[2]["claims_without_new_action"] == 2
        assert protocols[3]["action_seq"] == 1
        assert protocols[3]["rejected_at_current_physical_revision"] is False
        assert protocols[3]["claims_without_new_action"] == 0
        assert len(coordinator.submitted) == 1
        assert coordinator.verify_calls == 4
        assert pipeline.checkpoint_calls == []
        prompts = events(store, "fast.final_stage_prompt")
        assert [item["rejected_at_current_physical_revision"] for item in prompts] == [False, True, True, False]
        assert fast.false_claims == 3
        assert context.action_seq == 1

    asyncio.run(run())


@pytest.mark.parametrize("enabled,final_stage", [(False, True), (True, False)])
def test_disabled_or_intermediate_prompt_keeps_stage_handoff_contract(enabled, final_stage):
    async def run():
        store, _context, model, _coordinator, _slow, pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "propose current outcome"}], ["not_verified"],
            enabled=enabled, final_stage=final_stage)

        assert await fast.run(1) == "decision_budget"
        assert "ACTIVE MILESTONE CONTRACT" in model.calls[0]["prompt"]
        assert "FINAL MISSION OUTCOME CONTRACT" not in model.calls[0]["prompt"]
        assert "ONLY the current milestone" in model.calls[0]["prompt"]
        assert "final_completion_protocol" not in model.calls[0]["projection"]
        assert events(store, "fast.final_stage_prompt") == []
        assert pipeline.checkpoint_calls == ([] if final_stage else ["fast_completion_proposal"])

    asyncio.run(run())


def test_final_prompt_flag_does_not_change_rejection_review_or_checkpoint_behavior():
    async def run_case(enabled):
        store, _context, model, coordinator, slow, pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "claim one"},
             {"kind": "complete", "reason": "claim two"}], ["not_verified", "not_verified"], enabled=enabled)
        assert await fast.run(2) == "decision_budget"
        return {
            "reviews": slow.review_requests,
            "local_reviews": pipeline.local_reviews,
            "checkpoints": pipeline.checkpoint_calls,
            "verifications": coordinator.verify_calls,
            "false_claims": fast.false_claims,
            "steps": fast.steps,
            "refreshes": len(events(store, "recovery.observation_refreshed")),
            "waits": events(store, "completion.review_wait_started"),
            "calls": len(model.calls),
        }

    baseline, enabled = asyncio.run(run_case(False)), asyncio.run(run_case(True))
    assert enabled == baseline
    assert enabled["checkpoints"] == []
    assert enabled["waits"] == []
    assert enabled["false_claims"] == 2
    assert enabled["calls"] == enabled["steps"] == 2


def test_true_official_success_does_not_enter_final_recovery_or_checkpoint():
    async def run():
        store, _context, _model, coordinator, slow, pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "finished"}], ["verified"], enabled=True)
        assert await fast.run(1) == "success"
        assert coordinator.verify_calls == 1
        assert fast.false_claims == 0
        assert slow.review_requests == []
        assert pipeline.checkpoint_calls == []
        assert events(store, "pipeline.final_completion_rejected") == []

    asyncio.run(run())


def test_pending_checkpoint_scout_override_does_not_claim_final_protocol():
    async def run():
        store, _context, model, _coordinator, _slow, pipeline, fast = await make_agent(
            [{"kind": "tool", "tool": "observe", "args": {}, "reason": "inspect for the preview"}],
            [], enabled=True)
        pending = asyncio.Event()
        pipeline.checkpoint_task = asyncio.create_task(pending.wait())
        try:
            assert await fast.run(1) == "decision_budget"
            assert "PREPARATION-ONLY SCOPE OVERRIDE" in model.calls[0]["prompt"]
            assert "CHECKPOINT PREPARATION" in model.calls[0]["prompt"]
            assert "FINAL MISSION OUTCOME CONTRACT" not in model.calls[0]["prompt"]
            assert "final_completion_protocol" not in model.calls[0]["projection"]
            assert events(store, "fast.final_stage_prompt") == []
            assert len(events(store, "pipeline.preparation_inference_started")) == 1
            assert pipeline.checkpoint_calls == []
            assert pipeline.preparation_used is True
        finally:
            pipeline.checkpoint_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pipeline.checkpoint_task

    asyncio.run(run())


def test_pause_cut_in_still_preempts_a_final_prompt_decision():
    async def run():
        holder = {}

        def pause_during_inference(call_index, _prompt, _projection):
            if call_index == 1:
                holder["coordinator"].paused = True

        store, _context, model, coordinator, _slow, pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "stale claim"}], [], enabled=True,
            on_decide=pause_during_inference)
        holder["coordinator"] = coordinator

        assert await fast.run(2) == "paused"
        assert len(model.calls) == 1
        assert coordinator.verify_calls == 0
        assert pipeline.checkpoint_calls == []
        assert len(events(store, "fast.final_stage_prompt")) == 1
        assert len(events(store, "fast.decision_stale")) == 1

    asyncio.run(run())


def test_contract_handoff_switches_back_to_intermediate_prompt_and_resets_rejection():
    async def run():
        store, context, model, _coordinator, _slow, pipeline, fast = await make_agent(
            [{"kind": "complete", "reason": "final stage claim"},
             {"kind": "complete", "reason": "next stage claim"}],
            ["not_verified", "not_verified"], enabled=True)
        next_stage = Milestone("next-stage", 1, context.milestone.mission, "acquire object",
                               capabilities=("move_to",), final_stage=False)
        pipeline.handoff_on_decision = (1, next_stage)

        assert await fast.run(2) == "decision_budget"
        assert "FINAL MISSION OUTCOME CONTRACT" in model.calls[0]["prompt"]
        assert model.calls[1]["prompt"].count("ACTIVE MILESTONE CONTRACT") == 1
        assert "FINAL MISSION OUTCOME CONTRACT" not in model.calls[1]["prompt"]
        assert "final_completion_protocol" not in model.calls[1]["projection"]
        assert model.calls[1]["projection"]["milestone"]["id"] == "next-stage"
        assert events(store, "fast.final_stage_prompt")[0]["milestone_id"] == "final-stage"
        assert fast.rejected_completion_count == 0

    asyncio.run(run())


@pytest.mark.parametrize("mode,pipeline,recovery", [
    ("single_fast", True, True),
    ("dual_async", False, True),
    ("dual_async", True, False),
])
def test_app_rejects_final_prompt_without_required_dual_async_pipeline(mode, pipeline, recovery):
    from embodied_harness.app import episode

    args = SimpleNamespace(final_stage_prompt=True, mode=mode, pipeline_supervision=pipeline,
                           async_completion_recovery=recovery)

    with pytest.raises(ValueError, match="final_stage_prompt requires"):
        asyncio.run(episode(args))


def _runner_config(repo: Path):
    scripts = repo / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_round22_targeted as runner

    flags = copy.deepcopy(runner.R21_FLAGS["dual_async"])
    flags.update(async_completion_recovery=True, final_stage_prompt=True)
    config = {
        "schema_version": 1,
        "protocol_id": "round22-targeted-development",
        "status": "prepared_not_launched",
        "output_root": "/tmp/round22g-final-prompt-test",
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


def test_runner_validates_and_records_final_prompt_option_in_job_and_cli_args():
    repo = Path(__file__).resolve().parents[1]
    runner, config = _runner_config(repo)
    runner.validate_config(config)
    catalog, _catalog_path, _catalog_hash = runner.load_catalog(
        config, repo / "configs" / "round21-benchmark.json")
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert job["feature_flags"]["final_stage_prompt"] is True
    args = runner.app_arguments(job, Path("/tmp/memory.json"))
    assert "--final-stage-prompt" in args

    config["variants"]["patched22"]["feature_flags_by_mode"]["dual_async"]["async_completion_recovery"] = False
    with pytest.raises(ValueError, match="final_stage_prompt requires"):
        runner.validate_config(config)
