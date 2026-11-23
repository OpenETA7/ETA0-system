"""Offline tests for triggering the bounded executor after real failed actions."""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import ActionRequest, Milestone, Observation, Receipt, uid
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class Store:
    def __init__(self):
        self.rows = []

    def emit(self, kind, payload, event_id=None):
        self.rows.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.rows[cursor:]

    def of(self, kind):
        return [row["payload"] for row in self.rows if row["kind"] == kind]

    def index(self, kind):
        return next(i for i, row in enumerate(self.rows) if row["kind"] == kind)


class PhysicalAdapter:
    generation = 1
    terminal = False
    schemas = {
        "observe": {"parameters": {"properties": {"camera": {"type": "string"}}}},
        "move_to": {"parameters": {"properties": {"delta_m": {"type": "array"}}}},
    }

    def __init__(self, context, *, verifier="not_verified"):
        self.context = context
        self.version = 1
        self.sim_time = 1.0
        self.verifier = verifier
        self.executed = []
        self.inspected = []

    def validate_args(self, tool, args):
        if tool not in self.schemas or not isinstance(args, dict):
            raise ValueError("unknown tool or malformed args")
        if tool == "move_to" and args.get("bad"):
            raise ValueError("fixture preflight rejection")

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, self.generation,
            time.monotonic(), self.sim_time, action_seq,
            {"candidate_scope": "fixture", "views": [], "robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}}, ())

    async def inspect(self, tool, args, action_seq):
        self.inspected.append((tool, copy.deepcopy(args), action_seq))
        return await self.observe(action_seq)

    async def execute(self, tool, args, fence, is_cancelled, ingest):
        self.executed.append({"tool": tool, "args": copy.deepcopy(args), "fence": fence})
        self.sim_time += 1.0
        self.version += 1
        obs = Observation(f"obs-{self.version}", self.version, self.generation,
            time.monotonic(), self.sim_time, self.context.action_seq,
            {"candidate_scope": "fixture", "views": [], "robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}}, ())
        ingest(obs)
        return {
            "macro_status": "partial", "observation_id": obs.id,
            "feedback": {"motion": {"execution_state": "partial", "physics_executed": True,
                "reason_code": "servo_horizon", "motion_summary": {"position_error_m": 0.1}}},
        }

    async def verify(self, observation_id):
        return {"status": self.verifier, "source": "official_libero_check_success",
                "observation_id": observation_id}


class Pipeline:
    pipelined = True
    pending_reason = None
    checkpoint_task = None
    preparation_used = False

    def __init__(self, remaining=8):
        self.remaining = remaining
        self.local_reviews = []

    async def before_decision(self, remaining):
        self.remaining = remaining
        return None

    def record_decision(self):
        self.remaining -= 1

    async def guard_proposal(self, _kind, _tool=None):
        return True, None

    def note_readonly(self, *_args):
        pass

    def note_preparation_points(self, *_args):
        pass

    def note_terminal(self, _evidence):
        pass

    def request_local_review(self, reason):
        self.local_reviews.append(reason)


class Actor:
    model = "fast-fixture"
    role = "fast"

    def __init__(self, decisions):
        self.decisions = deque(copy.deepcopy(list(decisions)))
        self.history = deque(maxlen=6)
        self.calls = []

    async def decide(self, prompt, projection, images=(), **kwargs):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images), "kwargs": kwargs})
        return copy.deepcopy(self.decisions.popleft())


class Executor:
    model = "strong-fixture"
    role = "recovery_executor"

    def __init__(self, decisions):
        self.decisions = deque(copy.deepcopy(list(decisions)))
        self.history = deque(maxlen=6)
        self.accounting = {"logical_requests": 0}
        self.calls = []

    async def decide(self, prompt, projection, images=()):
        self.accounting["logical_requests"] += 1
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection)})
        return copy.deepcopy(self.decisions.popleft())


def build(*, actor_decisions=(), executor_decisions=(), max_decisions=8, remaining=8,
          progress=True, physical=True, strong=True, verifier="not_verified"):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("final", 1, "place the object in the bowl", "complete final placement",
        capabilities=("observe", "move_to"), generation=1, final_stage=True)
    context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 1.0, 0,
        {"candidate_scope": "fixture", "views": [], "robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}}, ()))
    adapter = PhysicalAdapter(context, verifier=verifier)
    coordinator = Coordinator(adapter, context, store)
    actor = Actor(actor_decisions)
    executor = Executor(executor_decisions)
    fast = FastAgent(actor, context, coordinator, store, mode="dual_async",
        async_completion_recovery=True, progress_recovery=progress,
        recovery_executor_model=executor if strong else None,
        physical_failure_executor=physical)
    fast.milestones = Pipeline(remaining)
    return store, context, adapter, coordinator, actor, executor, fast


def failure_decision(args=None):
    return {"kind": "tool", "tool": "move_to", "args": args or {"delta_m": [0.0, 0.0, 0.1]}}


def observe_decision(camera="agentview"):
    return {"kind": "tool", "tool": "observe", "args": {"camera": camera}}


_DEFAULT_FEEDBACK = object()


def _direct_failure(fast, context, *, key="servo-key", action_id=None, request_id=None,
                    before_time=None, after_time=None, status="completed", feedback=_DEFAULT_FEEDBACK,
                    evidence=None, detail=None, action_seq=None):
    old = context.latest
    next_seq = context.action_seq + 1 if action_seq is None else action_seq
    next_time = old.sim_time + 1.0 if after_time is None else after_time
    context.action_seq = next_seq
    new = Observation(f"manual-{old.version + 1}", old.version + 1, context.milestone.generation,
        time.monotonic(), next_time, next_seq, copy.deepcopy(old.state), ())
    context.ingest(new)
    request_id = request_id or uid()
    req = ActionRequest(request_id, context.milestone.id, context.milestone.revision,
        context.milestone.generation, old.id, "move_to", {"delta_m": [0.0, 0.0, 0.1]})
    receipt = Receipt(action_id or uid(), request_id, status, 1, new.id, detail or {})
    recovery_feedback = ({
        "failure_key": key, "tool": "move_to", "failure_kind": "servo_horizon",
        "stagnant_attempts": 1, "measured_progress": False,
    } if feedback is _DEFAULT_FEEDBACK else feedback)
    verification = evidence if evidence is not None else {
        "status": "not_verified", "source": "official_libero_check_success", "observation_id": new.id}
    fast.note_executor_physical_failure(req, receipt, recovery_feedback,
        old.sim_time if before_time is None else before_time, verification)
    return req, receipt, recovery_feedback, new, verification


def test_run_arms_after_second_real_failure_only_and_replaces_remaining_slots():
    decisions = [
        failure_decision(),
        observe_decision("wrist"),
        failure_decision({"bad": True}),  # rejected before physical dispatch
        failure_decision(),
    ]
    strong = [failure_decision(), observe_decision("agentview"), observe_decision("overhead")]
    store, context, adapter, _coordinator, actor, executor, fast = build(
        actor_decisions=decisions, executor_decisions=strong, max_decisions=7)
    assert asyncio.run(fast.run(7)) == "decision_budget"
    assert fast.steps == 7 and len(actor.calls) == 4 and len(executor.calls) == 3
    assert context.action_seq == 2 and len(adapter.executed) == 2
    assert len(adapter.inspected) == 3

    physical = store.of("executor_recovery.physical_failure_recorded")
    assert len(physical) == 2
    assert physical[0]["stagnant_attempts"] == 1
    assert len(physical[0]["physical_receipts"]) == 1
    assert physical[1]["stagnant_attempts"] == 2
    rows = physical[1]["physical_receipts"]
    assert len(rows) == 2
    assert len({row["action_id"] for row in rows}) == len({row["request_id"] for row in rows}) == 2
    assert len({row["action_seq"] for row in rows}) == 2
    assert rows[0]["after_sim_time"] > rows[0]["before_sim_time"]
    assert store.of("executor_recovery.armed")[-1]["trigger"] == "repeated_physical_failure"
    assert len(store.of("executor_recovery.armed")[-1]["physical_evidence"]["physical_receipts"]) == 2
    physical_indices = [i for i, row in enumerate(store.rows)
        if row["kind"] == "executor_recovery.physical_failure_recorded"]
    strong_indices = [i for i, row in enumerate(store.rows)
        if row["kind"] == "executor_recovery.request_started"]
    assert physical_indices[-1] < store.index("executor_recovery.armed") < strong_indices[0]
    # Strong's first same-key correction proposal is blocked by existing local
    # evidence; it consumes a decision slot but cannot create another receipt.
    assert len(store.of("recovery.equivalent_attempt_blocked")) == 1
    assert len(store.of("executor_recovery.request_started")) == len(executor.calls) == 3
    assert len(store.of("executor_recovery.physical_failure_recorded")) == 2
    assert fast.milestones.remaining == 0


def test_physical_receipt_trigger_respects_two_future_decision_minimum():
    store, context, adapter, _coordinator, actor, executor, fast = build(
        actor_decisions=[failure_decision(), failure_decision(), observe_decision()],
        executor_decisions=[observe_decision()], max_decisions=3)
    assert asyncio.run(fast.run(3)) == "decision_budget"
    assert len(adapter.executed) == 2
    assert len(store.of("executor_recovery.physical_failure_recorded")) == 2
    assert not fast.recovery_executor_armed
    assert not executor.calls
    assert store.of("executor_recovery.skipped")[-1]["reason"] == "fewer_than_two_future_decisions"
    assert len(actor.calls) == 3


def test_flag_off_keeps_same_physical_failure_history_without_executor_arm():
    store, context, adapter, _coordinator, actor, executor, fast = build(
        actor_decisions=[failure_decision(), failure_decision(), observe_decision()],
        executor_decisions=[observe_decision()], max_decisions=3, physical=False)
    assert asyncio.run(fast.run(3)) == "decision_budget"
    assert len(adapter.executed) == 2 and len(actor.calls) == 3
    assert not executor.calls
    assert not store.of("executor_recovery.physical_failure_recorded")
    assert not store.of("executor_recovery.armed")


def test_flag_off_preserves_original_official_negative_strong_executor_trigger():
    store, context, _adapter, _coordinator, _actor, _executor, fast = build(physical=False)
    evidence = {"status": "not_verified", "source": "official_libero_check_success",
                "observation_id": context.latest.id}
    fast.arm_recovery_executor(evidence)
    assert fast.recovery_executor_armed and fast.recovery_executor_remaining == 4
    assert "trigger" not in store.of("executor_recovery.armed")[0]
    assert "physical_evidence" not in store.of("executor_recovery.armed")[0]


def test_verified_physical_action_never_arms_failure_executor():
    store, _context, adapter, _coordinator, _actor, executor, fast = build(
        actor_decisions=[failure_decision()], executor_decisions=[observe_decision()],
        max_decisions=2, verifier="verified")
    assert asyncio.run(fast.run(2)) == "success"
    assert len(adapter.executed) == 1
    assert not store.of("executor_recovery.physical_failure_recorded")
    assert not store.of("executor_recovery.armed")
    assert not executor.calls


@pytest.mark.parametrize("invalid", [
    "paused", "running", "unknown", "pending", "checkpoint", "wrong_status", "wrong_source",
    "stale_observation", "stop_confirmed", "zero_clock_delta", "nan_clock", "infinite_clock", "host_error",
    "state_unknown", "depth_discontinuity", "nonstagnant", "wrong_tool", "action_mismatch",
])
def test_nonqualifying_second_receipt_cannot_arm(invalid):
    store, context, _adapter, coordinator, _actor, _executor, fast = build()
    _direct_failure(fast, context)
    receipt_status = "completed"
    feedback = {"failure_key": "servo-key", "tool": "move_to", "failure_kind": "servo_horizon",
        "stagnant_attempts": 2, "measured_progress": False}
    before, after = context.latest.sim_time, context.latest.sim_time + 1.0
    evidence = {"status": "not_verified", "source": "official_libero_check_success",
                "observation_id": None}
    detail = {}
    action_seq = None
    if invalid == "paused":
        coordinator.paused = True
    elif invalid == "running":
        coordinator.running = SimpleNamespace(action_id="active")
    elif invalid == "unknown":
        coordinator.unknown = True
    elif invalid == "pending":
        coordinator.contract_review_pending = True
    elif invalid == "checkpoint":
        fast.milestones.checkpoint_task = SimpleNamespace(done=lambda: False)
    elif invalid == "wrong_status":
        evidence["status"] = "verified"
    elif invalid == "wrong_source":
        evidence["source"] = "fast_claim"
    elif invalid == "stop_confirmed":
        receipt_status = "stop_confirmed"
    elif invalid == "zero_clock_delta":
        after = before
    elif invalid == "nan_clock":
        after = float("nan")
    elif invalid == "infinite_clock":
        after = float("inf")
    elif invalid == "host_error":
        detail["host_error"] = "preflight"
    elif invalid == "state_unknown":
        detail["state_unknown"] = True
    elif invalid == "depth_discontinuity":
        feedback["failure_kind"] = "depth_discontinuity"
    elif invalid == "nonstagnant":
        feedback["measured_progress"] = True
    elif invalid == "wrong_tool":
        feedback["tool"] = "grasp_point"
    elif invalid == "action_mismatch":
        action_seq = context.action_seq - 1
    elif invalid == "stale_observation":
        evidence["observation_id"] = "old-observation"
    _direct_failure(fast, context, before_time=before, after_time=after, status=receipt_status,
                    feedback=feedback, evidence=evidence, detail=detail, action_seq=action_seq)
    assert len(store.of("executor_recovery.physical_failure_recorded")) == 1
    assert not store.of("executor_recovery.armed")
    assert not fast.recovery_executor_armed


def test_duplicate_receipt_ids_do_not_count_as_a_second_failed_action():
    store, context, _adapter, _coordinator, _actor, _executor, fast = build()
    first = _direct_failure(fast, context)
    request, receipt = first[0], first[1]
    duplicate = _direct_failure(fast, context, action_id=receipt.action_id,
        request_id=request.request_id, feedback={"failure_key": "servo-key", "tool": "move_to",
            "failure_kind": "servo_horizon", "stagnant_attempts": 2, "measured_progress": False})
    assert len(store.of("executor_recovery.physical_failure_recorded")) == 1
    assert not store.of("executor_recovery.armed")
    _direct_failure(fast, context, feedback={"failure_key": "servo-key", "tool": "move_to",
        "failure_kind": "servo_horizon", "stagnant_attempts": 3, "measured_progress": False})
    assert store.of("executor_recovery.armed")
    assert len(store.of("executor_recovery.armed")[-1]["physical_evidence"]["physical_receipts"]) == 2


@pytest.mark.parametrize("changed", ["id", "revision", "generation", "mission"])
def test_new_contract_clears_staged_physical_failures_and_cannot_rearm(changed):
    store, context, _adapter, _coordinator, _actor, _executor, fast = build()
    _direct_failure(fast, context)
    previous = context.milestone
    updates = {"id": previous.id, "revision": previous.revision,
               "generation": previous.generation, "mission": previous.mission}
    if changed == "id":
        updates.update(id="new-final-contract", revision=1)
    elif changed == "revision":
        updates["revision"] += 1
    elif changed == "generation":
        updates["generation"] += 1
    else:
        updates["mission"] = "changed mission"
    context.milestone = replace(previous, **updates)
    _direct_failure(fast, context, key="servo-key", feedback={"failure_key": "servo-key",
        "tool": "move_to", "failure_kind": "servo_horizon", "stagnant_attempts": 2,
        "measured_progress": False})
    assert not store.of("executor_recovery.armed")
    assert len(fast.executor_failure_receipts["servo-key"]) == 1
    _direct_failure(fast, context, key="servo-key", feedback={"failure_key": "servo-key",
        "tool": "move_to", "failure_kind": "servo_horizon", "stagnant_attempts": 3,
        "measured_progress": False})
    assert fast.recovery_executor_armed
    count = len(store.of("executor_recovery.armed"))
    context.milestone = replace(context.milestone, revision=2)
    _direct_failure(fast, context, key="servo-key", feedback={"failure_key": "servo-key",
        "tool": "move_to", "failure_kind": "servo_horizon", "stagnant_attempts": 4,
        "measured_progress": False})
    assert len(store.of("executor_recovery.armed")) == count
    assert fast.recovery_executor_remaining == 4


@pytest.mark.parametrize("bad_feedback", [None, {}, {"failure_key": "", "tool": "move_to",
    "failure_kind": "servo_horizon", "stagnant_attempts": 2, "measured_progress": False},
    {"failure_key": "key", "tool": "move_to", "failure_kind": "unresolved_held_object",
     "stagnant_attempts": 2, "measured_progress": False},
])
def test_malformed_or_nonphysical_failure_feedback_is_not_staged(bad_feedback):
    store, context, _adapter, _coordinator, _actor, _executor, fast = build()
    _direct_failure(fast, context, feedback=bad_feedback)
    assert not store.of("executor_recovery.physical_failure_recorded")
    assert not store.of("executor_recovery.armed")


def test_app_cli_and_runner_require_strong_route_plus_progress_and_record_policy(tmp_path, monkeypatch):
    from embodied_harness import app

    parsed = []

    async def fake_episode(args):
        parsed.append((args.strong_recovery_executor, args.physical_failure_executor))

    monkeypatch.setattr(app, "episode", fake_episode)
    monkeypatch.setattr(sys, "argv", ["embodied-eval", "--mission", "fixture", "--output", "/tmp/offline"])
    app.main()
    monkeypatch.setattr(sys, "argv", ["embodied-eval", "--mission", "fixture", "--output", "/tmp/offline",
        "--strong-recovery-executor", "--physical-failure-executor"])
    app.main()
    assert parsed == [(False, False), (True, True)]

    import copy as _copy
    from test_round22_runner import fixture_config, load_fixture_catalog, valid_result_for
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    flags = _copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True, strong_recovery_executor=True,
        physical_failure_executor=True, progress_recovery=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = next(row for row in runner.make_jobs(config, catalog, Path(config["output_root"]))
        if row["variant"] == "patched22" and row["mode"] == "dual_async")
    arguments = runner.app_arguments(job, tmp_path / "memory.json")
    assert "--physical-failure-executor" in arguments
    assert job["expected_active_models"]["recovery_executor"] == job["slow_profile"]["model"]
    result = valid_result_for(job)
    result["models"].append({"role": "recovery_executor", "model": job["slow_profile"]["model"],
        "request_accounting": {"logical_requests": 1}})
    result["recovery_executor_policy"] = {
        "model": job["slow_profile"]["model"], "options": job["slow_profile"]["options"],
        "role": "recovery_executor", "max_requests": 4,
        "trigger": "first_official_negative_final_claim_or_two_same_key_physical_failures",
        "scheduling": "replace_actor_calls_within_existing_decision_budget",
        "scope": "current_final_contract_id_revision_generation_mission",
        "history": "copy_primary_executed_history_and_commit_only_returned_decisions",
        "cost": "separate_strong_executor_accounting_no_planner_tool_access"}
    assert runner.check_result(result, job, {"package_sha256": "b" * 64}) == []


@pytest.mark.parametrize("bad", ["no_strong", "no_progress"])
def test_app_rejects_incomplete_physical_executor_flags_before_environment_load(tmp_path, monkeypatch, bad):
    from embodied_harness import app

    args = SimpleNamespace(mode="dual_async", backend="direct_libero", pipeline_supervision=True,
        async_completion_recovery=True, strong_recovery_executor=(bad != "no_strong"),
        progress_recovery=(bad != "no_progress"), physical_failure_executor=True,
        output=str(tmp_path / bad))
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("environment must not load"))
    with pytest.raises(ValueError, match="physical_failure_executor requires"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("conflict", ["completion_schema_guard", "completion_contract_repair",
    "urgent_completion_review", "completion_rendezvous", "recovery_reasoning", "final_stage_prompt"])
def test_app_keeps_physical_executor_separate_from_other_completion_ablations(tmp_path, monkeypatch, conflict):
    from embodied_harness import app

    args = SimpleNamespace(mode="dual_async", backend="direct_libero", pipeline_supervision=True,
        async_completion_recovery=True, strong_recovery_executor=True, progress_recovery=True,
        physical_failure_executor=True, output=str(tmp_path / conflict), **{conflict: True})
    monkeypatch.setattr(app, "load_env", lambda _path: pytest.fail("environment must not load"))
    with pytest.raises(ValueError, match="strong_recovery_executor must be tested separately"):
        asyncio.run(app.episode(args))
    assert not Path(args.output).exists()


@pytest.mark.parametrize("bad", ["no_strong", "no_progress"])
def test_runner_rejects_incomplete_physical_executor_feature_combinations(bad):
    from test_round22_runner import fixture_config
    import copy as _copy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    flags = _copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True,
        strong_recovery_executor=(bad != "no_strong"),
        progress_recovery=(bad != "no_progress"), physical_failure_executor=True)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    with pytest.raises(ValueError, match="physical_failure_executor requires"):
        runner.validate_config(config)


@pytest.mark.parametrize("conflict", ["completion_schema_guard", "completion_contract_repair",
    "urgent_completion_review", "completion_rendezvous", "recovery_reasoning", "final_stage_prompt"])
def test_runner_keeps_physical_executor_separate_from_other_completion_ablations(conflict):
    from test_round22_runner import fixture_config
    import copy as _copy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner

    config = fixture_config()
    flags = _copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"].update(async_completion_recovery=True, strong_recovery_executor=True,
        physical_failure_executor=True, progress_recovery=True, **{conflict: True})
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    with pytest.raises(ValueError, match="strong_recovery_executor must be a separate completion ablation"):
        runner.validate_config(config)
