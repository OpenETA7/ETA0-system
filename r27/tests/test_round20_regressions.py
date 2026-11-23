"""Offline Round20 regressions for scheduler priority, audit economy, and recovery."""
import asyncio
import copy
import time
from collections import deque
from types import SimpleNamespace

import pytest

from embodied_harness.agents.slow import SlowAgent
from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.milestones import MilestoneManager
from embodied_harness.runtime.recovery import ProgressRecovery


MISSION = "put the red cup inside the bowl"


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


class FakeAdapter:
    generation = 1

    def __init__(self):
        self.version = 0
        self.schemas = {
            "move_to": {"parameters": {"properties": {"xyz_m": {"type": "array"}}}},
            "grasp_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
            "place_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
        }

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, 1, time.monotonic(),
                           float(self.version), action_seq, {})

    def validate_args(self, _capability, _args):
        return None

    async def execute(self, _capability, _args, _fence, _cancelled, _publish):
        return {}

    async def verify(self, observation_id):
        return {"status": "not_verified", "source": "fake", "observation_id": observation_id}


class QueuedModel:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    async def decide(self, prompt, projection, images):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images)})
        value = self.responses.popleft()
        if callable(value):
            value = value(prompt, projection, images)
            if asyncio.iscoroutine(value):
                value = await value
        return copy.deepcopy(value)


def two_stage_plan():
    return {
        "kind": "plan",
        "fundamental_constraints": [],
        "milestones": [
            {"goal": "lift the red cup clear of the table", "constraints": [], "preconditions": [],
             "success_conditions": [{"predicate": "visual", "description": "the red cup is lifted clear of the table"}]},
            {"goal": "place the red cup inside the bowl", "constraints": [], "preconditions": [],
             "success_conditions": [{"predicate": "visual", "description": "the red cup is inside the bowl"}]},
        ],
    }


def unknown_advance():
    return {"kind": "checkpoint", "decision": "advance", "reason": "The outcome is uncertain.",
            "assessments": [{"condition_id": "success-0", "status": "unknown",
                             "evidence": "The current image does not establish the required relation."}],
            "invariants": []}


def supported_advance():
    return {"kind": "checkpoint", "decision": "advance", "reason": "The visual outcome is supported.",
            "assessments": [{"condition_id": "success-0", "status": "satisfied",
                             "evidence": "The current image shows the red cup lifted above the table."}],
            "invariants": []}


async def make_manager(responses, *, review_economy=False):
    store = MemoryStore()
    context = Context(store)
    adapter = FakeAdapter()
    coordinator = Coordinator(adapter, context, store)
    coordinator.authorize_mission(MISSION, capabilities=("move_to", "grasp_point", "place_point"))
    await coordinator.observe()
    model = QueuedModel(responses)
    slow = SlowAgent(model, context, coordinator, store, review_economy=review_economy)
    manager = MilestoneManager(slow, context, coordinator, store, max_decisions=12,
                               deadline=time.monotonic() + 10)
    return store, context, adapter, coordinator, slow, manager


def test_foreground_slow_inference_preempts_only_review_and_reviewer_retries():
    async def run():
        from embodied_harness.agents.prompts import SLOW_REVIEW

        class BlockingReviewModel:
            def __init__(self):
                self.background_calls = 0
                self.review_entered = asyncio.Event()
                self.review_cancelled = asyncio.Event()
                self.review_retried = asyncio.Event()

            async def decide(self, prompt, _projection, _images):
                if prompt != SLOW_REVIEW:
                    return {"kind": "foreground_answer"}
                self.background_calls += 1
                if self.background_calls == 1:
                    self.review_entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.review_cancelled.set()
                        raise
                self.review_retried.set()
                return {"kind": "noop"}

        store = MemoryStore()
        context = Context(store)
        context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 1.0, 0, {}))
        model = BlockingReviewModel()
        slow = SlowAgent(model, context, SimpleNamespace(running=None), store,
                         priority_foreground=True)
        reviewer = asyncio.create_task(slow.run())
        slow.request_review("initial_background_review")
        await asyncio.wait_for(model.review_entered.wait(), timeout=1)

        async def foreground_request():
            async with slow.foreground("checkpoint"):
                return await model.decide("foreground", {}, ())

        answer = await asyncio.wait_for(foreground_request(), timeout=1)
        assert answer == {"kind": "foreground_answer"}
        assert model.review_cancelled.is_set()
        await asyncio.wait_for(model.review_retried.wait(), timeout=1)
        assert model.background_calls == 2
        assert not reviewer.done()
        assert any(event["kind"] == "slow.background_preempted" for event in store.events)

        reviewer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reviewer

    asyncio.run(run())


def test_review_economy_memoizes_only_negative_checkpoint_for_exact_stamp():
    async def run():
        _store, context, _adapter, _coordinator, slow, manager = await make_manager(
            [two_stage_plan(), unknown_advance(), supported_advance()], review_economy=True)
        await manager.initialize()
        active = context.milestone

        # Unknown semantic evidence cannot auto-advance, and is the only result memoized.
        assert await manager.checkpoint() == "continue"
        assert context.milestone == active
        assert manager.completed == []
        assert len(slow.model.calls) == 2

        assert await manager.checkpoint() == "continue"
        assert len(slow.model.calls) == 2

        assert context.ingest(Observation("obs-new", 2, 1, time.monotonic(), 2.0,
                                          context.action_seq, {}))
        assert await manager.checkpoint() == "advanced"
        assert len(slow.model.calls) == 3
        assert context.milestone.stage_index == 1
        assert len(manager.completed) == 1

    asyncio.run(run())


def partial_motion(position_error_m):
    return {"feedback": {"motion": {
        "execution_state": "partial", "reason_code": "servo_horizon",
        "motion_summary": {"position_error_m": position_error_m},
    }}}


def not_started_receipt():
    return {"feedback": {"motion": {
        "execution_state": "not_started", "reason_code": "invalid_arguments",
        "physics_executed": False,
    }}}


def test_recovery_key_uses_target_geometry_not_cosmetic_motion_labels():
    recovery = ProgressRecovery()
    points = {
        "p1": {"xyz_m": [0.100, 0.200, 0.300]},
        "p2": {"xyz_m": [0.101, 0.201, 0.301]},
    }
    move = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0]}
    cosmetic_retry = {**move, "motion_mode": "recovery", "position_tolerance_m": 0.02,
                      "max_steps": 8000}

    original_key = recovery.key("move_to", move, points)
    assert recovery.key("move_to", cosmetic_retry, points) == original_key
    assert recovery.key("move_to", {**move, "xyz_m": [0.13, 0.2, 0.3]}, points) != original_key
    assert recovery.key("move_to", {**move, "approach_world": [1, 0, 0]}, points) != original_key
    assert recovery.key("grasp_point", {"point_id": "p1", "approach_world": [0, 0, -1],
                                         "jaw_world": [0, 1, 0]}, points) == recovery.key(
        "grasp_point", {"point_id": "p2", "approach_world": [0, 0, -1], "jaw_world": [0, 1, 0]}, points)


def test_repeated_servo_horizon_failures_block_without_progress_then_reopen_on_progress():
    recovery = ProgressRecovery()
    args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0], "servo_horizon": 80}
    points = {}

    assert recovery.record("move_to", args, points, partial_motion(0.169))["stagnant_attempts"] == 1
    assert recovery.blocked("move_to", args, points) is None
    stalled = recovery.record("move_to", {**args, "motion_mode": "different_label"},
                              points, partial_motion(0.168))
    assert stalled["stagnant_attempts"] == 2
    assert stalled["failure_kind"] == "servo_horizon"
    assert recovery.blocked("move_to", args, points)["stagnant_attempts"] == 2

    progressed = recovery.record("move_to", args, points, partial_motion(0.10))
    assert progressed["measured_progress"] is True
    assert recovery.blocked("move_to", args, points) is None


def test_not_started_receipt_does_not_count_as_a_physical_target_failure():
    recovery = ProgressRecovery()
    args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0]}

    assert recovery.record("move_to", args, {}, not_started_receipt()) is None
    assert recovery.failures == {}
    assert recovery.blocked("move_to", args, {}) is None


def test_not_started_receipt_preserves_prior_physical_failure_memory():
    recovery = ProgressRecovery()
    args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0]}
    recovery.record("move_to", args, {}, partial_motion(0.169))
    original = recovery.record("move_to", args, {}, partial_motion(0.168))
    assert recovery.blocked("move_to", args, {})["stagnant_attempts"] == 2

    assert recovery.record("move_to", args, {}, not_started_receipt()) is None
    retained = recovery.blocked("move_to", args, {})
    assert retained == original
    assert retained["failure_kind"] == "servo_horizon"
    assert retained["position_error_m"] == pytest.approx(0.168)


def test_fast_unexecuted_receipt_keeps_blocked_proposals_and_recovery_feedback():
    async def run():
        class PreflightAdapter(FakeAdapter):
            async def execute(self, _capability, _args, _fence, _cancelled, _publish):
                return not_started_receipt()

        class OneMoveModel:
            history = []

            async def decide(self, _prompt, _projection, _images):
                return {"kind": "tool", "tool": "move_to",
                        "args": {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
                                 "jaw_world": [0, 1, 0]}}

        store = MemoryStore()
        context = Context(store)
        adapter = PreflightAdapter()
        coordinator = Coordinator(adapter, context, store)
        coordinator.authorize_mission(MISSION, capabilities=("move_to",))
        coordinator.delegate(Milestone("milestone-1", 1, MISSION, "move to the measured target",
                                        capabilities=("move_to",)))
        await coordinator.observe()
        agent = FastAgent(OneMoveModel(), context, coordinator, store, progress_recovery=True)
        args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
                "jaw_world": [0, 1, 0]}
        key = agent.progress_recovery.key("move_to", args, {})
        old_feedback = {"failure_key": key, "failure_kind": "servo_horizon",
                        "stagnant_attempts": 2, "instruction": "Change the physical approach."}
        context.local_state["recovery_feedback"] = old_feedback
        agent.blocked_proposals[key] = 2

        assert await agent.run(1) == "decision_budget"
        assert agent.blocked_proposals[key] == 2
        assert context.local_state["recovery_feedback"] == old_feedback
        assert agent.progress_recovery.blocked("move_to", args, {}) is None

    asyncio.run(run())


def test_measured_reposition_resets_stagnation_but_small_motion_does_not():
    recovery = ProgressRecovery()
    args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0]}
    state = {"robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}}
    recovery.observe(state)
    recovery.record("move_to", args, {}, partial_motion(0.169))
    recovery.record("move_to", args, {}, partial_motion(0.168))
    key = recovery.key("move_to", args, {})
    assert recovery.blocked("move_to", args, {}) is not None

    assert recovery.observe({"robot": {"grip_xyz_m": [0.02, 0.0, 0.0]}}) == []
    assert recovery.blocked("move_to", args, {}) is not None

    # A measured pose update alone can reflect a failed or incidental drift.
    corrective_args = {**args, "xyz_m": [0.2, 0.2, 0.3]}
    corrective_key = recovery.key("move_to", corrective_args, {})
    assert recovery.observe({"robot": {"grip_xyz_m": [0.04, 0.0, 0.0]}}) == []
    assert recovery.blocked("move_to", args, {}) is not None

    # Only a distinct, completed corrective action may reopen the old target.
    assert recovery.observe({"robot": {"grip_xyz_m": [0.04, 0.0, 0.0]}},
                            corrective=True, exclude_key=corrective_key) == [key]
    assert recovery.blocked("move_to", args, {}) is None


def test_failed_target_motion_does_not_clear_stagnation_by_reposition_alone():
    recovery = ProgressRecovery()
    args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0]}
    recovery.observe({"robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}})
    recovery.record("move_to", args, {}, partial_motion(0.169))

    # The robot moved at least 4 cm, but this was only an observation after the
    # failed attempt, not a distinct completed corrective command.
    assert recovery.observe({"robot": {"grip_xyz_m": [0.04, 0.0, 0.0]}}) == []
    repeated_failure = recovery.record("move_to", args, {}, partial_motion(0.170))
    assert repeated_failure["stagnant_attempts"] == 2
    assert repeated_failure["measured_progress"] is False
    assert recovery.blocked("move_to", args, {}) is not None


def test_possible_held_object_errors_do_not_poison_grasp_target_after_release():
    recovery = ProgressRecovery()
    points = {"p1": {"xyz_m": [0.1, 0.2, 0.3]}}
    args = {"point_id": "p1", "approach_world": [0, 0, -1], "jaw_world": [0, 1, 0]}

    for _ in range(3):
        feedback = recovery.record_error("grasp_point", args, points, "Possible held object detected")
        assert feedback["failure_kind"] == "unresolved_held_object"
        assert "precondition" in feedback["instruction"]
        assert feedback["stagnant_attempts"] == 0
        assert recovery.blocked("grasp_point", args, points) is None
    assert recovery.failures == {}


def test_recovery_feedback_clears_only_on_explicit_no_possible_hold():
    store = MemoryStore()
    context = Context(store)
    context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 1.0, 0,
                               {"robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}}))
    agent = FastAgent(object(), context, SimpleNamespace(adapter=SimpleNamespace(points={})),
                      store, progress_recovery=True)
    recovery = agent.progress_recovery

    grasp_args = {"point_id": "grasp-site", "approach_world": [0, 0, -1],
                  "jaw_world": [0, 1, 0]}
    held = recovery.record_error("grasp_point", grasp_args, {}, "Possible held object detected")
    context.local_state["recovery_feedback"] = held

    move_args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
                 "jaw_world": [0, 1, 0]}
    recovery.record("move_to", move_args, {}, partial_motion(0.169))
    recovery.record("move_to", move_args, {}, partial_motion(0.168))
    assert recovery.blocked("move_to", move_args, {}) is not None

    # Unknown and positive hold state leave the state precondition in view.
    agent.refresh_recovery_pose()
    assert context.local_state["recovery_feedback"] == held
    context.local_state["possible_hold"] = True
    agent.refresh_recovery_pose()
    assert context.local_state["recovery_feedback"] == held

    # An explicit adapter measurement of no possible hold clears only that
    # precondition feedback; an unrelated failed motion target remains blocked.
    context.local_state["possible_hold"] = False
    agent.refresh_recovery_pose()
    assert "recovery_feedback" not in context.local_state
    assert recovery.blocked("move_to", move_args, {}) is not None
    assert any(event["kind"] == "recovery.precondition_resolved" for event in store.events)


def test_recovery_threshold_respects_stage_retry_limit():
    recovery = ProgressRecovery()
    args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
            "jaw_world": [0, 1, 0]}
    recovery.record("move_to", args, {}, partial_motion(0.169))
    assert recovery.blocked("move_to", args, {}, max_attempts=1) is not None
    assert recovery.blocked("move_to", args, {}, max_attempts=2) is None
    recovery.record("move_to", args, {}, partial_motion(0.168))
    assert recovery.blocked("move_to", args, {}, max_attempts=2) is not None


def test_failure_memory_and_escalation_survive_milestone_contract_change():
    async def run():
        store = MemoryStore()
        context = Context(store)

        class Model:
            def __init__(self):
                self.history = ["repeated failed proposal"]

        class RecoveryManager:
            def __init__(self):
                self.calls = []

            async def recover(self, reason):
                self.calls.append(reason)

        model = Model()
        coordinator = SimpleNamespace(adapter=SimpleNamespace(points={}, schemas={}))
        slow = SimpleNamespace()
        agent = FastAgent(model, context, coordinator, store, slow=slow, progress_recovery=True)
        recovery_state = agent.progress_recovery
        manager = RecoveryManager()
        agent.milestones = manager

        first = Milestone("stage-0", 1, MISSION, "lift the red cup")
        context.milestone = first
        agent.sync_contract()
        args = {"xyz_m": [0.1, 0.2, 0.3], "approach_world": [0, 0, -1],
                "jaw_world": [0, 1, 0]}
        recovery_state.record("move_to", args, {}, partial_motion(0.169))
        recovery_state.record("move_to", args, {}, partial_motion(0.168))
        feedback = recovery_state.blocked("move_to", args, {})
        assert feedback is not None

        context.milestone = Milestone("stage-1", 1, MISSION, "place the red cup")
        agent.sync_contract()
        assert agent.progress_recovery is recovery_state
        assert recovery_state.blocked("move_to", args, {}) is not None

        # Repeated model proposals cannot spend the remaining action budget on the
        # same failed physical target: clear stale model history, ask Slow to help,
        # and stop after bounded repeated rejection.
        assert await agent.blocked_attempt(feedback, deadline=None) is None
        assert model.history == []
        assert await agent.blocked_attempt(feedback, deadline=None) is None
        assert len(manager.calls) == 1
        assert await agent.blocked_attempt(feedback, deadline=None) == "recovery_exhausted"
        assert any(event["kind"] == "recovery.stalled" for event in store.events)

    asyncio.run(run())
