"""Offline protocol tests for the V18 milestone manager.

These exercise contracts and handoffs with a small fake adapter and queued
model responses. They do not start a simulator or contact a model service.
"""
import asyncio
import copy
import time
from collections import deque

import pytest

from embodied_harness.domain.models import ActionRequest, Observation
from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator, GateError
from embodied_harness.runtime.milestones import MilestoneManager, PlanError


MISSION = "put the red cup inside the bowl"
FUNDAMENTAL = "preserve the identity of the red cup"


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


class FakeAdapter:
    generation = 1

    def __init__(self, tactile=None):
        self.version = 0
        self.executions = []
        self.schemas = {
            "move_to": {"parameters": {"properties": {"x": {"type": "number"}}}},
            "grasp_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
        }
        self.tactile = tactile if tactile is not None else {
            "left_fingerpad_contact": True,
            "right_fingerpad_contact": True,
        }
        self.verify_calls = 0

    async def observe(self, action_seq):
        self.version += 1
        return Observation(
            id=f"obs-{self.version}", version=self.version, generation=1,
            captured_at=time.monotonic(), sim_time=float(self.version),
            action_seq=action_seq, state={"tactile": copy.deepcopy(self.tactile)},
        )

    def validate_args(self, capability, args):
        if capability not in self.schemas or not isinstance(args, dict):
            raise ValueError("invalid tool request")

    async def execute(self, capability, args, fence, cancelled, publish):
        self.executions.append((capability, copy.deepcopy(args)))
        return {}

    async def verify(self, observation_id):
        self.verify_calls += 1
        return {"status": "not_verified", "source": "environment_verifier",
                "observation_id": observation_id}


class QueuedModel:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    async def decide(self, prompt, projection, images):
        self.calls.append({"prompt": prompt, "projection": copy.deepcopy(projection),
                           "images": tuple(images)})
        response = self.responses.popleft()
        if callable(response):
            response = response(prompt, projection, images)
            if asyncio.iscoroutine(response):
                response = await response
        return copy.deepcopy(response)


def slow_stub(responses, context, coordinator, store):
    model = QueuedModel(responses)
    return SlowAgent(model, context, coordinator, store)


def stage(goal, visual_description, *, extra_conditions=(), decision_budget=8):
    return {
        "goal": goal,
        "constraints": [],
        "preconditions": [],
        "success_conditions": [
            {"predicate": "visual", "description": visual_description},
            *copy.deepcopy(list(extra_conditions)),
        ],
        "max_retries": 2,
        "decision_budget": decision_budget,
    }


def two_stage_plan(*, extra_first_conditions=(), first_budget=8, second_budget=8,
                   fundamental_constraints=None):
    if fundamental_constraints is None:
        fundamental_constraints = [FUNDAMENTAL]
    return {
        "kind": "plan",
        "fundamental_constraints": list(fundamental_constraints),
        "milestones": [
            stage("lift the red cup clear of the table",
                  "the red cup is visibly lifted clear of the tabletop",
                  extra_conditions=extra_first_conditions, decision_budget=first_budget),
            stage("place the red cup inside the bowl",
                  "the same red cup is visibly inside the bowl",
                  decision_budget=second_budget),
        ],
    }


def checkpoint_decision(manager, *, condition_statuses=None, invariant_statuses=None,
                        decision="advance"):
    milestone = manager.context.milestone
    condition_statuses = condition_statuses or {}
    invariant_statuses = invariant_statuses or {}
    return {
        "kind": "checkpoint",
        "decision": decision,
        "reason": "Evidence was inspected.",
        "assessments": [
            {"condition_id": f"success-{index}",
             "status": condition_statuses.get(index, "satisfied"),
             "evidence": f"Evidence for condition {index}."}
            for index, _ in enumerate(milestone.success_conditions)
        ],
        "invariants": [
            {"constraint_id": f"constraint-{index}",
             "status": invariant_statuses.get(index, "satisfied"),
             "evidence": f"Evidence for invariant {index}."}
            for index, _ in enumerate(milestone.constraints)
        ],
        "advice": "Inspect any remaining uncertainty.",
    }


async def harness(responses, *, max_decisions=20, tactile=None, max_replans=3):
    store = MemoryStore()
    context = Context(store)
    adapter = FakeAdapter(tactile=tactile)
    coordinator = Coordinator(adapter, context, store)
    coordinator.authorize_mission(MISSION, capabilities=("grasp_point", "move_to", "place_point"))
    await coordinator.observe()
    slow = slow_stub(responses, context, coordinator, store)
    manager = MilestoneManager(
        slow, context, coordinator, store, max_decisions=max_decisions,
        deadline=time.monotonic() + 30, max_replans=max_replans,
    )
    return store, context, adapter, coordinator, slow, manager


def test_valid_two_stage_plan_advances_only_after_evidence_checkpoint():
    async def run():
        _store, context, _adapter, _coordinator, _slow, manager = await harness(
            [two_stage_plan(), None]
        )
        await manager.initialize()
        first = context.milestone
        assert first.stage_index == 0
        assert first.final_stage is False
        assert first.success_predicate == "stage_conditions"
        assert first.capabilities == ("grasp_point", "move_to", "place_point")
        assert context.mission_plan["active_index"] == 0

        manager.slow.model.responses[0] = checkpoint_decision(manager)
        assert await manager.checkpoint() == "advanced"
        second = context.milestone
        assert second.id != first.id
        assert second.plan_id == first.plan_id
        assert second.stage_index == 1
        assert second.final_stage is True
        assert second.goal == "place the red cup inside the bowl"
        assert manager.completed[0]["provenance"] == "model_assessed_stage_completion"

    asyncio.run(run())


@pytest.mark.parametrize("case", [
    "unknown_condition", "contradicted_condition", "missing_condition", "duplicate_condition",
    "unknown_invariant", "contradicted_invariant", "missing_invariant", "duplicate_invariant",
])
def test_incomplete_or_negative_assessments_do_not_advance(case):
    async def run():
        store, context, _adapter, _coordinator, _slow, manager = await harness(
            [two_stage_plan()]
        )
        await manager.initialize()
        decision = checkpoint_decision(manager)
        collection = "invariants" if case.endswith("_invariant") else "assessments"
        if case.startswith("unknown"):
            decision[collection][0]["status"] = "unknown"
        elif case.startswith("contradicted"):
            decision[collection][0]["status"] = "contradicted"
        elif case.startswith("missing"):
            decision[collection].pop()
        else:
            decision[collection].append(copy.deepcopy(decision[collection][0]))
        manager.slow.model.responses.append(decision)

        active = context.milestone
        expected = "checkpoint_unavailable" if case.startswith(("missing", "duplicate")) else "continue"
        assert await manager.checkpoint() == expected
        assert context.milestone == active
        assert manager.index == 0
        assert manager.completed == []
        assert not any(event["kind"] == "milestone.completed" for event in store.events)

    asyncio.run(run())


def test_model_cannot_override_contradicted_public_sensor_evidence():
    async def run():
        store, context, _adapter, _coordinator, _slow, manager = await harness(
            [two_stage_plan(extra_first_conditions=[{"predicate": "bilateral_contact"}])],
            tactile={"left_fingerpad_contact": True, "right_fingerpad_contact": False},
        )
        await manager.initialize()
        decision = checkpoint_decision(manager)
        manager.slow.model.responses.append(decision)

        assert await manager.checkpoint() == "continue"
        assert context.milestone.stage_index == 0
        assessment = next(event for event in store.events
                          if event["kind"] == "milestone.assessment")
        assert assessment["payload"]["sensor_assessments"]["success-1"]["status"] == "contradicted"
        assert assessment["payload"]["provenance"] == "model_judgement_plus_separate_sensor_evidence"

    asyncio.run(run())


def test_final_stage_model_assessment_is_not_mission_success():
    async def run():
        responses = [
            two_stage_plan(),
            None,  # replaced with the first checkpoint after initialization
            None,  # replaced with the final checkpoint after stage advance
        ]
        store, context, adapter, coordinator, slow, manager = await harness(responses)
        await manager.initialize()
        slow.model.responses[0] = checkpoint_decision(manager)
        assert await manager.checkpoint() == "advanced"
        final_stage = context.milestone
        assert final_stage.final_stage
        slow.model.responses[0] = checkpoint_decision(manager)

        assert await manager.checkpoint() == "continue"
        assert context.milestone == final_stage
        assert not any(event["kind"] == "milestone.completed" and
                       event["payload"]["milestone_id"] == final_stage.id
                       for event in store.events)
        verifier_result = await coordinator.verify()
        assert verifier_result["status"] == "not_verified"
        assert adapter.verify_calls == 1

    asyncio.run(run())


def test_replan_retains_fundamental_constraints_and_completed_evidence():
    async def run():
        replan = {
            "kind": "plan",
            "fundamental_constraints": [],
            "milestones": [stage("finish placing the red cup",
                                  "the same red cup is visibly inside the bowl")],
        }
        store, context, _adapter, _coordinator, slow, manager = await harness(
            [two_stage_plan(), None, replan]
        )
        await manager.initialize()
        slow.model.responses[0] = checkpoint_decision(manager)
        assert await manager.checkpoint() == "advanced"
        previous_plan_id = manager.plan_id
        previous_completion = copy.deepcopy(manager.completed[0])

        assert manager.request_replan("new visual evidence changed the remaining route", context.milestone)
        assert await manager.recover("new visual evidence changed the remaining route") is None
        assert manager.fundamental == [FUNDAMENTAL]
        assert FUNDAMENTAL in context.milestone.constraints
        assert manager.plan_id != previous_plan_id
        assert manager.completed == [previous_completion]
        assert slow.model.calls[-1]["projection"]["retained_fundamental_constraints"] == [FUNDAMENTAL]

    asyncio.run(run())


def test_episode_decision_budget_is_shared_across_stage_activation():
    async def run():
        store, context, _adapter, _coordinator, slow, manager = await harness(
            [two_stage_plan(first_budget=8, second_budget=8), None], max_decisions=5
        )
        await manager.initialize()
        assert context.milestone.decision_budget == 5
        manager.record_decision()
        manager.record_decision()
        manager.record_decision()
        assert manager.remaining == 2

        slow.model.responses[0] = checkpoint_decision(manager)
        assert await manager.checkpoint() == "advanced"
        assert context.milestone.stage_index == 1
        assert manager.stage_decisions == 0
        assert manager.remaining == 2
        assert context.milestone.decision_budget == 2
        manager.record_decision()
        assert manager.remaining == 1
        assert context.mission_plan["episode_decisions_remaining"] == 1

    asyncio.run(run())


def test_exhausted_episode_budget_does_not_activate_or_plan_next_stage():
    async def run():
        _store, context, _adapter, _coordinator, slow, manager = await harness(
            [two_stage_plan(first_budget=8, second_budget=8), None], max_decisions=1
        )
        await manager.initialize()
        manager.record_decision()
        assert manager.remaining == 0
        slow.model.responses[0] = checkpoint_decision(manager)

        assert await manager.checkpoint() == "decision_budget"
        # The reviewed stage remains the active contract; no executable next-stage
        # contract is installed once the episode-wide action budget is exhausted.
        assert context.milestone.stage_index == 0
        assert manager.remaining == 0
        assert await manager.before_decision(remaining=0) == "decision_budget"
        # Exhaustion must not be routed into another Slow plan/recovery call.
        assert len(slow.model.calls) == 2

    asyncio.run(run())


def test_stale_checkpoint_model_answer_is_rejected_after_observation_changes():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_answer(_prompt, _projection, _images):
            entered.set()
            await release.wait()
            return {"kind": "checkpoint", "decision": "advance", "reason": "old frame",
                    "assessments": [{"condition_id": "success-0", "status": "satisfied",
                                     "evidence": "It looked complete in the old frame."}],
                    "invariants": [{"constraint_id": "constraint-0", "status": "satisfied",
                                    "evidence": "The old frame appeared consistent."}]}

        store, context, _adapter, _coordinator, slow, manager = await harness([two_stage_plan(), delayed_answer])
        await manager.initialize()
        old_milestone = context.milestone
        task = asyncio.create_task(manager.checkpoint())
        await entered.wait()
        assert context.ingest(Observation(
            id="obs-new", version=2, generation=1, captured_at=time.monotonic(),
            sim_time=2.0, action_seq=context.action_seq,
            state={"tactile": {"left_fingerpad_contact": True,
                               "right_fingerpad_contact": True}},
        ))
        release.set()

        assert await task == "checkpoint_unavailable"
        assert context.milestone == old_milestone
        assert manager.index == 0
        assert manager.completed == []
        assert any(event["kind"] == "milestone.checkpoint_error" for event in store.events)
        assert slow.foreground_pending == 0
        assert slow.foreground_done.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("arrival_phase", ["fast_inference", "identity_inference"])
def test_pending_replan_invalidates_fast_decision_before_action(arrival_phase):
    async def run():
        store, context, adapter, coordinator, slow, manager = await harness(
            [two_stage_plan()], max_decisions=3
        )
        await manager.initialize()
        entered, release = asyncio.Event(), asyncio.Event()

        class FastModel:
            history = []

            async def decide(self, _prompt, _projection, _images):
                if arrival_phase == "fast_inference":
                    entered.set()
                    await release.wait()
                    return {"kind": "tool", "tool": "move_to", "args": {"x": 0.1}}
                return {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "point-1"}}

        class IdentityModel:
            async def check(self, _point_id):
                entered.set()
                await release.wait()
                return True

        fast = FastAgent(FastModel(), context, coordinator, store, slow=slow, mode="dual_async")
        fast.milestones = manager
        if arrival_phase == "identity_inference":
            fast.identity_checker = IdentityModel()
        task = asyncio.create_task(fast.run(1))
        await entered.wait()
        assert manager.request_replan("background review found the remaining plan unsuitable",
                                      context.milestone)
        release.set()

        assert await task == "decision_budget"
        assert adapter.executions == []
        assert context.action_seq == 0
        assert manager.pending_reason is not None

    asyncio.run(run())


def test_replan_arriving_while_submit_waits_for_coordinator_lock_blocks_action():
    async def run():
        _store, context, adapter, coordinator, _slow, manager = await harness([two_stage_plan()])
        await manager.initialize()
        observation = context.latest
        milestone = context.milestone
        request = ActionRequest("queued-action", milestone.id, milestone.revision,
                                milestone.generation, observation.id, "move_to", {"x": 0.1})

        await coordinator.lock.acquire()
        task = asyncio.create_task(coordinator.submit(request))
        await asyncio.sleep(0)  # submit is now waiting at the physical-action gate
        assert manager.request_replan("background review invalidated the current contract", milestone)
        coordinator.lock.release()

        with pytest.raises(GateError, match="contract_review_pending"):
            await task
        assert adapter.executions == []
        assert context.action_seq == 0

    asyncio.run(run())


def test_stale_background_replan_is_rejected_after_new_observation():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_replan(_prompt, _projection, _images):
            entered.set()
            await release.wait()
            return {"kind": "replan", "reason": "old frame suggested the remaining plan was unsuitable"}

        store = MemoryStore()
        context = Context(store)
        adapter = FakeAdapter()
        coordinator = Coordinator(adapter, context, store)
        coordinator.authorize_mission(MISSION, capabilities=("grasp_point", "move_to", "place_point"))
        await coordinator.observe()
        slow = SlowAgent(QueuedModel([two_stage_plan(), delayed_replan]), context, coordinator, store)
        manager = MilestoneManager(slow, context, coordinator, store, max_decisions=20,
                                   deadline=time.monotonic() + 30)
        await manager.initialize()

        task = asyncio.create_task(slow.review())
        await entered.wait()
        assert context.ingest(Observation(
            id="obs-new", version=2, generation=1, captured_at=time.monotonic(),
            sim_time=2.0, action_seq=context.action_seq,
            state={"tactile": {"left_fingerpad_contact": True,
                               "right_fingerpad_contact": True}},
        ))
        release.set()
        await task

        assert manager.pending_reason is None
        assert coordinator.contract_review_pending is False
        assert not any(event["kind"] == "milestone.replan_requested" for event in store.events)
        assert any(event["kind"] == "milestone.replan_stale" for event in store.events)

    asyncio.run(run())


def test_pending_replan_during_checkpoint_handoff_is_not_cleared_or_overwritten():
    async def run():
        returned, release = asyncio.Event(), asyncio.Event()

        async def checkpoint_response(_prompt, _projection, _images):
            returned.set()
            await release.wait()
            return {"kind": "checkpoint", "decision": "advance", "reason": "apparently complete",
                    "assessments": [{"condition_id": "success-0", "status": "satisfied",
                                     "evidence": "The target appears lifted."}],
                    "invariants": [{"constraint_id": "constraint-0", "status": "satisfied",
                                    "evidence": "The target identity appears consistent."}]}

        _store, context, _adapter, coordinator, slow, manager = await harness(
            [two_stage_plan(), checkpoint_response]
        )
        await manager.initialize()
        await coordinator.lock.acquire()
        task = asyncio.create_task(manager.checkpoint())
        await returned.wait()
        release.set()
        await asyncio.sleep(0)  # checkpoint reaches its final coordinator-lock handoff
        reason = "background review says the remaining outcomes need replanning"
        assert manager.request_replan(reason, context.milestone)
        coordinator.lock.release()

        assert await task == "replan"
        assert context.milestone.stage_index == 0
        assert manager.pending_reason == reason
        assert coordinator.contract_review_pending is True
        assert manager.completed == []
        assert len(slow.model.calls) == 2

    asyncio.run(run())


def test_milestone_inference_waits_for_shared_slow_model_lock():
    async def run():
        store, context, _adapter, _coordinator, slow, manager = await harness([two_stage_plan()])
        await slow.model_lock.acquire()  # Represents an already-running background Slow review.
        task = asyncio.create_task(manager.initialize())
        await asyncio.sleep(0)
        assert slow.model.calls == []
        assert slow.foreground_pending == 1
        assert not slow.foreground_done.is_set()

        slow.model_lock.release()
        assert await task == "planned"
        assert len(slow.model.calls) == 1
        assert slow.foreground_pending == 0
        assert slow.foreground_done.is_set()
        assert any(event["kind"] == "milestone.inference_started" for event in store.events)

    asyncio.run(run())


@pytest.mark.parametrize("automatic", [False, True])
def test_unavailable_checkpoint_never_allows_fast_to_dispatch_next_action(automatic):
    from embodied_harness.agents.model import ModelError

    async def run():
        def unavailable(*_args):
            raise ModelError("synthetic assessment timeout")

        conditions = ({"predicate": "bilateral_contact"},) if automatic else ()
        store, context, adapter, coordinator, slow, manager = await harness(
            [two_stage_plan(extra_first_conditions=conditions), unavailable, unavailable])
        await manager.initialize()

        class Actor:
            calls = 0

            async def decide(self, *_args):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("Fast must not continue after unavailable assessment")
                return {"kind": "complete", "reason": "stage appears complete"}

        actor = Actor()
        fast = FastAgent(actor, context, coordinator, store, mode="dual_async")
        fast.milestones = manager
        assert await fast.run(3) == "checkpoint_unavailable"
        assert actor.calls == (0 if automatic else 1)
        assert adapter.executions == []
        assert manager.completed == []
        assert context.milestone.stage_index == 0
        assert len(slow.model.calls) == 3  # One plan and two bounded assessment attempts.
        assert any(e["kind"] == "milestone.checkpoint_unavailable" for e in store.events)

    asyncio.run(run())


def test_checkpoint_retry_refreshes_evidence_and_can_advance_without_fast_motion():
    from embodied_harness.agents.model import ModelError

    async def run():
        def unavailable(*_args):
            raise ModelError("synthetic transient timeout")

        store, context, adapter, _coordinator, slow, manager = await harness([two_stage_plan()])
        await manager.initialize()
        old_observation = context.latest.id
        slow.model.responses.extend([unavailable, lambda *_args: checkpoint_decision(manager)])
        assert await manager.checkpoint() == "advanced"
        assert context.latest.id != old_observation
        assert len(manager.completed) == 1
        assert adapter.executions == []
        assert any(e["kind"] == "milestone.checkpoint_retry" for e in store.events)

    asyncio.run(run())


def test_public_sensor_candidate_triggers_review_before_next_fast_action():
    async def run():
        plan = two_stage_plan(extra_first_conditions=[{"predicate": "bilateral_contact"}])
        store, context, adapter, _coordinator, slow, manager = await harness([plan, None])
        await manager.initialize()
        slow.model.responses[0] = checkpoint_decision(manager)
        assert await manager.before_decision(20) is None
        assert context.milestone.stage_index == 1
        assert adapter.executions == []
        assert manager.remaining == 20
        proposed = [e for e in store.events if e["kind"] == "milestone.completion_proposed"]
        assert proposed[-1]["payload"]["trigger"] == "sensor_conditions_candidate"
    asyncio.run(run())


def test_sensor_candidate_never_overrides_uncertain_identity_or_spams_reviews():
    async def run():
        plan = two_stage_plan(extra_first_conditions=[{"predicate": "bilateral_contact"}])
        _store, context, _adapter, _coordinator, slow, manager = await harness([plan, None])
        await manager.initialize()
        slow.model.responses[0] = checkpoint_decision(manager, condition_statuses={0: "unknown"})
        assert await manager.before_decision(20) is None
        assert context.milestone.stage_index == 0
        assert manager.completed == []
        assert await manager.before_decision(20) is None
        assert len(slow.model.calls) == 2  # Same physical state cannot trigger repeated automatic checks.
    asyncio.run(run())


def test_final_only_constraint_does_not_block_acquisition_but_is_retained():
    async def run():
        final_rule = "the cup ends inside the bowl"
        plan = two_stage_plan(fundamental_constraints=[FUNDAMENTAL,
            {"description": final_rule, "scope": "on_mission_completion"}])

        def assess(_prompt, projection, _images):
            required = projection["checkpoint_invariants"]
            assert [c["description"] for c in required] == [FUNDAMENTAL]
            assert projection["deferred_mission_completion_constraints"][0]["description"] == final_rule
            return {"kind": "checkpoint", "decision": "advance", "reason": "acquisition is supported",
                    "assessments": [{"condition_id": c["condition_id"], "status": "satisfied", "evidence": "visible lift"}
                                    for c in projection["checkpoint_conditions"]],
                    "invariants": [{"constraint_id": c["constraint_id"], "status": "satisfied", "evidence": "identity is preserved"}
                                   for c in required]}

        _store, context, _adapter, _coordinator, _slow, manager = await harness([plan, assess])
        await manager.initialize()
        assert await manager.checkpoint() == "advanced"
        assert final_rule in context.milestone.constraints
        assert context.milestone.constraint_scopes[final_rule] == "on_mission_completion"
        changed = two_stage_plan(fundamental_constraints=[
            {"description": FUNDAMENTAL, "scope": "on_mission_completion"}])
        with pytest.raises(PlanError, match="scope cannot be changed"):
            manager.validate_plan(changed)
    asyncio.run(run())
