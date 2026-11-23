"""Offline regressions for the opt-in pipelined supervision path."""
import asyncio
import copy
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest
import jsonschema

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Advice, Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.pipeline import PipelinedMilestoneManager


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
    terminal = False

    def __init__(self):
        self.version = 0
        self.sim_time = 1.0
        self.control_steps = 10
        self.schemas = {
            "observe": {"parameters": {"properties": {}}},
            "mark_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
            "locate_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
            "select_candidate": {"parameters": {"properties": {"candidate_id": {"type": "string"}}}},
            "move_to": {"parameters": {"properties": {"xyz_m": {"type": "array"}}}},
            "grasp_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
            "place_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
        }
        self.points = {}
        self.executions = []
        self.inspected = asyncio.Event()

    def validate_args(self, capability, args):
        if capability not in self.schemas or not isinstance(args, dict):
            raise ValueError("invalid fake arguments")

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"obs-{self.version}", self.version, self.generation,
                           time.monotonic(), self.sim_time, action_seq, self.state())

    async def inspect(self, capability, _args, action_seq):
        self.validate_args(capability, {})
        self.version += 1
        self.inspected.set()
        state = self.state()
        if capability == "select_candidate":
            state["feedback"] = {"point": {"point_id": "point-1"}}
        return Observation(f"obs-{self.version}", self.version, self.generation,
                           time.monotonic(), self.sim_time, action_seq, state)

    def state(self):
        return {"sensor": {"sim_time": self.sim_time, "control_steps": self.control_steps},
                "robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}}

    async def execute(self, capability, args, _fence, _cancelled, _publish):
        self.executions.append((capability, copy.deepcopy(args)))
        self.sim_time += 0.1
        self.control_steps += 1
        return {"macro_status": "reached"}

    async def verify(self, observation_id):
        return {"status": "not_verified", "source": "fake", "observation_id": observation_id}


class ScriptedSlow:
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


def plan(*, first_budget=8):
    return {
        "kind": "plan", "fundamental_constraints": [],
        "milestones": [
            {"goal": "lift the red cup clear of the table", "decision_budget": first_budget,
             "success_conditions": [{"predicate": "visual", "description": "the red cup is lifted clear of the table"}]},
            {"goal": "place the red cup inside the bowl",
             "success_conditions": [{"predicate": "visual", "description": "the red cup is inside the bowl"}]},
        ],
    }


def checkpoint_reply(status):
    return {"kind": "checkpoint", "decision": "advance", "reason": "Assess the current visual outcome.",
            "assessments": [{"condition_id": "success-0", "status": status,
                             "evidence": "The current visual relation was assessed."}],
            "invariants": []}


async def make_pipeline(slow_model):
    store = MemoryStore()
    context = Context(store)
    adapter = FakeAdapter()
    coordinator = Coordinator(adapter, context, store)
    coordinator.authorize_mission(MISSION, capabilities=("move_to", "grasp_point", "place_point"))
    await coordinator.observe()
    slow = SlowAgent(slow_model, context, coordinator, store)
    manager = PipelinedMilestoneManager(slow, context, coordinator, store, max_decisions=8,
                                       deadline=time.monotonic() + 20)
    return store, context, adapter, coordinator, slow, manager


def test_initial_plan_overlaps_one_scout_and_bridges_its_explicit_readonly_observation():
    async def run():
        plan_entered, release_plan = asyncio.Event(), asyncio.Event()

        async def delayed_plan(_prompt, _projection, _images):
            plan_entered.set()
            await release_plan.wait()
            return plan()

        class Scout:
            calls = 0

            async def decide(self, _prompt, _projection, _images):
                self.calls += 1
                return {"kind": "tool", "tool": "observe", "args": {}, "reason": "inspect scene"}

        store, context, adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([delayed_plan]))
        before = manager.stamp()
        scout = Scout()
        initializing = asyncio.create_task(manager.initialize_with_scout(scout))
        await asyncio.wait_for(plan_entered.wait(), timeout=1)
        await asyncio.wait_for(adapter.inspected.wait(), timeout=1)

        assert manager.scout_decisions == 1
        assert scout.calls == 1
        assert context.action_seq == 0
        assert context.milestone is None  # No Fast physical command precedes the contract.
        assert adapter.executions == []
        assert manager.matches_stamp(before)  # Explicit read-only edge preserves this plan.

        release_plan.set()
        assert await initializing == "planned"
        assert manager.plan_id is not None
        assert context.milestone.stage_index == 0

        # The scout consumes the single episode decision. With a total budget of
        # one, Fast receives no further cognition slots.
        fast = FastAgent(object(), context, _coordinator, store)
        fast.milestones = manager
        assert await fast.run(1 - manager.scout_decisions) == "decision_budget"
        assert fast.steps == 0
        assert manager.scout_decisions + fast.steps == 1

    asyncio.run(run())


def test_app_counts_initial_scout_against_global_decision_budget(monkeypatch, tmp_path):
    async def run():
        from embodied_harness import app
        import embodied_harness.adapters.libero_direct as direct
        scout_decided = asyncio.Event()

        class AppAdapter(FakeAdapter):
            cancellation = "synthetic_boundary"

            def __init__(self, *_args):
                super().__init__()
                self.host = None

            async def start(self):
                self.host = SimpleNamespace(execution_accounting={})

            async def close(self):
                pass

        class OfflineModel:
            def __init__(self, _base_url, _key, model, role, _output, options=None):
                self.model, self.role = model, role
                self.calls = 0
                self.usage = {}
                self.latencies = []
                self.accounting = {"responses_with_usage": 0}

            def bind_tools(self, _schemas):
                return None

            def bind_memory(self, _entries):
                return None

            def bind_mission_reference(self, _observation):
                return None

            async def decide(self, prompt, _projection, _images):
                self.calls += 1
                if self.role == "slow":
                    await scout_decided.wait()
                    return plan()
                if "Use ONE read-only observation/measurement" in prompt:
                    scout_decided.set()
                    return {"kind": "tool", "tool": "observe", "args": {},
                            "reason": "identify the target before the plan is ready"}
                raise AssertionError("Fast cognition should receive zero slots after the scout")

            async def close(self):
                pass

        monkeypatch.setattr(app, "OpenETAAdapter", AppAdapter)
        monkeypatch.setattr(direct, "DirectLiberoAdapter", AppAdapter)
        monkeypatch.setattr(app, "ModelClient", OfflineModel)
        monkeypatch.setattr(app, "load_env", lambda _path: None)
        for name, value in {
            "MODEL_BASE_URL": "http://unused.invalid",
            "FAST_API_KEY": "offline-fast",
            "SLOW_API_KEY": "offline-slow",
            "FAST_MODEL": "offline-fast-model",
            "SLOW_MODEL": "offline-slow-model",
        }.items():
            monkeypatch.setenv(name, value)

        args = SimpleNamespace(
            output=str(tmp_path / "pipeline-budget"), model_env="unused.env", memory=None,
            mission=MISSION, backend="direct_libero", sim_url="unused", env_id="unused",
            seed=1, timeout=10.0, max_decisions=1, mode="dual_async", hierarchy="milestones",
            review_every=3, pipeline_supervision=True, identity_check=False,
        )
        result = await app.episode(args)

        assert result["error"] is None
        assert result["scout_decisions"] == 1
        assert result["decisions"] == 1
        assert result["action_count"] == 0
        assert [model["calls"] for model in result["models"]] == [1, 1]

    asyncio.run(run())


def test_readonly_stamp_bridge_requires_explicit_unchanged_edge_and_keeps_hard_fences():
    async def run():
        _store, context, adapter, coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan()]))
        await manager.initialize()
        before = context.latest
        stamp = manager.stamp()
        after = Observation("read-only", before.version + 1, before.generation,
                            time.monotonic(), before.sim_time, context.action_seq, copy.deepcopy(before.state))
        manager.note_readonly(before, after)
        assert context.ingest(after)
        assert manager.matches_stamp(stamp)

        original_action_seq = context.action_seq
        context.action_seq += 1
        assert not manager.matches_stamp(stamp)
        context.action_seq = original_action_seq

        original_generation = adapter.generation
        adapter.generation += 1
        assert not manager.matches_stamp(stamp)
        adapter.generation = original_generation

        original_pause_epoch = coordinator.pause_epoch
        coordinator.pause_epoch += 1
        assert not manager.matches_stamp(stamp)
        coordinator.pause_epoch = original_pause_epoch

        original_contract_epoch = coordinator.contract_review_epoch
        coordinator.contract_review_epoch += 1
        assert not manager.matches_stamp(stamp)
        coordinator.contract_review_epoch = original_contract_epoch

        current = context.milestone
        context.milestone = replace(current, revision=current.revision + 1)
        assert not manager.matches_stamp(stamp)

    asyncio.run(run())


@pytest.mark.parametrize("refresh", ["unsolicited", "robot_changed", "sensor_changed", "clock_missing"])
def test_unclassified_or_state_changing_observation_does_not_bridge_plan_stamp(refresh):
    async def run():
        _store, context, _adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan()]))
        await manager.initialize()
        before = context.latest
        stamp = manager.stamp()
        state = copy.deepcopy(before.state)
        if refresh == "robot_changed":
            state["robot"] = {"grip_xyz_m": [0.2, 0.0, 0.0]}
        elif refresh == "sensor_changed":
            state["sensor"]["left_fingerpad_contact"] = True
        elif refresh == "clock_missing":
            state["sensor"].pop("sim_time")
        after = Observation(f"{refresh}-obs", before.version + 1, before.generation,
                            time.monotonic(), before.sim_time, context.action_seq, state)
        if refresh != "unsolicited":
            manager.note_readonly(before, after)  # Changed physical evidence is rejected.
        assert context.ingest(after)
        assert not manager.matches_stamp(stamp)

    asyncio.run(run())


@pytest.mark.parametrize("qualified,expected", [(True, True), (False, False)])
def test_replan_stamp_accepts_qualified_readonly_chain_but_rejects_unsolicited_refresh(qualified, expected):
    async def run():
        _store, context, _adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan()]))
        await manager.initialize()
        before = context.latest
        expected_stamp = manager.stamp()
        after = Observation("replan-refresh", before.version + 1, before.generation,
                            time.monotonic(), before.sim_time, context.action_seq, copy.deepcopy(before.state))
        if qualified:
            manager.note_readonly(before, after)
        assert context.ingest(after)
        assert manager.request_replan("Recheck remaining outcomes.", context.milestone,
                                      expected_stamp=expected_stamp) is expected

    asyncio.run(run())


@pytest.mark.parametrize("result,expected_stage,terminal", [
    ("unknown", 0, None), ("none", 0, "checkpoint_unavailable"), ("satisfied", 1, None),
])
def test_checkpoint_allows_one_readonly_preparation_and_discards_old_physical_proposal(result, expected_stage, terminal):
    async def run():
        checkpoint_entered, release_checkpoint = asyncio.Event(), asyncio.Event()

        async def delayed_checkpoint(_prompt, _projection, _images):
            checkpoint_entered.set()
            await release_checkpoint.wait()
            return None if result == "none" else checkpoint_reply(
                "unknown" if result == "unknown" else "satisfied")

        responses = [plan(), delayed_checkpoint]
        if result == "none":
            responses.append(None)
        _store, context, adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow(responses))
        await manager.initialize()

        assert await manager.checkpoint("fast_completion_proposal") == "checkpoint_pending"
        await asyncio.wait_for(checkpoint_entered.wait(), timeout=1)
        assert context.local_state["pipeline"]["phase"] == "checkpoint_pending"
        allowed, terminal_before = await manager.guard_proposal("tool", "observe")
        assert (allowed, terminal_before) == (True, None)
        assert manager.preparation_used

        physical = asyncio.create_task(manager.guard_proposal("tool", "move_to"))
        await asyncio.sleep(0)
        assert not physical.done()  # Physical proposals rendezvous with assessment.
        release_checkpoint.set()
        allowed, terminal_after = await physical

        assert not allowed  # The proposal was based on the pre-assessment contract.
        assert terminal_after == terminal
        assert context.milestone.stage_index == expected_stage
        assert adapter.executions == []
        assert "pipeline" not in context.local_state

        await manager.close()

    asyncio.run(run())


@pytest.mark.parametrize("control", ["routine_advice", "pause", "replan"])
def test_pipeline_routine_advice_does_not_invalidate_fast_but_hard_control_does(control):
    async def run():
        _store, context, adapter, coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan()]))
        await manager.initialize()
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockedFastModel:
            calls = 0

            async def decide(self, _prompt, _projection, _images):
                self.calls += 1
                entered.set()
                await release.wait()
                return {"kind": "tool", "tool": "move_to", "args": {"xyz_m": [0.1, 0.2, 0.3]}}

        model = BlockedFastModel()
        fast = FastAgent(model, context, coordinator, _store, slow=_slow, progress_recovery=True)
        fast.milestones = manager
        task = asyncio.create_task(fast.run(1))
        await asyncio.wait_for(entered.wait(), timeout=1)
        before_version = context.control_version

        if control == "routine_advice":
            m = context.milestone
            assert coordinator.apply_advice(Advice(m.id, m.revision, m.generation, "advice",
                "Recheck the next grasp angle at the next decision.", time.monotonic() + 60, priority=40))
            assert context.control_version == before_version
        elif control == "pause":
            coordinator.pause("Stop before continuing.")
        else:
            assert manager.request_replan("The remaining route needs review.", context.milestone)

        release.set()
        assert await task == "decision_budget"
        assert model.calls == 1
        if control == "routine_advice":
            assert len(adapter.executions) == 1
        else:
            assert adapter.executions == []
            assert any(event["kind"] == "fast.decision_stale" for event in _store.events)

        await manager.close()

    asyncio.run(run())


@pytest.mark.parametrize("route", ["soft_budget", "automatic_no_progress"])
def test_soft_budget_and_automatic_no_progress_route_to_background_review(route):
    async def run():
        _store, context, _adapter, coordinator, slow, manager = await make_pipeline(
            ScriptedSlow([plan(first_budget=1)]))
        await manager.initialize()
        fast = FastAgent(object(), context, coordinator, _store, slow=slow, mode="dual_async")
        fast.milestones = manager

        if route == "soft_budget":
            manager.record_decision()
            assert await manager.before_decision(remaining=5) is None
        else:
            await fast.seek_help("Repeated local failures", deadline=None)

        assert manager.recoveries == 0
        assert slow.pending.is_set()
        assert len(slow.model.calls) == 1  # No awaited replan/model call.
        assert any(event["kind"] == "pipeline.recovery_requested" for event in _store.events)

    asyncio.run(run())


@pytest.mark.parametrize("route", ["pause", "replan", "explicit_escalation"])
def test_hard_recovery_and_explicit_escalation_wait_for_fresh_plan(route):
    async def run():
        second_plan_entered, release_second_plan = asyncio.Event(), asyncio.Event()

        async def delayed_second_plan(_prompt, projection, _images):
            if projection.get("purpose") == "remaining_work_replan":
                second_plan_entered.set()
                await release_second_plan.wait()
            return plan()

        _store, context, _adapter, coordinator, slow, manager = await make_pipeline(
            ScriptedSlow([plan(), delayed_second_plan]))
        await manager.initialize()
        fast = FastAgent(object(), context, coordinator, _store, slow=slow, mode="dual_async")
        fast.milestones = manager

        async def recover():
            if route == "pause":
                coordinator.pause("Stop and reassess.")
                return await manager.before_decision(remaining=5)
            if route == "replan":
                manager.request_replan("A revised route is needed.", context.milestone)
                return await manager.before_decision(remaining=5)
            return await fast.seek_help("Explicit escalation", deadline=None, blocking=True)

        task = asyncio.create_task(recover())
        await asyncio.wait_for(second_plan_entered.wait(), timeout=1)
        assert not task.done()
        release_second_plan.set()
        outcome = await task

        assert outcome is None
        assert len(slow.model.calls) == 2
        assert manager.recoveries == 1
        assert manager.pending_reason is None
        assert not coordinator.contract_review_pending
        if route == "pause":
            assert not coordinator.paused

        await manager.close()

    asyncio.run(run())


def test_close_cancels_pending_checkpoint_task_and_clears_pipeline_projection():
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def blocking_checkpoint(_prompt, _projection, _images):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        _store, context, _adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan(), blocking_checkpoint]))
        await manager.initialize()
        assert await manager.checkpoint() == "checkpoint_pending"
        await asyncio.wait_for(entered.wait(), timeout=1)
        pending = manager.checkpoint_task

        await manager.close()
        assert cancelled.is_set()
        assert pending.done()
        assert manager.checkpoint_task is None
        assert "pipeline" not in context.local_state

    asyncio.run(run())


def test_pending_checkpoint_skips_redundant_candidate_crop_but_grasp_still_checks_identity():
    async def run():
        checkpoint_entered, release_checkpoint = asyncio.Event(), asyncio.Event()

        async def blocking_checkpoint(_prompt, _projection, _images):
            checkpoint_entered.set()
            await release_checkpoint.wait()
            return checkpoint_reply("unknown")

        store, context, adapter, coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan(), blocking_checkpoint]))
        await manager.initialize()
        assert await manager.checkpoint() == "checkpoint_pending"
        await asyncio.wait_for(checkpoint_entered.wait(), timeout=1)

        class IdentityChecker:
            calls = 0

            async def check(self, _point_id):
                self.calls += 1
                return True

        class CandidateFastModel:
            async def decide(self, _prompt, _projection, _images):
                return {"kind": "tool", "tool": "select_candidate",
                        "args": {"candidate_id": "candidate-1", "part": "top_surface"}}

        candidate_fast = FastAgent(CandidateFastModel(), context, coordinator, store)
        candidate_fast.milestones = manager
        candidate_fast.identity_checker = IdentityChecker()
        assert await candidate_fast.run(1) == "decision_budget"
        assert candidate_fast.identity_checker.calls == 0
        assert adapter.executions == []

        release_checkpoint.set()
        assert await manager.wait_checkpoint() == "continue"
        await manager.close()

        # A distinct Fast episode still performs the mandatory semantic check
        # immediately before grasping the measured point.
        _store2, context2, adapter2, coordinator2, _slow2, manager2 = await make_pipeline(
            ScriptedSlow([plan()]))
        await manager2.initialize()

        class GraspFastModel:
            async def decide(self, _prompt, _projection, _images):
                return {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "point-1"}}

        grasp_fast = FastAgent(GraspFastModel(), context2, coordinator2, _store2)
        grasp_fast.milestones = manager2
        grasp_fast.identity_checker = IdentityChecker()
        assert await grasp_fast.run(1) == "decision_budget"
        assert grasp_fast.identity_checker.calls == 1
        assert adapter2.executions[0][0] == "grasp_point"
        await manager2.close()

    asyncio.run(run())


def test_plan_rationale_is_optional_non_authoritative_and_unknown_action_fields_are_rejected():
    async def run():
        proposed = plan()
        proposed["reason"] = "Use the normal outcome stages; this explanation grants no action authority."
        _store, context, adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([proposed]))

        assert await manager.initialize() == "planned"
        assert context.milestone.goal == proposed["milestones"][0]["goal"]
        assert adapter.executions == []
        assert "reason" not in manager.stages[0]

        for field, value in (
            ("tool", "move_to"),
            ("action", {"tool": "move_to", "args": {"xyz_m": [0.1, 0.2, 0.3]}}),
        ):
            invalid = copy.deepcopy(proposed)
            invalid[field] = value
            with pytest.raises(jsonschema.ValidationError):
                manager.validate_plan(invalid)

    asyncio.run(run())


def test_fast_uses_pending_checkpoint_for_one_scoped_preparation_decision():
    async def run():
        checkpoint_entered = asyncio.Event()

        async def pending_checkpoint(_prompt, _projection, _images):
            checkpoint_entered.set()
            await asyncio.Event().wait()

        store, context, adapter, coordinator, slow, manager = await make_pipeline(
            ScriptedSlow([plan(), pending_checkpoint]))
        await manager.initialize()
        assert await manager.checkpoint("fast_completion_proposal") == "checkpoint_pending"
        await asyncio.wait_for(checkpoint_entered.wait(), timeout=1)

        class PreparationModel:
            def __init__(self):
                self.calls = []
                self.history = []

            async def decide(self, prompt, projection, _images):
                self.calls.append((prompt, copy.deepcopy(projection)))
                # The preview gives the model a concrete next-stage target to
                # inspect. Without that preparation context it repeats a
                # completion proposal, which must not be treated as progress.
                if projection.get("preparation_goal") == plan()["milestones"][1]["goal"]:
                    return {"kind": "tool", "tool": "observe", "args": {},
                            "reason": "inspect the destination for the next stage"}
                return {"kind": "complete", "reason": "repeat the current-stage completion proposal"}

        model = PreparationModel()
        fast = FastAgent(model, context, coordinator, store, slow=slow, mode="dual_async")
        fast.milestones = manager

        try:
            assert await asyncio.wait_for(fast.run(1), timeout=1) == "decision_budget"
            assert len(model.calls) == fast.steps == 1
            assert model.calls[0][1]["preparation_goal"] == plan()["milestones"][1]["goal"]
            assert manager.preparation_used
            assert adapter.inspected.is_set()
            assert adapter.executions == []
            started = [e for e in store.events if e["kind"] == "pipeline.preparation_inference_started"]
            assert len(started) == 1
            assert started[0]["payload"]["next_goal_preview"] == plan()["milestones"][1]["goal"]
        finally:
            await manager.close()

    asyncio.run(run())


def test_close_cancels_pending_initial_plan_and_scout_tasks():
    async def run():
        plan_entered, scout_entered = asyncio.Event(), asyncio.Event()

        async def blocking_plan(_prompt, _projection, _images):
            plan_entered.set()
            await asyncio.Event().wait()

        class BlockingScout:
            async def decide(self, _prompt, _projection, _images):
                scout_entered.set()
                await asyncio.Event().wait()

        _store, _context, _adapter, _coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([blocking_plan]))
        initializing = asyncio.create_task(manager.initialize_with_scout(BlockingScout()))
        await asyncio.wait_for(plan_entered.wait(), timeout=1)
        await asyncio.wait_for(scout_entered.wait(), timeout=1)
        initial_task, scout_task = manager.initial_task, manager.scout_task

        await manager.close()
        assert initial_task.done() and initial_task.cancelled()
        assert scout_task.done() and scout_task.cancelled()
        with pytest.raises(asyncio.CancelledError):
            await initializing

    asyncio.run(run())


def test_failed_hard_replan_keeps_contract_gate_closed_and_returns_unavailable():
    async def run():
        invalid = {"kind": "not_a_plan"}
        store, context, adapter, coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan(), invalid, invalid]))
        await manager.initialize()
        assert manager.request_replan("A hard contract conflict needs a revised plan.",
                                      context.milestone)

        class NeverCalledFast:
            calls = 0

            async def decide(self, _prompt, _projection, _images):
                self.calls += 1
                raise AssertionError("Fast must stop when hard supervisor recovery is unavailable")

        fast_model = NeverCalledFast()
        fast = FastAgent(fast_model, context, coordinator, store)
        fast.milestones = manager
        assert await fast.run(1) == "supervisor_unavailable"
        assert fast_model.calls == 0
        assert coordinator.contract_review_pending is True
        assert manager.pending_reason is not None
        assert adapter.executions == []
        assert any(event["kind"] == "pipeline.hard_recovery_unavailable" for event in store.events)

    asyncio.run(run())


def test_hard_replan_arriving_during_soft_recovery_stays_gated_if_fresh_plan_fails():
    async def run():
        replan_entered, release_replan = asyncio.Event(), asyncio.Event()

        async def delayed_invalid_plan(_prompt, projection, _images):
            if projection.get("purpose") == "remaining_work_replan":
                replan_entered.set()
                await release_replan.wait()
            return {"kind": "not_a_plan"}

        store, context, adapter, coordinator, _slow, manager = await make_pipeline(
            ScriptedSlow([plan(), delayed_invalid_plan, {"kind": "not_a_plan"}]))
        await manager.initialize()

        recovery = asyncio.create_task(manager.recover("A local review is waiting on Slow."))
        await asyncio.wait_for(replan_entered.wait(), timeout=1)
        assert not recovery.done()
        assert manager.request_replan("A hard contract conflict arrived during the review.",
                                      context.milestone)
        release_replan.set()

        assert await recovery == "supervisor_unavailable"
        assert coordinator.contract_review_pending is True
        assert manager.pending_reason is not None
        assert adapter.executions == []
        assert any(event["kind"] == "pipeline.hard_recovery_unavailable" for event in store.events)

    asyncio.run(run())


@pytest.mark.parametrize("mode,backend,hierarchy", [
    ("dual_sync", "direct_libero", "milestones"),
    ("dual_async", "eta", "milestones"),
    ("dual_async", "direct_libero", "legacy_review"),
])
def test_pipeline_app_rejects_unsupported_runtime_combinations(mode, backend, hierarchy):
    async def run():
        from embodied_harness.app import episode

        with pytest.raises(ValueError, match="pipeline_supervision"):
            await episode(SimpleNamespace(pipeline_supervision=True, mode=mode,
                                          backend=backend, hierarchy=hierarchy))

    asyncio.run(run())
