"""Exercise preflight recovery through the actor loop, without simulator physics."""
import asyncio
from collections import deque

from test_round20_regressions import MemoryStore, FakeAdapter, QueuedModel, MISSION
from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.recovery import ProgressRecovery


def test_reliable_interior_sample_is_allowed_inside_coarse_failure_bucket():
    recovery = ProgressRecovery()
    points = {"edge": {"xyz_m": [.1, .2, .3], "measurement": {"depth_edge": True}},
              "solid": {"xyz_m": [.102, .2, .3], "measurement": {"depth_edge": False}}}
    recovery.record_error("grasp_point", {"point_id": "edge"}, points, "Depth discontinuity")
    assert recovery.blocked("grasp_point", {"point_id": "edge"}, points)
    assert recovery.blocked("grasp_point", {"point_id": "solid"}, points) is None


def test_unreliable_depth_is_blocked_before_repeat_validation_but_new_site_executes():
    async def run():
        class Adapter(FakeAdapter):
            def __init__(self):
                super().__init__()
                self.points = {"edge": {"xyz_m": [0., 0., 0.]},
                               "renamed_edge": {"xyz_m": [0., 0., 0.]},
                               "interior": {"xyz_m": [.1, 0., 0.]}}
                self.validations = []
                self.executions = []

            def validate_args(self, tool, args):
                self.validations.append(args["point_id"])
                if args["point_id"] in {"edge", "renamed_edge"}:
                    raise ValueError("Depth discontinuity; remeasure an interior surface")

            async def execute(self, tool, args, fence, cancelled, publish):
                self.executions.append(args["point_id"])
                return {"macro_status": "lift_completed_verify_retention_visually"}

        store = MemoryStore()
        context = Context(store)
        adapter = Adapter()
        coordinator = Coordinator(adapter, context, store)
        coordinator.authorize_mission(MISSION, capabilities=("grasp_point",))
        coordinator.delegate(Milestone("stage", 1, MISSION, MISSION, capabilities=("grasp_point",)))
        await coordinator.observe()
        model = QueuedModel([{"kind": "tool", "tool": "grasp_point", "args": {"point_id": p}}
                             for p in ("edge", "renamed_edge", "interior")])
        model.history = deque([{"role": "assistant", "content": "repeat edge"}])
        actor = FastAgent(model, context, coordinator, store, mode="single_fast", progress_recovery=True)
        assert await actor.run(3) == "decision_budget"
        assert actor.steps == 3 and actor.errors == 0
        assert "renamed_edge" not in adapter.validations
        assert adapter.executions == ["interior"]
        assert any(e["kind"] == "recovery.equivalent_attempt_blocked" for e in store.events)
        assert not model.history
    asyncio.run(run())


def test_unclassified_schema_failure_still_hits_hard_error_limit():
    async def run():
        class Adapter(FakeAdapter):
            def validate_args(self, tool, args):
                raise ValueError("malformed physical command")
        store = MemoryStore()
        context = Context(store)
        coordinator = Coordinator(Adapter(), context, store)
        coordinator.authorize_mission(MISSION, capabilities=("grasp_point",))
        coordinator.delegate(Milestone("stage", 1, MISSION, MISSION, capabilities=("grasp_point",)))
        await coordinator.observe()
        model = QueuedModel([{"kind": "tool", "tool": "grasp_point", "args": {"point_id": "x"}}] * 7)
        actor = FastAgent(model, context, coordinator, store, mode="single_fast", progress_recovery=True)
        assert await actor.run(7) == "model_or_tool_errors"
        assert actor.steps == actor.errors == 6
    asyncio.run(run())
