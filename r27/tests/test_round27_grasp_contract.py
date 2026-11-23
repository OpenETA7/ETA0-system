"""Orientation errors stop before physics and survive read-only observations."""
import asyncio
import copy

import pytest

from embodied_harness.adapters.skills import grasp_point, validate_grasp, validate_grasp_orientation
from embodied_harness.domain.tool_errors import GraspOrientationError, orientation_receipt_error
from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, ActionRequest
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator, GateError
from embodied_harness.runtime.phase_feedback import grasp_phase_evidence
from test_grasp_skill import Adapter
from test_round20_regressions import FakeAdapter, MemoryStore, QueuedModel
from test_round22m_phase_feedback import phase_step, context_with_phase_feedback, receipt_context


@pytest.mark.parametrize("axes,code", [
    ({"jaw_world": [1, 0, 0]}, "grasp_orientation_pair_required"),
    ({"approach_world": [0, 0, -1]}, "grasp_orientation_pair_required"),
    ({"approach_world": [1, 0, 0], "jaw_world": [-1, 0, 0]}, "grasp_orientation_collinear"),
    ({"approach_world": [-.678, .735, 0], "jaw_world": [.678, -.735, 0]}, "grasp_orientation_collinear"),
    ({"approach_world": [0, 0, -1], "jaw_world": [0, 1e-10, 1]}, "grasp_orientation_collinear"),
    ({"approach_world": [0, 0, 0], "jaw_world": [1, 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": [0, 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": [0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": [True, 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": ["1", 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": [float("nan"), 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, float("inf")], "jaw_world": [1, 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": [10**400, 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": [0, 0, -1], "jaw_world": [1e200, 0, 0]}, "grasp_orientation_invalid"),
    ({"approach_world": None, "jaw_world": [1, 0, 0]}, "grasp_orientation_invalid"),
])
def test_invalid_axes_never_open_remeasure_or_dispatch(axes, code):
    adapter = Adapter()
    args = {"point_id": "p0", **axes}
    with pytest.raises(GraspOrientationError) as exc:
        validate_grasp(adapter, args)
    assert exc.value.code == code
    with pytest.raises(GraspOrientationError):
        asyncio.run(grasp_point(adapter, args, lambda: False, lambda _: None))
    assert adapter.calls == []
    assert adapter.possible_hold is False


@pytest.mark.parametrize("axes", [
    {}, {"approach_world": [0, 0, -1], "jaw_world": [1, 0, 0]},
    {"approach_world": [0, 0, -2], "jaw_world": [2, 0, .4]},
    {"approach_world": [0, 0, 1], "jaw_world": [1, 0, 1e9]},
    # A horizontal approach can be deliberate: the host must not silently fix it.
    {"approach_world": [-.992, .122, 0], "jaw_world": [0, 1, 0]},
])
def test_valid_axes_keep_original_actor_choice_and_macro_commands(axes):
    adapter = Adapter()
    args = {"point_id": "p0", **copy.deepcopy(axes)}
    before = copy.deepcopy(args)
    validate_grasp(adapter, args)
    result = asyncio.run(grasp_point(adapter, args, lambda: False, lambda _: None))
    assert result["macro_status"] == "lift_completed_verify_retention_visually"
    assert args == before
    for tool, parameters in adapter.calls:
        if tool == "move_to" and "point_id" in parameters:
            for field, value in axes.items():
                assert parameters[field] == value


def test_measured_default_still_uses_downward_approach_and_rim_jaw():
    adapter = Adapter()
    jaw = [-.9924799183, .1224075641, 0]
    adapter.points["p0"]["visual_candidate"] = {"suggested_jaw_world": jaw}
    result = asyncio.run(grasp_point(adapter, {"point_id": "p0"}, lambda: False, lambda _: None))
    assert result["macro_status"] == "lift_completed_verify_retention_visually"
    approach = adapter.calls[2][1]
    assert approach["approach_world"] == [0, 0, -1]
    assert approach["jaw_world"] == jaw


class ExecutionAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.schemas["observe"] = {"parameters": {"properties": {}}}
        self.executions = 0
        self.points = {"p": {"xyz_m": [0, 0, 0]}}

    def validate_args(self, tool, args):
        if tool == "grasp_point":
            validate_grasp_orientation(args)

    async def inspect(self, _tool, _args, action_seq):
        return await self.observe(action_seq)

    async def execute(self, *_args):
        self.executions += 1
        return {"macro_status": "lift_completed_verify_retention_visually"}


async def setup_actor(responses):
    store = MemoryStore()
    context = Context(store)
    adapter = ExecutionAdapter()
    coordinator = Coordinator(adapter, context, store)
    coordinator.delegate(Milestone("m", 1, "pick bowl", "pick bowl", capabilities=("grasp_point",)))
    await coordinator.observe()
    model = QueuedModel(responses)
    actor = FastAgent(model, context, coordinator, store, progress_recovery=True)
    return store, context, adapter, coordinator, model, actor


BAD = {"kind": "tool", "tool": "grasp_point", "args": {
    "point_id": "p", "approach_world": [-.678, .735, 0], "jaw_world": [.678, -.735, 0]}}


def test_actor_persists_rejection_through_new_observation_then_can_execute_valid_choice():
    async def run():
        store, context, adapter, coordinator, model, actor = await setup_actor([
            BAD, {"kind": "tool", "tool": "observe", "args": {}},
            {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "p"}},
            {"kind": "tool", "tool": "observe", "args": {}}])
        assert await actor.run(4) == "decision_budget"
        assert adapter.executions == 1 and actor.steps == 4 and actor.errors == 1
        before = model.calls[1]["projection"]["local_state"]["last_tool_argument_error"]
        after = model.calls[2]["projection"]["local_state"]["last_tool_argument_error"]
        assert before == after
        assert before["code"] == "grasp_orientation_collinear"
        assert before["physical_action_dispatched"] is False and before["action_seq"] == 0
        assert before["observation_id"] != model.calls[2]["projection"]["observation"]["id"]
        assert "repair" in before and "not evidence of collision" in before["summary"]
        assert "repair" not in context.projection("slow")["local_state"]["last_tool_argument_error"]
        # Executor-specific field instructions do not leak via generic error text.
        assert "approach_world" not in repr(context.projection("slow"))
        assert sum(e["kind"] == "action.accepted" for e in store.events) == 1
        assert context.action_seq == 1 and not coordinator.unknown
    asyncio.run(run())


def test_invalid_parameters_keep_original_six_error_limit_and_no_hidden_retry():
    async def run():
        store, context, adapter, _, model, actor = await setup_actor([BAD] * 8)
        assert await actor.run(8) == "model_or_tool_errors"
        assert actor.errors == actor.steps == len(model.calls) == 6
        assert adapter.executions == context.action_seq == 0
        assert not any(e["kind"] == "action.accepted" for e in store.events)
    asyncio.run(run())


def test_coordinator_preflight_rejects_without_allocating_action_or_fence():
    async def run():
        store, context, adapter, coordinator, _, _ = await setup_actor([])
        req = ActionRequest("bad", "m", 1, 1, context.latest.id, "grasp_point", BAD["args"])
        with pytest.raises(GateError, match="invalid_tool_arguments"):
            await coordinator.submit(req)
        assert adapter.executions == coordinator.fence == context.action_seq == 0
        assert not coordinator.receipts and not coordinator.unknown
    asyncio.run(run())


def test_old_primitive_error_keeps_specific_cause_without_erasing_earlier_physics():
    steps = [phase_step("open"), phase_step("remeasure"),
             phase_step("approach", state="not_started", physics=False)]
    steps[-1]["error"] = {"code": "direct_tool_rejected", "message": "Collinear orientation basis"}
    detail = {"skill": "grasp_point", "macro_status": "approach_failed", "feedback": {"substeps": steps}}
    original = copy.deepcopy(detail)
    evidence = grasp_phase_evidence(detail)
    assert evidence["phases"][-1]["error_cause"] == "grasp_orientation_collinear"
    assert "earlier macro phases may have executed" in evidence["interpretation"]
    assert evidence["close_attempted"] is False
    _, context = context_with_phase_feedback(True)
    receipt_context(context, detail)
    for _ in range(15):
        context.remember("readonly.result", {"point": "new"})
    fast = context.projection("fast")["local_state"]["last_action"]["feedback"]["grasp_phase_evidence"]
    slow = context.projection("slow")["local_state"]["last_action"]["feedback"]["grasp_phase_evidence"]
    assert fast["phases"][-1]["error_cause"] == "grasp_orientation_collinear"
    assert "repair" in fast and "repair" not in slow
    assert detail == original


@pytest.mark.parametrize("error", [None, {}, {"code": "direct_tool_rejected", "message": []},
    {"code": "host_execution_error", "message": "Collinear orientation basis"},
    {"code": "direct_tool_rejected", "message": "private free-form server message"}])
def test_unknown_or_malformed_host_errors_are_not_reinterpreted(error):
    assert orientation_receipt_error(error) is None
