import pytest
from embodied_harness.adapters.gripper_state import possible_hold_after


@pytest.mark.parametrize("motion,error,openness", [
    ({}, {"code": "rejected"}, 1.0),
    ({"execution_state": "not_started", "physics_executed": False}, None, 1.0),
    ({"execution_state": "partial", "physics_executed": True}, None, 1.0),
    ({"execution_state": "completed", "physics_executed": True}, None, .7),
    ({"execution_state": "completed", "physics_executed": True}, None, None),
    ({"execution_state": "unknown"}, {"code": "host_execution_error"}, 1.0),
])
def test_incomplete_open_never_clears_possible_load(motion, error, openness):
    assert possible_hold_after(True, "open", {"feedback": {"motion": motion}, "error": error,
                                             "robot": {"gripper": {"openness": openness}}})


def test_only_measured_completed_release_clears_load():
    assert not possible_hold_after(True, "open", {"feedback": {"motion": {
        "execution_state": "completed", "physics_executed": True}}, "robot": {"gripper": {"openness": .96}}})


def test_rejected_close_does_not_invent_held_load_but_unknown_close_does():
    assert not possible_hold_after(False, "close", {"feedback": {"motion": {
        "execution_state": "not_started", "physics_executed": False}}})
    assert possible_hold_after(False, "close", {"feedback": {"motion": {"execution_state": "unknown"}}})
