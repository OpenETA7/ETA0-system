import asyncio
from types import SimpleNamespace
import pytest
from embodied_harness.adapters.skills import grasp_point, validate_grasp


def point(id="p0", source="obs0", x=0.1):
    return {"point_id": id, "source_packet_id": source, "camera_frame_id": "agentview",
            "pixel_xy": [220, 300], "xyz_m": [x, 0.2, 0.9], "measurement": {"depth_edge": False}}


class Adapter:
    def __init__(self, bilateral=True, fail=None, moved=False, cancel_after=None, contact_residual=None):
        self.points = {"p0": point()}
        self.possible_hold = False
        self.last = {"views": [{"source_packet_id": "obs0", "camera_frame_id": "agentview"}]}
        self.cached = SimpleNamespace(id="obs0")
        self.calls = []
        self.bilateral, self.fail, self.moved, self.cancel_after = bilateral, fail, moved, cancel_after
        self.contact_residual = contact_residual
        self.host = SimpleNamespace(call=self.call)

    def call(self, name, args):
        self.calls.append((name, args))
        feedback = {"motion": {"execution_state": "reached"}}
        error = None
        if name == "gripper_control":
            feedback = {"motion": {"execution_state": "completed", "physics_executed": True, "motion_summary": {
                "gripper_contact": {"contact_pattern": "bilateral_pads" if self.bilateral else "no_contact"}}}}
        elif name == "mark_point":
            feedback = {"point": point("p1", "obs1", 0.2 if self.moved else 0.1)}
        elif name == "move_to" and "contact_point_id" in args and self.contact_residual is not None:
            feedback = {"motion": {"execution_state": "partial", "physics_executed": True,
                "motion_summary": {"position_error_m": self.contact_residual,
                                   "orientation_error_rad": 0.05}}}
        if len(self.calls) == self.fail:
            error = {"code": "host_execution_error"}
            feedback = {"motion": {"execution_state": "unknown"}}
        return {"views": [{"source_packet_id": "obs1", "camera_frame_id": "agentview"}],
                "robot": {"gripper": {"openness": 1.0}},
                "contact_state": {"close_binding_available": name == "move_to" and "contact_point_id" in args},
                "feedback": feedback, "error": error}

    async def _thread(self, fn, *args):
        return fn(*args)

    def _unpack(self, result):
        self.last = result
        self.cached = SimpleNamespace(id=f"frame{len(self.calls)}")
        return self.cached

    def cancelled(self):
        return self.cancel_after is not None and len(self.calls) >= self.cancel_after


def run(adapter):
    published = []
    result = asyncio.run(grasp_point(adapter, {"point_id": "p0"}, adapter.cancelled, published.append))
    assert len(published) == len(adapter.calls)
    return result


def test_skill_remeasures_after_open_and_authorizes_contact():
    a = Adapter()
    result = run(a)
    assert result["macro_status"] == "lift_completed_verify_retention_visually"
    assert result["feedback"]["retention_proven"] is False
    assert [name for name, _ in a.calls] == ["gripper_control", "mark_point", "move_to", "move_to", "gripper_control", "move_to"]
    assert a.calls[1][1]["source_packet_id"] == "obs1"
    assert a.calls[3][1]["contact_point_id"] == "p1"
    assert a.calls[4][1] == {"action": "close"}
    assert a.calls[-1][1] == {"delta_m": [0, 0, 0.12], "delta_frame": "world"}


def test_no_lift_without_bilateral_contact():
    a = Adapter(bilateral=False)
    assert run(a)["macro_status"] == "grasp_contact_not_established"
    assert len(a.calls) == 5
    assert a.possible_hold


def test_stop_at_confirmed_primitive_boundary():
    a = Adapter(cancel_after=3)
    result = run(a)
    assert result["stop_confirmed"]
    assert result["macro_status"] == "cancelled_before_contact"
    assert len(a.calls) == 3


def test_unknown_primitive_blocks_skill_remainder():
    a = Adapter(fail=3)
    result = run(a)
    assert result["state_unknown"]
    assert len(a.calls) == 3
    assert not result["stop_confirmed"]


def test_changed_surface_requires_reobservation():
    a = Adapter(moved=True)
    assert run(a)["macro_status"] == "surface_changed_reobserve"
    assert len(a.calls) == 2


def test_skill_rejects_old_point_or_possible_held_object():
    a = Adapter()
    validate_grasp(a, {"point_id": "p0"})
    a.possible_hold = True
    with pytest.raises(ValueError, match="Possible held"):
        validate_grasp(a, {"point_id": "p0"})
    a.possible_hold = False
    a.last["views"][0]["source_packet_id"] = "other"
    with pytest.raises(ValueError, match="old observation"):
        validate_grasp(a, {"point_id": "p0"})


def test_incomplete_aperture_stops_before_any_approach():
    a = Adapter()
    original = a.host.call
    def partial_open(name, args):
        result = original(name, args)
        result["robot"]["gripper"]["openness"] = .7
        return result
    a.host.call = partial_open
    assert run(a)["macro_status"] == "open_not_measured"
    assert len(a.calls) == 1


def test_missing_contact_binding_stops_before_closing():
    a = Adapter()
    original = a.host.call
    def no_binding(name, args):
        result = original(name, args)
        result["contact_state"]["close_binding_available"] = False
        return result
    a.host.call = no_binding
    assert run(a)["macro_status"] == "close_binding_unavailable"
    assert len(a.calls) == 4


@pytest.mark.parametrize("residual", [0.010, 0.014, 0.018])
def test_bounded_close_probe_accepts_small_contact_residual_but_requires_bilateral_contact(residual):
    a = Adapter(bilateral=True, contact_residual=residual)
    result = run(a)

    assert result["macro_status"] == "lift_completed_verify_retention_visually"
    contact_move = next(step for step in result["feedback"]["substeps"]
                        if step["tool"] == "move_to" and "contact_point_id" in step["args"])
    assert contact_move["feedback"]["motion"]["execution_state"] == "partial"
    assert contact_move["bounded_close_probe"]["allowed"] is True
    assert contact_move["bounded_close_probe"]["max_residual_m"] == pytest.approx(0.018)
    assert a.calls[-1][0] == "move_to" and "delta_m" in a.calls[-1][1]


def test_bounded_close_probe_never_lifts_without_bilateral_contact():
    a = Adapter(bilateral=False, contact_residual=0.014)

    result = run(a)

    assert result["macro_status"] == "grasp_contact_not_established"
    assert a.calls[-1] == ("gripper_control", {"action": "close"})
    assert not any(name == "move_to" and "delta_m" in args for name, args in a.calls)


def test_contact_residual_above_bounded_probe_tolerance_stops_before_close():
    a = Adapter(bilateral=True, contact_residual=0.0181)

    result = run(a)

    assert result["macro_status"] == "contact_failed"
    assert len(a.calls) == 4
    assert not any(name == "gripper_control" and args.get("action") == "close" for name, args in a.calls)
