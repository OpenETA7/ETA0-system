"""Offline regressions for retired preparation measurements and fresh references."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_harness.adapters.libero_direct import LiberoHost
from embodied_harness.adapters.openeta import OpenETAAdapter
from embodied_harness.adapters.skills import GRASP_SCHEMA
from embodied_harness.domain.models import Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.pipeline import PipelinedMilestoneManager


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})


def measured_point(point_id, packet="direct-1123", pixel=(358, 308)):
    return {
        "point_id": point_id,
        "source_packet_id": packet,
        "camera_frame_id": "agentview",
        "pixel_xy": list(pixel),
        "xyz_m": [0.2, 0.3, 0.4],
        "generation": 15,
        "measurement": {"depth_edge": False},
    }


def envelope(body):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(body))])


def openeta_adapter(tmp_path):
    """Create only the synchronous, sensor-decoding portion of the adapter."""
    adapter = object.__new__(OpenETAAdapter)
    adapter.output = tmp_path
    adapter.images = tmp_path / "frames"
    adapter.images.mkdir(parents=True)
    adapter.generation = 1
    adapter.version = 0
    adapter.action_seq = 0
    adapter.last = None
    adapter.cached = None
    adapter.points = {}
    adapter.retired_point_ids = set()
    adapter.schemas = {"grasp_point": GRASP_SCHEMA}
    adapter.possible_hold = False
    adapter.cancellation = "safe_boundary_only"
    return adapter


def test_retired_prep_ids_stay_retired_when_old_feedback_is_replayed(tmp_path):
    adapter = openeta_adapter(tmp_path)
    current = measured_point("current-physical-point", "direct-1140")
    preview = measured_point("point-18", "direct-1123")
    adapter.points = {current["point_id"]: current, preview["point_id"]: preview}

    manager = object.__new__(PipelinedMilestoneManager)
    manager.checkpoint_task = object()
    manager.preparation_used = True
    manager.preparation_points = {preview["point_id"]}
    manager.preparation_goal = "place the cup in the bowl"
    manager.context = SimpleNamespace(
        milestone=SimpleNamespace(goal="pick up the cup"),
        local_state={},
    )
    manager.coordinator = SimpleNamespace(adapter=adapter)
    manager.fast_model = None
    manager.store = Store()

    manager.finish_preparation("continue")

    assert adapter.retired_point_ids == {"point-18"}
    assert "point-18" not in adapter.points
    assert "current-physical-point" in adapter.points

    # Direct LIBERO observation packaging echoes the last measurement in feedback.
    # Re-ingesting that raw receipt must not restore its executable cache binding.
    old_feedback_observation = adapter._unpack(envelope({
        "feedback": {"point": preview},
        "views": [{"source_packet_id": "direct-1123", "camera_frame_id": "agentview"}],
    }))
    assert "point-18" not in adapter.points
    assert old_feedback_observation.state["feedback"]["point"] == preview

    context = Context(manager.store)
    context.latest = old_feedback_observation
    context.local_state.update(manager.context.local_state)
    projected = context.projection("fast")["observation"]["state"]["feedback"]
    assert "point" not in projected
    assert projected["retired_measurement"]["point_id"] == "point-18"
    assert projected["retired_measurement"]["reason"] == "unapproved_next_goal_preparation"
    assert context.latest.state["feedback"]["point"] == preview


@pytest.mark.parametrize(("tool", "args"), [
    ("grasp_point", {"point_id": "point-18"}),
    ("grasp_point", {"point_id": "current-physical-point", "object_center_point_id": "point-18"}),
    ("place_point", {"point_id": "point-18"}),
    ("pull_handle", {"point_id": "point-18"}),
    ("move_to", {"point_id": "point-18"}),
    ("move_to", {"contact_point_id": "point-18"}),
    ("gripper_control", {"contact_point_id": "point-18"}),
])
def test_every_tool_point_reference_rejects_retired_id_even_if_cache_is_repopulated(
        tmp_path, tool, args):
    adapter = openeta_adapter(tmp_path)
    adapter.retire_points({"point-18"})
    # Simulate a stale feedback path or accidental cache write after retirement.
    adapter.points["point-18"] = measured_point("point-18")
    adapter.points["current-physical-point"] = measured_point("current-physical-point")
    adapter.schemas.update({
        "place_point": {"parameters": {"type": "object", "properties": {
            "point_id": {"type": "string"}, "clearance_m": {"type": "number"}},
            "additionalProperties": True}},
        "pull_handle": {"parameters": {"type": "object", "properties": {
            "point_id": {"type": "string"}}, "additionalProperties": True}},
        "move_to": {"parameters": {"type": "object", "properties": {
            "point_id": {"type": "string"}, "contact_point_id": {"type": "string"}},
            "additionalProperties": True}},
        "gripper_control": {"parameters": {"type": "object", "properties": {
            "contact_point_id": {"type": "string"}}, "additionalProperties": True}},
    })
    if "object_center_point_id" in args:
        adapter.last = {"views": [{"source_packet_id": "direct-1123", "camera_frame_id": "agentview"}]}

    with pytest.raises(ValueError, match="(?i)retired"):
        adapter.validate_args(tool, args)


def direct_host_for_mark_point(tmp_path, monkeypatch):
    host = object.__new__(LiberoHost)
    host.packet = 1147
    host.generation = 16
    host.points = {}  # The retired cache entry has already been removed.
    host._point_seq = 18
    host.obs = {"agentview_depth": np.full((512, 512), 0.5, dtype=float)}
    host.sim = SimpleNamespace(
        model=SimpleNamespace(
            stat=SimpleNamespace(extent=1.0),
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.1, zfar=10.0)),
            camera_name2id=lambda _name: 0,
            cam_fovy=[60.0],
        ),
        data=SimpleNamespace(
            cam_xmat=[np.eye(3).reshape(-1)],
            cam_xpos=[np.zeros(3)],
        ),
    )
    host.reference_robot_position = np.array([0.0, 0.0, 1.0])
    host.execution_accounting = {
        "control_steps": 0,
        "physical_dispatches": 0,
        "physical_calls_with_steps": 0,
        "physical_unknown_outcomes": 0,
    }
    host.feedback = {}
    host.candidate_cache = None
    host.proposal_mode = "objects"
    host.step_failed = False
    host.cancelled = lambda: False
    host.on_sample = None
    host.trace = tmp_path / "direct-commands.jsonl"

    def result(**_kwargs):
        body = {
            "feedback": host.feedback,
            "views": [{"source_packet_id": f"direct-{host.packet}",
                       "camera_frame_id": "agentview"}],
        }
        return envelope(body)

    host.result = result
    monkeypatch.setattr("embodied_harness.adapters.libero_direct.support_height", lambda *a: None)
    monkeypatch.setattr("embodied_harness.adapters.libero_direct.unproject", lambda *a: np.array([0.2, 0.3, 0.4]))
    monkeypatch.setattr("embodied_harness.adapters.libero_direct.visible_normal", lambda *a: None)
    monkeypatch.setattr("embodied_harness.adapters.libero_direct.horizontal_bar_outward", lambda *a: None)
    return host


def test_physical_remeasurement_gets_a_new_monotonic_id_and_is_usable(tmp_path, monkeypatch):
    host = direct_host_for_mark_point(tmp_path, monkeypatch)
    raw = host.call("mark_point", {
        "source_packet_id": "direct-1147",
        "camera_frame_id": "agentview",
        "x": 358,
        "y": 308,
    })
    fresh_point = json.loads(raw.content[0].text)["feedback"]["point"]

    assert fresh_point["point_id"] == "point-19"
    assert fresh_point["point_id"] != "point-18"

    adapter = openeta_adapter(tmp_path / "adapter")
    adapter.retire_points({"point-18"})
    adapter._unpack(raw)
    adapter.validate_args("grasp_point", {"point_id": "point-19"})
    assert "point-18" in adapter.retired_point_ids
    assert "point-19" in adapter.points


def test_cut_in_sensor_refresh_is_still_not_a_qualified_readonly_bridge():
    from embodied_harness.domain.models import Observation

    manager = object.__new__(PipelinedMilestoneManager)
    manager.readonly_parents = {}
    before = Observation(
        "before", 1, 1, time.monotonic(), 0.0, 0,
        {"sensor": {"sim_time": 1.0, "control_steps": 10},
         "robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}, "tactile": {}},
    )
    after = Observation(
        "cut-in", 2, 1, time.monotonic(), 0.0, 0,
        {"sensor": {"sim_time": 1.1, "control_steps": 11},
         "robot": {"grip_xyz_m": [0.0, 0.0, 0.0]}, "tactile": {}},
    )
    manager.context = SimpleNamespace(latest=after, action_seq=0, control_version=0,
                                       milestone=None)
    manager.coordinator = SimpleNamespace(
        adapter=SimpleNamespace(generation=1, fence=0),
        pause_epoch=0,
        contract_review_epoch=0,
    )
    stamp = (None, None, before.id, 0, 1, 0, 0)

    manager.note_readonly(before, after)

    assert manager.readonly_parents == {}
    assert not manager.matches_stamp(stamp)
