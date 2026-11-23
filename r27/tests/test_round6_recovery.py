"""Regression checks for contact-aware placement and dual_async startup."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_harness.adapters.libero_direct import DirectLiberoAdapter, LiberoHost, horizontal_bar_outward
from embodied_harness.adapters.placement import PLACE_SCHEMA, place_point
from embodied_harness.adapters.skills import GRASP_SCHEMA
from embodied_harness.adapters import vision_geometry
from embodied_harness.adapters.vision_geometry import estimate_placement_opening, propose_horizontal_bars
from test_grasp_skill import Adapter as PrimitiveAdapter
from test_grasp_skill import point


def make_placement_adapter():
    adapter = PrimitiveAdapter()
    adapter.possible_hold = True
    adapter.held_center_in_grip = None
    adapter.schemas = {
        "move_to": {"parameters": {"properties": {"maintain_grip_contact": {}}}}
    }
    adapter.last["robot"] = {"grip_xyz_m": [0.0, 0.0, 1.2]}
    adapter.last["tactile"] = {
        "left_fingerpad_contact": True,
        "right_fingerpad_contact": True,
    }
    return adapter


def test_direct_place_preflight_rejects_initially_missing_fingerpad_contact():
    """The direct adapter must reject a place request before it dispatches motion."""
    adapter = object.__new__(DirectLiberoAdapter)
    adapter.schemas = {"place_point": PLACE_SCHEMA}
    adapter.possible_hold = True
    adapter.retired_point_ids = set()
    adapter.points = {"p0": point()}
    adapter.last = {
        "views": [{"source_packet_id": "obs0", "camera_frame_id": "agentview"}],
        "tactile": {
            "left_fingerpad_contact": False,
            "right_fingerpad_contact": False,
        },
    }

    with pytest.raises(ValueError, match="Current fingerpad contact is absent"):
        adapter.validate_args("place_point", {"point_id": "p0"})


def test_inferred_placement_opening_is_rejected_as_a_grasp_surface():
    adapter = object.__new__(DirectLiberoAdapter)
    adapter.schemas = {"grasp_point": GRASP_SCHEMA}
    adapter.possible_hold = False
    adapter.retired_point_ids = set()
    inferred = {
        **point(),
        "placement_anchor": "estimated_visible_region_opening",
        "measurement": {
            "depth_edge": False,
            "provenance": "inferred_from_visible_component_3d_extent_and_upper_edge",
            "not_a_measured_floor": True,
        },
    }
    adapter.points = {inferred["point_id"]: inferred}
    adapter.last = {"views": [{"source_packet_id": "obs0", "camera_frame_id": "agentview"}]}

    with pytest.raises(ValueError, match="cannot be used as a measured grasp surface"):
        adapter.validate_args("grasp_point", {"point_id": inferred["point_id"]})


def test_rotated_container_opening_uses_visible_3d_footprint_not_bbox_center():
    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    angle = np.deg2rad(31.0)
    dx, dy = xx - 64, yy - 64
    along = np.cos(angle) * dx + np.sin(angle) * dy
    across = -np.sin(angle) * dx + np.cos(angle) * dy
    outer = (along / 35.0) ** 2 + (across / 20.0) ** 2 <= 1.0
    inner = (along / 25.0) ** 2 + (across / 11.0) ** 2 < 1.0
    ring = outer & ~inner

    # The visible rim surrounds an occluded opening: the image-box center is
    # background at table height, while the component's upper edge is visible.
    cloud = np.empty((height, width, 3), dtype=float)
    cloud[..., 0] = 0.4 + 0.004 * (np.cos(angle) * dx - np.sin(angle) * dy)
    cloud[..., 1] = -0.2 + 0.004 * (np.sin(angle) * dx + np.cos(angle) * dy)
    cloud[..., 2] = 0.1
    cloud[..., 2][ring] = 0.34 + 0.04 * along[ring] / 35.0
    labels = np.zeros((height, width), dtype=int)
    labels[ring] = 1

    ys, xs = np.nonzero(ring)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    center_pixel = [(bbox[0] + bbox[2] - 1) // 2, (bbox[1] + bbox[3] - 1) // 2]
    assert labels[center_pixel[1], center_pixel[0]] == 0
    candidate = {"component": 1, "bbox_xyxy": bbox, "center_pixel_xy": center_pixel,
                 "center_xyz_m": cloud[center_pixel[1], center_pixel[0]].tolist()}

    estimate = estimate_placement_opening(candidate, labels, cloud)
    alternate_candidate = {**candidate, "bbox_xyxy": [0, 0, width, height],
                           "center_pixel_xy": [8, 8], "center_xyz_m": [9.0, 9.0, 9.0]}
    alternate_estimate = estimate_placement_opening(alternate_candidate, labels, cloud)

    assert estimate == pytest.approx(alternate_estimate)
    assert estimate[2] > candidate["center_xyz_m"][2] + 0.2
    assert estimate[:2] == pytest.approx([0.4, -0.2], abs=0.025)


def test_oversized_merged_component_is_rejected_for_placement_estimation():
    yy, xx = np.mgrid[:80, :80]
    labels = np.ones((80, 80), dtype=int)
    cloud = np.stack((xx * 0.01, yy * 0.01, np.full_like(xx, 0.3)), axis=-1).astype(float)
    candidate = {"component": 1}

    with pytest.raises(ValueError, match="too large/merged"):
        estimate_placement_opening(candidate, labels, cloud)


def test_horizontal_bar_outward_is_perpendicular_to_local_bar_axis(monkeypatch):
    yy, xx = np.mgrid[:128, :128]
    along = (xx - 64) * 0.004
    across = (yy - 64) * 0.003
    radius = 0.025
    top = np.sqrt(np.maximum(radius**2 - across**2, 0.0))
    cloud = np.stack((along, across, 0.4 + top), axis=-1)
    monkeypatch.setattr(vision_geometry, "world_cloud", lambda *args: cloud)

    outward = horizontal_bar_outward(64, 64, np.ones((128, 128)), 60.0,
                                     np.eye(3), np.array([0.0, -1.0, 1.0]))

    assert outward is not None
    assert abs(outward[0]) < 0.1
    assert outward[1] < -0.9
    assert outward[2] == pytest.approx(0.0)


def test_horizontal_bar_proposal_without_support_returns_no_semantic_candidates():
    depth = np.full((64, 64), 0.8)
    candidates, labels, cloud = propose_horizontal_bars(
        depth, 60.0, np.eye(3), np.array([0.0, 0.0, 1.0]), None, [0.0, 0.0, 0.6]
    )

    assert candidates == []
    assert labels is None
    assert cloud.shape == (64, 64, 3)


def test_observe_horizontal_bars_mode_invalidates_and_uses_bar_proposals(monkeypatch, tmp_path):
    host = object.__new__(LiberoHost)
    host.execution_accounting = {"control_steps": 0}
    host.step_failed = False
    host.proposal_mode = "objects"
    host.candidate_cache = (1, "stale candidates")
    host.packet = 1
    host.trace = tmp_path / "commands.jsonl"
    host.feedback = {}
    host.result = lambda **kwargs: "observation result"
    host.depth = lambda name: np.ones((32, 32))
    host.reference_robot_position = np.array([0.0, 0.35, 0.8])
    host.sim = SimpleNamespace(
        model=SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=[60.0]),
        data=SimpleNamespace(cam_xmat=[np.eye(3)], cam_xpos=[[0.0, 0.0, 1.0]]),
    )
    calls = []

    monkeypatch.setattr("embodied_harness.adapters.libero_direct.support_height",
                        lambda *args: 0.0)
    monkeypatch.setattr(vision_geometry, "propose_horizontal_bars",
                        lambda *args: calls.append("horizontal_bars") or ([], None, None))
    monkeypatch.setattr(vision_geometry, "propose_surfaces",
                        lambda *args: calls.append("objects") or ([], None, None))

    result = host.call("observe", {"candidate_mode": "horizontal_bars"})
    candidates = host.candidates()

    assert result == "observation result"
    assert host.proposal_mode == "horizontal_bars"
    assert host.candidate_cache == (host.packet, ([], None, None))
    assert candidates == ([], None, None)
    assert calls == ["horizontal_bars"]


def test_horizontal_bar_proposal_finds_perspective_horizontal_elongated_patch():
    height = width = 256
    fovy = 60.0
    focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    tilt = np.deg2rad(25.0)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(tilt), -np.sin(tilt)],
        [0.0, np.sin(tilt), np.cos(tilt)],
    ])
    position = np.array([0.0, 0.0, 1.0])
    yy, xx = np.mgrid[:height, :width]
    ray_y = -(yy - height / 2) / focal
    ray_z = ray_y * np.sin(tilt) - np.cos(tilt)
    # Rays outside the bar hit a lower support surface; rays through the bar
    # hit a horizontal plane at z=.30. The tilted camera makes depth vary across
    # the bar while its reconstructed world surface stays horizontal.
    depth = (-0.35 - position[2]) / ray_z
    bar_depth = (0.30 - position[2]) / ray_z
    depth[112:120, 102:152] = bar_depth[112:120, 102:152]
    world = vision_geometry.world_cloud(depth, fovy, rotation, position)
    robot_position = [float(world[116, 127, 0]), float(world[116, 127, 1]), 0.8]

    candidates, labels, cloud = propose_horizontal_bars(
        depth, fovy, rotation, position, 0.0, robot_position
    )

    assert candidates, "the visible thin support should yield at least one geometry candidate"
    candidate = candidates[0]
    assert candidate["geometry_type"] == "horizontal_elongated_surface_not_semantic_handle"
    assert candidate["height_range_m"] == pytest.approx([0.30, 0.30], abs=1e-6)
    assert np.count_nonzero(labels == candidate["component"]) >= 40
    assert "semantic_label" not in candidate and "object_name" not in candidate


@pytest.mark.parametrize(
    ("lost_on_move", "expected_status", "expected_move_count"),
    [
        (1, "retention_lost_during_transport", 1),
        (2, "retention_lost_during_lower", 2),
    ],
)
def test_placement_stops_substeps_after_host_reports_lost_retention(
    lost_on_move, expected_status, expected_move_count
):
    adapter = make_placement_adapter()
    original_call = adapter.host.call

    def report_retention_loss(name, args):
        body = original_call(name, args)
        if name == "move_to":
            move_count = sum(call_name == "move_to" for call_name, _ in adapter.calls)
            assert args["maintain_grip_contact"] is True
            if move_count == lost_on_move:
                body["feedback"]["motion"]["execution_state"] = "retention_lost"
        return body

    adapter.host.call = report_retention_loss
    result = asyncio.run(place_point(adapter, {"point_id": "p0"}, adapter.cancelled, lambda _: None))

    assert result["macro_status"] == expected_status
    assert sum(name == "move_to" for name, _ in adapter.calls) == expected_move_count
    assert not any(name == "gripper_control" for name, _ in adapter.calls)
    assert adapter.possible_hold


def test_placement_does_not_claim_release_from_tool_success_without_open_gripper_measurement():
    adapter = make_placement_adapter()
    original_call = adapter.host.call

    def under_opened_gripper(name, args):
        body = original_call(name, args)
        if name == "gripper_control" and args.get("action") == "open":
            # The primitive returned success, but its measured aperture is incomplete.
            body["robot"]["gripper"]["openness"] = 0.7
        return body

    adapter.host.call = under_opened_gripper
    result = asyncio.run(place_point(adapter, {"point_id": "p0"}, adapter.cancelled, lambda _: None))

    assert result["macro_status"] == "release_not_measured"
    assert adapter.possible_hold
    assert not any(name == "move_to" and "delta_m" in args for name, args in adapter.calls)


def test_legacy_dual_async_starts_fast_while_initial_slow_review_is_pending(monkeypatch, tmp_path):
    """Exercise app startup wiring without simulator, model APIs, or worker processes."""
    from embodied_harness import app

    slow_entered = asyncio.Event()
    fast_entered = asyncio.Event()
    mission_start_observation = SimpleNamespace(
        id="mission-start", images=("mission-start-agentview.png",),
        state={"views": [{"camera_frame_id": "agentview"}]},
    )
    older_frame = SimpleNamespace(id="older-frame", images=("older.png",), state={"views": []})
    constructed = []

    class Store:
        def __init__(self, path):
            self.path = path

        def emit(self, *_args, **_kwargs):
            pass

        def since(self, cursor):
            return []

        def close(self):
            pass

    class Context:
        def __init__(self, store):
            self.action_seq = 0
            self.frames = [older_frame, mission_start_observation]
            self.latest = mission_start_observation

    class Adapter:
        cancellation = "synthetic_boundary"

        def __init__(self, *args):
            self.schemas = {"move_to": {}, "observe": {}}
            self.host = None

        async def start(self):
            self.host = SimpleNamespace(execution_accounting={})

        async def close(self):
            pass

    class Coordinator:
        def __init__(self, adapter, context, store):
            self.adapter = adapter

        async def observe(self):
            pass

        def delegate(self, milestone):
            pass

        async def verify(self):
            return {"status": "not_verified", "evidence": []}

    class Model:
        def __init__(self, base_url, key, model, role, output, options=None):
            self.role, self.model = role, model
            self.calls, self.usage, self.latencies = 0, {}, []
            self.accounting = {"responses_with_usage": 0}
            self.mission_references = []
            constructed.append(self)

        def bind_tools(self, schemas):
            pass

        def bind_memory(self, memory):
            pass

        def bind_mission_reference(self, observation):
            assert observation is mission_start_observation
            self.mission_references.append(observation)

        async def close(self):
            pass

    class Slow:
        def __init__(self, *args, **kwargs):
            self.reviews = 0

        def request_review(self, reason):
            pass

        async def delegate(self, mission):
            raise AssertionError("dual_async must not await a Slow delegate before Fast starts")

        async def run(self):
            slow_entered.set()
            await asyncio.Future()

    class Fast:
        def __init__(self, *args, **kwargs):
            self.steps, self.false_claims, self.errors = 0, 0, 0

        async def run(self, max_decisions, deadline=None):
            fast_entered.set()
            await asyncio.wait_for(slow_entered.wait(), timeout=0.5)
            return "decision_limit"

    monkeypatch.setattr(app, "EventStore", Store)
    monkeypatch.setattr(app, "Context", Context)
    monkeypatch.setattr(app, "OpenETAAdapter", Adapter)
    monkeypatch.setattr(app, "Coordinator", Coordinator)
    monkeypatch.setattr(app, "ModelClient", Model)
    monkeypatch.setattr(app, "SlowAgent", Slow)
    monkeypatch.setattr(app, "FastAgent", Fast)
    monkeypatch.setattr(app, "load_env", lambda path: None)
    for key, value in {
        "MODEL_BASE_URL": "http://unused.invalid",
        "FAST_API_KEY": "unused",
        "SLOW_API_KEY": "unused",
        "FAST_MODEL": "fast-test",
        "SLOW_MODEL": "slow-test",
    }.items():
        monkeypatch.setenv(key, value)

    args = SimpleNamespace(
        output=str(tmp_path / "episode"), model_env="unused.env", memory=None,
        mission="synthetic mission", backend="eta", sim_url="unused",
        env_id="unused", seed=0, timeout=1.0, max_decisions=1,
        mode="dual_async", review_every=1,
    )
    args.hierarchy = "legacy_review"
    result = asyncio.run(app.episode(args))

    assert fast_entered.is_set()
    assert slow_entered.is_set()
    assert result["error"] is None
    assert [model.role for model in constructed] == ["fast", "slow"]
    assert all(model.mission_references == [mission_start_observation] for model in constructed)
