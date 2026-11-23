"""Synthetic regressions for RGB-D geometry and verified appearance memory."""
import asyncio
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_harness.adapters.libero_direct import support_height, visible_normal
from embodied_harness.adapters.placement import place_point
from embodied_harness.adapters.skills import grasp_point
from embodied_harness.adapters.vision_geometry import candidate_surface, propose_surfaces
from embodied_harness.runtime.memory import retrieve


def test_flat_depth_plane_has_camera_facing_normal_and_dominant_support_height():
    depth = np.ones((128, 128), dtype=float)
    fovy = 60.0
    rotation = np.eye(3)
    camera_position = np.array([0.0, 0.0, 1.0])

    normal = visible_normal(64, 64, depth, fovy, rotation, camera_position)
    support = support_height(depth, fovy, rotation, camera_position,
                             robot_position=[0.0, 0.0, 0.6])

    assert normal is not None
    assert np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)
    assert support == pytest.approx(0.0, abs=1e-6)


def synthetic_adjacent_depth_patches():
    """Two touching image regions at different ranges, on a flat support plane."""
    depth = np.ones((128, 128), dtype=float)
    depth[42:82, 24:58] = 0.82
    depth[42:82, 58:92] = 0.92
    return propose_surfaces(depth, 60.0, np.eye(3), [0.0, 0.0, 1.0],
                            support=0.0, robot_position=[0.0, 0.0, 0.0])


def test_touching_image_patches_with_3d_depth_discontinuity_remain_separate():
    candidates, labels, cloud = synthetic_adjacent_depth_patches()

    assert len(candidates) == 2
    first, second = candidates
    assert first["component"] != second["component"]
    assert first["bbox_xyxy"][2] == second["bbox_xyxy"][0], "the image regions touch"
    assert np.linalg.norm(np.asarray(first["center_xyz_m"]) - second["center_xyz_m"]) > 0.05
    assert all("semantic_label" not in c and "object_name" not in c for c in candidates)


def test_top_surface_selection_returns_a_pixel_inside_its_depth_component():
    candidates, labels, cloud = synthetic_adjacent_depth_patches()
    selected = candidates[0]

    x, y, jaw = candidate_surface(selected, labels, cloud, "top_surface")
    component_pixels = labels == selected["component"]
    heights = cloud[:, :, 2][component_pixels]

    assert component_pixels[y, x]
    assert cloud[y, x, 2] >= np.quantile(heights, 0.9) - 0.003
    assert jaw is None


def test_rim_grasp_automatically_carries_center_offset_into_placement():
    class Adapter:
        def __init__(self):
            self.points = {
                "rim": {
                    "point_id": "rim", "source_packet_id": "frame-0",
                    "camera_frame_id": "agentview", "pixel_xy": [40, 55],
                    "xyz_m": [0.10, 0.20, 0.90],
                    "measurement": {"depth_edge": False},
                    "visual_candidate": {
                        "center_xyz_m": [0.15, 0.20, 0.90],
                        "suggested_jaw_world": [1.0, 0.0, 0.0],
                    },
                }
            }
            self.possible_hold = False
            self.held_center_in_grip = None
            self.cached = SimpleNamespace(id="frame-0")
            self.last = {
                "views": [{"source_packet_id": "frame-0", "camera_frame_id": "agentview"}],
                "robot": {"grip_xyz_m": [0.0, 0.0, 1.2], "jaw_world": [1.0, 0.0, 0.0],
                          "approach_world": [0.0, 0.0, -1.0], "gripper": {"openness": 1.0}},
            }
            self.calls = []
            self.host = SimpleNamespace(call=self.call)

        def call(self, name, args):
            self.calls.append((name, dict(args)))
            openness = 1.0 if name == "gripper_control" and args.get("action") == "open" else 0.0
            if name == "mark_point":
                motion = {"execution_state": "completed", "physics_executed": True}
                feedback = {"point": {
                    "point_id": "measured-rim", "source_packet_id": "frame-1",
                    "camera_frame_id": "agentview", "pixel_xy": [40, 55],
                    "xyz_m": [0.10, 0.20, 0.90], "measurement": {"depth_edge": False},
                }}
            elif name == "gripper_control" and args.get("action") == "close":
                motion = {"execution_state": "completed", "physics_executed": True,
                          "motion_summary": {"gripper_contact": {"contact_pattern": "bilateral_pads"}}}
                feedback = {"motion": motion}
            elif name == "gripper_control":
                motion = {"execution_state": "completed", "physics_executed": True,
                          "motion_summary": {"gripper_contact": {"contact_pattern": "no_contact"}}}
                feedback = {"motion": motion}
            else:
                motion = {"execution_state": "reached", "physics_executed": True}
                feedback = {"motion": motion}
            return {
                "views": [{"source_packet_id": "frame-1", "camera_frame_id": "agentview"}],
                "robot": {"grip_xyz_m": [0.0, 0.0, 1.2], "jaw_world": [1.0, 0.0, 0.0],
                          "approach_world": [0.0, 0.0, -1.0], "gripper": {"openness": openness}},
                "contact_state": {"close_binding_available": name == "move_to" and "contact_point_id" in args},
                "feedback": feedback, "error": None,
            }

        async def _thread(self, fn, *args):
            return fn(*args)

        def _unpack(self, result):
            self.last = result
            self.cached = SimpleNamespace(id=f"frame-{len(self.calls)}")
            return self.cached

        def cancelled(self):
            return False

    async def run():
        adapter = Adapter()
        grasp = await grasp_point(adapter, {"point_id": "rim"}, adapter.cancelled, lambda _: None)

        assert grasp["macro_status"] == "lift_completed_verify_retention_visually"
        assert adapter.held_center_in_grip == pytest.approx([0.05, 0.0, 0.0])
        grasp_moves = [args for name, args in adapter.calls if name == "move_to"]
        assert len(grasp_moves) == 3
        assert grasp_moves[0]["jaw_world"] == [1.0, 0.0, 0.0]
        assert grasp_moves[1]["jaw_world"] == [1.0, 0.0, 0.0]

        adapter.points["destination"] = {
            "point_id": "destination", "source_packet_id": "frame-1",
            "camera_frame_id": "agentview", "xyz_m": [0.50, 0.30, 0.40],
        }
        adapter.last["robot"]["grip_xyz_m"] = [0.0, 0.0, 1.2]
        placement = await place_point(adapter, {"point_id": "destination"}, adapter.cancelled, lambda _: None)

        assert placement["macro_status"] == "released_verify_placement_visually"
        placement_phases = placement["feedback"]["substeps"]
        transport = next(step for step in placement_phases if step["phase"] == "transport")
        assert transport["args"]["xyz_m"] == pytest.approx([0.45, 0.30, 1.2])
        assert adapter.held_center_in_grip is None

    asyncio.run(run())


def write_manifest(path, entries):
    path.write_text(json.dumps({"entries": entries}))
    return path


def test_appearance_memory_is_opt_in_and_skips_nonmatching_entries(tmp_path):
    assert retrieve(None, "pick up the cup") == []
    manifest = write_manifest(tmp_path / "manifest.json", [{
        "match_text": "basket", "image": "missing.png",
        "provenance": {"official_success": False},
    }])

    assert retrieve(manifest, "pick up the cup") == []


def test_appearance_memory_rejects_matching_unverified_source(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json", [{
        "match_text": "cup", "image": "prior.png",
        "provenance": {"official_success": False},
    }])

    with pytest.raises(ValueError, match="verified source episode"):
        retrieve(manifest, "pick up the cup")


def test_appearance_memory_rejects_image_path_traversal(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json", [{
        "match_text": "cup", "image": "../outside.png",
        "provenance": {"official_success": True},
    }])

    with pytest.raises(ValueError, match="within manifest directory"):
        retrieve(manifest, "pick up the cup")


def test_verified_matching_memory_returns_image_and_integrity_hashes(tmp_path):
    image = tmp_path / "cup.png"
    image.write_bytes(b"synthetic prior appearance image")
    manifest = write_manifest(tmp_path / "manifest.json", [{
        "match_text": "cup", "image": "cup.png", "caption": "blue cup on a counter",
        "provenance": {"official_success": True, "episode_id": "prior-episode"},
    }])

    result = retrieve(manifest, "place the cup in the basket")

    assert len(result) == 1
    assert result[0]["image"] == str(image.resolve())
    assert result[0]["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert result[0]["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
