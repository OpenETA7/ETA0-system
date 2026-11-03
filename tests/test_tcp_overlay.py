"""Independent projection checks and regressions from two archived benchmarks."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tcp_overlay import (
    CameraCalibration,
    OverlayStyle,
    RobotGeometry,
    project_points,
    render_tcp_overlay,
)
from tcp_overlay_adapter import (
    from_legacy_snapshot,
    render_session_overlays,
    session_geometry,
)


def camera(**kwargs):
    return CameraCalibration(**dict(
        {"K_image_px": [[100, 0, 160], [0, 120, 120], [0, 0, 1]],
         "T_reference_from_camera_optical": np.eye(4), "image_size_wh": (320, 240),
         "reference_frame": "chassis", "observation_id": "frame-7"}, **kwargs,
    ))


def robot(**kwargs):
    return RobotGeometry(**dict(
        {"tcp_reference_m": [0, 0, 1], "reference_frame": "chassis",
         "observation_id": "frame-7", "pose_origin": "tool_tip"}, **kwargs,
    ))


@pytest.mark.parametrize("sign", [1, -1])
def test_pinhole_signed_focal_length_and_pixel_centers(sign):
    cam = camera(K_image_px=[[sign * 100, 0, 159.5], [0, 120, 119.5], [0, 0, 1]])
    uv, z = project_points([[.2, .1, 2]], cam)
    np.testing.assert_allclose(uv, [[159.5 + sign * 10, 125.5]])
    np.testing.assert_allclose(z, [2])


def test_rotated_translated_camera_and_moving_reference_invariance():
    # Optical (0.2, 0.1, 2) -> reference (0.9, 2.2, 5).
    transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.]])
    cam = camera(T_reference_from_camera_optical=transform)
    point = np.array([[.9, 2.2, 5.]])
    uv, z = project_points(point, cam)
    np.testing.assert_allclose(uv, [[170, 126]])
    np.testing.assert_allclose(z, [2])
    # Re-express both point and camera in a new moving chassis frame.
    change = np.array([[1, 0, 0, 3], [0, 0, -1, 4], [0, 1, 0, 5], [0, 0, 0, 1.]])
    moved = replace(cam, T_reference_from_camera_optical=change @ transform)
    moved_point = point @ change[:3, :3].T + change[:3, 3]
    np.testing.assert_allclose(project_points(moved_point, moved)[0], uv)


def test_non_square_image_stays_in_original_pixels_without_mutation():
    image = Image.new("RGB", (320, 240), (20, 30, 40))
    before = image.tobytes()
    rendered, meta = render_tcp_overlay(image, camera(), robot())
    assert rendered.size == image.size and image.tobytes() == before
    np.testing.assert_allclose(meta["tcp_pixel_xy"], [160, 120])
    np.testing.assert_allclose(meta["axes"]["X"]["pixel_xy"][-1], [165, 120])
    assert "jaw" not in meta and not meta["depth_tested"] and not meta["is_grasp_evidence"]
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("change,match", [
    ({"reference_frame": "world"}, "frames differ"),
    ({"observation_id": "older"}, "observation IDs differ"),
])
def test_reject_mismatched_frame_or_observation(change, match):
    with pytest.raises(ValueError, match=match):
        render_tcp_overlay(Image.new("RGB", (320, 240)), camera(), robot(**change))


def test_reject_resize_without_calibration_and_non_metric_geometry():
    with pytest.raises(ValueError, match="dimensions"):
        render_tcp_overlay(Image.new("RGB", (512, 512)), camera(), robot())
    with pytest.raises(ValueError, match="metres"):
        robot(units="mm")


@pytest.mark.parametrize("updates", [
    {"K_image_px": np.zeros((3, 3))},
    {"T_reference_from_camera_optical": np.diag([-1, 1, 1, 1])},
    {"T_reference_from_camera_optical": np.eye(4) * 2},
    {"distortion_model": "fisheye"},
])
def test_reject_invalid_calibration(updates):
    with pytest.raises(ValueError):
        camera(**updates)


def test_behind_camera_and_camera_plane_are_json_safe_and_not_drawn():
    for z in [-1., 0.]:
        _, meta = render_tcp_overlay(Image.new("RGB", (320, 240)), camera(),
                                     robot(tcp_reference_m=[0, 0, z]))
        json.dumps(meta, allow_nan=False)
        assert not meta["axes"]["X"]["drawn"]
        if z < 0:
            assert not any(axis["drawn"] for axis in meta["axes"].values())
        else:
            assert meta["tcp_pixel_xy"] == [None, None]
            assert meta["axes"]["Z"]["near_plane_clipped"]


def test_jaw_crossing_near_plane_and_axis_outside_frame():
    _, meta = render_tcp_overlay(
        Image.new("RGB", (320, 240)), camera(),
        robot(tcp_reference_m=[.015, 0, .01],
              jaw_centers_reference_m=[[0, 0, -.01], [.01, 0, .1]]),
    )
    assert meta["jaw"]["drawn"]
    assert meta["axes"]["X"]["drawn"]
    assert not meta["axes"]["X"]["endpoint_in_image"]
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("name", ["libero", "robocasa"])
def test_archived_benchmark_projection_matches_original_renderer(name):
    fixture = Path(__file__).parent / "fixtures" / "tcp_overlay" / f"{name}.json"
    archived = json.loads(fixture.read_text())
    for view, expected in archived["views"].items():
        cam, geometry = from_legacy_snapshot(archived, view, "recorded-frame")
        _, meta = render_tcp_overlay(Image.new("RGB", cam.image_size_wh), cam, geometry)
        np.testing.assert_allclose(meta["tcp_pixel_xy"], expected["tcp_pixel_xy"], atol=1e-9)
        np.testing.assert_allclose(meta["jaw"]["pixel_xy"], expected["jaw_pixels"], atol=1e-9)
        for axis in "XYZ":
            np.testing.assert_allclose(meta["axes"][axis]["pixel_xy"],
                                       expected["axes"][axis]["pixel_xy"], atol=1e-9)
        assert cam.reference_frame == archived["axes_frame"]
    assert (cam.K_image_px[0, 0] < 0) == (name == "libero")


def test_session_bridge_rejects_stale_state_before_reading_simulator():
    session = SimpleNamespace(context={"frame": 7}, p=SimpleNamespace(sim_steps=8))
    with pytest.raises(ValueError, match="stale"):
        session_geometry(session, "chassis")


@pytest.mark.parametrize("frame", ["robot0_base", "mobilebase0_base"])
def test_session_bridge_uses_named_measured_frame_and_current_capture(tmp_path, frame):
    image_path = tmp_path / "camera.png"
    Image.new("RGB", (320, 240)).save(image_path)
    mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_BODY="body", mjOBJ_SITE="site", mjOBJ_GEOM="geom"),
        mj_id2name=lambda model, kind, index: frame if kind == "body" else "measured_tip",
        mj_name2id=lambda model, kind, name: {"left_pad": 0, "right_pad": 1}[name],
    )
    provider = SimpleNamespace(
        sim_steps=7, mujoco=mujoco, _model=None, _base_id=0, _site_id=0,
        _data=SimpleNamespace(site_xpos=np.array([[1., 2., 4.]]),
                              geom_xpos=np.array([[.98, 2., 4.], [1.02, 2., 4.]])),
        to_base=lambda point: np.asarray(point) - [1., 2., 3.],
    )
    context = dict(frame=7, observation_id="frame-7", coordinate_frame=frame,
                   eef_pose_base={"position_m": [0, 0, 1]},
                   cameras={"wrist": {"K_image_px": camera().K_image_px.tolist(),
                                      "T_robot_base_from_camera_optical": np.eye(4).tolist(),
                                      "image_size_wh": [320, 240]}})
    session = SimpleNamespace(
        p=provider, context=context, captures={"wrist": {"rgb_path": str(image_path)}},
        env=SimpleNamespace(robots=[SimpleNamespace(gripper=SimpleNamespace(
            important_geoms={"left_fingerpad": ["left_pad"], "right_fingerpad": "right_pad"}
        ))]),
    )
    paths, records = render_session_overlays(session, tmp_path / "out", reference_frame=frame)
    assert Image.open(paths["wrist"]).size == (320, 240)
    np.testing.assert_allclose(records["wrist"]["robot"]["tcp_reference_m"], [0, 0, 1])
    np.testing.assert_allclose(records["wrist"]["robot"]["jaw_centers_reference_m"],
                               [[-.02, 0, 1], [.02, 0, 1]])
    assert provider.sim_steps == 7 and context["coordinate_frame"] == frame
    with pytest.raises(ValueError, match="requested frame"):
        session_geometry(session, "wrong_base")


def test_tick_length_validation():
    with pytest.raises(ValueError):
        OverlayStyle(tick_spacing_m=0)
