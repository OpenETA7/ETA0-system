"""Bridge existing LIBERO/RoboCasa sessions to the benchmark-independent overlay.

No simulation step, observation refresh, global monkey patch, or model API call.
Call immediately after session.observe(), before executing another action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tcp_overlay import CameraCalibration, RobotGeometry, render_tcp_overlay


def from_legacy_snapshot(metadata, view, observation_id):
    """Convert archived tcp_axes.json, using its authoritative axes_frame.

    RoboCasa's legacy calibration can retain a stale translation_frame label
    even though its matrix is in mobilebase0_base. Do not infer the frame from K.
    """
    frame = metadata["axes_frame"]
    calibration = metadata["views"][view]["calibration"]
    raw = metadata["robot"]
    pads = raw.get("fingerpad_centers_base_m")
    return (
        CameraCalibration(calibration["K_image_px"],
                          calibration["T_robot_base_from_camera_optical"],
                          calibration["image_size_wh"], frame, observation_id, view),
        RobotGeometry(raw["tcp_base_m"], frame, observation_id, raw["pose_origin"],
                      None if pads is None else [pads["left"], pads["right"]]),
    )


def session_geometry(session, reference_frame):
    """Read robot-only geometry from the already captured simulation state.

    The two existing sessions expose the selected arm's gripper as robots[0].
    For other robots (including suction/multi-arm), supply RobotGeometry yourself.
    """
    provider = session.p
    context = session.context
    if context is None or context["frame"] != provider.sim_steps:
        raise ValueError("observe before rendering; session state is stale")
    if context.get("coordinate_frame", reference_frame) != reference_frame:
        raise ValueError("requested frame differs from session context")
    mujoco = provider.mujoco
    base_name = mujoco.mj_id2name(provider._model, mujoco.mjtObj.mjOBJ_BODY, provider._base_id)
    if base_name != reference_frame:
        raise ValueError("requested frame differs from measured provider base body")
    origin_name = mujoco.mj_id2name(provider._model, mujoco.mjtObj.mjOBJ_SITE, provider._site_id)
    tcp = provider.to_base(provider._data.site_xpos[provider._site_id])
    if not np.allclose(tcp, context["eef_pose_base"]["position_m"], atol=1e-6, rtol=0):
        raise ValueError("measured TCP differs from captured observation")
    pads = []
    groups = session.env.robots[0].gripper.important_geoms
    for side in ("left", "right"):
        names = groups.get(side + "_fingerpad")
        if not names:
            pads = None
            break
        if isinstance(names, str):
            names = [names]
        ids = [mujoco.mj_name2id(provider._model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
        if any(i < 0 for i in ids):
            raise ValueError("unknown robot finger-pad geometry")
        pads.append(provider.to_base(np.mean(provider._data.geom_xpos[ids], axis=0)))
    return RobotGeometry(tcp, reference_frame, context["observation_id"], origin_name, pads)


def render_session_overlays(session, output_dir, *, reference_frame, views=("wrist",)):
    """Render only requested views, with current K/T and actual image dimensions.

    LIBERO: reference_frame='robot0_base'.
    RoboCasa mobile v3: reference_frame='mobilebase0_base'.
    Returns image paths and per-view audit records; does not alter model history.
    """
    robot = session_geometry(session, reference_frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, records = {}, {}
    for view in views:
        calibration = session.context["cameras"][view]
        camera = CameraCalibration(
            calibration["K_image_px"], calibration["T_robot_base_from_camera_optical"],
            calibration["image_size_wh"], reference_frame, robot.observation_id, view,
        )
        with Image.open(session.captures[view]["rgb_path"]) as source:
            image, metadata = render_tcp_overlay(source, camera, robot)
        # A caller's camera name is a label, never an output path.
        path = output_dir / f"view-{len(paths):02d}_tcp_axes.png"
        image.save(path)
        paths[view], records[view] = str(path), metadata
    (output_dir / "tcp_axes.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return paths, records
