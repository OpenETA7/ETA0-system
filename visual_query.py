"""Model-selected pixel; exact optical-depth unprojection, zero physics."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from tcp_overlay import CameraCalibration

QUERY_STAGE = "mark_point"
QUERY_NOTICE = '''Optional read-only query (exact fields):
{"stage":"mark_point","observation_id":"CURRENT","camera":"wrist","point_xy_px":[u,v]}
Coordinates are integer pixels in the original image. The tool marks YOUR point.
If registered metric optical depth is available, it backprojects that exact pixel
into the current reference frame. No nearest-valid-pixel search, depth snapping,
object mask or object membership test is performed. A visible surface point is
not a TCP target, grasp pose, safe clearance or proof of success. Local depth
statistics are unsegmented context only; never substitute them for invalid depth
at the chosen pixel. Queries execute no ticks. Without depth only annotation is returned.
'''


@dataclass
class QueryView:
    rgb: Image.Image
    camera: CameraCalibration | None = None
    depth_optical_m: object = None  # Registered optical Z, not range or grayscale.
    max_depth_m: float = 5.0


def get_view(action, observation):
    if action.get("observation_id") != observation.robot.observation_id:
        raise ValueError("stale query observation_id")
    name = action.get("camera")
    if not isinstance(name, str):
        raise ValueError("camera must be a string")
    view = observation.query_views.get(name)
    if view is None and name == "wrist":
        view = QueryView(observation.wrist_rgb, observation.wrist_camera)
    if view is None and name == "external":
        view = QueryView(observation.external_rgb)
    if view is None:
        raise ValueError("unknown query camera")
    if view.camera is not None:
        if (view.camera.observation_id != observation.robot.observation_id
                or view.camera.reference_frame != observation.robot.reference_frame):
            raise ValueError("query calibration frame/observation mismatch")
        if view.rgb.size != view.camera.image_size_wh:
            raise ValueError("query image/calibration size mismatch")
    if not np.isfinite(view.max_depth_m) or view.max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    depth = None
    if view.depth_optical_m is not None:
        if view.camera is None:
            raise ValueError("metric query requires calibration")
        depth = np.asarray(view.depth_optical_m, dtype=float)
        if depth.shape != (view.rgb.height, view.rgb.width):
            raise ValueError("depth must be registered to the query RGB")
    return view, depth


def unproject(pixels, depth, camera):
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera.K_image_px).T
    optical = rays * np.asarray(depth)[:, None]
    transform = camera.T_reference_from_camera_optical
    return optical @ transform[:3, :3].T + transform[:3, 3]


def query(action, observation):
    fields = {"stage", "observation_id", "camera", "point_xy_px"}
    if set(action) != fields or action["stage"] != QUERY_STAGE:
        raise ValueError("mark_point requires exactly " + ", ".join(sorted(fields)))
    view, depth = get_view(action, observation)
    point = action["point_xy_px"]
    if (not isinstance(point, list) or len(point) != 2
            or any(type(v) is not int for v in point)
            or not 0 <= point[0] < view.rgb.width or not 0 <= point[1] < view.rgb.height):
        raise ValueError("invalid original-image pixel")
    x, y = point
    image = view.rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline="#ff5555", width=2)
    draw.line([(x - 9, y), (x + 9, y)], fill="#ff5555", width=2)
    draw.line([(x, y - 9), (x, y + 9)], fill="#ff5555", width=2)
    feedback = dict(stage=QUERY_STAGE, observation_id=action["observation_id"],
                    camera=action["camera"], point_xy_px=point, geometry_valid=False,
                    invalid_reason="metric_depth_not_supplied", optical_depth_mm=None,
                    surface_point_reference_mm=None, tcp_to_surface_mm=None,
                    coordinate_frame=observation.robot.reference_frame, units="mm",
                    is_motion_target=False, is_grasp_evidence=False, privileged=False,
                    target_membership_verified=False, executed_ticks=0,
                    local_depth_p10_p50_p90_mm=None, instruction=QUERY_NOTICE)
    if depth is not None:
        z = float(depth[y, x])
        valid = bool(np.isfinite(z) and 0 < z < view.max_depth_m)
        patch = depth[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        samples = patch[np.isfinite(patch) & (patch > 0) & (patch < view.max_depth_m)]
        feedback.update(geometry_valid=valid,
                        invalid_reason=None if valid else "invalid_selected_pixel_depth",
                        local_statistics_are_unsegmented=True,
                        local_depth_p10_p50_p90_mm=(np.percentile(samples, [10, 50, 90])
                                                   * 1000).tolist() if samples.size else None)
        if valid:
            xyz = unproject([point], [z], view.camera)[0]
            feedback.update(optical_depth_mm=z * 1000,
                            surface_point_reference_mm=(xyz * 1000).tolist(),
                            tcp_to_surface_mm=((xyz - observation.robot.tcp_reference_m)
                                               * 1000).tolist())
    return image, feedback


