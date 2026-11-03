"""Read-only bbox and exact-pixel queries over the same synchronized observation."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from tcp_overlay import CameraCalibration

QUERY_STAGES = ("bbox", "mark_point")
BBOX_NOTICE = """Optional read-only query (exact fields):
{"stage":"bbox","observation_id":"CURRENT","camera":"wrist","bbox_xyxy_px":[x0,y0,x1,y1]}
camera is wrist or external, or another name supplied by the adapter. Coordinates
are integer original-image pixels; the box is half-open [x0,x1) x [y0,y1).
The tool draws YOUR box. If metric optical depth is available it reports separate
XYZ quantiles of ALL valid pixels, including robot/table/background; no mask,
object identity, stereo fusion or grasp success is inferred. Quantiles on different
axes are not a single surface point or a motion target. Queries execute no ticks.
Without depth it returns the annotation and explicitly marks geometry unavailable.
"""

POINT_NOTICE = """Optional read-only query (exact fields):
{"stage":"mark_point","observation_id":"CURRENT","camera":"wrist","point_xy_px":[u,v]}
Coordinates are integer pixels in the original image. The tool marks YOUR point.
If registered metric optical depth is available, it backprojects that exact pixel
into the current reference frame. No nearest-valid-pixel search, depth snapping,
object mask or object membership test is performed. A visible surface point is
not a TCP target, grasp pose, safe clearance or proof of success. Local depth
statistics are unsegmented context only; never substitute them for invalid depth
at the chosen pixel. Queries execute no ticks. Without depth only annotation is returned.
"""

QUERY_NOTICE = (
    BBOX_NOTICE
    + "\n"
    + POINT_NOTICE
    + "\nBoth queries share one per-observation budget; neither executes physics.\n"
)


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
        if (
            view.camera.observation_id != observation.robot.observation_id
            or view.camera.reference_frame != observation.robot.reference_frame
        ):
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


def query_bbox(action, observation):
    fields = {"stage", "observation_id", "camera", "bbox_xyxy_px"}
    if set(action) != fields or action["stage"] != "bbox":
        raise ValueError("bbox requires exactly " + ", ".join(sorted(fields)))
    view, depth = get_view(action, observation)
    box = action["bbox_xyxy_px"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(v) is not int for v in box)
        or not 0 <= box[0] < box[2] <= view.rgb.width
        or not 0 <= box[1] < box[3] <= view.rgb.height
    ):
        raise ValueError("invalid half-open bbox")
    x0, y0, x1, y1 = box
    image = view.rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline="#ffd166", width=2)
    feedback = dict(
        stage="bbox",
        observation_id=action["observation_id"],
        camera=action["camera"],
        bbox_xyxy_px=box,
        geometry_valid=False,
        invalid_reason="metric_depth_not_supplied",
        axis_percentiles_mm=None,
        coordinate_frame=observation.robot.reference_frame,
        units="mm",
        segmented=False,
        axes_are_independent=True,
        is_motion_target=False,
        is_grasp_evidence=False,
        privileged=False,
        executed_ticks=0,
        bbox_pixel_count=(x1 - x0) * (y1 - y0),
        valid_points=0,
        instruction=BBOX_NOTICE,
    )
    if depth is not None:
        region = depth[y0:y1, x0:x1]
        valid = np.isfinite(region) & (region > 0) & (region < view.max_depth_m)
        ys, xs = np.nonzero(valid)
        count = len(xs)
        feedback.update(
            valid_points=count,
            valid_depth_fraction=count / region.size,
            geometry_valid=bool(count),
            invalid_reason=None if count else "no_valid_depth_in_bbox",
        )
        if count:
            points = unproject(np.c_[xs + x0, ys + y0], region[valid], view.camera) * 1000
            percentiles = [0, 1, 5, 10, 50, 90, 95, 99, 100]
            labels = ["min", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max"]
            values = np.percentile(points, percentiles, axis=0)
            feedback["axis_percentiles_mm"] = {
                axis: {label: float(values[j, i]) for j, label in enumerate(labels)}
                for i, axis in enumerate("xyz")
            }
    return image, feedback


def query_mark_point(action, observation):
    fields = {"stage", "observation_id", "camera", "point_xy_px"}
    if set(action) != fields or action["stage"] != "mark_point":
        raise ValueError("mark_point requires exactly " + ", ".join(sorted(fields)))
    view, depth = get_view(action, observation)
    point = action["point_xy_px"]
    if (
        not isinstance(point, list)
        or len(point) != 2
        or any(type(v) is not int for v in point)
        or not 0 <= point[0] < view.rgb.width
        or not 0 <= point[1] < view.rgb.height
    ):
        raise ValueError("invalid original-image pixel")
    x, y = point
    image = view.rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline="#ff5555", width=2)
    draw.line([(x - 9, y), (x + 9, y)], fill="#ff5555", width=2)
    draw.line([(x, y - 9), (x, y + 9)], fill="#ff5555", width=2)
    feedback = dict(
        stage="mark_point",
        observation_id=action["observation_id"],
        camera=action["camera"],
        point_xy_px=point,
        geometry_valid=False,
        invalid_reason="metric_depth_not_supplied",
        optical_depth_mm=None,
        surface_point_reference_mm=None,
        tcp_to_surface_mm=None,
        coordinate_frame=observation.robot.reference_frame,
        units="mm",
        is_motion_target=False,
        is_grasp_evidence=False,
        privileged=False,
        target_membership_verified=False,
        executed_ticks=0,
        local_depth_p10_p50_p90_mm=None,
        instruction=POINT_NOTICE,
    )
    if depth is not None:
        z = float(depth[y, x])
        valid = bool(np.isfinite(z) and 0 < z < view.max_depth_m)
        patch = depth[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
        samples = patch[np.isfinite(patch) & (patch > 0) & (patch < view.max_depth_m)]
        feedback.update(
            geometry_valid=valid,
            invalid_reason=None if valid else "invalid_selected_pixel_depth",
            local_statistics_are_unsegmented=True,
            local_depth_p10_p50_p90_mm=(np.percentile(samples, [10, 50, 90]) * 1000).tolist()
            if samples.size
            else None,
        )
        if valid:
            xyz = unproject([point], [z], view.camera)[0]
            feedback.update(
                optical_depth_mm=z * 1000,
                surface_point_reference_mm=(xyz * 1000).tolist(),
                tcp_to_surface_mm=((xyz - observation.robot.tcp_reference_m) * 1000).tolist(),
            )
    return image, feedback


def query(action, observation):
    """Dispatch one explicit selection. Both tools are available each turn."""
    if not isinstance(action, dict):
        raise ValueError("visual query must be a JSON object")
    if action.get("stage") == "bbox":
        return query_bbox(action, observation)
    if action.get("stage") == "mark_point":
        return query_mark_point(action, observation)
    raise ValueError("visual query stage must be bbox or mark_point")
