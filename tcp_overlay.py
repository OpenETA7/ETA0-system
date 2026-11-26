"""Calibrated TCP/reference-frame overlay; depends only on NumPy and Pillow.

Pixels must already match K (including any crop, resize or mirror). All geometry
and the camera pose use one named reference frame and one observation ID. This
module neither reads simulator state nor interprets object/contact information.
It can also run as a standalone file: ``python tcp_overlay.py snapshot.json out.png``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _array(value, shape, name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


@dataclass
class CameraCalibration:
    K_image_px: object
    T_reference_from_camera_optical: object
    image_size_wh: tuple[int, int]
    reference_frame: str
    observation_id: str
    camera_name: str = "wrist"
    distortion_model: str = "none"

    def __post_init__(self):
        self.K_image_px = _array(self.K_image_px, (3, 3), "K_image_px")
        self.T_reference_from_camera_optical = _array(
            self.T_reference_from_camera_optical, (4, 4), "camera pose"
        )
        if not np.allclose(self.K_image_px[2], [0, 0, 1]):
            raise ValueError("K must be a pinhole projection with last row [0,0,1]")
        if abs(np.linalg.det(self.K_image_px)) < 1e-12:
            raise ValueError("K must be invertible; signed focal lengths are allowed")
        transform = self.T_reference_from_camera_optical
        rotation = transform[:3, :3]
        if (not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            raise ValueError("camera pose must be a rigid right-handed transform")
        if (len(self.image_size_wh) != 2
                or any(type(n) is not int or n < 16 for n in self.image_size_wh)):
            raise ValueError("image_size_wh must contain two integers >=16")
        self.image_size_wh = tuple(self.image_size_wh)
        if not self.reference_frame or not self.observation_id or not self.camera_name:
            raise ValueError("frame, observation ID and camera name are required")
        if self.distortion_model != "none":
            raise ValueError("rectify distorted images and update K before rendering")


@dataclass
class RobotGeometry:
    tcp_reference_m: object
    reference_frame: str
    observation_id: str
    pose_origin: str
    jaw_centers_reference_m: object = None
    units: str = "m"

    def __post_init__(self):
        self.tcp_reference_m = _array(self.tcp_reference_m, (3,), "measured TCP")
        if self.jaw_centers_reference_m is not None:
            self.jaw_centers_reference_m = _array(
                self.jaw_centers_reference_m, (2, 3), "jaw centers"
            )
        if not self.reference_frame or not self.observation_id or not self.pose_origin:
            raise ValueError("frame, observation ID and measured pose origin are required")
        if self.units != "m":
            raise ValueError("overlay geometry and transform translations must use metres")


@dataclass(frozen=True)
class OverlayStyle:
    axis_length_m: float = 0.05
    tick_spacing_m: float = 0.01
    near_plane_m: float = 0.001
    supersample: int = 3

    def __post_init__(self):
        values = [self.axis_length_m, self.tick_spacing_m, self.near_plane_m]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("axis length, tick spacing and near plane must be positive")
        if not 1 <= self.axis_length_m / self.tick_spacing_m <= 100:
            raise ValueError("axis must contain between 1 and 100 tick intervals")
        if type(self.supersample) is not int or not 1 <= self.supersample <= 8:
            raise ValueError("supersample must be an integer from 1 to 8")


def project_points(points_reference_m, camera):
    """Return image pixels and optical z; non-projectable pixels are NaN.

    T maps camera -> reference, so row-vector inverse is (p - t) @ R.
    Optical axes are +x right, +y down, +z forward. Signed K encodes image flips.
    """
    points = np.asarray(points_reference_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array in metres")
    transform = camera.T_reference_from_camera_optical
    optical = (points - transform[:3, 3]) @ transform[:3, :3]
    homogeneous = optical @ camera.K_image_px.T
    pixels = np.full((len(points), 2), np.nan)
    np.divide(homogeneous[:, :2], homogeneous[:, 2, None], out=pixels,
              where=np.abs(homogeneous[:, 2, None]) > 1e-12)
    return pixels, optical[:, 2]


def _clip_pixels(a, b, width, height, margin=2):
    delta = b - a
    low, high = 0., 1.
    for p, q in [(-delta[0], a[0] - margin),
                 (delta[0], width - 1 - margin - a[0]),
                 (-delta[1], a[1] - margin),
                 (delta[1], height - 1 - margin - a[1])]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
    return (a + low * delta, a + high * delta) if low <= high else None


def _project_segment(a, b, camera, near):
    """Clip in 3D before projection, including segments crossing the camera plane."""
    points = np.array([a, b], dtype=float)
    _, depths = project_points(points, camera)
    if max(depths) < near:
        return None
    if min(depths) < near:
        index = int(np.argmin(depths))
        other = 1 - index
        weight = (near - depths[index]) / (depths[other] - depths[index])
        points[index] += weight * (points[other] - points[index])
    pixels, _ = project_points(points, camera)
    return _clip_pixels(*pixels, *camera.image_size_wh)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@lru_cache(maxsize=16)
def _font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0
        return ImageFont.load_default()


DEFAULT_STYLE = OverlayStyle()


def overlay_notice(camera, robot, style=DEFAULT_STYLE):
    point_name = (
        "configured Franka EEF reference (not the fingertip grasp center)"
        if "configured Franka EEF" in robot.pose_origin
        else "TCP"
    )
    return (
        f"Robot-only overlay: origin is measured {robot.pose_origin}. Red +X, green +Y, "
        f"blue +Z are {robot.reference_frame} directions translated to that {point_name}, not "
        f"tool-local rotation axes. Axes are {style.axis_length_m * 100:g} cm long; "
        f"ticks are {style.tick_spacing_m * 100:g} cm apart. Projection uses this "
        f"observation's actual calibration ({camera.observation_id}). "
        + ("White line joins measured finger-pad centers; it is not an object axis. "
           if robot.jaw_centers_reference_m is not None else "No jaw axis is available. ")
        + "Clipped endpoints are labeled. Overlay is not depth-tested and does not "
        "prove visibility, contact or grasp. Image pixel coordinates are preserved."
    )


def render_tcp_overlay(image, camera, robot, style=DEFAULT_STYLE):
    """Return a new PIL RGB image and JSON-safe audit metadata; input is untouched."""
    if camera.reference_frame != robot.reference_frame:
        raise ValueError("camera and robot reference frames differ")
    if camera.observation_id != robot.observation_id:
        raise ValueError("camera and robot observation IDs differ (stale geometry)")
    if image.size != camera.image_size_wh:
        raise ValueError("image dimensions do not match calibration; update K after resize/crop")
    width, height = image.size
    point_name = "EEF ref" if "configured Franka EEF" in robot.pose_origin else "TCP"
    scale = style.supersample
    canvas = image.convert("RGB").resize((width * scale, height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    origin = robot.tcp_reference_m
    tcp, depth = project_points([origin], camera)
    tcp = tcp[0]
    metadata = dict(schema_version="tcp-overlay-v1", camera=asdict(camera),
                    robot=asdict(robot), style=asdict(style), axes_frame=robot.reference_frame,
                    tcp_pixel_xy=tcp, tcp_optical_depth_m=float(depth[0]), axes={},
                    is_grasp_evidence=False, depth_tested=False,
                    notice=overlay_notice(camera, robot, style))

    def xy(point):
        return tuple(float(x) * scale for x in point)

    def inside(point, z):
        return bool(z >= style.near_plane_m and np.isfinite(point).all()
                    and 2 <= point[0] <= width - 3 and 2 <= point[1] <= height - 3)

    def line(a, b, color, thickness=2):
        draw.line([xy(a), xy(b)], fill=color, width=thickness * scale)

    def label(point, text, color="white", size=12):
        font = _font(size * scale)
        box = draw.textbbox((0, 0), text, font=font, stroke_width=2 * scale)
        x = max(3, min(width - 5 - (box[2] - box[0]) / scale, point[0]))
        y = max(3, min(height - 5 - (box[3] - box[1]) / scale, point[1]))
        draw.text(xy((x, y)), text, fill=color, font=font,
                  stroke_width=2 * scale, stroke_fill="#101319")

    jaw = robot.jaw_centers_reference_m
    if jaw is not None:
        pixels, z = project_points(jaw, camera)
        segment = _project_segment(*jaw, camera, style.near_plane_m)
        metadata["jaw"] = dict(pixel_xy=pixels, optical_depth_m=z,
                               drawn=segment is not None)
        if segment is not None:
            line(*segment, "#eeeece", 1)
            for point, point_z in zip(pixels, z, strict=True):
                if inside(point, point_z):
                    draw.ellipse([xy(point - 2), xy(point + 2)], fill="#eeeece")
            label(segment[1] + [5, 14], "jaw axis", size=11)

    distances = np.arange(0, style.axis_length_m, style.tick_spacing_m)
    distances = np.r_[distances, style.axis_length_m]
    colors = zip("XYZ", ["#ff6767", "#65e791", "#68baff"], strict=True)
    for index, (name, color) in enumerate(colors):
        points = origin + distances[:, None] * np.eye(3)[index]
        pixels, z = project_points(points, camera)
        endpoint_inside = inside(pixels[-1], z[-1])
        segments = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            segment = _project_segment(a, b, camera, style.near_plane_m)
            if segment is not None:
                line(*segment, "#101319", 4)
                line(*segment, color, 2)
                segments.append(segment)
        for point, point_z in zip(pixels[1:], z[1:], strict=True):
            if inside(point, point_z):
                draw.ellipse([xy(point - 2.3), xy(point + 2.3)], fill=color)
        metadata["axes"][name] = dict(points_reference_m=points, pixel_xy=pixels,
                                      optical_depth_m=z, endpoint_in_image=endpoint_inside,
                                      near_plane_clipped=bool((z < style.near_plane_m).any()),
                                      drawn=bool(segments))
        if segments:
            start, endpoint = segments[-1]
            direction = endpoint - start
            norm = np.linalg.norm(direction)
            if norm > 6:
                unit = direction / norm
                normal = np.array([-unit[1], unit[0]])
                draw.polygon([xy(endpoint), xy(endpoint - unit * 8 + normal * 4),
                              xy(endpoint - unit * 8 - normal * 4)], fill=color)
            suffix = "" if endpoint_inside else " [clipped]"
            label(endpoint + [8, -18], f"+{name} {style.axis_length_m * 100:g} cm{suffix}", color)
    if inside(tcp, depth[0]):
        line(tcp - [6, 0], tcp + [6, 0], "#ffd66b")
        line(tcp - [0, 6], tcp + [0, 6], "#ffd66b")
        label(tcp + [9, 8], point_name, "#ffd66b")
    draw.rectangle([0, 0, width * scale - 1, 27 * scale], fill="#101319")
    title = (f"{camera.camera_name} | {robot.reference_frame} XYZ @ {point_name} | "
             f"ticks {style.tick_spacing_m * 100:g} cm")
    draw.text(xy((7, 5)), title, fill="white", font=_font(12 * scale))
    return canvas.resize((width, height), Image.Resampling.LANCZOS), _jsonable(metadata)


def try_render_tcp_overlay(image, camera, robot, style=DEFAULT_STYLE):
    """Render overlay when calibration is valid; fall back cleanly to raw RGB otherwise."""
    if camera is None or robot is None:
        return image.convert("RGB"), dict(
            schema_version="tcp-overlay-v1",
            status="unavailable",
            reason="missing_calibration_or_robot_geometry",
            notice="TCP overlay unavailable for this view; using unmodified RGB image.",
        )
    try:
        rendered, metadata = render_tcp_overlay(image, camera, robot, style=style)
        metadata["status"] = "available"
        return rendered, metadata
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return image.convert("RGB"), dict(
            schema_version="tcp-overlay-v1",
            status="unavailable",
            reason=str(exc),
            notice=f"TCP overlay unavailable ({exc}); using unmodified RGB image.",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path,
                        help="JSON containing image, camera, robot, optional style")
    parser.add_argument("output", type=Path, help="Output PNG (and same-stem JSON audit)")
    args = parser.parse_args()
    record = json.loads(args.snapshot.read_text())
    image_path = args.snapshot.parent / record["image"]
    with Image.open(image_path) as source:
        result, metadata = render_tcp_overlay(
            source, CameraCalibration(**record["camera"]), RobotGeometry(**record["robot"]),
            OverlayStyle(**record.get("style", {})),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output, format="PNG")
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
