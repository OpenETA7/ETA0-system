"""Measured camera transforms and robot-base orientations."""

import math

import numpy as np

RESOLUTION = 512


def quaternion_to_matrix(quat_wxyz) -> list[list[float]]:
    """Scalar-first quaternion -> rotation matrix (column-major safe, matches MuJoCo)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_quaternion_wxyz(m) -> list[float]:
    """Rotation matrix -> scalar-first quaternion (same as the preview builder)."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]


DEPTH_CLIP_MM = 5000.0


def log_depth_gray(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 millimetres -> the exact log1p visualization the model was trained on."""
    d = np.clip(depth_mm.astype(np.float32), 0, DEPTH_CLIP_MM)
    return (np.log1p(d) / math.log1p(DEPTH_CLIP_MM) * 255).astype(np.uint8)


class CameraGeometry:
    """Projection helpers for one MuJoCo camera expressed in robot0_base.

    ``rotation_base_camera``/``position_base_camera`` come from
    ``data.cam_xmat``/``data.cam_xpos`` premultiplied by base_from_world.
    MuJoCo cameras look along -z with x right and y up (OpenGL frame).
    """

    def __init__(
        self,
        position_base_camera,
        rotation_base_camera,
        fovy_deg: float,
        resolution: int = RESOLUTION,
        pixel_center: float | None = None,
    ):
        self.position = np.asarray(position_base_camera, dtype=np.float64)
        self.rotation = np.asarray(rotation_base_camera, dtype=np.float64).reshape(3, 3)
        self.resolution = resolution
        self.fovy_deg = float(fovy_deg)
        self.focal = 0.5 * resolution / math.tan(math.radians(fovy_deg) / 2.0)
        self.center = resolution / 2.0 if pixel_center is None else pixel_center

    def project(self, points_base) -> tuple[np.ndarray, np.ndarray]:
        """robot0_base points -> (pixel xy, positive depth in metres).

        The pixel space is the observation image (rotate-180 ``read_pixels``
        output), which is u-mirrored relative to the raw OpenGL projection:
        calibrating against the oracle mask on 2026-09-30 put the bottle body
        centre 18 px from the mask centroid with the u mirror and ~110 px
        without it.
        """
        p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
        cam = (p - self.position) @ self.rotation
        depth = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.resolution - (cam[:, 0] * self.focal / depth + self.center)
            v = -cam[:, 1] * self.focal / depth + self.center
        return np.stack([u, v], axis=1), depth

    def unproject(self, pixel_xy, depth_m: float) -> np.ndarray:
        """One pixel + metric depth -> robot0_base surface point (diagnostics)."""
        u, v = float(pixel_xy[0]), float(pixel_xy[1])
        if depth_m <= 0:
            raise ValueError("unproject needs positive metric depth")
        u_gl = self.resolution - u
        cam = np.array(
            [
                (u_gl - self.center) / self.focal * depth_m,
                -(v - self.center) / self.focal * depth_m,
                -float(depth_m),
            ]
        )
        return self.rotation @ cam + self.position

    def as_dict(self) -> dict:
        return {
            "position_base_camera_m": [round(float(v), 6) for v in self.position],
            "rotation_base_camera": [[round(float(v), 6) for v in row] for row in self.rotation],
            "fovy_deg": self.fovy_deg,
            "focal_px": self.focal,
            "center_px": self.center,
            "resolution": self.resolution,
            "coordinate_frame": "robot0_base",
            "camera_axes": "OpenGL: +x right, +y up, -z forward",
            "pixel_projection": "u=resolution-(fx*x/(-z)+cx); v=-fy*y/(-z)+cy",
            "depth_convention": "positive optical-axis depth in metres, not Euclidean range",
        }


class PixelCenterCamera(CameraGeometry):
    def unproject(self, point, depth):
        return super().unproject((point[0] + 0.5, point[1] + 0.5), depth)

    def project(self, points):
        uv, d = super().project(points)
        return uv - 0.5, d

    def as_dict(self):
        d = super().as_dict()
        d["pixel_sample_offset_xy"] = [0.5, 0.5]
        d["pixel_projection"] = "u=res-(fx*x/(-z)+cx)-0.5; v=-fy*y/(-z)+cy-0.5"
        return d


def camera_contract(camera):
    """Exact K/T for the already mirrored pixel-centre images, proper SO(3) pose."""
    shift = camera.as_dict().get("pixel_sample_offset_xy", [0, 0])
    n, f, c = camera.resolution, camera.focal, camera.center
    K = np.array([[-f, 0, n - c - shift[0]], [0, f, c - shift[1]], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = camera.rotation @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = camera.position
    return dict(
        K_image_px=K.tolist(),
        T_robot_base_from_camera_optical=T.tolist(),
        transform_direction=(
            "p_robot_base = T_robot_base_from_camera_optical @ [p_camera_optical,1]"
        ),
        unprojection="p_camera_optical = depth_optical_m * inverse(K_image_px) @ [u,v,1]",
        image_size_wh=[n, n],
        pixel_origin="top_left_integer_pixel_centres",
        horizontal_image_mirror=True,
        camera_optical_axes=(
            "right-handed +x optical right, +y down, +z forward; displayed image is "
            "horizontally mirrored"
        ),
        translation_frame="robot0_base",
        depth_convention="positive camera optical-axis metres, not Euclidean range",
    )
