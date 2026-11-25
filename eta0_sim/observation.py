"""Read synchronized robot state and render calibrated views."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import CameraGeometry, log_depth_gray, matrix_to_quaternion_wxyz

OBS_RGB = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}


OBS_DEPTH = {"agentview": "agentview_depth", "wrist": "robot0_eye_in_hand_depth"}


EEF_SITE = "gripper0_grip_site"


BASE_BODY = "robot0_base"


def metric_depth(model, raw_depth) -> np.ndarray:
    """MuJoCo normalized z-buffer -> linear metres (identical to the replay tool)."""
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if not (0.0 < near < far):
        raise ValueError(f"Invalid MuJoCo depth clip planes: near={near}, far={far}")
    depth = near / (1.0 - depth * (1.0 - near / far))
    return np.nan_to_num(depth, nan=0.0, posinf=far, neginf=0.0).clip(0.0, far)


class ObservationProvider:
    """Captures synchronized RGB-D views and robot state from the live env."""

    def __init__(self, env: Any, mujoco: Any, output_dir: Path, pixel_center: float | None = None):
        self.env = env
        self.mujoco = mujoco
        self.out = Path(output_dir)
        (self.out / "obs").mkdir(parents=True, exist_ok=True)
        self.pixel_center = pixel_center
        self.sim_steps = 0
        self._sim = env.sim
        # Raw mujoco structs (the robosuite wrappers lack the mj_name2id API).
        self._model = self._sim.model._model
        self._data = self._sim.data._data
        self._site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        self._base_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        if self._site_id < 0 or self._base_id < 0:
            raise ValueError(f"missing {EEF_SITE} site or {BASE_BODY} body in model")
        self._cam_ids = {
            label: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for label, name in (("agentview", "agentview"), ("wrist", "robot0_eye_in_hand"))
        }
        if any(cid < 0 for cid in self._cam_ids.values()):
            raise ValueError("missing agentview or robot0_eye_in_hand camera in model")

    # -- frame transforms ---------------------------------------------------

    def world_from_base(self) -> np.ndarray:
        return np.asarray(self._data.xmat[self._base_id], dtype=np.float64).reshape(3, 3)

    def base_from_world(self) -> np.ndarray:
        return self.world_from_base().T

    def to_base(self, position_world, rotation_world=None):
        base = self.base_from_world()
        p = base @ (
            np.asarray(position_world, dtype=np.float64)
            - np.asarray(self._data.xpos[self._base_id], dtype=np.float64)
        )
        if rotation_world is None:
            return p
        return p, base @ np.asarray(rotation_world, dtype=np.float64).reshape(3, 3)

    def eef_pose_base(self) -> dict:
        position, rotation = self.to_base(
            self._data.site_xpos[self._site_id], self._data.site_xmat[self._site_id]
        )
        return {
            "position_m": [round(float(v), 6) for v in position],
            "quaternion_wxyz": [round(float(v), 6) for v in matrix_to_quaternion_wxyz(rotation)],
        }

    def eef_position_world(self) -> np.ndarray:
        return np.asarray(self._data.site_xpos[self._site_id], dtype=np.float64).copy()

    def camera_geometry(self, camera_label: str) -> CameraGeometry:
        cam_id = self._cam_ids[camera_label]
        position, rotation = self.to_base(self._data.cam_xpos[cam_id], self._data.cam_xmat[cam_id])
        return CameraGeometry(
            position, rotation, float(self._model.cam_fovy[cam_id]), pixel_center=self.pixel_center
        )

    def gripper_open_width_m(self) -> float:
        # Panda two-finger: sum of both finger joint positions in metres.
        names = [
            self.mujoco.mj_id2name(self._model, self.mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self._model.njnt)
        ]
        widths = []
        for index, name in enumerate(names):
            if name and "finger" in name and "joint" in name:
                qposadr = int(self._model.jnt_qposadr[index])
                widths.append(abs(float(self._data.qpos[qposadr])))
        if not widths:
            return float("nan")
        return float(sum(widths))

    # -- observation capture --------------------------------------------------

    def capture(self, camera_label: str, obs: dict, tag: str) -> dict:
        """Render-free capture: consumes an observation dict returned by the env."""
        rgb = np.ascontiguousarray(np.asarray(obs[OBS_RGB[camera_label]])[::-1, ::-1])
        depth_m = metric_depth(self._model, np.asarray(obs[OBS_DEPTH[camera_label]])[::-1, ::-1])
        depth_mm = np.round(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
        stem = f"{tag}-{camera_label}"
        rgb_path = self.out / "obs" / f"{stem}-rgb.png"
        depth_path = self.out / "obs" / f"{stem}-depthgray.png"
        metric_path = self.out / "obs" / f"{stem}-depth-mm.png"
        Image.fromarray(rgb).save(rgb_path, optimize=True)
        Image.fromarray(log_depth_gray(depth_mm)).save(depth_path, optimize=True)
        Image.fromarray(depth_mm).save(metric_path)
        return {
            "camera": camera_label,
            "rgb": rgb,
            "depth_mm": depth_mm,
            "rgb_path": str(rgb_path),
            "depth_gray_path": str(depth_path),
            "metric_depth_path": str(metric_path),
            "camera_geometry": self.camera_geometry(camera_label),
            "eef_pose_base": self.eef_pose_base(),
            "gripper_open_width_m": round(self.gripper_open_width_m(), 6),
        }

    def fresh_observation(self) -> dict:
        """Refresh sensors without re-setting simulator state during contact.

        regenerate_obs_from_state re-applies qpos/qvel and runs forward; even
        restoring the current pose can change the ongoing contact trajectory.
        Actual state restoration belongs only in offline demonstration replay.
        """
        runtime = self.env.env
        runtime._update_observables(force=True)
        return runtime._get_observations()


def synchronize_gripper(env, provider):
    """After restoring qpos, align Panda integrator + ctrl without stepping/resetting."""
    robot = env.robots[0]
    g = robot.gripper
    m = provider._model
    d = provider._data
    if type(g).__name__ != "PandaGripper":
        raise ValueError("unsupported_gripper_sync")
    aids = [env.sim.model.actuator_name2id(n) for n in g.actuators]
    before = np.asarray(g.current_action).copy()
    desired = []
    for aid in aids:
        if int(m.actuator_trntype[aid]) != 0:
            raise ValueError("gripper_nonjoint_actuator")
        jid = int(m.actuator_trnid[aid, 0])
        q = float(d.qpos[int(m.jnt_qposadr[jid])])
        lo, hi = m.actuator_ctrlrange[aid]
        if not lo - 1e-5 <= q <= hi + 1e-5:
            raise ValueError("restored_gripper_out_of_range")
        if abs(m.actuator_gainprm[aid, 0] + m.actuator_biasprm[aid, 1]) > 1e-5:
            raise ValueError("not_position_servo")
        desired.append(float(np.clip(q, lo, hi)))
    limits = m.actuator_ctrlrange[aids]
    bias = limits.mean(axis=1)
    weight = (limits[:, 1] - limits[:, 0]) / 2
    targets = np.asarray(desired)
    normalized = (targets - bias) / weight
    g.current_action = normalized.copy()
    d.ctrl[aids] = targets
    return dict(
        profile="gripper-guard-v1",
        normalized_before=before.tolist(),
        normalized_after=normalized.tolist(),
        actuator_targets=targets.tolist(),
        physical_state_modified=False,
        sim_steps=0,
    )
