"""Bind LIBERO to the two-camera, millimetre/degree action protocol."""

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def vector(value):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
    ):
        raise ValueError("expected a finite numeric three-vector")
    return np.asarray(value, dtype=float)


def quaternion(rotvec):
    v = vector(rotvec)
    if np.linalg.norm(v) > 180.2:
        raise ValueError("rotation vector must use the principal degree branch")
    q = Rotation.from_rotvec(v, degrees=True).as_quat()
    return q[[3, 0, 1, 2]].tolist()


def to_controller(action, context):
    from eta0_sim.control import VERSION, validate_action

    a = copy.deepcopy(action)
    n = a.get("horizon_ticks")
    if type(n) is not int or not 1 <= n <= 60:
        raise ValueError("horizon_ticks must be an integer from 1 to 60")
    if a.get("stage") == "move_to":
        if set(a) != {
            "stage",
            "observation_id",
            "delta_position_mm",
            "delta_rotation_vector_deg",
            "gripper_command",
            "horizon_ticks",
        }:
            raise ValueError("move_to requires the six documented fields")
        pose = context["eef_pose_base"]
        current = Rotation.from_quat(np.asarray(pose["quaternion_wxyz"])[[1, 2, 3, 0]])
        # Left multiplication: increment axes are robot-base axes.
        delta = Rotation.from_quat(
            np.asarray(quaternion(a["delta_rotation_vector_deg"]))[[1, 2, 3, 0]]
        )
        q = (delta * current).as_quat()[[3, 0, 1, 2]].tolist()
        target = (np.asarray(pose["position_m"]) + vector(a["delta_position_mm"]) / 1000).tolist()
        knots = [dict(tick=n - 1, position_base_m=target, quaternion_wxyz=q)]
        if n > 1:
            knots.insert(
                0,
                dict(
                    tick=0,
                    position_base_m=pose["position_m"],
                    quaternion_wxyz=pose["quaternion_wxyz"],
                ),
            )
        a = dict(
            stage="motion_traj",
            schema_version=VERSION,
            observation_id=a["observation_id"],
            coordinate_frame="robot0_base",
            control_hz=20,
            horizon_ticks=n,
            gripper_command=a["gripper_command"],
            knots=knots,
        )
    elif a.get("stage") == "motion_traj":
        if a.get("schema_version") != "interactive-keynodes-mm-rotvec-v1":
            raise ValueError("unsupported millimetre/degree trajectory schema")
        a["schema_version"] = VERSION
        for knot in a.get("knots", []):
            if set(knot) != {"tick", "position_base_mm", "rotation_vector_deg"}:
                raise ValueError("trajectory knot fields do not match the documented schema")
            knot["position_base_m"] = (vector(knot.pop("position_base_mm")) / 1000).tolist()
            knot["quaternion_wxyz"] = quaternion(knot.pop("rotation_vector_deg"))
    else:
        raise ValueError("available actions are move_to, motion_traj and finish")
    a = validate_action(a, context)
    if any(np.max(np.abs(k["position_base_m"])) > 2 for k in a["knots"]):
        raise ValueError("goal is outside the declared +/-2000 mm workspace bounds")
    return a


def public_mm(value):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k.endswith("_m"):
                result[k[:-2] + "_mm"] = (np.asarray(v) * 1000).round(3).tolist()
            else:
                result[k] = public_mm(v)
        return result
    if isinstance(value, list):
        return [public_mm(x) for x in value]
    return value


class LiberoAdapter:
    def __init__(
        self, task_id, initial_index, seed, output, visual_queries=False, suite_name="libero_object"
    ):
        import mujoco
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        from eta0_sim.control import AbsoluteOSC, TrajectoryExecutor
        from eta0_sim.observation import ObservationProvider, synchronize_gripper

        self.out = output
        self.visual_queries = visual_queries
        output.mkdir(parents=True, exist_ok=False)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_path = (
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        states = torch.load(init_path, map_location="cpu", weights_only=False)
        state = np.asarray(states[initial_index])
        self.instruction = task.language
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=512,
            camera_widths=512,
            camera_depths=True,
            horizon=10000,
        )
        self.env.seed(seed)
        self.env.reset()
        self.env.set_init_state(state)
        self.p = ObservationProvider(self.env, mujoco, output)
        sync = synchronize_gripper(self.env, self.p)
        self.executor = TrajectoryExecutor(
            AbsoluteOSC(self.env, self.p), self.p, max_tracking_error_m=0.18, settle_ticks=0
        )
        self.previous = None
        self.context = None
        self.view = 0
        self.validated = None
        self.meta = dict(
            suite=suite_name,
            task_id=task_id,
            task_name=task.name,
            instruction=task.language,
            initial_index=initial_index,
            seed=seed,
            initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            initial_file_sha256=hashlib.sha256(init_path.read_bytes()).hexdigest(),
            bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            gripper_initialization=sync,
            initial_success=self.success(),
        )
        (output / "task.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        import imageio.v2 as imageio

        self.video = imageio.get_writer(
            str(output / "rollout.mp4"), fps=5, codec="libx264", ffmpeg_params=["-threads", "1"]
        )

    def success(self):
        return bool(self.env.check_success())

    def observe(self):
        from eta0_sim.geometry import PixelCenterCamera, camera_contract
        from harness import Observation
        from tcp_overlay import CameraCalibration
        from tcp_overlay_adapter import session_geometry

        raw = self.p.fresh_observation()
        captures = {
            c: self.p.capture(c, raw, f"view-{self.view:03d}") for c in ["agentview", "wrist"]
        }
        cameras = {}
        for name, capture in captures.items():
            c = capture["camera_geometry"]
            cameras[name] = camera_contract(
                PixelCenterCamera(c.position, c.rotation, c.fovy_deg, c.resolution, c.center)
            )
        self.context = dict(
            frame=self.p.sim_steps,
            observation_id=f"view{self.view}-tick{self.p.sim_steps}",
            coordinate_frame="robot0_base",
            eef_pose_base=self.p.eef_pose_base(),
            cameras=cameras,
        )
        robot = session_geometry(
            SimpleNamespace(p=self.p, context=self.context, env=self.env), "robot0_base"
        )
        c = cameras["wrist"]
        wrist = CameraCalibration(
            c["K_image_px"],
            c["T_robot_base_from_camera_optical"],
            c["image_size_wh"],
            "robot0_base",
            robot.observation_id,
            "wrist",
        )
        public_cameras = {}
        for name, c in cameras.items():
            transform = np.array(c["T_robot_base_from_camera_optical"])
            transform[:3, 3] *= 1000
            # Public calibration names must match the camera names accepted by
            # visual_query. The internal PhysLM source remains 'agentview'.
            public_name = "external" if name == "agentview" else name
            public_cameras[public_name] = dict(
                K_image_px=c["K_image_px"],
                image_size_wh=c["image_size_wh"],
                T_robot_base_from_camera_optical_mm=transform.tolist(),
            )
        q = np.asarray(self.context["eef_pose_base"]["quaternion_wxyz"])
        state = dict(
            eef_pose_base=public_mm(self.context["eef_pose_base"]),
            rotation_vector_deg=Rotation.from_quat(q[[1, 2, 3, 0]])
            .as_rotvec(degrees=True)
            .round(3)
            .tolist(),
            gripper_open_width_mm=round(self.p.gripper_open_width_m() * 1000, 3),
            previous_execution=public_mm(self.previous),
            cameras=public_cameras,
        )
        with Image.open(captures["agentview"]["rgb_path"]) as im:
            external = im.copy()
        with Image.open(captures["wrist"]["rgb_path"]) as im:
            wrist_rgb = im.copy()
        observation = Observation(external, wrist_rgb, wrist, robot, state)
        if self.visual_queries:
            from eta0_sim.observation import OBS_DEPTH, metric_depth
            from visual_query import QueryView

            depths = {}
            for name, source, image in [
                ("external", "agentview", external),
                ("wrist", "wrist", wrist_rgb),
            ]:
                c = cameras[source]
                calibration = CameraCalibration(
                    c["K_image_px"],
                    c["T_robot_base_from_camera_optical"],
                    c["image_size_wh"],
                    "robot0_base",
                    robot.observation_id,
                    name,
                )
                # Exactly the capture() RGB rotation, but retain optical Z at
                # float precision instead of using the rounded uint16-mm PNG.
                depth = metric_depth(self.p._model, np.asarray(raw[OBS_DEPTH[source]])[::-1, ::-1])
                depths[name] = depth
                observation.query_views[name] = QueryView(image, calibration, depth)
            np.savez_compressed(
                self.out / "obs" / f"view-{self.view:03d}-query-depth.npz", **depths
            )
        return observation

    def validate_action(self, action, observation):
        if action["observation_id"] != self.context["observation_id"]:
            raise ValueError("stale observation")
        self.validated = (copy.deepcopy(action), to_controller(action, self.context))

    def execute(self, action):
        if self.validated is None or self.validated[0] != action:
            raise RuntimeError("physical execution requires an unchanged validated action")
        resolved = self.validated[1]
        self.validated = None

        def frame(raw, tick, report):
            if tick % 4 == 0:
                self.video.append_data(
                    np.concatenate(
                        [
                            raw["agentview_image"][::-1, ::-1],
                            raw["robot0_eye_in_hand_image"][::-1, ::-1],
                        ],
                        axis=1,
                    )
                )

        result, reports = self.executor.execute(resolved, self.context, frame)
        with (self.out / "controller.jsonl").open("a") as f:
            f.write(
                json.dumps(dict(action=action, resolved=resolved, result=result, per_tick=reports))
                + "\n"
            )
        self.previous = result
        self.view += 1
        feedback = public_mm(result)
        (self.out / "latest.json").write_text(
            json.dumps(
                dict(
                    view=self.view,
                    ticks=self.p.sim_steps,
                    native_success=self.success(),
                    feedback=feedback,
                ),
                indent=2,
            )
        )
        return feedback

    def close(self):
        self.video.close()
        self.env.close()


ACTION_PROMPT = (Path(__file__).with_name("eta0_sim") / "actions.txt").read_text()
