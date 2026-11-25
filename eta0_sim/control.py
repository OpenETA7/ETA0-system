"""Validated absolute OSC goals, interpolation and execution receipts."""

import copy
import math

import numpy as np

from .geometry import matrix_to_quaternion_wxyz, quaternion_to_matrix

GRIPPER = {"open": -1.0, "hold": 0.0, "close": 1.0}


def slerp(a, b, t):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    b = np.asarray(b, float)
    b = b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b = -b
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        c = a + t * (b - a)
        return c / np.linalg.norm(c)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


class WorkspaceViolation(ValueError):
    def __init__(self, feedback):
        super().__init__("trajectory_outside_robot_chain_bound")
        self.feedback = feedback


def validate_trajectory_workspace(action, workspace):
    """Reject the whole path before stepping; the convex sphere covers linear interpolation."""
    if (
        workspace.get("coordinate_frame") != "robot0_base"
        or workspace.get("pose_origin") != "gripper0_grip_site"
    ):
        raise ValueError("invalid_workspace_frame_or_pose_origin")
    sphere = workspace["chain_outer_sphere"]
    center = np.asarray(sphere["center_base_m"], float)
    radius = float(sphere["radius_m"])
    if (
        center.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("invalid_workspace_outer_bound")
    tolerance = 1e-6
    offending = []
    for i, knot in enumerate(action["knots"]):
        point = np.asarray(knot["position_base_m"], float)
        distance = float(np.linalg.norm(point - center))
        if distance > radius + tolerance:
            offending.append(
                dict(
                    knot_index=i,
                    tick=knot["tick"],
                    position_base_m=point.tolist(),
                    radius_from_chain_origin_m=distance,
                    excess_m=distance - radius,
                )
            )
    if offending:
        raise WorkspaceViolation(
            dict(
                error_code="trajectory_outside_robot_chain_bound",
                executed_ticks=0,
                coordinate_frame="robot0_base",
                pose_origin="gripper0_grip_site",
                center_base_m=center.tolist(),
                radius_m=radius,
                tolerance_m=tolerance,
                offending_knots=offending,
                hard_rejection=True,
                coordinates_clipped=False,
                instruction=(
                    "No part of this segment executed. Replan every offending knot from current "
                    "measured eef_pose_base. Do not continue from a rejected goal. Inside the bound"
                    " is not an IK or collision guarantee."
                ),
            )
        )


VERSION = "interactive-keynodes-preview-v2"


MAX_TICKS = 60


def validate_action(action, context):
    fields = {
        "stage",
        "schema_version",
        "observation_id",
        "coordinate_frame",
        "control_hz",
        "horizon_ticks",
        "gripper_command",
        "knots",
    }
    if not isinstance(action, dict) or set(action) != fields:
        raise ValueError("motion_traj_fields")
    if context.get("scene_tracking_required") and not context.get("scene_tracking_complete"):
        raise ValueError("scene_tracking_required_for_current_observation")
    if action["stage"] != "motion_traj" or action["schema_version"] != VERSION:
        raise ValueError("motion_traj_schema")
    if action["observation_id"] != context["observation_id"]:
        raise ValueError("stale_observation")
    if action["coordinate_frame"] != "robot0_base":
        raise ValueError("motion_traj_coordinate_frame")
    if type(action["control_hz"]) is not int or action["control_hz"] != 20:
        raise ValueError("control_hz_must_be_20")
    n = action["horizon_ticks"]
    if type(n) is not int or not 1 <= n <= MAX_TICKS:
        raise ValueError("horizon_ticks_must_be_1_to_60")
    if action["gripper_command"] not in ("open", "close", "hold"):
        raise ValueError("gripper_command")
    knots = action["knots"]
    if not isinstance(knots, list) or not 1 <= len(knots) <= n:
        raise ValueError("knots_length")
    result = copy.deepcopy(action)
    previous = -1
    for k in result["knots"]:
        if not isinstance(k, dict) or set(k) != {"tick", "position_base_m", "quaternion_wxyz"}:
            raise ValueError("knot_fields")
        t = k["tick"]
        if type(t) is not int or not previous < t < n:
            raise ValueError("knot_ticks_strictly_increasing")
        previous = t
        for key, size in [("position_base_m", 3), ("quaternion_wxyz", 4)]:
            v = k[key]
            if (
                not isinstance(v, list)
                or len(v) != size
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
            ):
                raise ValueError("invalid_" + key)
        q = np.asarray(k["quaternion_wxyz"], float)
        length = np.linalg.norm(q)
        if abs(length - 1) > 0.01:
            raise ValueError("quaternion_must_be_unit_wxyz")
        # JSON rounding is the only accepted quaternion normalization error.
        k["quaternion_wxyz"] = (q / length).tolist()
    if knots[0]["tick"] != 0 or knots[-1]["tick"] != n - 1:
        raise ValueError("knots_must_cover_zero_to_horizon_minus_one")
    if context.get("workspace_hard_check"):
        validate_trajectory_workspace(result, context["robot_workspace"])
    return result


def expand_action(action, context):
    a = validate_action(action, context)
    ks = a["knots"]
    j = 0
    poses = []
    for t in range(a["horizon_ticks"]):
        while j + 1 < len(ks) and ks[j + 1]["tick"] <= t:
            j += 1
        lo, hi = ks[j], ks[min(j + 1, len(ks) - 1)]
        f = (t - lo["tick"]) / (hi["tick"] - lo["tick"]) if hi["tick"] != lo["tick"] else 0.0
        pos = (1 - f) * np.asarray(lo["position_base_m"]) + f * np.asarray(hi["position_base_m"])
        poses.append(
            dict(
                position_m=pos.tolist(),
                quaternion_wxyz=slerp(lo["quaternion_wxyz"], hi["quaternion_wxyz"], f).tolist(),
                gripper_command=a["gripper_command"],
            )
        )
    return poses


def pose_error(actual, goal):
    p = float(np.linalg.norm(np.asarray(actual["position_m"]) - goal["position_m"]))
    a = np.asarray(actual["quaternion_wxyz"])
    a = a / np.linalg.norm(a)
    b = np.asarray(goal["quaternion_wxyz"])
    b = b / np.linalg.norm(b)
    return dict(
        position_error_m=p,
        rotation_error_deg=float(np.degrees(2 * np.arccos(np.clip(abs(a @ b), 0.0, 1.0)))),
    )


class AbsoluteOSC:
    """Use OSC's absolute goal API; do not clip/re-encode an absolute goal as a delta."""

    def __init__(self, env, provider):
        self.env, self.p = env, provider
        ctrl = env.robots[0].controller
        if ctrl.impedance_mode != "fixed" or not ctrl.use_ori:
            raise ValueError("requires_fixed_impedance_OSC_POSE")
        physical_env = getattr(env, "env", env)
        if int(physical_env.control_freq) != 20:
            raise ValueError("requires_20Hz_environment")

    def step(self, pose):
        p = self.p
        ctrl = self.env.robots[0].controller
        rotation = p.world_from_base()
        position_world = rotation @ np.asarray(pose["position_m"]) + p._data.xpos[p._base_id]
        rotation_world = rotation @ np.asarray(quaternion_to_matrix(pose["quaternion_wxyz"]))
        original = ctrl.set_goal
        # Restore the controller method even if physics or a callback fails.
        had_instance_method = "set_goal" in ctrl.__dict__
        instance_method = ctrl.__dict__.get("set_goal")
        applied = []

        def absolute_goal(action, *args, **kwargs):
            original(action, set_pos=position_world, set_ori=rotation_world)
            xyz, rot = p.to_base(ctrl.goal_pos, ctrl.goal_ori)
            actual_goal = dict(
                position_m=xyz.tolist(), quaternion_wxyz=matrix_to_quaternion_wxyz(rot)
            )
            err = pose_error(actual_goal, pose)
            if err["position_error_m"] > 1e-6 or err["rotation_error_deg"] > 1e-3:
                raise RuntimeError("OSC_modified_requested_equilibrium_goal")
            applied.append(actual_goal)

        ctrl.set_goal = absolute_goal
        try:
            raw, _, _, _ = self.env.step(np.r_[np.zeros(6), GRIPPER[pose["gripper_command"]]])
        finally:
            if had_instance_method:
                ctrl.set_goal = instance_method
            else:
                del ctrl.set_goal
        p.sim_steps += 1
        if not applied:
            raise RuntimeError("OSC_goal_callback_not_called")
        return raw, applied[-1]


class TrajectoryExecutor:
    def __init__(self, servo, provider, max_tracking_error_m=0.18, settle_ticks=0):
        if not np.isfinite(max_tracking_error_m) or max_tracking_error_m <= 0:
            raise ValueError("tracking_limit_must_be_positive")
        self.servo, self.p = servo, provider
        self.max_tracking_error_m = max_tracking_error_m
        if type(settle_ticks) is not int or not 0 <= settle_ticks <= 40:
            raise ValueError("settle_ticks_must_be_0_to_40")
        self.settle_ticks = settle_ticks

    def execute(self, action, context, frame_sink=None):
        poses = expand_action(action, context)  # all validation before any physics
        reports = []
        before = copy.deepcopy(self.p.eef_pose_base())
        reason = "path_complete"
        next_tick = len(poses)
        for tick, pose in enumerate(poses):
            pre = self.p.eef_pose_base()
            error = pose_error(pre, pose)
            if error["position_error_m"] > self.max_tracking_error_m:
                reason = "tracking_lag_reobserve"
                next_tick = tick
                break
            try:
                raw, applied = self.servo.step(pose)
                actual = copy.deepcopy(self.p.eef_pose_base())
                if not np.isfinite(actual["position_m"] + actual["quaternion_wxyz"]).all():
                    raise RuntimeError("nonfinite_measured_pose")
                report = dict(
                    tick=tick,
                    requested_pose=copy.deepcopy(pose),
                    applied_goal_pose=applied,
                    actual_pose=actual,
                    **pose_error(actual, pose),
                )
                reports.append(report)
                if frame_sink:
                    frame_sink(raw, tick, report)
            except Exception as exc:
                # A physics failure is not a zero-tick model validation error.
                raise RuntimeError(
                    f"physical_execution_failed_after_{len(reports)}_reported_ticks:{exc}"
                ) from exc
        path_ticks = len(reports)
        # Optional diagnostic mode. Preserve the requested path and terminal goal;
        # allow physical convergence, but stop if a fixed goal makes no progress.
        settle_errors = []
        if reason == "path_complete" and self.settle_ticks:
            reason = "pose_settle_budget"
            for extra in range(self.settle_ticks + 1):
                err = pose_error(self.p.eef_pose_base(), poses[-1])
                if err["position_error_m"] <= 0.005 and err["rotation_error_deg"] <= 3:
                    reason = "path_complete"
                    break
                settle_errors.append(err)
                if len(settle_errors) >= 9:
                    old = settle_errors[-9]
                    if (
                        old["position_error_m"] - err["position_error_m"] < 0.001
                        and old["rotation_error_deg"] - err["rotation_error_deg"] < 0.5
                    ):
                        reason = "pose_stalled_reobserve"
                        break
                if extra == self.settle_ticks:
                    break
                try:
                    raw, applied = self.servo.step(poses[-1])
                    actual = copy.deepcopy(self.p.eef_pose_base())
                    if not np.isfinite(actual["position_m"] + actual["quaternion_wxyz"]).all():
                        raise RuntimeError("nonfinite_measured_pose")
                    report = dict(
                        tick=len(reports),
                        phase="terminal_settle",
                        requested_pose=copy.deepcopy(poses[-1]),
                        applied_goal_pose=applied,
                        actual_pose=actual,
                        **pose_error(actual, poses[-1]),
                    )
                    reports.append(report)
                    if frame_sink:
                        frame_sink(raw, report["tick"], report)
                except Exception as exc:
                    raise RuntimeError(
                        f"physical_execution_failed_after_{len(reports)}_reported_ticks:{exc}"
                    ) from exc
        actual = copy.deepcopy(self.p.eef_pose_base())
        final_error = pose_error(actual, poses[-1])
        result = dict(
            termination=reason,
            requested_ticks=len(poses),
            executed_ticks=len(reports),
            ticks=len(reports),
            elapsed_sim_seconds=len(reports) / 20.0,
            timing_mode="nominal_20Hz_bounded_terminal_settle"
            if self.settle_ticks
            else "nominal_20Hz_no_extra_settle",
            extra_ticks=len(reports) - path_ticks,
            path_ticks=path_ticks,
            time_scaled=False,
            before_eef_pose_base=before,
            actual_eef_pose_base=actual,
            requested_final_pose=poses[-1],
            last_applied_goal_pose=reports[-1]["applied_goal_pose"] if reports else None,
            final_goal_error=final_error,
            final_pose_within_tolerance=final_error["position_error_m"] <= 0.002
            and final_error["rotation_error_deg"] <= 2,
            settle_pose_within_tolerance=final_error["position_error_m"] <= 0.005
            and final_error["rotation_error_deg"] <= 3,
            last_executed_tick=path_ticks - 1 if path_ticks else None,
            next_tick=next_tick,
            completed_knot_count=sum(k["tick"] < path_ticks for k in action["knots"]),
            remaining_knots=[copy.deepcopy(k) for k in action["knots"] if k["tick"] >= path_ticks],
            automatic_continuation=False,
            grasp_verified=False,
            instruction=(
                "Path completion is not measured arrival, successful grasp or task success. Re-"
                "observe and plan from actual_eef_pose_base."
            ),
        )
        return result, reports
