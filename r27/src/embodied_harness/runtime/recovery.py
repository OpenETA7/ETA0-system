"""Evidence-based failure equivalence; no simulator object identities or poses."""
import copy
import json
import math


class ProgressRecovery:
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.failures = {}
        self.robot_position = None

    def observe(self, state, *, corrective=False, exclude_key=None):
        """Reset only after a distinct completed corrective action, never a failed drift."""
        position = (state.get("robot") or {}).get("grip_xyz_m")
        if (not isinstance(position, (list, tuple)) or len(position) != 3
                or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in position)):
            return []
        self.robot_position = list(position)
        reset = []
        if not corrective:
            return reset
        for key, failure in self.failures.items():
            old = failure.get("robot_position")
            if (key != exclude_key and old is not None and failure["stagnant_attempts"]
                    and math.dist(position, old) >= .03):
                failure["stagnant_attempts"] = 0
                failure["robot_position"] = list(position)
                failure["position_error_m"] = None
                reset.append(key)
        return reset

    def key(self, tool, args, points):
        values = copy.deepcopy(args)
        for name in ("point_id", "contact_point_id", "destination_point_id"):
            if name in values and values[name] in points:
                values[name] = {"measured_xyz": points[values[name]]["xyz_m"]}
        # These fields do not change the requested physical target. Increasing a
        # horizon or renaming a motion cannot erase evidence of an obstruction.
        for name in ("motion_mode", "position_tolerance_m", "orientation_tolerance_rad",
                     "max_steps", "timeout", "timeout_s"):
            values.pop(name, None)

        def normalize(value, field=""):
            if isinstance(value, dict):
                return {k: normalize(v, k) for k, v in sorted(value.items())}
            if isinstance(value, (list, tuple)):
                # Measured site tolerance is 1cm; changed approach/jaw vectors
                # remain distinct. This is a local failure key, not object_ref.
                precision = 1 if field in {"approach_world", "jaw_world", "quat", "quaternion"} else 2
                return [round(float(x), precision) if isinstance(x, (int, float)) else normalize(x, field) for x in value]
            if isinstance(value, float):
                return round(value, 3)
            return value
        return json.dumps([tool, normalize(values)], sort_keys=True)

    def blocked(self, tool, args, points, max_attempts=None):
        value = self.failures.get(self.key(tool, args, points))
        threshold = min(self.threshold, max_attempts) if max_attempts is not None else self.threshold
        if value and value["failure_kind"] == "depth_discontinuity":
            point = points.get(args.get("point_id"), {})
            if (point.get("measurement") or {}).get("depth_edge") is False:
                # The 1 cm motion-equivalence bucket can contain a nearby solid
                # sample. Fresh depth evidence outranks that coarse blacklist;
                # the adapter still checks this reference's frame freshness.
                return None
            threshold = 1  # A fresh label cannot repair the same invalid depth sample.
        return copy.deepcopy(value) if value and value["stagnant_attempts"] >= threshold else None

    @staticmethod
    def corrective_feedback(reason):
        if reason == "servo_horizon":
            return "Target was NOT reached. Repeated motion has no meaningful measured improvement. Inspect obstruction/contact; retreat or change approach/pose using current measurements. Changing motion_mode/tolerance alone is not progress."
        if reason == "depth_discontinuity":
            return "This depth site is unreliable. Inspect a solid interior surface and select a genuinely different measured site, not a renamed candidate at the same position."
        if "retention" in reason:
            return "Holding evidence was lost. Stop placement, observe the actual object, then reacquire with a different measured grasp."
        return "Repeated contact/approach attempt failed. Inspect the current scene and change the grasp site, jaw orientation or approach. Do not repeat the same failed physical target."

    def _failure(self, tool, args, points, reason, position_error=None):
        key = self.key(tool, args, points)
        previous = self.failures.get(key, {})
        old_error = previous.get("position_error_m")
        improved = (reason == "servo_horizon" and old_error is not None and position_error is not None
                    and old_error-position_error > max(0.003, abs(old_error)*0.10))
        count = 1 if improved else previous.get("stagnant_attempts", 0)+1
        value = {"failure_key": key, "tool": tool, "failure_kind": reason,
                 "stagnant_attempts": count, "position_error_m": position_error,
                 "robot_position": copy.deepcopy(self.robot_position),
                 "measured_progress": improved, "instruction": self.corrective_feedback(reason)}
        self.failures[key] = value
        return copy.deepcopy(value)

    def record(self, tool, args, points, detail):
        feedback = detail.get("feedback") or {}
        motion = feedback.get("motion") or {}
        status = detail.get("macro_status") or motion.get("execution_state")
        if status == "not_started" or (not detail.get("macro_status")
                                      and motion.get("physics_executed") is False):
            # The ordinary receipt retains the exact tool/preflight error. A
            # rejected argument envelope is no evidence about target geometry,
            # and must neither poison nor clear prior physical failure memory.
            return None
        if self.execution_completed(detail):
            self.failures.pop(self.key(tool, args, points), None)
            return None
        reason = motion.get("reason_code") if status == "partial" else status
        if not reason:
            return None
        error = (motion.get("motion_summary") or {}).get("position_error_m")
        if not isinstance(error, (int, float)) or not math.isfinite(error):
            error = None
        return self._failure(tool, args, points, str(reason), error)

    @staticmethod
    def execution_completed(detail):
        motion = (detail.get("feedback") or {}).get("motion") or {}
        status = detail.get("macro_status") or motion.get("execution_state")
        return status in {"reached", "completed", "lift_completed_verify_retention_visually",
                          "released_verify_placement_visually", "pull_completed_verify_visually"}

    def record_error(self, tool, args, points, message):
        text = str(message)
        if "depth discontinuity" in text.lower():
            return self._failure(tool, args, points, "depth_discontinuity")
        if "Possible held object" in text:
            # A hold precondition belongs to current robot state, not to a
            # permanently unsuitable grasp site. Releasing must permit a retry.
            return {"failure_key": self.key(tool, args, points), "tool": tool,
                    "failure_kind": "unresolved_held_object", "stagnant_attempts": 0,
                    "instruction": "Resolve the currently possible held load before another grasp: inspect and explicitly release if appropriate. This is a state precondition, not evidence that the target site is bad."}
        return None
