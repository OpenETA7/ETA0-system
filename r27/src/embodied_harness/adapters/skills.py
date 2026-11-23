"""Bounded visual-point grasp skill; every primitive passes through ETA Host.

This is a scripted motor skill, not a learned VLA. Target identity is supplied by
the model's measured pixel; no simulator object pose or reward is read here.
"""
import math
from .gripper_state import possible_hold_after, released
from embodied_harness.domain.tool_errors import GraspOrientationError


GRASP_SCHEMA = {
    "description": "Grasp a freshly marked visible object surface. This bounded skill opens an initially empty gripper, remeasures the pixel, approaches with clearance, descends with explicit contact authorization, closes, checks finger-contact evidence, and lifts only when bilateral contact is measured. Every primitive uses ETA gates and fresh feedback. It can stop at primitive boundaries. It does NOT identify the object or prove retention/task success. If holding an object already, use explicit release first. Prefer this for basic picks instead of manually sequencing open/descend/close/lift. Inspect the final images and contact feedback.",
    "parameters": {
        "type": "object", "additionalProperties": False, "required": ["point_id"],
        "properties": {
            "point_id": {"type": "string", "minLength": 1},
            "object_center_point_id":{"type":"string","minLength":1,"description":"Optional fresh measurement of the object's visible center, separate from a rim grasp point. Enables estimated XY center compensation during placement; no identity guarantee."},
            "contact_offset_z_m": {"type": "number", "minimum": -0.04, "maximum": 0.02,
                                   "description": "Grip-site offset from measured visible surface; default -0.02 m. Surface height is not necessarily grasp center."},
            "clearance_m": {"type": "number", "minimum": 0.05, "maximum": 0.18},
            "lift_m": {"type": "number", "minimum": 0, "maximum": 0.18},
            "approach_world":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3,
                "description":"Optional world-frame approach axis pointing TOWARD contact, not the finger-closing axis. A top-down approach is [0,0,-1]. Supply together with jaw_world; axes must be finite, nonzero and non-collinear. Prefer omitting BOTH when the current measured point has suggested_jaw_world: the skill uses that jaw estimate with top-down approach. Without an estimate, omission preserves the current wrist orientation; it does not infer a rim pinch."},
            "jaw_world":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3,
                "description":"Optional world-frame finger-closing axis ACROSS the grasp site, e.g. spanning one bowl-rim wall. suggested_jaw_world describes this axis, NEVER the approach axis. Supply together with approach_world; not parallel or antiparallel to it (orthogonal axes preferred). Override only using current geometry, or omit BOTH to use the documented default."},
            "contact_offset_m":{"type":"array","items":{"type":"number","minimum":-.05,"maximum":.05},"minItems":3,"maxItems":3},
        },
    },
}


def validate_grasp_orientation(args):
    """Reject impossible explicit orientations before any macro primitive runs.

    Valid non-orthogonal axes retain the host's orthogonalization behavior. This
    checks parameters only: it does not infer a replacement pose or clearance.
    """
    if ("approach_world" in args) != ("jaw_world" in args):
        raise GraspOrientationError("grasp_orientation_pair_required")
    if "approach_world" not in args:
        return
    axes = []
    for field in ("approach_world", "jaw_world"):
        values = args[field]
        if (not isinstance(values, (list, tuple)) or len(values) != 3
                or any(type(v) not in (int, float) for v in values)):
            raise GraspOrientationError("grasp_orientation_invalid")
        try:
            values = [float(v) for v in values]
        except (OverflowError, ValueError):
            raise GraspOrientationError("grasp_orientation_invalid") from None
        if not all(math.isfinite(v) for v in values):
            raise GraspOrientationError("grasp_orientation_invalid")
        norm = math.sqrt(sum(v * v for v in values))
        if not math.isfinite(norm) or norm < 1e-6:
            raise GraspOrientationError("grasp_orientation_invalid")
        axes.append(([v / norm for v in values], norm))
    approach, _ = axes[0]
    jaw, jaw_norm = axes[1]
    dot = sum(a * j for a, j in zip(approach, jaw))
    residual = math.sqrt(sum((j - dot * a) ** 2 for a, j in zip(approach, jaw)))
    # Match the host's projected-jaw magnitude threshold, without adding an
    # angular constraint on valid non-orthogonal, non-unit inputs.
    if residual * jaw_norm < 1e-6:
        raise GraspOrientationError("grasp_orientation_collinear")


def validate_grasp(adapter, args):
    point = adapter.points.get(args["point_id"])
    if point is None:
        raise ValueError("Unknown grasp point; measure it using mark_point first")
    if point.get("placement_anchor"):raise ValueError("Inferred placement opening cannot be used as a measured grasp surface")
    if adapter.possible_hold:
        raise ValueError("Possible held object: explicitly release or resolve it before starting another grasp")
    views = adapter.last.get("views") or []
    if not any(v["source_packet_id"] == point["source_packet_id"] and v["camera_frame_id"] == point["camera_frame_id"] for v in views):
        raise ValueError("Grasp point is from an old observation; mark a fresh visible surface")
    if point.get("measurement", {}).get("depth_edge"):
        raise ValueError("Depth discontinuity: select an interior object surface pixel")
    if args.get("object_center_point_id"):
        center=adapter.points.get(args["object_center_point_id"])
        if not center or center["source_packet_id"]!=point["source_packet_id"]:raise ValueError("Center and grasp must be measured in the same current frame")
        if math.dist(center["xyz_m"][:2],point["xyz_m"][:2])>.12:raise ValueError("Object center too far from grasp surface")
    for key in ("contact_offset_z_m", "clearance_m", "lift_m"):
        if key in args and not math.isfinite(args[key]):
            raise ValueError("Skill offsets must be finite")
    validate_grasp_orientation(args)
    if "contact_offset_m" in args and "contact_offset_z_m" in args:raise ValueError("Choose one contact offset representation")


async def grasp_point(adapter, args, cancelled, publish):
    validate_grasp_orientation(args)
    original = adapter.points[args["point_id"]]
    trace = []

    async def call(name, parameters, *, phase):
        result = await adapter._thread(adapter.host.call, name, parameters)
        obs = adapter._unpack(result)
        publish(obs)
        feedback = adapter.last.get("feedback") or {}
        error = adapter.last.get("error")
        if name == "gripper_control":
            adapter.possible_hold = possible_hold_after(adapter.possible_hold, parameters["action"], adapter.last)
        trace.append({"phase": phase, "tool": name, "args": parameters, "feedback": feedback, "error": error,
                      "observation_id": obs.id})
        return feedback, error

    def result(status, unknown=False):
        return {"skill": "grasp_point", "macro_status": status, "state_unknown": unknown,
                "stop_confirmed": cancelled() and not unknown,
                "observation_id": adapter.cached.id if adapter.cached else None,
                "feedback": {"skill_status": status, "substeps": trace, "retention_proven": False},
                "source": "eta_host_primitive_receipts"}

    def failed(feedback, error):
        motion = feedback.get("motion") or {}
        return bool(error or (isinstance(motion, dict) and
                    motion.get("execution_state") not in {"reached", "completed"}))

    def unknown(feedback, error):
        motion = feedback.get("motion") or {}
        return bool((error or {}).get("code") == "host_execution_error" or
                    isinstance(motion, dict) and motion.get("execution_state") in {"unknown", "state_unknown"})

    if cancelled():
        return result("cancelled_before_open")
    feedback, error = await call("gripper_control", {"action": "open"}, phase="open")
    if failed(feedback, error):
        return result("open_failed", unknown(feedback, error))
    if not released(adapter.last):
        return result("open_not_measured")
    if cancelled():
        return result("cancelled_after_open")
    views = adapter.last.get("views") or []
    source = next((v for v in views if v["camera_frame_id"] == original["camera_frame_id"]), None)
    if source is None:
        return result("camera_unavailable")
    feedback, error = await call("mark_point", {"source_packet_id": source["source_packet_id"],
        "camera_frame_id": source["camera_frame_id"], "x": original["pixel_xy"][0], "y": original["pixel_xy"][1]}, phase="remeasure")
    point = feedback.get("point")
    if error or not point or point.get("measurement", {}).get("depth_edge"):
        return result("remeasure_failed")
    displacement = math.sqrt(sum((a-b)**2 for a, b in zip(point["xyz_m"], original["xyz_m"])))
    if displacement > 0.025:
        return result("surface_changed_reobserve")
    point_id = point["point_id"]
    approach=args.get("approach_world",[0,0,-1])
    length=math.sqrt(sum(x*x for x in approach))
    if length<1e-6:return result("invalid_approach")
    approach=[x/length for x in approach]
    orientation={key:args[key] for key in ("approach_world","jaw_world") if key in args}
    visual=original.get("visual_candidate") or {}
    if not orientation and visual.get("suggested_jaw_world"):
        orientation={"approach_world":[0,0,-1],"jaw_world":visual["suggested_jaw_world"]}
    offset=args.get("contact_offset_m",[0,0,args.get("contact_offset_z_m",-.02)])
    commands = [
        ("approach", "move_to", {"point_id": point_id, "offset_m": [-x*args.get("clearance_m",.10) for x in approach],**orientation}),
        ("contact", "move_to", {"point_id": point_id, "offset_m": offset,
                                "contact_point_id": point_id,**orientation}),
        # Contact motion retires the point generation. Host authorizes this
        # immediate close through its retained, checked close-contact binding.
        ("close", "gripper_control", {"action": "close"}),
    ]
    if "position_tolerance_m" in getattr(adapter,"schemas",{}).get("move_to",{}).get("parameters",{}).get("properties",{}):
        commands[0][2]["position_tolerance_m"]=.025
    if "orientation_tolerance_rad" in getattr(adapter,"schemas",{}).get("move_to",{}).get("parameters",{}).get("properties",{}):
        commands[0][2]["orientation_tolerance_rad"]=.15
    for phase, tool, parameters in commands:
        if cancelled():
            return result("cancelled_before_" + phase)
        if phase == "close" and not (adapter.last.get("contact_state") or {}).get("close_binding_available"):
            return result("close_binding_unavailable")
        feedback, error = await call(tool, parameters, phase=phase)
        contact_stop=(phase=="contact" and not error and isinstance(feedback.get("motion"),dict) and feedback["motion"].get("execution_state")=="contact_reached")
        # Bounded closure can test a near-contact pose without misreporting
        # the original servo as reached. Handle skills use a wider internal bound.
        motion=feedback.get("motion") or {}
        near_contact=(phase=="contact" and not error and isinstance(motion,dict) and motion.get("execution_state")=="partial"
            and (motion.get("motion_summary") or {}).get("position_error_m",float("inf"))<=args.get("contact_position_tolerance_m",.018)
            and (motion.get("motion_summary") or {}).get("orientation_error_rad",float("inf"))<=.15)
        if near_contact:
            trace[-1]['bounded_close_probe']={'allowed':True,'max_residual_m':args.get('contact_position_tolerance_m',.018),
                'note':'Original servo remains partial; next close must establish bilateral pad contact before lift'}
        if failed(feedback, error) and not contact_stop and not near_contact:
            return result(phase + "_failed", unknown(feedback, error))
    contact = ((feedback.get("motion") or {}).get("motion_summary") or {}).get("gripper_contact") or {}
    bilateral = (contact.get("contact_pattern") == "bilateral_pads" or
                 contact.get("left_fingerpad_contact") is True and contact.get("right_fingerpad_contact") is True)
    if not bilateral:
        return result("grasp_contact_not_established")
    if args.get("object_center_point_id") or visual.get("center_xyz_m"):
        import numpy as np
        center=adapter.points[args["object_center_point_id"]]["xyz_m"] if args.get("object_center_point_id") else visual["center_xyz_m"]
        # Visual pre-grasp estimate, recorded in the current jaw frame so later
        # wrist rotation can transform it. It is not simulator object state.
        robot=adapter.last["robot"];x=np.asarray(robot["jaw_world"]);z=np.asarray(robot["approach_world"])
        basis=np.column_stack((x,np.cross(z,x),z))
        offset=np.array([center[0]-point["xyz_m"][0],center[1]-point["xyz_m"][1],0.])
        adapter.held_center_in_grip=(basis.T@offset).tolist()
    else:adapter.held_center_in_grip=None
    if args.get("lift_m",.12)==0:return result("contact_established_no_lift")
    if cancelled():
        return result("cancelled_before_lift")
    feedback, error = await call("move_to", {"delta_m": [0, 0, args.get("lift_m", 0.12)], "delta_frame": "world"}, phase="lift")
    if failed(feedback, error):
        return result("lift_failed", unknown(feedback, error))
    return result("lift_completed_verify_retention_visually")
