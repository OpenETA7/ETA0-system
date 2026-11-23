"""Bounded, simulation-only measured-surface placement primitive."""
import math
from .gripper_state import possible_hold_after, released


PLACE_SCHEMA = {"description": "SIMULATION ONLY. Place/release a possibly held object over a fresh destination anchor. A select_candidate placement_center anchor estimates the visible opening/upper edge, NOT the container floor: its default grip-site release offset is 0.06m above that opening. A manually marked surface has a default offset of 0.12m above that surface. Usually OMIT release_height_m to use the anchor-aware default; adding 0.12m above an already elevated opening can drop a bowl too high and cause bounce or sideways displacement. Uses clearance, transport with estimated held-object center compensation, descent, opening and retreat. No object identity or retention proof. Inspect the destination first; physical contacts are simulated, not collision-certified.",
 "parameters": {"type":"object", "additionalProperties":False, "required":["point_id"], "properties":{
     "point_id":{"type":"string","minLength":1,"description":"A fresh measured feedback.point.point_id returned by a completed selection/measurement call. A visible candidate_id such as p200o-c1 is NOT a measured point_id. Measure the destination/support surface, not the currently held object."},
     "clearance_m":{"type":"number","minimum":.12,"maximum":.3},
     "release_height_m":{"type":"number","minimum":.04,"maximum":.2,
                         "description":"Optional override of grip-site height above point.xyz_m (NOT necessarily a floor). Default 0.06m for an inferred placement_center opening, 0.12m for a manually measured surface. Omit unless visible object geometry requires a different offset."}}}}


def release_height(point, args):
    return args.get("release_height_m", .06 if point.get("placement_anchor") else .12)


def validate_place(adapter, args):
    if not adapter.possible_hold: raise ValueError("No possibly held load; establish grasp before placement")
    tactile=adapter.last.get('tactile')
    if tactile and not tactile.get('left_fingerpad_contact') and not tactile.get('right_fingerpad_contact'):
        raise ValueError('Current fingerpad contact is absent; inspect whether the object was lost before placement')
    point=adapter.points.get(args["point_id"])
    if point is None:
        raise ValueError("Measure destination first: the supplied ID is not a known measured point_id. "
                         "Candidate IDs cannot be used as point IDs. Select or visually measure the chosen "
                         "destination/support surface, wait for its result, then copy feedback.point.point_id. "
                         "No placement motion was executed.")
    if not any(v["source_packet_id"]==point["source_packet_id"] and v["camera_frame_id"]==point["camera_frame_id"] for v in adapter.last.get("views",[])):
        raise ValueError("Remeasure destination in current view")
    if point.get("measurement",{}).get("depth_edge"):raise ValueError("Select an interior destination surface")
    for key in ("clearance_m","release_height_m"):
        if key in args and not math.isfinite(args[key]):raise ValueError("Offsets must be finite")
    if release_height(point,args)>args.get("clearance_m",.25):
        raise ValueError("Release height exceeds transport clearance")


async def place_point(adapter,args,cancelled,publish):
    point=adapter.points[args["point_id"]]
    x,y,z=point["xyz_m"]
    offset=getattr(adapter,"held_center_in_grip",None)
    if offset is not None:
        import numpy as np
        robot=adapter.last["robot"];axis_x=np.asarray(robot["jaw_world"]);axis_z=np.asarray(robot["approach_world"])
        world_offset=np.column_stack((axis_x,np.cross(axis_z,axis_x),axis_z))@np.asarray(offset)
        x-=float(world_offset[0]);y-=float(world_offset[1])
    current=adapter.last["robot"]["grip_xyz_m"]
    height=max(current[2],z+args.get("clearance_m",.25))
    commands=[]
    if height-current[2]>.006:commands.append(("clearance","move_to",{"delta_m":[0,0,height-current[2]]}))
    commands += [("transport","move_to",{"xyz_m":[x,y,height]}),
                 ("lower","move_to",{"xyz_m":[x,y,z+release_height(point,args)]}),
                 ("release","gripper_control",{"action":"open"}),
                 ("retreat","move_to",{"delta_m":[0,0,.1]})]
    trace=[]
    def result(status,unknown=False):
        return {"skill":"place_point","macro_status":status,"state_unknown":unknown,
                "stop_confirmed":cancelled() and not unknown,"observation_id":adapter.cached.id,
                "feedback":{"skill_status":status,"substeps":trace,"placement_proven":False},
                "source":"direct_libero_bounded_osc"}
    for phase,name,params in commands:
        if cancelled():return result("cancelled_before_"+phase)
        if phase in {"clearance","transport","lower"} and "maintain_grip_contact" in getattr(adapter,"schemas",{}).get("move_to",{}).get("parameters",{}).get("properties",{}):
            params={**params,"maintain_grip_contact":True}
        raw=await adapter._thread(adapter.host.call,name,params)
        obs=adapter._unpack(raw);publish(obs)
        body=adapter.last;feedback=body.get("feedback") or {};error=body.get("error") or {}
        trace.append({"phase":phase,"tool":name,"args":params,"feedback":feedback,"error":error,"observation_id":obs.id})
        if name=="gripper_control":adapter.possible_hold=possible_hold_after(adapter.possible_hold,params["action"],body)
        motion=feedback.get("motion") or {}
        unknown=error.get("code")=="host_execution_error" or motion.get("execution_state") in {"unknown","state_unknown"}
        if motion.get('execution_state')=='retention_lost':return result('retention_lost_during_'+phase)
        if error or motion.get("execution_state") not in {"reached","completed"}:
            return result(phase+"_failed",unknown)
        if phase=="release":
            if not released(body):return result("release_not_measured")
            adapter.held_center_in_grip=None
    return result("released_verify_placement_visually")
