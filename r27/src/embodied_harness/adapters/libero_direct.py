"""Simulator-only OSC adapter. No ETA controller or hidden object-state policy.

This is a distinct control condition, NOT a replacement claiming ETA collision
coverage. MuJoCo simulates contacts; action magnitudes and horizons are bounded.
Only robot proprioception, rendered RGB-D and fingerpad contact bits are public.
"""
import asyncio
import base64
import io
import json
import re
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from .openeta import OpenETAAdapter
from .skills import GRASP_SCHEMA


def unproject(x, y, depth, fovy, camera_rotation, camera_position):
    h, w = depth.shape
    f = .5 * h / np.tan(np.deg2rad(fovy) / 2)
    camera_xyz = np.array([(x - w / 2) / f, (y - h / 2) / f, 1.]) * depth[y, x]
    # robosuite camera convention: right/down/forward -> MuJoCo right/up/back.
    return np.asarray(camera_position) + np.asarray(camera_rotation) @ (camera_xyz * [1, -1, -1])


def bounded_control(delta_xyz, delta_rotation, output_max):
    scale = np.asarray(output_max, dtype=float)
    if scale.shape != (6,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Unsupported OSC action scale")
    delta = np.r_[np.clip(delta_xyz, -.02, .02), np.clip(delta_rotation, -.12, .12)]
    return np.clip(delta / scale, -1, 1)


def visible_normal(x,y,depth,fovy,rotation,position):
    """Local RGB-D surface normal, oriented toward the observing camera."""
    pixels=[(u,v) for u in range(max(0,x-3),min(depth.shape[1],x+4))
                  for v in range(max(0,y-3),min(depth.shape[0],y+4))
                  if abs(float(depth[v,u])-float(depth[y,x]))<.015]
    if len(pixels)<9:return None
    points=np.array([unproject(u,v,depth,fovy,rotation,position) for u,v in pixels])
    _,singular,axes=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
    if singular[1]<1e-5 or singular[2]/singular[1]>.5:return None
    normal=axes[-1]
    if np.dot(normal,np.asarray(position)-points.mean(axis=0))<0:normal=-normal
    return normal.tolist()


def support_height(depth,fovy,rotation,position,robot_position):
    """Dominant visible horizontal surface below the robot; geometric estimate only."""
    h,w=depth.shape;v,u=np.mgrid[4:h:8,4:w:8];f=.5*h/np.tan(np.deg2rad(fovy)/2)
    d=depth[v,u];camera=np.stack(((u-w/2)/f*d,-(v-h/2)/f*d,-d),axis=-1)
    world=camera@np.asarray(rotation).T+np.asarray(position)
    du=world[1:-1,2:]-world[1:-1,:-2];dv=world[2:,1:-1]-world[:-2,1:-1]
    normals=np.cross(du,dv);norm=np.linalg.norm(normals,axis=-1)
    centers=world[1:-1,1:-1]
    valid=(np.abs(normals[...,2])>.97*norm)&(norm>1e-8)&(centers[...,2]<robot_position[2]-.06)
    valid &= np.linalg.norm(centers[...,:2]-np.asarray(robot_position)[:2],axis=-1)<.8
    z=centers[...,2][valid]
    if len(z)<12:return None
    bins,counts=np.unique(np.round(z/.005).astype(int),return_counts=True)
    best=bins[np.argmax(counts)]
    if counts.max()<8:return None
    return float(np.median(z[np.round(z/.005).astype(int)==best]))


def horizontal_bar_outward(x,y,depth,fovy,rotation,position):
    """Estimate a horizontal elongated surface axis; curved normals are unstable.

    Used only by the explicitly requested handle skill, never semantic identity.
    """
    from .vision_geometry import world_cloud
    cloud=world_cloud(depth,fovy,rotation,position);point=cloud[y,x]
    local=cloud[max(0,y-40):y+41,max(0,x-40):x+41].reshape(-1,3)
    local=local[(np.linalg.norm(local-point,axis=-1)<.07)&(np.abs(local[:,2]-point[2])<.012)]
    if len(local)<30:return None
    values,vectors=np.linalg.eigh(np.cov(local[:,:2].T))
    if values[-1]<.0001 or values[-1]<4*max(values[0],1e-12):return None
    # The visible side of a cylinder is perspective-biased. Fit its dominant
    # horizontal line by support, rather than the average of curved point normals.
    xy=local[:,:2];best_count=0;best_mask=None
    for angle in np.linspace(0,np.pi,180,endpoint=False):
        n=np.array([np.cos(angle),np.sin(angle)]);distance=xy@n
        bins,counts=np.unique(np.round(distance/.003).astype(int),return_counts=True)
        index=int(np.argmax(counts))
        if counts[index]>best_count:
            best_count=int(counts[index]);best_mask=np.abs(distance-bins[index]*.003)<.004
    axis=vectors[:,-1]
    if best_count>=max(30,.18*len(local)):
        values,fit=np.linalg.eigh(np.cov(xy[best_mask].T))
        if values[-1]>.0001:axis=fit[:,-1]
    normal=np.array([-axis[1],axis[0],0.])
    if np.dot(normal,np.asarray(position)-point)<0:normal=-normal
    return normal.tolist()


def schema(description, properties, required=()):
    return SimpleNamespace(description=description, inputSchema={"type": "object", "properties": properties,
        "required": list(required), "additionalProperties": False})


VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
POINT = {"type": "string", "minLength": 1}


class LiberoHost:
    cameras = ("agentview", "robot0_eye_in_hand")

    def __init__(self, env_id, seed, output):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from pathlib import Path
        match = re.fullmatch(r"openeta/libero_(libero_\w+)_task(\d+)-v0", env_id)
        if not match:
            raise ValueError("Direct adapter only accepts official LIBERO task IDs")
        suite_name, task_id = match.group(1), int(match.group(2))
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        if seed < 0 or seed >= len(states):
            raise ValueError("Initial-state index unavailable")
        self.env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_names=list(self.cameras), camera_heights=512, camera_widths=512, camera_depths=True,
            controller="OSC_POSE", horizon=100000, ignore_done=True)
        self.env.seed(seed)
        self.env.reset()
        self.obs = self.env.set_init_state(states[seed])
        self.gripper = -1.
        for _ in range(10):
            self.obs, _, _, _ = self.env.step(np.r_[np.zeros(6), self.gripper])
        self.sim = self.env.env.sim
        self.robot = self.env.env.robots[0]
        self.reference_robot_position=np.asarray(self.obs["robot0_eef_pos"]).copy()
        self.packet = 1
        self.generation = 1
        self.points = {}
        self._point_seq = 0
        self.binding = False
        self.closed = False
        self.official_success = bool(self.env.check_success())
        self.runner = SimpleNamespace(terminated=False, truncated=False, failure_reason=None)
        self.execution_accounting = {"preflight_calls": 0, "physical_dispatches": 0,
            "physical_calls_with_steps": 0, "physical_unknown_outcomes": 0, "control_steps": 0}
        self.feedback = {}
        self.candidate_cache=None
        self.proposal_mode="objects"
        self.cancelled = lambda: False
        self.on_sample = None
        self.last_sample = 0.
        self.step_failed = False
        self.trace = Path(output) / "direct-commands.jsonl"
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        import hashlib
        initial = np.asarray(states[seed])
        (Path(output) / "direct-initialization.json").write_text(json.dumps({"suite": suite_name,
            "task_index": task_id, "initial_state_index": seed, "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "task_language": task.language, "controller": "OSC_POSE", "settling_steps": 10,
            "sensing": "RGB-D, robot proprioception, fingerpad contact bits", "object_state_policy": False}, indent=2))
        self.schemas = {
            "select_candidate":schema("Select a numbered RGB-D surface component visible in the CURRENT agentview image. Copy the COMPLETE versioned ID, e.g. p138o-c3; old IDs expire after motion or proposal-mode change and never track objects. IDs are GEOMETRY ONLY, not semantic identity. Choose by actual appearance and task relations. For picking a wide bowl choose a rim part; for a can choose top_surface and for a destination container choose placement_center (an inferred opening from visible 3D footprint, NOT a measured floor). Returns measured point_id, estimated center offset and rim jaw direction. This avoids manually transcribing pixel coordinates. Components can merge/miss objects; use observe/crop or mark_point when ambiguous.",{
                "candidate_id":{"type":"string","pattern":"^p[0-9]+[ob]-c[0-9]+$"},
                "part":{"type":"string","enum":["center","placement_center","top_surface","top_rim","bottom_rim","left_rim","right_rim"]}},("candidate_id","part")),
            "locate_point":schema("Preferred visual measurement using explicit NORMALIZED 0-1000 coordinates: u=0 left,1000 right; v=0 top,1000 bottom. The host converts to canonical pixels and returns point_id, xyz and an annotated marker. This does not recognize object identity. Use the canonical full image, not zoom-preview-relative coordinates.",{
                "u":{"type":"number","minimum":0,"maximum":1000},"v":{"type":"number","minimum":0,"maximum":1000},
                "camera_frame_id":{"type":"string","enum":list(self.cameras)}},("u","v")),
            "observe": schema("Read scene. candidate_mode=horizontal_bars proposes thin horizontal-world-axis surface regions for handle search; objects uses whole elevated components. IDs never encode semantic identity. crop_xyxy uses ORIGINAL 512x512 coordinates and returns the full canonical frame PLUS a magnified crop with original-coordinate rulers. Select mark_point pixels on the canonical full image, never on the zoom preview.", {
                "candidate_mode":{"type":"string","enum":["objects","horizontal_bars"]},
                "camera": {"type": "string", "enum": [*self.cameras, "wrist"]},
                "crop_xyxy": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 512}, "minItems": 4, "maxItems": 4}}),
            "mark_point": schema("Measure a visible surface from the CURRENT calibrated depth image, original pixel x,y. Gives world xyz_m, never object identity. Remeasure after object motion.", {
                "source_packet_id": POINT, "camera_frame_id": {"type": "string", "enum": list(self.cameras)},
                "x": {"type": "integer", "minimum": 0, "maximum": 511},
                "y": {"type": "integer", "minimum": 0, "maximum": 511}}, ("source_packet_id", "camera_frame_id", "x", "y")),
            "move_to": schema("SIMULATION ONLY. Bounded OSC end-effector servo, with MuJoCo contacts, no collision-free path guarantee or IK preflight. Choose one xyz_m, point_id+offset_m, delta_m, or rotation-only. Preserve orientation by omitting both approach_world and jaw_world; otherwise supply both. Translation per call <=0.8m. Plan clearance yourself. Contact point records intent, not identity. Each control step can stop on cancellation; progress samples arrive during movement.", {
                "xyz_m": VECTOR, "point_id": POINT, "offset_m": VECTOR, "delta_m": VECTOR,
                "delta_frame": {"type": "string", "enum": ["world", "grip_site"]},
                "approach_world": VECTOR, "jaw_world": VECTOR, "contact_point_id": POINT,
                "position_tolerance_m":{"type":"number","minimum":.003,"maximum":.025,"description":"Clearance waypoint tolerance; default6mm. Use up to25mm only for free-space approach, not final placement."},
                "orientation_tolerance_rad":{"type":"number","minimum":.03,"maximum":.15,"description":"Orientation tolerance for waypoint approach. Default .08 rad; final grasp still uses contact and retention checks."},
                "maintain_grip_contact":{"type":"boolean","description":"Stop motion if both fingerpad contacts disappear for three control steps while carrying."},
                "motion_mode": {"type": "string", "enum": ["strict", "recovery"]}}),
            "gripper_control": schema("Open or close gripper in simulated physics, with measured fingerpad contacts. Closing is not proof of grasp. Open retires contact marks; an immediate close may reuse preceding contact-move binding.", {
                "action": {"type": "string", "enum": ["open", "close"]}, "contact_point_id": POINT}, ("action",)),
        }

    def pose(self):
        # robosuite's observation quaternion belongs to the EEF BODY, whereas
        # OSC tracks the grip SITE. Their fixed yaw differs on Panda.
        return np.asarray(self.obs["robot0_eef_pos"]), np.asarray(self.sim.data.site_xmat[self.robot.eef_site_id]).reshape(3,3).copy()

    def contact(self):
        groups = self.robot.gripper.important_geoms
        ids = {k: {self.sim.model.geom_name2id(n) for n in groups[k]} for k in ("left_fingerpad", "right_fingerpad")}
        all_gripper = {self.sim.model.geom_name2id(n) for n in self.robot.gripper.contact_geoms}
        touched = {k: False for k in ids}
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            for k, group in ids.items():
                if c.geom1 in group and c.geom2 not in all_gripper or c.geom2 in group and c.geom1 not in all_gripper:
                    touched[k] = True
        return {"left_fingerpad_contact": touched["left_fingerpad"], "right_fingerpad_contact": touched["right_fingerpad"],
            "contact_pattern": "bilateral_pads" if all(touched.values()) else "unilateral_pad" if any(touched.values()) else "no_contact",
            "retention_proven": False}

    def state(self):
        pos, rot = self.pose()
        q = np.asarray(self.obs["robot0_gripper_qpos"])
        openness = float(np.clip((abs(q[0]) + abs(q[1])) / .08, 0, 1))
        return {"grip_xyz_m": pos.tolist(), "approach_world": rot[:, 2].tolist(), "jaw_world": rot[:, 0].tolist(),
                "gripper": {"openness": openness, "aperture_m": float(abs(q[0]) + abs(q[1]))}}

    def result(self, error=None, camera=None, crop=None, telemetry_only=False):
        body = {"robot": self.state(), "feedback": self.feedback, "error": error,
            "tactile":self.contact(),
            "sensor": {"captured_at": time.monotonic(), "clock": "host_monotonic", "sim_time": float(self.sim.data.time),
                       "control_steps": self.execution_accounting["control_steps"]},
            "contact_state": {"close_binding_available": self.binding, "measurement_generation": self.generation,
                              "retention_proven": False}, "views": [],
            "point_semantics": "Calibrated visible surface only; object identity and held state unverified."}
        body["sensor"]["visual_sample"] = not telemetry_only
        if telemetry_only:
            # Do not run region proposals / PNG encoding on the control-critical path.
            # The next boundary observation supplies fresh visual evidence independently.
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(body))])
        content = []
        candidates,_,_=self.candidates()
        body["candidate_scope"]=f"direct-{self.packet}/{self.proposal_mode}"
        body["visible_candidates"]=[{k:v for k,v in c.items() if k!='component'} for c in candidates]
        body["candidate_semantics"]="Unlabelled elevated RGB-D components, may merge/miss objects. IDs refer to current canonical agentview only. Select by image appearance, never by a presumed semantic ID."
        for name in self.cameras:
            if camera and name != ("robot0_eye_in_hand" if camera == "wrist" else camera):
                continue
            rgb = np.asarray(self.obs[name + "_image"])[::-1].copy()
            frame=Image.fromarray(rgb)
            if name=="agentview":
                draw=ImageDraw.Draw(frame)
                for candidate in candidates:
                    x0,y0,x1,y1=candidate["bbox_xyxy"]
                    draw.rectangle((x0,y0,x1-1,y1-1),outline="#00ccff",width=1)
                    draw.text((x0,max(0,y0-12)),candidate["candidate_id"],fill="white",stroke_width=1,stroke_fill="black")
            mark=(self.feedback or {}).get("point")
            if mark and mark["source_packet_id"]==f"direct-{self.packet}" and mark["camera_frame_id"]==name:
                draw=ImageDraw.Draw(frame);x,y=mark["pixel_xy"]
                draw.ellipse((x-5,y-5,x+5,y+5),outline="red",width=2)
                draw.text((min(x+8,390),max(0,y-15)),f"{mark['point_id']} ({x},{y})",fill="red",stroke_width=1,stroke_fill="white")
            stream = io.BytesIO(); frame.save(stream, format="PNG")
            body["views"].append({"source_packet_id": f"direct-{self.packet}", "camera_frame_id": name,
                                  "pixel_origin": "top_left", "is_current_observation": True})
            content.append(SimpleNamespace(type="image", data=base64.b64encode(stream.getvalue()).decode()))
            # Destination selection needs the same readable ID-to-appearance
            # binding as source selection. Holding an object must not silently
            # remove these cards. Wrist remains available by explicit observe.
            if name=='agentview' and not crop and candidates:
                from .visual_cards import candidate_gallery
                gallery=candidate_gallery(rgb,candidates);stream=io.BytesIO();gallery.save(stream,format='PNG')
                body['views'].append({'source_packet_id':f'direct-{self.packet}','camera_frame_id':'candidate_gallery',
                    'source_camera':'agentview','measurement_allowed':False,
                    'layout':'Each large c-ID labels its exact cropped region. Card positions are NOT scene spatial relations. Select IDs; never mark gallery pixels.'})
                content.append(SimpleNamespace(type='image',data=base64.b64encode(stream.getvalue()).decode()))
            if crop:
                x0,y0,x1,y1=crop
                if not (0<=x0<x1<=512 and 0<=y0<y1<=512):raise ValueError("Invalid crop")
                zoom=Image.fromarray(rgb).crop(tuple(crop)).resize((512,512))
                draw=ImageDraw.Draw(zoom)
                for t in (0,128,256,384):
                    draw.text((t,0),f"x={round(x0+t*(x1-x0)/512)}",fill="white",stroke_width=1,stroke_fill="black")
                    if t:draw.text((0,t),f"y={round(y0+t*(y1-y0)/512)}",fill="white",stroke_width=1,stroke_fill="black")
                selected_id=None
                visual=(mark or {}).get('visual_candidate') or {};bbox=visual.get('bbox_xyxy')
                if bbox and crop==[max(0,bbox[0]-8),max(0,bbox[1]-8),min(512,bbox[2]+8),min(512,bbox[3]+8)]:
                    selected_id=visual.get('candidate_id')
                    draw.rectangle((0,474,511,511),fill='#182633')
                    draw.text((12,480),f'CURRENT SELECTED REGION: {selected_id}',font=ImageFont.load_default(size=22),fill='white')
                stream=io.BytesIO();zoom.save(stream,format="PNG")
                body["views"].append({"source_packet_id":f"direct-{self.packet}","camera_frame_id":"zoom_preview","source_camera":name,"original_crop_xyxy":crop,
                    "measurement_allowed":False,"selected_candidate_id":selected_id,"coordinate_rule":"mark_point always uses original full-frame pixels"})
                content.append(SimpleNamespace(type="image",data=base64.b64encode(stream.getvalue()).decode()))
                break
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(body)), *content])

    def depth(self,name):
        normalized=np.asarray(self.obs[name+"_depth"])[::-1].squeeze()
        extent=self.sim.model.stat.extent;near=self.sim.model.vis.map.znear*extent;far=self.sim.model.vis.map.zfar*extent
        return near/(1-normalized*(1-near/far))

    def candidates(self):
        if self.candidate_cache and self.candidate_cache[0]==self.packet:return self.candidate_cache[1]
        from .vision_geometry import propose_surfaces,propose_horizontal_bars
        depth=self.depth("agentview");cam=self.sim.model.camera_name2id("agentview")
        rotation=self.sim.data.cam_xmat[cam].reshape(3,3);position=self.sim.data.cam_xpos[cam];fovy=self.sim.model.cam_fovy[cam]
        support=support_height(depth,fovy,rotation,position,self.reference_robot_position)
        proposer=propose_horizontal_bars if self.proposal_mode=="horizontal_bars" else propose_surfaces
        result=proposer(depth,fovy,rotation,position,support,self.reference_robot_position)
        for candidate in result[0]:
            candidate['candidate_id']=f"p{self.packet}{'b' if self.proposal_mode=='horizontal_bars' else 'o'}-"+candidate['candidate_id']
        self.candidate_cache=(self.packet,result)
        return result

    def step(self, arm):
        try:
            self.obs, _, _, _ = self.env.step(np.r_[arm, self.gripper])
        except BaseException:
            self.step_failed = True
            raise
        self.packet += 1
        self.execution_accounting["control_steps"] += 1
        # Evaluator-only terminal latch, matching benchmark stop-on-success.
        # Neither reward nor this bit enters the policy's state projection.
        self.official_success = self.official_success or bool(self.env.check_success())
        now = time.monotonic()
        if self.on_sample and now - self.last_sample >= .5 and not self.cancelled():
            self.last_sample = now
            # Public robot/contact telemetry remains live without expensive visual packaging.
            self.feedback = {"motion": {"execution_state": "running", "physics_executed": True,
                "motion_summary": {"control_steps_total": self.execution_accounting["control_steps"], "gripper_contact": self.contact()}}}
            self.on_sample(self.result(telemetry_only=True))

    def point(self, point_id, contact=False):
        point = self.points[point_id]
        if contact and point["generation"] != self.generation:
            raise ValueError("Contact point expired; remeasure")
        return point

    def call(self, tool, args):
        started = time.time()
        before = self.execution_accounting["control_steps"]
        self.step_failed = False
        selected=None;selected_jaw=None;selected_part=None
        try:
            if tool=="select_candidate":
                from .vision_geometry import candidate_surface
                candidates,labels,cloud=self.candidates()
                selected=next((c for c in candidates if c['candidate_id']==args['candidate_id']),None)
                if selected is None:raise ValueError("Expired or unknown visual candidate ID: use the complete ID from the CURRENT frame/card; old IDs are never rebound to a different object")
                selected_part=args['part']
                x,y,selected_jaw=candidate_surface(selected,labels,cloud,args['part'],
                    reliable=getattr(self, "reliable_surfaces", False))
                tool="mark_point";args={"source_packet_id":f"direct-{self.packet}","camera_frame_id":"agentview","x":x,"y":y}
            if tool=="locate_point":
                tool="mark_point"
                args={"source_packet_id":f"direct-{self.packet}","camera_frame_id":args.get("camera_frame_id","agentview"),
                      "x":int(round(float(args["u"])*511/1000)),"y":int(round(float(args["v"])*511/1000))}
            if tool == "observe":
                if args.get("candidate_mode"):
                    self.proposal_mode=args["candidate_mode"];self.candidate_cache=None
                return self.result(camera=args.get("camera"), crop=args.get("crop_xyxy"))
            if tool == "mark_point":
                if args["source_packet_id"] != f"direct-{self.packet}":
                    raise ValueError("Old camera packet")
                name, x, y = args["camera_frame_id"], args["x"], args["y"]
                normalized = np.asarray(self.obs[name + "_depth"])[::-1].squeeze()
                extent = self.sim.model.stat.extent
                near, far = self.sim.model.vis.map.znear * extent, self.sim.model.vis.map.zfar * extent
                depth = near / (1 - normalized * (1 - near / far))
                camera = self.sim.model.camera_name2id(name)
                xyz = unproject(x, y, depth, self.sim.model.cam_fovy[camera], self.sim.data.cam_xmat[camera].reshape(3,3), self.sim.data.cam_xpos[camera])
                patch = depth[max(0,y-1):y+2,max(0,x-1):x+2]
                support=support_height(depth,self.sim.model.cam_fovy[camera],self.sim.data.cam_xmat[camera].reshape(3,3),
                                       self.sim.data.cam_xpos[camera],self.reference_robot_position)
                nearby=depth[max(0,y-12):y+13,max(0,x-12):x+13]
                self._point_seq += 1
                point = {"point_id": f"point-{self._point_seq}", "source_packet_id": args["source_packet_id"],
                    "camera_frame_id": name, "pixel_xy": [x,y], "xyz_m": xyz.tolist(), "generation": self.generation,
                    "normalized_uv":[round(x/511*1000,1),round(y/511*1000,1)],
                    "measurement": {"depth_edge": bool(np.ptp(patch) > .04), "depth_m": float(depth[y,x]),
                        "camera_facing_surface_normal_world":visible_normal(x,y,depth,self.sim.model.cam_fovy[camera],
                            self.sim.data.cam_xmat[camera].reshape(3,3),self.sim.data.cam_xpos[camera]),
                        "horizontal_bar_outward_world":horizontal_bar_outward(x,y,depth,self.sim.model.cam_fovy[camera],
                            self.sim.data.cam_xmat[camera].reshape(3,3),self.sim.data.cam_xpos[camera]),
                        "dominant_support_height_m":support,"height_above_support_m":None if support is None else float(xyz[2]-support),
                        "local_depth_prominence_m":float(np.median(nearby)-depth[y,x]),
                        "geometry_note":"Visible geometry estimate, not semantic object identity. Flat support is not a graspable object."}}
                if not np.all(np.isfinite(xyz)):
                    raise ValueError("Non-finite depth")
                if selected:
                    if getattr(self, "reliable_surfaces", False):
                        point["measurement"]["selection_depth_continuity"] = "bounded_3x3_world_patch_not_semantic_identity"
                    point["visual_candidate"]={"candidate_id":selected["candidate_id"],"bbox_xyxy":selected["bbox_xyxy"],"center_xyz_m":xyz.tolist() if selected_part=='top_surface' else selected["center_xyz_m"],"suggested_jaw_world":selected_jaw,"provenance":"unlabelled_visible_depth_component"}
                    if selected_part=='placement_center':
                        from .vision_geometry import estimate_placement_opening
                        estimate=estimate_placement_opening(selected,labels,cloud)
                        point['xyz_m']=estimate
                        point['placement_anchor']='estimated_visible_region_opening'
                        point['measurement']={'depth_edge':False,'provenance':'inferred_from_visible_component_3d_extent_and_upper_edge',
                            'not_a_measured_floor':True,'uncertainty':'Region may be merged, irregular or occluded; visually verify container identity and footprint'}
                        cp=self.sim.data.cam_xmat[camera].reshape(3,3).T@(np.asarray(estimate)-self.sim.data.cam_xpos[camera])
                        focal=256/np.tan(np.deg2rad(self.sim.model.cam_fovy[camera])/2)
                        point['pixel_xy']=[int(np.clip(round(256+focal*cp[0]/-cp[2]),0,511)),int(np.clip(round(256-focal*cp[1]/-cp[2]),0,511))]
                        point.pop('normalized_uv',None)
                self.points[point["point_id"]] = point
                self.feedback = {"point": point}
            elif tool == "move_to":
                self.move(args)
            elif tool == "gripper_control":
                if args.get("contact_point_id"):
                    self.point(args["contact_point_id"], contact=True)
                self.gripper = -1. if args["action"] == "open" else 1.
                self.execution_accounting["physical_dispatches"] += 1
                for _ in range(24):
                    if self.cancelled() or self.official_success: break
                    self.step(np.zeros(6))
                self.generation += 1; self.binding = False
                steps = self.execution_accounting["control_steps"] - before
                self.feedback = {"gripper_command": args["action"], "attachment_proven": False,
                    "motion": {"execution_state": "partial" if self.cancelled() else "completed", "physics_executed": steps > 0,
                    "motion_summary": {"steps_executed": steps, "gripper_contact": self.contact()}}}
            else:
                raise ValueError("Unsupported direct tool")
            if selected:
                x0,y0,x1,y1=selected["bbox_xyxy"]
                return self.result(camera="agentview",crop=[max(0,x0-8),max(0,y0-8),min(512,x1+8),min(512,y1+8)])
            return self.result()
        except Exception as exc:
            steps = self.execution_accounting["control_steps"] - before
            # An env.step exception could have mutated physics without returning.
            unknown = self.step_failed or not isinstance(exc, (ValueError, KeyError))
            self.feedback = {"motion": {"execution_state": "unknown" if unknown else "partial" if steps else "not_started",
                                        "physics_executed": None if unknown else steps > 0}}
            if unknown: self.execution_accounting["physical_unknown_outcomes"] += 1
            return self.result(error={"code": "host_execution_error" if unknown else "direct_tool_rejected", "message": str(exc)[:500]})
        finally:
            steps = self.execution_accounting["control_steps"] - before
            if steps: self.execution_accounting["physical_calls_with_steps"] += 1
            with self.trace.open("a") as f:
                f.write(json.dumps({"tool": tool, "args": args, "started_at": started, "elapsed_s": time.time()-started,
                                    "control_steps": steps, "feedback": self.feedback}) + "\n")

    def move(self, args):
        pos, rotation = self.pose()
        if sum(k in args for k in ("xyz_m", "point_id", "delta_m")) > 1:
            raise ValueError("Choose one position representation")
        target = pos.copy()
        if "xyz_m" in args: target = np.asarray(args["xyz_m"], dtype=float)
        if "point_id" in args: target = np.asarray(self.point(args["point_id"])["xyz_m"]) + args.get("offset_m", [0,0,0])
        if "delta_m" in args:
            delta = np.asarray(args["delta_m"], dtype=float)
            target += rotation @ delta if args.get("delta_frame") == "grip_site" else delta
        if not np.all(np.isfinite(target)) or np.linalg.norm(target-pos) > .8:
            raise ValueError("Translation exceeds bounded simulator action")
        if ("approach_world" in args) != ("jaw_world" in args):
            raise ValueError("Supply both approach_world and jaw_world, or omit both")
        if "approach_world" in args:
            z, x = np.asarray(args["approach_world"], dtype=float), np.asarray(args["jaw_world"], dtype=float)
            if not np.isfinite(z).all() or not np.isfinite(x).all() or np.linalg.norm(z) < 1e-6:
                raise ValueError("Invalid orientation basis")
            z /= np.linalg.norm(z); x -= z * np.dot(x,z)
            if np.linalg.norm(x) < 1e-6: raise ValueError("Collinear orientation basis")
            x /= np.linalg.norm(x); rotation = np.column_stack((x,np.cross(z,x),z))
        if args.get("contact_point_id"):
            contact = self.point(args["contact_point_id"], contact=True)
            if np.linalg.norm(target-np.asarray(contact["xyz_m"])) > .12:
                raise ValueError("Contact intent too far from measured surface")
        self.execution_accounting["physical_dispatches"] += 1
        before = self.execution_accounting["control_steps"]
        reached = False
        contact_stop = False
        lost_contact=False;contact_gap=0
        tolerance=float(args.get("position_tolerance_m",.006))
        angle_tolerance=float(args.get("orientation_tolerance_rad",.08))
        if not .003<=tolerance<=.025:raise ValueError("Position tolerance outside supported range")
        if not .03<=angle_tolerance<=.15:raise ValueError("Orientation tolerance outside supported range")
        for _ in range(120):
            current, current_rot = self.pose()
            delta = target - current
            rot_delta = Rotation.from_matrix(rotation @ current_rot.T).as_rotvec()
            if args.get("maintain_grip_contact"):
                pads=self.contact()
                contact_gap=0 if pads["left_fingerpad_contact"] or pads["right_fingerpad_contact"] else contact_gap+1
                if contact_gap>=3:lost_contact=True;break
            if args.get("contact_point_id") and np.linalg.norm(current-np.asarray(contact["xyz_m"]))<=.06 and np.linalg.norm(rot_delta)<=.12:
                pads=self.contact()
                if pads["left_fingerpad_contact"] or pads["right_fingerpad_contact"]:
                    contact_stop=True;break
            if np.linalg.norm(delta) <= tolerance and np.linalg.norm(rot_delta) <= angle_tolerance:
                reached = True; break
            if self.cancelled() or self.official_success: break
            self.step(bounded_control(delta, rot_delta, self.robot.controller.output_max))
        steps = self.execution_accounting["control_steps"] - before
        self.binding = bool(args.get("contact_point_id")) and (steps > 0 or contact_stop)
        if self.binding: self.generation += 1
        current, current_rot = self.pose()
        reached = bool(np.linalg.norm(target-current) <= tolerance and
                       np.linalg.norm(Rotation.from_matrix(rotation @ current_rot.T).as_rotvec()) <= angle_tolerance)
        self.feedback = {"target": {"xyz_m": target.tolist()}, "motion": {
            "execution_state": "retention_lost" if lost_contact else "contact_reached" if contact_stop else "reached" if reached else "partial", "physics_executed": steps > 0,
            "reason_code": "fingerpad_contact_lost" if lost_contact else "fingerpad_contact_stop" if contact_stop else "target_reached" if reached else "cancelled" if self.cancelled() else "servo_horizon",
            "motion_summary": {"steps_executed": steps, "position_error_m": float(np.linalg.norm(target-current)),
                "orientation_error_rad":float(np.linalg.norm(Rotation.from_matrix(rotation @ current_rot.T).as_rotvec())),
                "gripper_contact": self.contact(), "collision_coverage": "simulated_contacts_only_no_path_certificate"}}}

    def success_evidence(self):
        return [{"source": "official_libero_check_success", "sim_time": float(self.sim.data.time)}] if self.official_success else []

    def close(self, reason=None):
        if not self.closed:
            self.env.close(); self.closed = True


class DirectLiberoAdapter(OpenETAAdapter):
    cancellation = "control_tick_boundary"

    def _unpack(self, result):
        observation = super()._unpack(result)
        sensor = self.last["sensor"]
        observation = replace(observation, captured_at=sensor["captured_at"], sim_time=sensor["sim_time"],
                              state={**observation.state, "sensor": sensor}, provenance="libero_rgbd_proprioception_tactile")
        self.cached = observation
        return observation

    async def start(self):
        _, env, _, seed, _, _ = self.config
        self.host = await self._thread(LiberoHost, env, seed, self.output)
        self.host.reliable_surfaces = getattr(self, "reliable_surfaces", False)
        self.schemas = {k: {"description": v.description, "parameters": v.inputSchema} for k,v in self.host.schemas.items()}
        self.schemas["grasp_point"] = {**GRASP_SCHEMA, "description": GRASP_SCHEMA["description"].replace(
            "Every primitive uses ETA gates and fresh feedback.",
            "Every primitive uses bounded OSC and fresh feedback; MuJoCo simulates contacts, with no collision-free path guarantee.")}
        from .placement import PLACE_SCHEMA
        self.schemas["place_point"] = PLACE_SCHEMA
        from .manipulation import PULL_SCHEMA
        self.schemas["pull_handle"] = PULL_SCHEMA
        self._unpack(await self._thread(self.host.result))
        self.initial_packet_pending = True
        return self

    def validate_args(self, capability, args):
        super().validate_args(capability,args)
        if capability=="place_point":
            from .placement import validate_place
            validate_place(self,args)
        if capability=="pull_handle":
            from .manipulation import validate_pull
            validate_pull(self,args)

    async def execute(self, capability, args, fence, cancelled, publish):
        loop = asyncio.get_running_loop()
        cancel_observed_at = None
        def controller_cancelled():
            nonlocal cancel_observed_at
            requested = cancelled()
            if requested and cancel_observed_at is None:
                cancel_observed_at = time.monotonic()
            return requested
        def sampled(result):
            observation = self._unpack(result)
            publish(observation)
        self.host.cancelled = controller_cancelled
        self.host.on_sample = lambda result: loop.call_soon_threadsafe(sampled, result)
        try:
            detail = await super().execute(capability, args, fence, cancelled, publish)
            detail["source"] = "direct_libero_bounded_osc"
            if cancel_observed_at is not None:
                detail["controller_cancel_observed_at"] = cancel_observed_at
            return detail
        finally:
            self.host.on_sample = None

    async def verify(self, observation_id):
        evidence = await self._thread(self.host.success_evidence)
        return {"status": "verified" if evidence else "not_verified", "evidence": evidence,
                "observation_id": observation_id, "source": "official_libero_check_success"}
