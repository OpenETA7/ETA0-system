"""Adapter uses the actual ETA CodexHost ingress, retaining schema/IK/contact gates.

There is no simulator-private object pose access in the cognitive plane.
Cancellation is safe-boundary-only: an in-flight ETA call is never claimed stopped.
"""
import asyncio
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from embodied_harness.domain.models import Observation, uid


class OpenETAAdapter:
    cancellation = "safe_boundary_only"

    def __init__(self, output, sim_url, env_id, mission, seed=0, timeout=1800, max_requests=500):
        self.output = Path(output)
        self.images = self.output / "frames"
        self.images.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eta-host")
        self.config = (sim_url, env_id, mission, seed, timeout, max_requests)
        self.host = None
        self.generation = 1
        self.version = 0
        self.fence = 0
        self.cached = None
        self.last = None
        self.action_seq = 0
        self.initial_packet_pending = False
        self.points = {}
        self.retired_point_ids = set()
        self.possible_hold = False

    def retire_points(self, point_ids):
        """Revocation survives cached feedback, history resets and later previews."""
        self.retired_point_ids.update(point_ids)
        for point_id in point_ids:
            self.points.pop(point_id, None)

    async def _thread(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def start(self):
        def build():
            from tools.codex_host import parser, build_host
            url, env, mission, seed, timeout, limit = self.config
            args = parser().parse_args(["--sim-url", url, "--env-id", env, "--task", mission,
                                       "--seed", str(seed), "--output", str(self.output / "eta"),
                                       "--tool-profile", "atomic", "--timeout", str(timeout),
                                       "--max-requests", str(limit)])
            return build_host(args)
        self.host = await self._thread(build)
        self.schemas = {k: {"description": v.description, "parameters": v.inputSchema}
                        for k, v in self.host.schemas.items()
                        if k in {"observe", "mark_point", "move_to", "gripper_control"}}
        from .skills import GRASP_SCHEMA
        self.schemas["grasp_point"] = GRASP_SCHEMA
        self._unpack(await self._thread(self.host.result))
        self.initial_packet_pending = True
        return self

    def validate_args(self, capability, args):
        import jsonschema
        if capability not in self.schemas:
            raise ValueError("unknown tool")
        jsonschema.validate(args, self.schemas[capability]["parameters"])
        for field in ("point_id", "contact_point_id", "object_center_point_id", "destination_point_id"):
            if args.get(field) in self.retired_point_ids:
                raise ValueError(f"Retired measurement reference: {field}={args[field]}; measure again under the current contract")
        if capability == "grasp_point":
            from .skills import validate_grasp
            validate_grasp(self, args)

    def _unpack(self, result):
        body = json.loads(result.content[0].text)
        paths = []
        self.version += 1
        for i, item in enumerate(result.content):
            if item.type == "image":
                path = self.images / f"obs-{self.version:05d}-{i}.png"
                path.write_bytes(base64.b64decode(item.data))
                paths.append(str(path))
        self.last = body
        point = (body.get("feedback") or {}).get("point")
        if (isinstance(point, dict) and point.get("point_id")
                and point["point_id"] not in self.retired_point_ids):
            self.points[point["point_id"]] = point
        # Explicit allowlist: episode reward/status/hidden objects do not enter model input.
        state = {k: body[k] for k in ("robot", "views", "feedback", "interventions", "error", "repair",
                                    "contact_state", "recent_motion_progress", "point_semantics","tactile","visible_candidates","candidate_semantics","candidate_scope") if k in body}
        state["tool_schemas"] = self.schemas
        from PIL import Image
        for view, path in zip(state.get("views", []), paths):
            with Image.open(path) as im:
                view["width_px"], view["height_px"] = im.size
        state["cancellation"] = self.cancellation
        observation = Observation(uid(), self.version, self.generation, time.monotonic(),
                                  -1.0, self.action_seq, state, tuple(paths), "openeta_rgbd_and_robot_receipt")
        self.cached = observation
        return observation

    async def observe(self, action_seq):
        self.action_seq = action_seq
        if self.initial_packet_pending and action_seq == 0:
            self.initial_packet_pending = False
            return self.cached
        if self.host.runner.terminated or self.host.runner.truncated or self.host.closed:
            if not self.cached:
                raise RuntimeError("terminated before observation")
            return self.cached
        result = await self._thread(self.host.call, "observe", {})
        return self._unpack(result)

    async def inspect(self, tool, args, action_seq):
        if tool not in {"mark_point", "observe","locate_point","select_candidate"}:
            raise ValueError("read_only_tool_required")
        self.validate_args(tool, args)
        self.action_seq = action_seq
        return self._unpack(await self._thread(self.host.call, tool, args))

    async def execute(self, capability, args, fence, cancelled, publish):
        if fence <= self.fence:
            raise RuntimeError("stale simulator lease")
        self.fence = fence
        if cancelled():
            return {"stop_confirmed": True, "dispatched": False, "source": "host_boundary"}
        self.action_seq += 1
        if capability == "place_point":
            from .placement import place_point
            return await place_point(self, args, cancelled, publish)
        if capability == "grasp_point":
            from .skills import grasp_point
            return await grasp_point(self, args, cancelled, publish)
        if capability == "pull_handle":
            from .manipulation import pull_handle
            return await pull_handle(self,args,cancelled,publish)
        result = await self._thread(self.host.call, capability, args)
        observation = self._unpack(result)
        publish(observation)
        error = self.last.get("error") or {}
        feedback = self.last.get("feedback") or {}
        motion = feedback.get("motion") or {}
        if not isinstance(motion, dict):
            motion = {}
        unknown = (error.get("code") == "host_execution_error" or
                   motion.get("execution_state") in {"unknown", "state_unknown"})
        if capability == "gripper_control":
            from .gripper_state import possible_hold_after
            self.possible_hold = possible_hold_after(self.possible_hold, args["action"], self.last)
        return {"host_error": self.last.get("error"), "feedback": self.last.get("feedback"),
                "observation_id": observation.id,
                "state_unknown": unknown,
                "stop_confirmed": bool(cancelled()), "source": "eta_call_returned_at_boundary"}

    async def verify(self, observation_id):
        evidence = await self._thread(self.host.success_evidence)
        # These fields are evaluator-only. Fast can propose completion but cannot forge them.
        return {"status": "verified" if evidence else "not_verified", "evidence": evidence,
                "observation_id": observation_id, "source": "openeta_official_environment_receipt"}

    @property
    def terminal(self):
        return bool(self.host and (self.host.closed or self.host.runner.terminated or self.host.runner.truncated
                                  or self.host.runner.failure_reason))

    async def close(self):
        if self.host:
            await self._thread(self.host.close, "harness_shutdown")
        self.executor.shutdown(wait=True)
