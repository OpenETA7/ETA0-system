"""Real simulator smoke: depth, short movement, telemetry and tick-boundary cancel."""
import asyncio
import json
import sys
from scipy.spatial.transform import Rotation
from pathlib import Path
from embodied_harness.adapters.libero_direct import DirectLiberoAdapter


async def main():
    output = Path(sys.argv[1])
    if output.exists(): raise RuntimeError("Use a new output directory")
    a = DirectLiberoAdapter(output, "unused", "openeta/libero_libero_spatial_task0-v0", "controller diagnostic", 0)
    try:
        await a.start()
        initial = await a.observe(0)
        view = initial.state["views"][0]
        marked = await a.inspect("mark_point", {"source_packet_id": view["source_packet_id"],
            "camera_frame_id": view["camera_frame_id"], "x": 385, "y": 295}, 0)
        samples = []
        move = await a.execute("move_to", {"delta_m": [0,0,.03]}, 1, lambda: False, samples.append)
        robot=a.cached.state["robot"]
        yaw=Rotation.from_rotvec([0,0,.15])
        rotated=await a.execute("move_to", {"approach_world":yaw.apply(robot["approach_world"]).tolist(),
            "jaw_world":yaw.apply(robot["jaw_world"]).tolist()}, 2, lambda:False, samples.append)
        before_cancel = a.cached.state["robot"]["grip_xyz_m"]
        stop = asyncio.Event()
        cancel_samples = []
        def publish(obs):
            cancel_samples.append(obs)
            stop.set()
        cancelled = await a.execute("move_to", {"delta_m": [0,0,.12]}, 3, stop.is_set, publish)
        report = {"task_success_claimed": False, "initial_robot": initial.state["robot"],
            "measured_surface": marked.state["feedback"], "short_move": move, "orientation_move":rotated,
            "stream_samples": len(samples), "cancel_samples": len(cancel_samples),
            "before_cancel": before_cancel, "after_cancel": a.cached.state["robot"]["grip_xyz_m"],
            "cancelled_move": cancelled, "accounting": a.host.execution_accounting}
        report["checks"]={"short_move_reached":move["feedback"]["motion"]["execution_state"]=="reached",
            "orientation_reached":rotated["feedback"]["motion"]["execution_state"]=="reached",
            "streamed_samples":len(samples)>2,"stop_confirmed":cancelled["stop_confirmed"],
            "stopped_before_full_target":cancelled["feedback"]["motion"]["reason_code"]=="cancelled"}
        (output/"smoke.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        assert all(report["checks"].values()), report["checks"]
    finally:
        await a.close()


asyncio.run(main())
