"""Rebuild the two-mug audit on a local CPU from historical inputs."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from contextlib import closing
from functools import partial
from itertools import groupby
from pathlib import Path
from unittest.mock import patch

TASK = (
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate"
    "_and_put_the_yellow_and_white_mug_on_the_right_plate"
)
CROSSED = (("porcelain_mug_1", "plate_2"), ("white_yellow_mug_1", "plate_1"))
GRIP = {"open": -1.0, "hold": 0.0, "close": 1.0}


def native_pair(tool, episode, index):
    receipt = tool({"episode_id": episode, "action_index": index})
    feedback = receipt["feedback"]
    if receipt["status"] != "checked" or feedback["native_evaluator_calls_added"] != 0:
        raise ValueError("missing passive native receipt")
    conditions = {
        c["requirement_id"]: c["satisfied_in_last_evaluation"] for c in feedback["conditions"]
    }
    return [conditions["goal_1"], conditions["goal_2"]]


def crossed_pair(env):
    # Deliberately outside the adapter's transition; these are privileged audit probes.
    return [bool(env.env._eval_predicate(["on", mug, plate])) for mug, plate in CROSSED]


def demo_inputs(source):
    import numpy as np

    if source.is_dir():
        files = sorted(source.glob("demo_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
        for f in files:
            with np.load(f) as data:
                yield f.stem, str(data["xml"]), data["states"], data["actions"]
    else:
        import h5py

        with h5py.File(source) as h5:
            for name in sorted(h5["data"], key=lambda n: int(n.split("_")[1])):
                group = h5["data"][name]
                yield name, group.attrs["model_file"], group["states"][:], group["actions"][:]


def local_xml(xml, root):
    import robosuite

    scene = ET.fromstring(xml)
    for node in scene.findall("asset/*[@file]"):
        old, asset = node.get("file").rsplit("assets/", 1)
        base = (
            Path(robosuite.__file__).parent / "models/assets"
            if "robosuite" in old
            else root / "libero/libero/assets"
        )
        node.set("file", str((base / asset).resolve(strict=True)))
    return ET.tostring(scene, encoding="unicode")


def observer(env, pack, root, episode):
    from openeta_bench.adapters.libero import LiberoAdapter
    from openeta_bench.tools import CompletionTool

    adapter = LiberoAdapter(pack, TASK, env.env, root)
    adapter.begin_episode(episode)
    return adapter, CompletionTool(adapter, profile="native_goal_conditions_v1")


def demo_controls(env, state, index, adapter, tool, name):
    checks = {}
    for mode in ("original", "swapped", "yellow_only"):
        index += 1
        env.set_init_state(state)
        data = env.sim.data
        if mode == "swapped":
            left, right = "plate_1_joint0", "plate_2_joint0"
            a, b = (data.get_joint_qpos(key).copy() for key in (left, right))
            data.set_joint_qpos(left, b)
            data.set_joint_qpos(right, a)
        elif mode == "yellow_only":
            joint = "porcelain_mug_1_joint0"
            pose = data.get_joint_qpos(joint).copy()
            pose[2] += 0.2
            data.set_joint_qpos(joint, pose)
        with adapter.observe_transition(index):
            env.sim.forward()
            env.check_success()
        checks[mode] = dict(flags=native_pair(tool, name, index), cross=crossed_pair(env))
    return checks


def collect_demos(env, pack, root, source, output):
    env.reset()
    demos = []
    for name, xml, states, actions in demo_inputs(source):
        env.reset_from_xml_string(local_xml(xml, root))
        adapter, tool = observer(env, pack, root, name)
        rows = []
        for index, state in enumerate(states, 1):
            with adapter.observe_transition(index):
                env.set_state(state)
                env.sim.forward()
                env.check_success()
            rows.append(
                dict(
                    i=index,
                    flags=native_pair(tool, name, index),
                    cross=crossed_pair(env),
                    cmd=float(actions[index - 2, 6]) if index > 1 else None,
                )
            )
        if name == "demo_0":
            frame = next(row["i"] for row in rows if row["flags"] == [True, True])
            cases = demo_controls(env, states[frame - 1], len(rows), adapter, tool, name)
            (output / "swap-control.json").write_text(
                json.dumps(dict(demo=name, frame=frame, **cases), indent=2) + "\n"
            )
        demos.append(dict(demo=name, mode="recorded_states", rows=rows))
    if not demos:
        raise ValueError("no demonstrations")
    (output / "demo-states.json").write_text(json.dumps(demos, separators=(",", ":")))


def finish_actions(source, initial):
    if source is None:
        return set()
    for line in source.read_text().splitlines():
        trace = json.loads(line)
        if (trace["arm"], trace["suite"], trace["task"], trace["initial"]) != (
            "base",
            "libero_10",
            4,
            initial,
        ):
            continue
        actions, finish = 0, set()
        for event in trace["events"]:
            if event[0] == "execution":
                actions += 1
            elif event[0] == "finish_feedback":
                finish.add(actions)
        return finish
    raise ValueError(f"missing event trace for initial {initial}")


def reconstruct(env, pack, root, directory, state, finishes):
    import numpy as np
    from scipy.spatial.transform import Rotation

    meta = json.loads((directory / "task.json").read_text())
    if (
        meta["task_name"] != TASK
        or hashlib.sha256(state.tobytes()).hexdigest() != (meta["initial_state_sha256"])
    ):
        raise ValueError("archived initial state mismatch")
    env.seed(meta["seed"])
    env.reset()
    env.set_init_state(state)
    robot = env.robots[0]
    data, model = env.sim.data, env.sim.model
    jaw = meta["gripper_initialization"]
    robot.gripper.current_action = np.array(jaw["normalized_after"])
    data.ctrl[[model.actuator_name2id(n) for n in robot.gripper.actuators]] = jaw[
        "actuator_targets"
    ]
    base = model.body_name2id("robot0_base")
    site = model.site_name2id("gripper0_grip_site")
    episode = f"recorded-{meta['initial_index']}"
    adapter, tool = observer(env, pack, root, episode)
    rows, elapsed = [], 0
    with (directory / "controller.jsonl").open() as file:
        for index, line in enumerate(file, 1):
            segment = json.loads(line)
            errors, angles, commands = [], [], []
            with adapter.observe_transition(index):
                for tick in segment["per_tick"]:
                    rotation = data.body_xmat[base].reshape(3, 3)
                    goal = tick["requested_pose"]
                    pos = rotation @ goal["position_m"] + data.body_xpos[base]
                    ori = (
                        rotation
                        @ Rotation.from_quat(np.roll(goal["quaternion_wxyz"], -1)).as_matrix()
                    )
                    cmd = goal["gripper_command"]
                    commands.append(cmd)
                    controller = robot.controller
                    absolute_goal = partial(controller.set_goal, set_pos=pos, set_ori=ori)
                    with patch.object(controller, "set_goal", absolute_goal):
                        env.step(np.r_[np.zeros(6), GRIP[cmd]])
                    actual = tick["actual_pose"]
                    measured_pos = rotation.T @ (data.site_xpos[site] - data.body_xpos[base])
                    errors.append(1000 * np.linalg.norm(measured_pos - actual["position_m"]))
                    target = Rotation.from_quat(np.roll(actual["quaternion_wxyz"], -1))
                    got = Rotation.from_matrix(rotation.T @ data.site_xmat[site].reshape(3, 3))
                    angles.append(float(np.rad2deg((target.inv() * got).magnitude())))
                env.check_success()
            elapsed += len(commands)
            rows.append(
                dict(
                    action=index,
                    tick=elapsed,
                    flags=native_pair(tool, episode, index),
                    cross=crossed_pair(env),
                    max_eef_error_mm=max(errors, default=0),
                    max_angle_error_deg=max(angles, default=0),
                    finish=index in finishes,
                    commands=[key for key, _ in groupby(commands)],
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    demos = modes.add_parser("demos")
    demos.add_argument("source", type=Path, help="official HDF5 or extracted demo_*.npz")
    archive = modes.add_parser("archived")
    archive.add_argument(
        "directory", type=Path, help="0,1,2 subdirectories with controller and task"
    )
    archive.add_argument("states", type=Path, help="native initial states exported to NPY")
    archive.add_argument("--events", type=Path, help="historical base-events.jsonl")
    args = parser.parse_args()

    import numpy as np
    from libero.libero.envs.env_wrapper import ControlEnv
    from openeta_bench.packs import load_pack

    pack = load_pack("libero")
    bddl = pack.source_path(args.libero_root, pack.task(TASK)["binding"]["task_source"])
    factory = partial(
        ControlEnv,
        str(bddl),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=10000,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "demos":
        with closing(factory()) as env:
            collect_demos(env, pack, args.libero_root, args.source, args.output)
    else:
        states = np.load(args.states)
        for i in range(3):
            with closing(factory()) as env:
                rows = reconstruct(
                    env,
                    pack,
                    args.libero_root,
                    args.directory / str(i),
                    states[i],
                    finish_actions(args.events, i),
                )
            (args.output / f"reconstructed-{i}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
