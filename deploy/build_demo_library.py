"""Freeze successful LIBERO executor rollouts into compact, portable examples."""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request_harness import PROTOCOL


def build(episodes, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if not episodes:
        raise ValueError("expected at least one successful example")
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "library"
        stage.mkdir()
        export(episodes, stage)
        stage.rename(output)


def export(episodes, output):
    examples = []
    for episode in episodes:
        rollout = episode / "rollout"
        result = json.loads((rollout / "RESULT.json").read_text())
        if result.get("success") is not True or result.get("reason") != "native_success":
            raise ValueError(f"demo requires recorded native success: {episode}")
        meta = json.loads((episode / "simulation/task.json").read_text())
        latest = json.loads((episode / "simulation/latest.json").read_text())
        if latest.get("native_success") is not True:
            raise ValueError("terminal simulation record disagrees with result")
        key = f"{meta['suite']}-{meta['task_id']:02d}-{meta['initial_index']:02d}"
        if any(e["id"] == key for e in examples):
            raise ValueError(f"duplicate demo: {key}")
        count = result["actions"]
        if count < 1:
            raise ValueError("example has no executed actions")
        samples = sorted({0, count // 2, count - 1})
        segments, frames = [], []
        with (episode / "simulation/controller.jsonl").open() as f:
            length = 0
            for index, line in enumerate(f):
                length += 1
                if index not in samples:
                    continue
                c = json.loads(line)
                end = c["result"]
                segments.append(
                    dict(
                        action=index + 1,
                        stage=c["action"]["stage"],
                        gripper_command=end["requested_final_pose"]["gripper_command"],
                        executed_ticks=end["executed_ticks"],
                        measured_displacement_mm=[
                            round((after - before) * 1000, 1)
                            for before, after in zip(
                                end["before_eef_pose_base"]["position_m"],
                                end["actual_eef_pose_base"]["position_m"],
                                strict=True,
                            )
                        ],
                        termination=end["termination"],
                    )
                )
                source = rollout / f"observation-{index:03d}" / "external.png"
                dest = output / f"{key}-{index:03d}.jpg"
                with Image.open(source) as im:
                    rgb = im.convert("RGB")
                    rgb.thumbnail((320, 320))
                    rgb.save(dest, quality=80)
                frames.append(
                    dict(
                        file=dest.name,
                        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        before_action=index + 1,
                    )
                )
        if length != count:
            raise ValueError("controller length disagrees with result")
        examples.append(
            dict(
                id=key,
                suite=meta["suite"],
                task_id=meta["task_id"],
                goal=meta["instruction"],
                initial_state_sha256=meta["initial_state_sha256"],
                task_name=meta["task_name"],
                source_result_sha256=hashlib.sha256(
                    (rollout / "RESULT.json").read_bytes()
                ).hexdigest(),
                success=True,
                reason="native_success",
                segments=segments,
                frames=frames,
            )
        )
    (output / "library.json").write_text(
        json.dumps(dict(version=1, protocol=PROTOCOL, examples=examples), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("episodes", nargs="+", type=Path)
    args = p.parse_args()
    build(args.episodes, args.output)
