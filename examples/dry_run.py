"""Offline wiring example. This is a fake robot/policy, not a benchmark rollout."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from harness import Observation, run  # noqa: E402
from tcp_overlay import CameraCalibration, RobotGeometry  # noqa: E402


class FakeRobot:
    def __init__(self):
        self.done = False

    def observe(self):
        source = Path(__file__).with_name("libero_snapshot.json")
        snapshot = json.loads(source.read_text())
        rgb = Image.open(source.parent / snapshot["image"]).convert("RGB")
        return Observation(rgb, rgb, CameraCalibration(**snapshot["camera"]),
                           RobotGeometry(**snapshot["robot"]), {"note": "offline wiring only"})

    def validate_action(self, action, observation):
        if set(action) != {"stage", "observation_id"} or action["stage"] != "fake_step":
            raise ValueError("this offline example only supports fake_step")

    def execute(self, action):
        self.done = True
        return {"simulated": True}

    def success(self):
        return self.done


def fake_policy(messages):
    context = json.loads(messages[-1]["content"][-1]["text"])
    action = dict(stage="fake_step", observation_id=context["observation_id"])
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(action),
    }}]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(FakeRobot(), fake_policy, instruction="Test offline wiring",
                         action_prompt="Offline fake action example", output_dir=args.output)))
