"""Two-subgoal offline wiring example using bundled camera fixtures and scripted replies."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.dry_run import FakeRobot  # noqa: E402
from harness import run  # noqa: E402
from hierarchy import Hierarchy  # noqa: E402


def reply(data):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(data)}}]}


class TwoStageRobot(FakeRobot):
    def __init__(self):
        self.actions = 0

    def execute(self, action):
        self.actions += 1
        return {"placed": self.actions}

    def success(self):
        return self.actions == 2


def upper(messages):
    system = messages[0]["content"]
    if system.startswith("Interpret"):
        return reply(
            {
                "steps": [
                    {
                        "instruction": "Place soup in the basket.",
                        "done_when": "Soup rests in basket.",
                    },
                    {
                        "instruction": "Place butter in basket.",
                        "done_when": "Both items rest in basket.",
                    },
                ]
            }
        )
    if system.startswith("Generate"):
        return reply(
            {
                "trajectory": [
                    {
                        "cue": "Target is visible",
                        "skill": "Lift target",
                        "expected": "Target rises",
                    },
                    {
                        "cue": "Target held above basket",
                        "skill": "Release",
                        "expected": "Target rests inside",
                    },
                ]
            }
        )
    return reply(
        dict(
            decision="advance",
            evidence=["observation-001"],
            reason="Scripted first placement confirmed.",
            hint="Locate the butter.",
        )
    )


def lower(messages):
    # Read the active subgoal and use the actual current observation ID.
    instruction = json.loads(messages[0]["content"].splitlines()[-1])
    assert instruction["hypothetical_demonstration"]
    context = json.loads(messages[-1]["content"][-1]["text"])
    return reply(dict(stage="fake_step", observation_id=context["observation_id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        run(
            TwoStageRobot(),
            lower,
            instruction="Place soup and butter in the basket.",
            action_prompt="Use fake_step.",
            output_dir=args.output,
            hierarchy=Hierarchy(upper, review_every=1),
        )
    )
