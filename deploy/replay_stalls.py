"""Replay compact, ordered LIBERO event traces through the live executor-led scheduler."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_led import EpisodeBudget, ExecutorLed  # noqa: E402
from hierarchy import Frame  # noqa: E402
from libero_semantics import task_semantics  # noqa: E402


class ReplayAdvisor(ExecutorLed):
    def ask(self, role, prompt, data, event):
        self.calls += 1
        event("upper_request", role=role)
        return {"plan": "Recorded planning opportunity.", "hint": "Recorded review opportunity."}


def replay(row, pack, failures, plateau, infer_single=False, *, detector=None):
    goals = task_semantics(pack, row["name"], row["suite"])["native_goals"]
    long_task = len(goals) > 1
    director = ReplayAdvisor(
        None,
        EpisodeBudget(2700, 900),
        planner=True,
        supervise=True,
        teacher=False,
        long_task=long_task,
        failures=failures,
        plateau=plateau,
    )
    if detector is not None:
        director.detector = detector
    decisions = actions = 0
    frame = Frame(Path("observation-000"), {}, None)
    trigger = None
    requests = Counter()
    hits = []

    def record(kind, **data):
        if kind == "upper_request":
            requests[data["role"]] += 1
        if kind == "stall_intervention":
            hits.append(dict(before_call=decisions, actions=actions, reason=data["trigger"]))

    for kind, *values in row["events"]:
        if kind == "response":
            decisions += 1
            director.prepare("Historical task", frame, actions, trigger, record)
            trigger = None
        elif kind == "observation":
            xyz, quat, width = values[0]
            frame.context["measured_state"] = {
                "eef_pose_base": {"position_mm": xyz, "quaternion_wxyz": quat},
                "gripper_open_width_mm": width,
            }
            # A subsequent recorded executor reply implies the native success guard was false.
            # Only a single-predicate task permits recovering a specific condition from that bit.
            if infer_single and len(goals) == 1:
                frame.context["completion"] = {
                    "status": "checked",
                    "feedback": {
                        "conditions": [
                            {
                                "requirement_id": goals[0]["requirement_id"],
                                "satisfied_in_last_evaluation": False,
                            }
                        ]
                    },
                }
        elif kind == "execution":
            actions += 1
            frame = Frame(Path(f"observation-{actions:03d}"), {}, {"feedback": values[0]})
        elif kind == "rejection":
            trigger = "rejected_action"
        elif kind == "finish_feedback":
            trigger = "executor_finish"
    return dict(
        arm=row["arm"],
        suite=row["suite"],
        task=row["task"],
        initial=row["initial"],
        lower_responses=decisions,
        planner=requests["planner"],
        supervisor=requests["supervisor"],
        hits=hits,
    )


def main():
    from openeta_bench.packs import load_pack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stall-failures", type=int, default=3)
    parser.add_argument("--stall-plateau", type=int, default=8)
    parser.add_argument("--infer-single-condition", action="store_true")
    args = parser.parse_args()
    pack = load_pack("libero")
    arms = defaultdict(Counter)
    episodes = []
    with args.traces.open() as stream:
        for line in stream:
            row = replay(
                json.loads(line),
                pack,
                args.stall_failures,
                args.stall_plateau,
                args.infer_single_condition,
            )
            episodes.append(row)
            arms[row["arm"]].update(
                episodes=1,
                with_stall=int(bool(row["hits"])),
                triggers=len(row["hits"]),
                planner=row["planner"],
                supervisor=row["supervisor"],
                lower_responses=row["lower_responses"],
            )
    args.output.write_text(
        json.dumps(
            dict(inferred_single=args.infer_single_condition, arms=arms, episodes=episodes),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()
