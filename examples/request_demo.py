"""Exercise the request boundary offline with a scripted two-step robot."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dry_run import fake_policy  # noqa: E402
from examples.hierarchy_demo import TwoStageRobot, reply  # noqa: E402
from harness import run  # noqa: E402
from request_harness import DemoLibrary, run_request  # noqa: E402


def demo(output, library=None, gate=None):
    goal = "Put those two objects in the basket."
    robot = TwoStageRobot()
    robot.meta = {"initial_state_sha256": "offline-fixture"}

    def upper(messages, **kwargs):
        data = json.loads(messages[-1]["content"][0]["text"])
        if not data["dialogue"]:
            return reply({"question": "Which two objects?"})
        items = data["dialogue"][0]["answer"].rstrip(".")
        schema = messages[0]["content"]
        answer = {}
        if '"instruction":' in schema:
            answer["instruction"] = f"Put these items in the basket: {items}."
        if '"demo_id":' in schema:
            candidates = data["candidates"]
            answer["demo_id"] = (
                candidates[0]["id"] if candidates and not data["previous_attempt"] else None
            )
        return reply(answer)

    def attempt(path, budget, obs, context, number):
        return run(
            robot,
            fake_policy,
            instruction=goal,
            action_prompt="Use fake_step.",
            output_dir=path,
            max_actions=1,
            budget=budget,
            initial_observation=obs,
            attempt_context=context,
        )

    return run_request(
        robot,
        gate=gate,
        goal=goal,
        output=output,
        chat=upper,
        execute_attempt=attempt,
        rewrite=True,
        clarify=True,
        ask=lambda _: "The soup and the butter.",
        retrieve=bool(library),
        library=library,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-library", type=Path)
    parser.add_argument("--gate", type=int, metavar="PORT")
    args = parser.parse_args()
    library = DemoLibrary(args.demo_library) if args.demo_library else None
    gate = None
    if args.gate is not None:
        from eta0_gate import RequestGate

        gate = RequestGate(args.gate, "Offline fixture: two scripted actions")
    try:
        result = demo(args.output, library, gate)
        if gate:
            gate.finish(result, (args.output / "RESULT.json").resolve())
        print(json.dumps(result, indent=2))
    finally:
        if gate:
            gate.close()
