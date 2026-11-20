"""Run the ETA0 hierarchy on the pinned LIBERO adapter, or run the executor alone."""

import argparse
import json
import os
import time
from pathlib import Path

from harness import run
from hierarchy import Hierarchy
from rollout_context import HistoryCompactor, bounded_cleanup


class ChatEndpoint:
    def __init__(self, base_url, model, key, deadline, *, direct=False):
        self.url = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.deadline = deadline
        self.direct = direct

    def post(self, url, payload):
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("episode wall time exhausted")
        handlers = [urllib.request.ProxyHandler({})] if self.direct else []
        opener = urllib.request.build_opener(*handlers)
        with opener.open(request, timeout=min(remaining, 600)) as response:
            return json.load(response)

    def chat(self, messages, **options):
        return self.post(
            self.url + "/chat/completions", dict(model=self.model, messages=messages, **options)
        )

    def count(self, messages):
        info = self.post(
            self.url.removesuffix("/v1") + "/tokenize",
            dict(
                model=self.model,
                messages=messages,
                chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
            ),
        )
        if info["max_model_len"] != 262144:
            raise ValueError("executor context differs from the pinned serving profile")
        return info["count"]


class ExecutorPolicy:
    def __init__(self, endpoint, seed, recover_length=False):
        self.endpoint = endpoint
        self.seed = seed
        self.recover_length = recover_length
        self.recover_next = False

    def __call__(self, messages):
        recovery = self.recover_next
        if recovery:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "The last generation was truncated. "
                    "Use the current observation and return exactly one valid action JSON.",
                },
            ]
        reply = self.endpoint.chat(
            messages,
            seed=self.seed,
            temperature=0.8,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.08,
            presence_penalty=0,
            max_tokens=16384,
            chat_template_kwargs={
                "enable_thinking": not recovery,
                "preserve_thinking": not recovery,
            },
        )
        self.seed += 1
        self.recover_next = (
            self.recover_length
            and not recovery
            and reply["choices"][0]["finish_reason"] == "length"
        )
        return reply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--initial-index", type=int, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--executor-client",
        type=Path,
        required=True,
        help="private client.json written by the deployment",
    )
    p.add_argument("--max-actions", type=int, default=120)
    p.add_argument("--max-calls", type=int, default=360)
    p.add_argument("--wall-seconds", type=int, default=2700)
    p.add_argument(
        "--mode",
        choices=["base", "plan", "judge", "teacher"],
        default="teacher",
        help="cumulative ablation: executor, +plan, +supervision, +one-shot",
    )
    p.add_argument("--instruction", help="user wording for an intent trial")
    p.add_argument(
        "--recover-length",
        action="store_true",
        help="use one direct generation after a truncated thinking reply",
    )
    p.add_argument("--interactive", action="store_true", help="answer planner clarification")
    args = p.parse_args()
    deadline = time.monotonic() + args.wall_seconds
    config = json.loads(args.executor_client.read_text())
    lower = ChatEndpoint(
        config["base_url"], config["model"], config["api_key"], deadline, direct=True
    )
    hierarchy = None
    if args.mode != "base":
        upper = ChatEndpoint(
            os.environ["ETA0_UPPER_BASE_URL"],
            os.environ["ETA0_UPPER_MODEL"],
            os.environ["ETA0_UPPER_API_KEY"],
            deadline,
        )
        hierarchy = Hierarchy(
            upper.chat,
            teacher=args.mode == "teacher",
            supervise=args.mode in {"judge", "teacher"},
            clarify=input if args.interactive else None,
        )

    # The pinned adapter loads its simulator and PhysLM dependencies in this interpreter.
    from rollout_libero_minimal import ACTION_PROMPT, LiberoAdapter

    args.output.mkdir(parents=True, exist_ok=False)
    adapter = LiberoAdapter(
        args.task,
        args.initial_index,
        args.seed,
        args.output / "simulation",
        visual_queries=True,
        suite_name=args.suite,
    )
    try:
        task = args.instruction or adapter.instruction
        (args.output / "run.json").write_text(
            json.dumps(
                dict(
                    mode=args.mode,
                    recover_length=args.recover_length,
                    user_goal=task,
                    native_goal=adapter.instruction,
                    executor=lower.model,
                    upper=upper.model if hierarchy else None,
                    max_actions=args.max_actions,
                    max_calls=args.max_calls,
                    wall_seconds=args.wall_seconds,
                    review_every=hierarchy.review_every if hierarchy else None,
                ),
                ensure_ascii=False,
            )
        )

        def record(event):
            with (args.output / "compactions.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")

        compactor = HistoryCompactor(
            task=task,
            count_tokens=lower.count,
            record=record,
            summarize=lambda messages, attempt, schema: lower.chat(
                messages, max_tokens=4096, chat_template_kwargs={"enable_thinking": False}
            ),
        )

        execute = ExecutorPolicy(lower, args.seed, args.recover_length)
        prompt = ACTION_PROMPT.replace(
            "40 motion segments and 120 model calls",
            f"{args.max_actions} motion segments and {args.max_calls} model calls",
        )
        if hierarchy and hierarchy.teacher:
            prompt = prompt.replace(
                "No demonstration or target oracle is provided.",
                "A hypothetical teacher demonstration may accompany each subgoal.",
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("episode deadline expired during initialization")
        result = run(
            adapter,
            execute,
            instruction=task,
            action_prompt=prompt,
            output_dir=args.output / "rollout",
            max_calls=args.max_calls,
            max_actions=args.max_actions,
            native_finish_feedback=True,
            bind_observation_ids=True,
            hierarchy=hierarchy,
            context_compactor=compactor,
            wall_time_s=remaining,
        )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        (args.output / "cleanup.json").write_text(json.dumps(bounded_cleanup(adapter.close)))


if __name__ == "__main__":
    main()
