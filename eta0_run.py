"""Run the ETA0 hierarchy on the pinned LIBERO adapter, or run the executor alone."""

import argparse
import http.client
import json
import logging
import os
import random
import socket
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

from harness import run
from hierarchy import Hierarchy
from rollout_context import HistoryCompactor, bounded_cleanup

UPPER_RETRIES = 4


def http_error_codes(response):
    with response:
        try:
            error = json.loads(response.read(8192)).get("error")
        except (ValueError, AttributeError, OSError, http.client.HTTPException):
            return []
    if not isinstance(error, dict):
        return []
    return [str(error[k]).lower() for k in ("code", "type") if error.get(k)]


def retry_after_seconds(header):
    if not header:
        return 0
    try:
        return max(0, int(header))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(header).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError, OSError):
            return 0


class ChatEndpoint:
    def __init__(
        self, base_url, model, key, deadline, *, direct=False, allowed_models=(), retries=0
    ):
        self.url = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.deadline = deadline
        self.direct = direct
        self.allowed_models = allowed_models
        self.retries = retries

    def post(self, url, payload):
        headers = {"Content-Type": "application/json", "User-Agent": "ETA0-System/0.1"}
        request = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        if self.key:
            request.add_unredirected_header("Authorization", "Bearer " + self.key)
        handlers = [urllib.request.ProxyHandler({})] if self.direct else []
        opener = urllib.request.build_opener(*handlers)
        for attempt in range(1, self.retries + 2):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("episode wall time exhausted")
            retry_after = 0
            try:
                with opener.open(
                    request, timeout=min(remaining, 120 if self.retries else 600)
                ) as r:
                    return json.load(r)
            except urllib.error.HTTPError as exc:
                codes = http_error_codes(exc)
                code = codes[0] if codes else "request_error"
                reason = f"HTTP {exc.code}"
                failure = RuntimeError(f"Model {self.model} returned {reason} ({code})")
                if exc.code not in {408, 429, 500, 502, 503, 504} or set(codes).intersection(
                    {"insufficient_quota", "insufficient_balance", "billing_hard_limit_reached"}
                ):
                    raise failure from None
                retry_after = retry_after_seconds(exc.headers.get("Retry-After"))
            except (OSError, http.client.IncompleteRead) as exc:
                cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                if not (
                    isinstance(cause, (TimeoutError, ConnectionError, http.client.IncompleteRead))
                    or isinstance(cause, socket.gaierror)
                    and cause.errno == socket.EAI_AGAIN
                ):
                    raise
                failure = exc
                reason = type(cause).__name__

            if attempt > self.retries:
                raise failure
            delay = max(retry_after, 2 ** (attempt - 1) + random.random())
            if delay >= self.deadline - time.monotonic():
                raise failure
            logging.getLogger(__name__).warning(
                "Model %s attempt %d/%d failed (%s), retrying in %.1fs",
                self.model,
                attempt,
                self.retries + 1,
                reason,
                delay,
            )
            time.sleep(delay)

    def chat(self, messages, **options):
        reply = self.post(
            self.url + "/chat/completions", dict(model=self.model, messages=messages, **options)
        )
        if self.allowed_models and reply.get("model") not in self.allowed_models:
            raise ValueError("upper endpoint returned a model outside the configured allowlist")
        return reply

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


def upper_endpoint(boyue_config, deadline):
    if boyue_config is not None:
        config = json.loads(boyue_config.read_text())["boyue"]
        allowed = config.get("allowed_returned_models", [config["model"]])
        if not (isinstance(allowed, list) and allowed and all(isinstance(m, str) for m in allowed)):
            raise ValueError("invalid Boyue allowed_returned_models")
        return ChatEndpoint(
            config["base_url"],
            config["model"],
            config["api_key"],
            deadline,
            allowed_models=allowed,
            retries=UPPER_RETRIES,
        )
    fields = ("_BASE_URL", "_MODEL", "_API_KEY")
    prefix = (
        "ETA0_UPPER" if any("ETA0_UPPER" + field in os.environ for field in fields) else "KIMI_CODE"
    )
    expected = os.environ.get("ETA0_UPPER_EXPECTED_MODEL")
    return ChatEndpoint(
        *(os.environ[prefix + field] for field in fields),
        deadline,
        allowed_models=[expected] if expected else [],
        retries=UPPER_RETRIES,
    )


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
    p.add_argument("--boyue-config", type=Path, help="private Boyue JSON client configuration")
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
    p.add_argument(
        "--upstream-base",
        action="store_true",
        help="use the frozen upstream module in the paired evaluation bundle",
    )
    args = p.parse_args()
    if args.upstream_base and args.mode != "base":
        p.error("--upstream-base requires --mode base")
    deadline = time.monotonic() + args.wall_seconds
    config = json.loads(args.executor_client.read_text())
    lower = ChatEndpoint(
        config["base_url"], config["model"], config["api_key"], deadline, direct=True
    )
    hierarchy = None
    if args.mode != "base":
        upper = upper_endpoint(args.boyue_config, deadline)
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
        options = dict(
            instruction=task,
            action_prompt=prompt,
            output_dir=args.output / "rollout",
            max_calls=args.max_calls,
            max_actions=args.max_actions,
            native_finish_feedback=True,
            bind_observation_ids=True,
            context_compactor=compactor,
        )
        if args.upstream_base:
            from upstream_harness import run as upstream_run

            try:
                result = upstream_run(adapter, execute, **options)
            except TimeoutError:
                if time.monotonic() < deadline:
                    raise
                result = {"reason": "wall_time_budget", "success": bool(adapter.success())}
                (args.output / "rollout/RESULT.json").write_text(json.dumps(result))
        else:
            result = run(adapter, execute, hierarchy=hierarchy, wall_time_s=remaining, **options)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        (args.output / "cleanup.json").write_text(json.dumps(bounded_cleanup(adapter.close)))


if __name__ == "__main__":
    main()
