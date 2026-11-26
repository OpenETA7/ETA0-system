"""Run the ETA0 hierarchy on the pinned LIBERO adapter, or run the executor alone."""

import argparse
import http.client
import json
import logging
import os
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

from executor_led import EpisodeBudget, ExecutorLed
from harness import run
from libero_semantics import SemanticEpisode, use_libero_checkout
from request_harness import DemoLibrary, InvalidPreparation, run_request
from rollout_context import HistoryCompactor, bounded_cleanup

UPPER_RETRIES = 4
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_MAX_OUTPUT_TOKENS = 16384


def adaptive_context_budget(max_model_len, requested_output=DEFAULT_MAX_OUTPUT_TOKENS):
    """Preserve the pinned 262K profile verbatim and scale safely on smaller contexts."""
    if max_model_len == DEFAULT_MAX_MODEL_LEN:
        return DEFAULT_MAX_MODEL_LEN, min(requested_output, DEFAULT_MAX_OUTPUT_TOKENS), 200000
    if max_model_len < 8192:
        raise ValueError(f"executor max_model_len too small ({max_model_len}); need at least 8192")
    max_output = min(requested_output, max(2048, max_model_len // 4))
    threshold = min(int(max_model_len * 0.76), max_model_len - max_output - 2048)
    return max_model_len, max_output, max(1024, threshold)


def estimate_tokens(messages):
    """Conservative local token estimate when an OpenAI-compatible server has no /tokenize."""
    total = 0
    for message in messages:
        total += 16
        content = message.get("content")
        if isinstance(content, str):
            total += max(1, (len(content) + 1) // 2)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    total += max(1, (len(str(part)) + 1) // 2)
                elif part.get("type") == "image_url":
                    total += 1024
                elif part.get("type") == "text":
                    total += max(1, (len(part.get("text") or "") + 1) // 2)
                else:
                    total += max(1, (len(json.dumps(part, ensure_ascii=False)) + 1) // 2)
        for key in ("reasoning", "reasoning_content"):
            extra = message.get(key)
            if isinstance(extra, str) and extra:
                total += max(1, (len(extra) + 1) // 2)
    return total


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


class UpperRateLimit(RuntimeError):
    """Keep the server's retry time available to parallel callers."""

    def __init__(self, message, retry_after):
        super().__init__(message)
        self.retry_at = time.time() + retry_after


class ChatEndpoint:
    def __init__(
        self,
        base_url,
        model,
        key,
        deadline,
        *,
        direct=False,
        allowed_models=(),
        retries=0,
        strict_context=False,
        context_length=None,
        on_context_len=None,
    ):
        self.url = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.deadline = deadline
        self.direct = direct
        self.allowed_models = allowed_models
        self.retries = retries
        self.strict_context = strict_context
        self.context_length = context_length
        self.on_context_len = on_context_len
        self.tokenize_supported = True
        self.chat_calls = 0
        self.chat_seconds = 0.0

    def remaining(self):
        deadline = self.deadline() if callable(self.deadline) else self.deadline
        return deadline - time.monotonic()

    def post(self, url, payload):
        headers = {"Content-Type": "application/json", "User-Agent": "ETA0-System/0.1"}
        request = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        if self.key:
            request.add_unredirected_header("Authorization", "Bearer " + self.key)
        handlers = [urllib.request.ProxyHandler({})] if self.direct else []
        opener = urllib.request.build_opener(*handlers)
        for attempt in range(1, self.retries + 2):
            remaining = self.remaining()
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
                if exc.code == 429 and retry_after:
                    failure = UpperRateLimit(str(failure), retry_after)
            except (OSError, http.client.IncompleteRead) as exc:
                cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                if not (
                    isinstance(
                        cause,
                        (
                            TimeoutError,
                            ConnectionError,
                            http.client.IncompleteRead,
                            ssl.SSLEOFError,
                        ),
                    )
                    or isinstance(cause, socket.gaierror)
                    and cause.errno == socket.EAI_AGAIN
                ):
                    raise
                failure = exc
                reason = type(cause).__name__

            if attempt > self.retries:
                raise failure
            delay = max(retry_after, 2 ** (attempt - 1) + random.random())
            if delay >= self.remaining():
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
        self.chat_calls += 1
        start = time.monotonic()
        try:
            reply = self.post(
                self.url + "/chat/completions", dict(model=self.model, messages=messages, **options)
            )
        finally:
            self.chat_seconds += time.monotonic() - start
        if self.allowed_models:
            if not isinstance(reply, dict) or not reply.get("model"):
                raise InvalidPreparation("invalid_response")
            if reply["model"] not in self.allowed_models:
                raise ValueError("upper endpoint returned a model outside the configured allowlist")
        return reply

    def count(self, messages):
        if not self.tokenize_supported:
            return estimate_tokens(messages)
        try:
            info = self.post(
                self.url.removesuffix("/v1") + "/tokenize",
                dict(
                    model=self.model,
                    messages=messages,
                    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
                ),
            )
        except RuntimeError as exc:
            msg = str(exc)
            if not self.strict_context and any(
                code in msg for code in ("HTTP 404", "HTTP 405", "HTTP 501")
            ):
                self.tokenize_supported = False
                logging.getLogger(__name__).warning(
                    "Executor %s has no /tokenize endpoint (%s); using local token estimate",
                    self.model,
                    msg,
                )
                return estimate_tokens(messages)
            raise
        if not isinstance(info, dict) or "count" not in info:
            if self.strict_context:
                raise ValueError("executor /tokenize returned an unexpected payload")
            self.tokenize_supported = False
            return estimate_tokens(messages)
        reported = info.get("max_model_len")
        if reported is not None:
            reported = int(reported)
            if self.strict_context and reported != DEFAULT_MAX_MODEL_LEN:
                raise ValueError("executor context differs from the pinned serving profile")
            if reported != self.context_length:
                self.context_length = reported
                if self.on_context_len is not None:
                    self.on_context_len(reported)
        return int(info["count"])


class RelayedUpper(ChatEndpoint):
    """Use the same response checks and budget as a directly connected upper endpoint."""

    def __init__(self, mailbox, model, deadline, allowed):
        super().__init__("", model, None, deadline, allowed_models=allowed)
        self.mailbox = Path(mailbox)

    def post(self, url, payload):
        from preparation_relay import request_preparation

        return request_preparation(self.mailbox, payload, self.remaining)


def upper_endpoint(boyue_config, deadline):
    if boyue_config is not None:
        config = json.loads(boyue_config.read_text())
        config = config.get("boyue", config)
        allowed = config.get("allowed_returned_models", [config["model"]])
        if not (isinstance(allowed, list) and allowed and all(isinstance(m, str) for m in allowed)):
            raise ValueError("invalid Boyue allowed_returned_models")
        if "preparation_mailbox" in config:
            return RelayedUpper(config["preparation_mailbox"], config["model"], deadline, allowed)
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
    def __init__(
        self,
        endpoint,
        seed,
        recover_length=False,
        *,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        api_style="vllm",
    ):
        self.endpoint = endpoint
        self.seed = seed
        self.recover_length = recover_length
        self.max_tokens = max_tokens
        self.api_style = api_style
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
        options = dict(
            seed=self.seed,
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0,
            max_tokens=self.max_tokens,
        )
        if self.api_style != "openai":
            options.update(
                top_k=20,
                repetition_penalty=1.08,
                chat_template_kwargs={
                    "enable_thinking": not recovery,
                    "preserve_thinking": not recovery,
                },
            )
        reply = self.endpoint.chat(messages, **options)
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
        choices=["request", "base", "plan", "judge", "teacher"],
        help="role preset; individual switches override it",
    )
    for feature in ("rewrite", "retrieve", "clarify"):
        p.add_argument("--" + feature, action=argparse.BooleanOptionalAction)
    p.add_argument("--demo-library", type=Path)
    p.add_argument("--skills-dir", type=Path, help="directory of read-only Markdown skill cards")
    p.add_argument(
        "--skill",
        action="append",
        default=[],
        help="repeatable read-only skill card ID to attach from --skills-dir",
    )
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--interactive", action="store_true", help="read clarification answers on stdin")
    p.add_argument("--gate", type=int, metavar="PORT", help="local browser handoff (0: free port)")
    p.add_argument("--instruction", help="user wording for an intent trial")
    p.add_argument(
        "--recover-length",
        action="store_true",
        help="use one direct generation after a truncated thinking reply",
    )
    for role in ("planner", "supervisor", "teacher"):
        p.add_argument("--" + role, action=argparse.BooleanOptionalAction)
    p.add_argument("--upper-wall-seconds", type=int, default=900)
    p.add_argument("--max-upper-calls", type=int, default=8)
    p.add_argument("--stall-failures", type=int, default=3)
    p.add_argument("--stall-plateau", type=int, default=8)
    p.add_argument("--task-length", choices=["auto", "long", "short"], default="auto")
    p.add_argument("--instruction-mode", choices=["original", "revised"], default="revised")
    p.add_argument("--completion-query", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--progress-query-ablation",
        action="store_true",
        help="cache native evidence in both base arms and expose it only on request",
    )
    p.add_argument("--libero-source-root", type=Path, default=os.environ.get("ETA0_LIBERO_ROOT"))
    p.add_argument(
        "--strict-serving-profile",
        action="store_true",
        help="require vLLM /tokenize and max_model_len == 262144 for pinned benchmark runs",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        help="override executor context length (defaults to client.json or /tokenize)",
    )
    p.add_argument(
        "--upstream-base",
        action="store_true",
        help="use the frozen upstream module in the paired evaluation bundle",
    )
    args = p.parse_args()
    if args.mode is None:
        args.mode = (
            "judge"
            if any(getattr(args, r) is not None for r in ("teacher", "supervisor", "planner"))
            else "request"
        )
    if args.mode == "request" and any(
        getattr(args, r) for r in ("teacher", "supervisor", "planner")
    ):
        p.error(
            "request mode owns the attempt boundary; select a legacy preset to enable those roles"
        )
    if args.mode != "request" and (
        any(getattr(args, r) is not None for r in ("rewrite", "retrieve", "clarify"))
        or args.demo_library
        or args.skills_dir
        or args.skill
        or args.gate is not None
        or args.interactive
        or args.max_attempts != 2
    ):
        p.error("request features require --mode request")
    if args.skill and not args.skills_dir:
        p.error("--skill requires --skills-dir")
    if args.gate is not None and (args.interactive or not 0 <= args.gate <= 65535):
        p.error("--gate requires port 0–65535 and owns the interactive handoff")
    if args.max_attempts < 1:
        p.error("--max-attempts must be positive")
    rewrite = args.rewrite is not False
    clarify = args.clarify is not False
    retrieve = bool(args.demo_library) if args.retrieve is None else args.retrieve
    if retrieve and not args.demo_library:
        p.error("--retrieve requires --demo-library")
    roles = dict(
        planner=args.mode in {"plan", "judge", "teacher"},
        supervisor=args.mode in {"judge", "teacher"},
        teacher=args.mode == "teacher",
    )
    roles.update({r: getattr(args, r) for r in roles if getattr(args, r) is not None})
    if args.progress_query_ablation and (
        args.mode != "base" or any(roles.values()) or args.upstream_base
    ):
        p.error("--progress-query-ablation requires the executor-only base harness")
    if args.upstream_base and (args.mode != "base" or any(roles.values()) or args.completion_query):
        p.error("--upstream-base requires base mode without advisory roles or completion query")
    if min(args.max_upper_calls, args.stall_failures, args.stall_plateau) < 1:
        p.error("call and stall thresholds must be positive")
    completion_enabled = (
        not args.upstream_base if args.completion_query is None else args.completion_query
    )
    collect_progress = completion_enabled or args.progress_query_ablation
    progress_access = "default"
    if args.progress_query_ablation:
        progress_access = "on_request" if completion_enabled else "hidden"
    if args.mode == "request":
        progress_access = "on_request" if completion_enabled else "hidden"
    budget = EpisodeBudget(args.wall_seconds, args.upper_wall_seconds)
    config = json.loads(args.executor_client.read_text())
    api_style = config.get("api_style", "vllm")
    strict_profile = args.strict_serving_profile or bool(config.get("strict_serving_profile"))
    initial_context_len = (
        args.max_model_len
        if args.max_model_len is not None
        else int(config.get("max_model_len", DEFAULT_MAX_MODEL_LEN))
    )
    requested_output = int(config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    ctx_len, max_out, compact_threshold = adaptive_context_budget(
        initial_context_len, requested_output
    )
    serving_state = dict(
        max_model_len=ctx_len,
        max_output_tokens=max_out,
        threshold=compact_threshold,
        compactor=None,
        policy=None,
    )

    def sync_context_len(reported_len):
        new_len, new_out, new_threshold = adaptive_context_budget(reported_len, requested_output)
        serving_state.update(
            max_model_len=new_len, max_output_tokens=new_out, threshold=new_threshold
        )
        if serving_state["compactor"] is not None:
            serving_state["compactor"].update_limits(
                max_model_len=new_len, max_output_tokens=new_out, threshold=new_threshold
            )
        if serving_state["policy"] is not None:
            serving_state["policy"].max_tokens = new_out

    lower = ChatEndpoint(
        config["base_url"],
        config["model"],
        config["api_key"],
        budget.lower_deadline,
        direct=True,
        strict_context=strict_profile,
        context_length=initial_context_len,
        on_context_len=sync_context_len,
    )
    # The pinned adapter loads its simulator and PhysLM dependencies in this interpreter.
    args.output.mkdir(parents=True, exist_ok=False)
    if args.libero_source_root:
        use_libero_checkout(args.libero_source_root, args.output)
    from rollout_libero_minimal import ACTION_PROMPT, LiberoAdapter

    adapter = LiberoAdapter(
        args.task,
        args.initial_index,
        args.seed,
        args.output / "simulation",
        visual_queries=True,
        suite_name=args.suite,
    )
    try:
        native_goal = adapter.instruction
        semantic = None
        if args.instruction_mode == "revised" or collect_progress or roles["planner"]:
            semantic = SemanticEpisode(
                adapter, args.instruction_mode, collect_progress, args.libero_source_root
            )
            adapter = semantic
        task = args.instruction or adapter.instruction
        long_task = (
            bool(semantic and semantic.long_task)
            if args.task_length == "auto"
            else args.task_length == "long"
        )
        need_upper = (roles["planner"] and long_task) or roles["teacher"] or roles["supervisor"]
        if args.mode == "request":
            need_upper = rewrite or clarify or retrieve
        upper = (
            upper_endpoint(args.boyue_config, lambda: budget.upper_deadline) if need_upper else None
        )
        hierarchy = (
            ExecutorLed(
                upper.chat,
                budget,
                planner=roles["planner"],
                supervise=roles["supervisor"],
                teacher=roles["teacher"],
                long_task=long_task,
                failures=args.stall_failures,
                plateau=args.stall_plateau,
                max_upper_calls=args.max_upper_calls,
            )
            if upper and args.mode != "request"
            else None
        )
        (args.output / "run.json").write_text(
            json.dumps(
                dict(
                    mode=args.mode,
                    recover_length=args.recover_length,
                    user_goal=task,
                    native_goal=native_goal,
                    instruction_mode="user" if args.instruction else args.instruction_mode,
                    benchmark_goal=adapter.instruction,
                    completion_query=completion_enabled,
                    progress_access=progress_access,
                    pack_version=semantic.pack.version if semantic else None,
                    pack_sha256=semantic.pack.sha256 if semantic else None,
                    long_task=long_task,
                    roles=roles,
                    stall_failures=args.stall_failures,
                    stall_plateau=args.stall_plateau,
                    upper_wall_seconds=args.upper_wall_seconds,
                    max_upper_calls=args.max_upper_calls,
                    executor=lower.model,
                    upper=upper.model if upper else None,
                    request_features=dict(
                        rewrite=rewrite,
                        retrieve=retrieve,
                        clarify=clarify,
                        max_attempts=args.max_attempts,
                        gui_gate=args.gate is not None,
                        skills_dir=str(args.skills_dir) if args.skills_dir else None,
                        skill_ids=list(args.skill),
                    )
                    if args.mode == "request"
                    else None,
                    max_actions=args.max_actions,
                    max_calls=args.max_calls,
                    wall_seconds=args.wall_seconds,
                ),
                ensure_ascii=False,
            )
        )

        def record(event):
            with (args.output / "compactions.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")

        def new_compactor():
            def summarize(messages, attempt, schema):
                opts = dict(max_tokens=min(4096, serving_state["max_output_tokens"]))
                if api_style != "openai":
                    opts["chat_template_kwargs"] = {"enable_thinking": False}
                return lower.chat(messages, **opts)

            compactor = HistoryCompactor(
                task=task,
                count_tokens=lower.count,
                record=record,
                summarize=summarize,
                threshold=serving_state["threshold"],
                max_model_len=serving_state["max_model_len"],
                max_output_tokens=serving_state["max_output_tokens"],
            )
            serving_state["compactor"] = compactor
            return compactor

        execute = ExecutorPolicy(
            lower,
            args.seed,
            args.recover_length,
            max_tokens=serving_state["max_output_tokens"],
            api_style=api_style,
        )
        serving_state["policy"] = execute
        prompt = ACTION_PROMPT.replace(
            "40 motion segments and 120 model calls",
            f"{args.max_actions} motion segments and {args.max_calls} model calls",
        )
        if hierarchy and hierarchy.teacher:
            prompt = prompt.replace(
                "No demonstration or target oracle is provided.",
                "A hypothetical recovery example may be provided after repeated difficulty.",
            )
        if args.mode == "request":
            if retrieve:
                prompt = prompt.replace(
                    "No demonstration or target oracle is provided.",
                    "A recorded example may illustrate a different scene. Use current geometry.",
                )

            def ask(question):
                try:
                    frames = args.output / "rollout/preparation-01"
                    print(f"External view: {frames / 'external.jpg'}")
                    print(f"Wrist view: {frames / 'wrist.jpg'}")
                    return input(question + " ")
                except EOFError:
                    return None

            def chat(messages, *, deadline, **options):
                upper.deadline = deadline
                return upper.chat(messages, **options)

            def attempt(path, current, observation, context, number):
                lower.deadline = current.lower_deadline
                policy = ExecutorPolicy(
                    lower,
                    args.seed + (number - 1) * args.max_calls,
                    args.recover_length,
                    max_tokens=serving_state["max_output_tokens"],
                    api_style=api_style,
                )
                serving_state["policy"] = policy
                return run(
                    adapter,
                    policy,
                    instruction=task,
                    action_prompt=prompt,
                    output_dir=path,
                    max_calls=args.max_calls,
                    max_actions=args.max_actions,
                    native_finish_feedback=True,
                    bind_observation_ids=True,
                    context_compactor=new_compactor(),
                    initial_observation=observation,
                    attempt_context=context,
                    budget=current,
                    progress_access=progress_access,
                    on_observation=gate.observe_live if gate else None,
                    should_stop=gate.stop_requested if gate else None,
                )

            request_dir = args.output / "rollout"
            gate = None
            if args.gate is not None:
                from eta0_gate import RequestGate

                gate = RequestGate(args.gate, native_goal)
            skills_lib = None
            if args.skills_dir is not None:
                from skill_cards import SkillLibrary

                skills_lib = SkillLibrary(args.skills_dir)
            try:
                result = run_request(
                    adapter,
                    goal=task,
                    output=request_dir,
                    execute_attempt=attempt,
                    chat=chat if upper else None,
                    rewrite=rewrite,
                    retrieve=retrieve,
                    clarify=clarify,
                    ask=ask if args.interactive else None,
                    library=DemoLibrary(args.demo_library) if retrieve else None,
                    skills=skills_lib,
                    skill_ids=args.skill,
                    max_attempts=args.max_attempts,
                    gate=gate,
                    wall_seconds=args.wall_seconds,
                    upper_seconds=args.upper_wall_seconds,
                    max_upper_calls=args.max_upper_calls,
                )
                result.update(
                    lower_chat_calls=lower.chat_calls,
                    lower_chat_s=lower.chat_seconds,
                    upper_calls=upper.chat_calls if upper else 0,
                    episode_wall_s=time.monotonic() - budget.start,
                )
                (request_dir / "RESULT.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
                )
                if gate:
                    gate.finish(result, (request_dir / "RESULT.json").resolve())
            except Exception as exc:
                if gate:
                    gate.failed(type(exc).__name__)
                raise
            finally:
                if gate:
                    gate.close()
            print(json.dumps(result, ensure_ascii=False))
            return
        remaining = budget.lower_deadline() - time.monotonic()
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
            context_compactor=new_compactor(),
        )
        if args.upstream_base:
            from upstream_harness import run as upstream_run

            try:
                result = upstream_run(adapter, execute, **options)
            except TimeoutError:
                if time.monotonic() < budget.lower_deadline():
                    raise
                result = {"reason": "wall_time_budget", "success": bool(adapter.success())}
                (args.output / "rollout/RESULT.json").write_text(json.dumps(result))
        else:
            result = run(
                adapter,
                execute,
                hierarchy=hierarchy,
                budget=budget,
                progress_access=progress_access,
                **options,
            )
        result.update(
            budget.metrics(), lower_chat_calls=lower.chat_calls, lower_chat_s=lower.chat_seconds
        )
        (args.output / "rollout/RESULT.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        (args.output / "cleanup.json").write_text(json.dumps(bounded_cleanup(adapter.close)))


if __name__ == "__main__":
    main()
