"""Prepare complete tasks at attempt boundaries and run a single physical environment."""

import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path

from executor_led import EpisodeBudget
from harness import parse_action, text
from hierarchy import phrase

RETRYABLE = {"wall_time_budget", "call_budget", "action_budget"}
PREPARATION_RETRIES = 4
PROTOCOL = "libero-osc-two-view-v1"
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "and",
    "with",
    "for",
    "from",
    "into",
    "onto",
    "please",
}
ZH_EN_ALIASES = {
    "抽屉": ("drawer",),
    "柜": ("cabinet",),
    "中间": ("middle",),
    "顶": ("top",),
    "最上": ("top",),
    "底": ("bottom",),
    "最下": ("bottom",),
    "碗": ("bowl",),
    "杯": ("mug", "cup"),
    "盘": ("plate",),
    "炉": ("stove",),
    "摩卡壶": ("moka", "pot"),
    "壶": ("pot", "moka"),
    "微波炉": ("microwave",),
    "篮": ("basket",),
    "打开": ("open",),
    "关": ("close",),
    "放": ("put", "place"),
    "拿": ("pick",),
    "抓": ("pick",),
    "捡": ("pick",),
    "黄油": ("butter",),
    "汤": ("soup",),
    "番茄": ("tomato", "ketchup"),
    "奶酪": ("cheese",),
    "牛奶": ("milk",),
    "橙汁": ("orange", "juice"),
    "巧克力": ("chocolate",),
    "布丁": ("pudding",),
    "酒": ("wine",),
    "书": ("book",),
    "架": ("rack", "caddy"),
    "盒": ("box",),
    "左": ("left",),
    "右": ("right",),
    "黑": ("black",),
    "白": ("white",),
    "红": ("red",),
    "黄": ("yellow",),
}


def _goal_terms(text_value):
    lowered = text_value.casefold()
    terms = {w for w in re.findall(r"[a-z0-9]+", lowered) if w not in STOPWORDS}
    for zh, en_words in ZH_EN_ALIASES.items():
        if zh in lowered:
            terms.update(en_words)
    return terms


def _stratified_sample(rows, limit=8):
    buckets = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        suite = row.get("suite") or row["id"].split("-", 1)[0]
        buckets.setdefault(suite, []).append(row)
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        next_keys = []
        for k in keys:
            if buckets[k] and len(selected) < limit:
                selected.append(buckets[k].pop(0))
            if buckets[k]:
                next_keys.append(k)
        keys = next_keys
    return selected
BRIEF = """Prepare one complete robot task before an autonomous attempt. The original user goal
and the user's actual clarifications are authoritative. Preserve the object identity, relations,
ordering and end state. Only add visible distinguishing cues. Maintain the user's language.
For a clear goal keep its wording. Never give an action sequence, assume grasp/contact, invent a
demo, or issue a robot action. On a retry the images show the current scene after the entire
previous attempt. Preserve completed work and use measured evidence when changing the brief.
Choose at most one supplied successful example if its manipulation and scene constraints fit.
Return only the requested JSON fields. Historical examples are evidence about other scenes."""


def image_url(image, path=None):
    rgb = image.convert("RGB")
    rgb.thumbnail((512, 512))
    stream = io.BytesIO()
    rgb.save(stream, format="JPEG", quality=85)
    encoded = stream.getvalue()
    if path is not None:
        path.write_bytes(encoded)
    data = base64.b64encode(encoded).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}


class DemoLibrary:
    """A frozen index produced from natively scored rollouts by build_demo_library.py."""

    def __init__(self, path):
        self.root = Path(path).resolve().parent
        raw = Path(path).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if data["protocol"] != PROTOCOL or data["version"] != 1:
            raise ValueError("incompatible demo library")
        self.entries = {}
        for row in data["examples"]:
            key = phrase(row["id"])
            if (
                key in self.entries
                or row["success"] is not True
                or row["reason"] != "native_success"
            ):
                raise ValueError("duplicate or unverified demonstration")
            self.entries[key] = row

    def candidates(self, goal, state_hash):
        words = _goal_terms(goal)
        rows = [r for r in self.entries.values() if r["initial_state_sha256"] != state_hash]
        if not rows:
            return []

        def overlap(row):
            terms = _goal_terms(row["goal"])
            if not words or not terms:
                return 0.0
            return len(words & terms) / max(1, len(words | terms))

        scored = [(overlap(r), r) for r in rows]
        if max(s for s, _ in scored) <= 0.0:
            picked = _stratified_sample(rows, limit=8)
        else:
            picked = [r for _, r in sorted(scored, key=lambda item: (-item[0], item[1]["id"]))[:8]]
        return [{k: r[k] for k in ("id", "goal", "segments")} for r in picked]

    def example(self, key):
        from PIL import Image

        row = self.entries[key]
        parts = [
            text(
                {
                    "historical_success_example": row["goal"],
                    "recorded_segments": row["segments"],
                    "source_id": key,
                    "outcome": "native_success",
                    "limits": (
                        "Compare manipulation pattern and visible appearance only; this is a "
                        "historical scene and supplies no current coordinates, pose, contact, "
                        "or task completion."
                    ),
                }
            )
        ]
        for frame in row["frames"]:
            path = (self.root / frame["file"]).resolve()
            if (
                not path.is_relative_to(self.root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]
            ):
                raise ValueError("demo frame provenance mismatch")
            with Image.open(path) as img:
                parts.extend(
                    [
                        text(f"Historical external frame before action {frame['before_action']}"),
                        image_url(img),
                    ]
                )
        return parts


class InvalidPreparation(ValueError):
    """A reply that can be regenerated without changing the confirmed goal."""


class GateCancelled(Exception):
    """The user cancelled the request before the first physical action."""


class PreparationFallback(Exception):
    """The executor can continue from the original goal and confirmed answers."""


def parse_preparation(
    response, fields, candidates, clarify, log, retry, *, skill_candidates=()
):
    try:
        choice = response["choices"][0]
        reason = choice["finish_reason"]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise InvalidPreparation("invalid_response") from None
    usage = response.get("usage")
    log(
        "preparation_response",
        retry=retry,
        finish_reason=reason,
        completion_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
        content_chars=len(content) if isinstance(content, str) else 0,
    )
    if choice["message"].get("refusal") or reason == "content_filter":
        raise ValueError("upper preparation declined")
    if reason == "length":
        raise InvalidPreparation("truncated")
    if reason != "stop":
        raise InvalidPreparation("invalid_finish_reason")
    if not isinstance(content, str) or not content.strip():
        raise InvalidPreparation("empty_content")
    try:
        data = parse_action(content)
        if clarify and set(data) == {"question"}:
            return {"question": phrase(data["question"])}
        if set(data) != set(fields):
            raise ValueError("unexpected fields")
        if "instruction" in data:
            data["instruction"] = phrase(data["instruction"])
        if data.get("demo_id") is not None and data["demo_id"] not in candidates:
            raise ValueError("unknown demo id")
        if data.get("skill_id") is not None and data["skill_id"] not in skill_candidates:
            raise ValueError("unknown skill id")
    except (ValueError, IndexError):
        raise InvalidPreparation("invalid_payload") from None
    return data


def prepare(
    chat,
    observation,
    goal,
    dialogue,
    previous,
    receipt,
    candidates,
    *,
    rewrite,
    retrieve,
    clarify,
    ask,
    budget,
    log,
    frame_dir=None,
    skill_candidates=(),
):
    fields = {}
    if rewrite:
        fields["instruction"] = "one complete grounded goal"
    if retrieve:
        fields["demo_id"] = "a listed id, or null"
    if skill_candidates:
        fields["skill_id"] = "a listed read-only skill id, or null"
    prompt = BRIEF + "\nReturn an object with exactly these fields: " + json.dumps(fields)
    if clarify:
        prompt += (
            "\nIf intent is genuinely ambiguous, you may instead return "
            '{"question":"one specific question"}.'
        )
    if frame_dir is not None:
        frame_dir.mkdir()
    views = [
        text("Current external RGB"),
        image_url(observation.external_rgb, frame_dir / "external.jpg" if frame_dir else None),
        text("Current wrist RGB"),
        image_url(observation.wrist_rgb, frame_dir / "wrist.jpg" if frame_dir else None),
    ]
    candidate_ids = [r["id"] for r in candidates]
    skill_candidate_ids = [r["id"] for r in skill_candidates]
    while True:
        payload = dict(
            original_goal=goal,
            dialogue=dialogue,
            previous_attempt=previous,
            candidates=candidates,
            reference_frame=observation.robot.reference_frame,
        )
        if skill_candidates:
            payload["skill_cards"] = skill_candidates
        if previous:
            payload["latest_public_completion"] = receipt
            payload["latest_measured_state"] = observation.state
        message = [text(payload), *views]
        messages = [dict(role="system", content=prompt), dict(role="user", content=message)]
        max_tokens = 1500
        for retry in range(PREPARATION_RETRIES + 1):
            try:
                with budget.upper_interval():
                    response = chat(messages, deadline=budget.upper_deadline, max_tokens=max_tokens)
                    data = parse_preparation(
                        response,
                        fields,
                        candidate_ids,
                        clarify,
                        log,
                        retry,
                        skill_candidates=skill_candidate_ids,
                    )
                break
            except (InvalidPreparation, json.JSONDecodeError, UnicodeDecodeError) as exc:
                cause = str(exc) if isinstance(exc, InvalidPreparation) else "invalid_response_json"
                log("preparation_invalid", retry=retry, cause=cause, max_tokens=max_tokens)
                if retry == PREPARATION_RETRIES:
                    raise PreparationFallback(cause) from exc
                if cause == "truncated":
                    max_tokens = 4096
            except TimeoutError as exc:
                raise PreparationFallback("upper_timeout") from exc
        if "question" in data:
            question = data["question"]
            log("clarification_question", question=question)
            if ask is None or len(dialogue) >= 2:
                return None, None, question, None
            answer = ask(question)
            if not answer or not answer.strip():
                return None, None, question, None
            dialogue.append(dict(question=question, answer=phrase(answer)))
            log("clarification_answer", **dialogue[-1])
            continue
        return data.get("instruction", goal), data.get("demo_id"), None, data.get("skill_id")


def run_request(
    adapter,
    *,
    goal,
    output,
    execute_attempt,
    chat=None,
    rewrite=False,
    retrieve=False,
    clarify=False,
    ask=None,
    library=None,
    skills=None,
    skill_ids=(),
    max_attempts=2,
    wall_seconds=2700,
    upper_seconds=900,
    max_upper_calls=8,
    gate=None,
):
    """execute_attempt receives frozen context and a per-attempt budget; it owns no upper client."""
    if max_attempts < 1 or wall_seconds <= 0 or upper_seconds < 0 or max_upper_calls < 1:
        raise ValueError("invalid request budget")
    if (rewrite or retrieve or clarify) and chat is None:
        raise ValueError("upper endpoint required for enabled features")
    if retrieve and library is None:
        raise ValueError("retrieval requires a verified library")
    pinned_skills = list(dict.fromkeys(skill_ids))
    if pinned_skills and skills is None:
        raise ValueError("skill_ids require a verified SkillLibrary")
    for sid in pinned_skills:
        skills.get(sid)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def log(event, **data):
        with (output / "events.jsonl").open("a") as f:
            print(
                json.dumps(dict(event=event, time=time.time(), **data), ensure_ascii=False), file=f
            )

    attempts, dialogue = [], []
    upper_s = 0.0
    question = None
    reason = "native_success"
    previous = None
    upper_calls = 0
    preparation_fallbacks = 0
    initial_view = None
    error_type = None

    fixed_instruction = None
    remove_demo = False

    def call_upper(*args, **kwargs):
        nonlocal upper_calls
        if upper_calls >= max_upper_calls:
            raise PreparationFallback("upper_call_budget")
        upper_calls += 1
        return chat(*args, **kwargs)

    for number in range(1, max_attempts + 1):
        if adapter.success():
            reason = "native_success"
            break
        if gate and callable(getattr(gate, "stop_requested", None)) and gate.stop_requested():
            reason = "user_stopped"
            break
        budget = EpisodeBudget(wall_seconds, max(0.0, upper_seconds - upper_s))
        obs = adapter.observe()
        if number == 1:
            initial_view = hashlib.sha256(obs.external_rgb.convert("RGB").tobytes()).hexdigest()
        if gate and number == 1:
            gate.scene(obs, goal)
        instruction, demo_id, selected_skill = goal, None, None
        may_ask = clarify and number == 1
        may_rewrite = rewrite and fixed_instruction is None
        may_retrieve = retrieve and not remove_demo
        may_select_skill = skills is not None and not pinned_skills and chat is not None
        if may_rewrite or may_retrieve or may_ask or may_select_skill:
            candidates = (
                library.candidates(goal, adapter.meta.get("initial_state_sha256"))
                if may_retrieve
                else []
            )
            skill_candidates = skills.catalog() if may_select_skill else []
            log(
                "preparation_started",
                attempt=number,
                candidates=[r["id"] for r in candidates],
                skill_candidates=[r["id"] for r in skill_candidates],
            )

            try:
                instruction, demo_id, question, selected_skill = prepare(
                    call_upper,
                    obs,
                    goal,
                    dialogue,
                    previous,
                    getattr(adapter, "completion", None),
                    candidates,
                    rewrite=may_rewrite,
                    retrieve=may_retrieve,
                    clarify=may_ask,
                    ask=gate.ask if gate else ask,
                    budget=budget,
                    log=log,
                    frame_dir=output / f"preparation-{number:02d}",
                    skill_candidates=skill_candidates,
                )
            except GateCancelled:
                reason = "user_cancelled"
                log(reason, attempt=number)
                break
            except PreparationFallback as exc:
                instruction, demo_id, question, selected_skill = goal, None, None, None
                preparation_fallbacks += 1
                log("preparation_fallback", attempt=number, cause=str(exc), upper_calls=upper_calls)
            except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as exc:
                reason = "preparation_error"
                error_type = type(exc).__name__
                log(reason, attempt=number, error_type=error_type)
                break
            finally:
                upper_s += budget.upper_elapsed
            if question:
                reason = "needs_clarification"
                break
        if gate and number == 1:
            try:
                confirmed, keep_demo = gate.review(instruction, demo_id, library, dialogue)
            except GateCancelled:
                reason = "user_cancelled"
                log(reason, attempt=number)
                break
            log(
                "handoff_reviewed",
                proposed_instruction=instruction,
                confirmed=confirmed,
                proposed_demo=demo_id,
                keep_demo=keep_demo,
            )
            if confirmed != instruction:
                fixed_instruction = confirmed
                dialogue.append(dict(question="Confirmed execution instruction", answer=confirmed))
            remove_demo = not keep_demo
        if fixed_instruction:
            instruction = fixed_instruction
        if remove_demo:
            demo_id = None
        active_skills = list(pinned_skills)
        if selected_skill and selected_skill not in active_skills:
            active_skills.append(selected_skill)
        if gate:
            gate.running(number, instruction, demo_id)
        brief = dict(
            original_goal=goal,
            dialogue=dialogue,
            instruction=instruction,
            demo_id=demo_id,
            skill_ids=active_skills,
        )
        log("attempt_started", attempt=number, **brief)
        context = []
        if dialogue or instruction != goal:
            context.append(
                text(
                    dict(
                        user_clarifications=dialogue,
                        grounded_instruction=instruction,
                        authority="Preserve the original task and user answers.",
                    )
                )
            )
        if active_skills and skills is not None:
            context.extend(skills.context_parts(active_skills))
        if demo_id:
            context.extend(library.example(demo_id))
        path = output / f"attempt-{number:02d}"
        # Start the executor clock at handoff. Serialization, demo I/O and human wait
        # are still included in request latency.
        lower_budget = EpisodeBudget(wall_seconds, 0)
        result = execute_attempt(path, lower_budget, obs, context, number)
        reason = result["reason"]
        attempts.append(result)
        previous = dict(
            instruction=instruction,
            demo_id=demo_id,
            reason=reason,
            actions=result["actions"],
            calls=result["calls"],
        )
        log("attempt_finished", attempt=number, **previous)
        if result["success"] or reason not in RETRYABLE:
            break
    result = dict(
        reason=reason,
        success=bool(adapter.success()),
        question=question,
        attempts=attempts,
        first_attempt_success=bool(attempts[0]["success"] if attempts else adapter.success()),
        upper_calls=upper_calls,
        error_type=error_type,
        preparation_fallbacks=preparation_fallbacks,
        lower_wall_s=sum(a.get("lower_wall_s", 0) for a in attempts),
        upper_wall_s=upper_s,
        elapsed_s=time.monotonic() - started,
        initial_external_rgb_sha256=initial_view,
        library_sha256=library.digest if retrieve else None,
        skills_sha256=skills.digest if skills is not None else None,
    )
    for key in ("actions", "calls", "completion_queries"):
        result[key] = sum(a[key] for a in attempts)
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
