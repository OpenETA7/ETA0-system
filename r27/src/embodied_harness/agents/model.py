import asyncio
import base64
import copy
import hashlib
import contextlib
import json
import re
import time
from collections import deque
from pathlib import Path
import httpx


class ModelError(RuntimeError):
    pass


class InferencePreempted(ModelError):
    """A newer accepted control message invalidated a cognitive request."""


def decode_object(text, finish_reason=None):
    """Accept one complete object; recover explanatory prefixes without guessing actions."""
    if finish_reason == "length":
        raise ModelError("model output was truncated; no decision may execute")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
    try:
        value = json.loads(cleaned)
        recovered = False
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        if start < 0:
            raise ModelError("model must return one JSON object") from exc
        try:
            value, end = json.JSONDecoder().raw_decode(cleaned, start)
        except json.JSONDecodeError as inner:
            raise ModelError("model must return one complete JSON object") from inner
        # A second object, array wrapper or any trailing instruction is ambiguous.
        if cleaned[end:].strip():
            raise ModelError("ambiguous text after model decision") from exc
        recovered = True
    if not isinstance(value, dict):
        raise ModelError("model decision must be object")
    return value, recovered


class ModelClient:
    def __init__(self, base_url, key, model, role, output, timeout=120, options=None):
        self.model, self.role = model, role
        self.timeout_s = timeout
        self._key = key
        self.options = options or {}
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", timeout=timeout,
                                       headers={"Authorization": f"Bearer {key}"}, trust_env=False)
        self.trace = Path(output) / f"{role}-model.jsonl"
        self.request_trace = Path(output) / f"{role}-requests.jsonl"
        self.accounting = {key: 0 for key in (
            "logical_requests", "attempts_started", "responses", "retries",
            "transport_errors", "cancelled_attempts", "http_errors",
            "logical_completed", "logical_failed", "logical_cancelled", "logical_timeouts",
            "responses_with_usage", "responses_without_usage")}
        self.accounting.update(attempt_latencies_s=[], logical_latencies_s=[])
        self.accounting["logical_preempted"] = 0
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.latencies = []
        self.history = deque(maxlen=6)
        self.tools_prefix = ""
        self.reference_messages = []
        self.reference_metadata = []
        self.mission_reference_messages = []
        self.mission_reference_metadata = None
        self.control_context = None
        self.input_instruction = "Continue from the executed previous decision. This is the NEW tool result/current snapshot, not a new episode. Choose the NEXT decision.\n"
        if role == "slow":
            self.input_instruction = "Review the current mission, scene evidence and execution outcomes. Return a planning or supervisory decision; the executor chooses how to act.\n"

    def bind_tools(self, schemas):
        if self.role == "slow":
            raise ValueError("Slow receives evidence and planning contracts, not execution tool interfaces")
        self.tools_prefix = "\nAvailable tools (stable schema):\n" + json.dumps(schemas,sort_keys=True,ensure_ascii=False)

    def bind_memory(self, entries):
        for entry in entries:
            self.reference_metadata.append({k:entry[k] for k in ('image','image_sha256','caption','manifest_sha256')})
            self.reference_messages.append({"role":"user","content":[
                {"type":"text","text":"PRIOR EXPERIENCE REFERENCE, NOT THE CURRENT SCENE. "+entry['caption']},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(Path(entry['image']).read_bytes()).decode()}}]})

    def bind_mission_reference(self, observation):
        """Pin one public start frame for identity continuity, never current pose."""
        if self.mission_reference_metadata is not None or not observation.images:
            return
        views = observation.state.get("views") or []
        index = next((i for i, view in enumerate(views)
                      if view.get("camera_frame_id") == "agentview" and i < len(observation.images)), None)
        if index is None:
            return
        path = Path(observation.images[index])
        data = path.read_bytes()
        self.mission_reference_metadata = {
            "observation_id": observation.id, "action_seq": observation.action_seq,
            "captured_at": observation.captured_at, "image": str(path),
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "camera_frame_id": views[index].get("camera_frame_id") if index < len(views) else None,
            "semantics": "task_start_reference_not_current_pose_or_retention",
        }
        self.mission_reference_messages = [{"role": "user", "content": [
            {"type": "text", "text": "THIS TASK'S START FRAME, NOT THE CURRENT SCENE. "
             "Use initial appearance and spatial relations to distinguish the mission target from distractors across later frames. "
             "Region IDs are local to this old frame; they cannot ground a current action. "
             "This image never proves current position, possession or completion. " + json.dumps(self.mission_reference_metadata)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
        ]}]

    def request_event(self, kind, **payload):
        # No headers, credentials, response bodies or exception text here.
        with self.request_trace.open("a") as stream:
            stream.write(json.dumps({"kind": kind, "at": time.time(), "model": self.model,
                                     "role": self.role, **payload}) + "\n")

    def effective_options(self, recovery_reasoning=False, exclude_completion=False):
        if type(recovery_reasoning) is not bool:
            raise ValueError("recovery_reasoning must be a boolean")
        if type(exclude_completion) is not bool:
            raise ValueError("exclude_completion must be a boolean")
        options = {"temperature": 0.1, "max_tokens": 4096, **copy.deepcopy(self.options)}
        if exclude_completion:
            if self.role != 'fast' or 'response_format' in options:
                raise ValueError('completion exclusion requires Fast without another response format')
            options['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'nonterminal_executor_decision', 'strict': True,
                'schema': {'type': 'object', 'properties': {
                    'kind': {'type': 'string', 'enum': ['tool', 'wait', 'escalate']}},
                    'required': ['kind'], 'additionalProperties': True}}}
        if recovery_reasoning:
            template = options.get("chat_template_kwargs")
            if (self.role != "fast" or self.model != "Qwen3.6-35B-A3B"
                    or not isinstance(template, dict) or template.get("enable_thinking") is not False):
                raise ValueError("recovery reasoning requires the non-thinking Qwen Fast profile")
            if type(options.get("max_tokens")) is not int or options["max_tokens"] != 4096:
                raise ValueError("recovery reasoning preserves the 4096-token output budget")
            options["chat_template_kwargs"] = {**template, "enable_thinking": True}
        return options

    async def decide(self, system, context, images=(), *, recovery_reasoning=False, exclude_completion=False):
        self.effective_options(recovery_reasoning, exclude_completion)  # validate before dispatch or accounting
        overrides = ({'recovery_reasoning': True} if recovery_reasoning else {})
        if exclude_completion:
            overrides['exclude_completion'] = True
        self.accounting["logical_requests"] += 1
        request_id = self.accounting["logical_requests"]
        started = time.monotonic()
        self.request_event("request.started", request_id=request_id,
                           observation_id=(context.get("observation") or {}).get("id"),
                           recovery_reasoning=recovery_reasoning, completion_excluded=exclude_completion)
        outcome = "failed"
        failure_type = None
        try:
            # Includes all HTTP attempts and retry backoff; a retry cannot
            # silently double the configured logical inference deadline.
            async with asyncio.timeout(self.timeout_s):
                result = await self._controlled_decide(system, context, images, request_id, **overrides)
            outcome = "completed"
            return result
        except InferencePreempted:
            outcome = "cancelled"
            self.accounting["logical_preempted"] += 1
            raise
        except TimeoutError as exc:
            self.accounting["logical_timeouts"] += 1
            failure_type = "logical_timeout"
            raise ModelError(f"{self.model}: logical inference deadline exceeded") from exc
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            failure_type = type(exc).__name__
            raise
        finally:
            elapsed = time.monotonic() - started
            self.accounting[f"logical_{outcome}"] += 1
            self.accounting["logical_latencies_s"].append(elapsed)
            self.request_event("request.finished", request_id=request_id, outcome=outcome,
                               failure_type=failure_type, elapsed_s=elapsed,
                               recovery_reasoning=recovery_reasoning, completion_excluded=exclude_completion)

    async def _controlled_decide(self, system, context, images, request_id, *, recovery_reasoning=False, exclude_completion=False):
        overrides = ({'recovery_reasoning': True} if recovery_reasoning else {})
        if exclude_completion:
            overrides['exclude_completion'] = True
        controls = self.control_context
        if controls is None:
            return await self._decide(system, context, images, request_id, **overrides)
        version = context.get("control_version", controls.control_version)
        if version != controls.control_version:
            notice = await controls.wait_for_control(version)
            controls.store.emit("control.inference_preempted", {"role": self.role, "request_id": request_id,
                "notice": notice, "wake_latency_s": time.monotonic() - notice["received_monotonic"],
                "cancel_cleanup_s": 0.0, "phase": "before_provider_dispatch"})
            raise InferencePreempted("control snapshot was superseded before inference dispatch")
        history_before = list(self.history)
        inference = asyncio.create_task(self._decide(system, context, images, request_id, **overrides))
        notification = asyncio.create_task(controls.wait_for_control(version))
        try:
            await asyncio.wait((inference, notification), return_when=asyncio.FIRST_COMPLETED)
            if controls.control_version != version:
                notice = await notification
                woke_at = time.monotonic()
                inference.cancel()
                with contextlib.suppress(asyncio.CancelledError, ModelError):
                    await inference
                self.history = deque(history_before, maxlen=self.history.maxlen)
                controls.store.emit("control.inference_preempted", {
                    "role": self.role, "request_id": request_id, "notice": notice,
                    "wake_latency_s": woke_at - notice["received_monotonic"],
                    "cancel_cleanup_s": time.monotonic() - woke_at,
                    "provider_cost": "may remain billable; use reported usage and unknown-attempt counters"})
                raise InferencePreempted("new accepted control message; discard the old cognitive decision")
            return await inference
        finally:
            for task in (inference, notification):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _decide(self, system, context, images, request_id, *, recovery_reasoning=False, exclude_completion=False):
        effective_options = self.effective_options(recovery_reasoning, exclude_completion)
        context=copy.deepcopy(context)
        state=(context.get('observation') or {}).get('state') or {}
        visual = context.get('attached_visual') or state
        views=visual.get('views') or []
        selected_indices = list(range(min(2, len(images))))
        # Acquisition checkpoints need evidence at the gripper. The candidate
        # gallery can omit an object that has moved to the edge of the scene.
        # Use only views actually present in this observation; never fabricate
        # a camera or treat old reference images as current evidence.
        if self.role == 'slow' and context.get('purpose') == 'checkpoint':
            by_camera = {view.get('camera_frame_id'): i for i, view in enumerate(views)
                         if i < len(images)}
            if all(camera in by_camera for camera in ('agentview', 'robot0_eye_in_hand')):
                selected_indices = [by_camera['agentview'], by_camera['robot0_eye_in_hand']]
        transmitted_images = [images[i] for i in selected_indices]
        if len(views) > len(selected_indices) or selected_indices != list(range(len(selected_indices))):
            visual['omitted_views']=[v.get('camera_frame_id') for i, v in enumerate(views)
                                     if i not in selected_indices]
            visual['views']=[views[i] for i in selected_indices if i < len(views)]
            visual['image_note'] = (
                'Only the listed views are attached. Treat an unseen view as missing evidence; describe the uncertainty to the executor.'
                if self.role == 'slow' else
                'Only the listed views are attached. Request observe(camera=wrist) if an omitted wrist view is needed.'
            )
        content = [{"type": "text", "text": self.input_instruction + json.dumps(context, ensure_ascii=False)}]
        for path in transmitted_images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()}})
        started = time.monotonic()
        request = {"model": self.model, "messages": [{"role": "system", "content": system+self.tools_prefix},
                     *self.mission_reference_messages, *self.reference_messages,*list(self.history), {"role": "user", "content": content}], **effective_options}
        # Both ablation arms record hashes from the actual outgoing system
        # message. No prompt text or credentials are added to this metadata.
        system_metadata = {
            'completion_excluded': exclude_completion,
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "tools_prefix_sha256": hashlib.sha256(self.tools_prefix.encode("utf-8")).hexdigest(),
            "transmitted_system_sha256": hashlib.sha256(
                request["messages"][0]["content"].encode("utf-8")).hexdigest(),
        }
        for attempt in range(2):
            self.calls += 1
            self.accounting["attempts_started"] += 1
            self.accounting["retries"] += int(attempt > 0)
            attempt_started = time.monotonic()
            attempt_outcome = "error"
            status_code = None
            response = None
            self.request_event("attempt.started", request_id=request_id, attempt=attempt+1, **system_metadata)
            try:
                response = await self.client.post("chat/completions", json=request)
                status_code = response.status_code
                self.accounting["responses"] += 1
                attempt_outcome = "response"
                if status_code != 200:
                    self.accounting["http_errors"] += 1
            except asyncio.CancelledError:
                self.accounting["cancelled_attempts"] += 1
                attempt_outcome = "cancelled"
                raise
            except (httpx.TimeoutException, httpx.TransportError):
                self.accounting["transport_errors"] += 1
                attempt_outcome = "transport_error"
                if attempt == 1:
                    raise ModelError(f"{self.model}: inference transport/timeout after 2 attempts")
            finally:
                elapsed_attempt = time.monotonic()-attempt_started
                self.accounting["attempt_latencies_s"].append(elapsed_attempt)
                self.request_event("attempt.finished", request_id=request_id, attempt=attempt+1,
                                   outcome=attempt_outcome, status_code=status_code, elapsed_s=elapsed_attempt)
            if response is None:
                await asyncio.sleep(2 ** attempt)
                continue
            if response.status_code not in {429, 502, 503, 504} or attempt == 1:
                break
            await asyncio.sleep(2 ** attempt)
        if response.status_code != 200:
            # Never log request headers or credential-bearing exceptions.
            safe_text = response.text.replace(self._key, "[redacted]")[:300]
            raise ModelError(f"{self.model}: HTTP {response.status_code}: {safe_text}")
        data = response.json()
        text = data["choices"][0]["message"].get("content") or ""
        elapsed = time.monotonic() - started
        self.latencies.append(elapsed)
        usage = data.get("usage") or {}
        reasoning = data["choices"][0]["message"].get("reasoning_content")
        reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
        details = usage.get("completion_tokens_details")
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if type(reasoning_tokens) is not int or reasoning_tokens < 0:
            reasoning_tokens = None
        if recovery_reasoning:
            self.request_event("response.reasoning_metadata", request_id=request_id,
                recovery_reasoning=True, reasoning_content_chars=reasoning_chars,
                reported_reasoning_tokens=reasoning_tokens)
        # Partial compatible-server usage is a lower bound, never evidence of zero cost.
        reported = {key: value for key in self.usage
                    if type(value := usage.get(key)) is int and value >= 0}
        complete_usage = len(reported) == len(self.usage)
        self.accounting["responses_with_usage" if complete_usage else "responses_without_usage"] += 1
        for key in self.usage:
            self.usage[key] += reported.get(key, 0)
        with self.trace.open("a") as f:
            transmitted_manifest = ([{"kind": "task_start_reference", **self.mission_reference_metadata}]
                                    if self.mission_reference_metadata else [])
            transmitted_manifest += [{"kind": "prior_experience_reference", **entry}
                                     for entry in self.reference_metadata]
            transmitted_manifest += [{"kind": "current_or_declared_visual", "image": str(images[i]),
                                      "source_image_index": i}
                                     for i in selected_indices]
            f.write(json.dumps({"model": self.model, "options": self.options, "latency_s": elapsed, "usage": usage,
                                "request_id": request_id, "effective_options": effective_options, **system_metadata,
                                "recovery_reasoning": recovery_reasoning, "reasoning_content_chars": reasoning_chars,
                                "finish_reason": data["choices"][0].get("finish_reason"),
                                "context": context, "images": images,
                                "transmitted_images": transmitted_images,
                                "transmitted_image_indices": selected_indices,
                                "appearance_references":self.reference_metadata,
                                "mission_reference": self.mission_reference_metadata,
                                "transmitted_image_manifest": transmitted_manifest,
                                "transmitted_total_image_count": len(transmitted_manifest),
                                "response": text}, ensure_ascii=False) + "\n")
        value, recovered = decode_object(text, data["choices"][0].get("finish_reason"))
        if exclude_completion and value.get('kind') not in {'tool', 'wait', 'escalate'}:
            self.request_event('response.constraint_violation', request_id=request_id,
                               constraint='exclude_completion')
            raise ModelError('Provider response violates requested nonterminal decision schema')
        if recovered:
            self.request_event("response.envelope_recovered", request_id=request_id,
                               recovery="single_trailing_json_object")
        cleaned = json.dumps(value, ensure_ascii=False)
        observation = context.get("observation") or {}
        state = observation.get("state") or {}
        self.history.append({"role": "user", "content": json.dumps({"milestone": context.get("milestone"),
            "observed_robot": state.get("robot"), "previous_tool_result": state.get("feedback"),
            "advice": context.get("advice"), "error": state.get("error")}, ensure_ascii=False)})
        self.history.append({"role": "assistant", "content": cleaned})
        return value

    async def close(self):
        await self.client.aclose()
