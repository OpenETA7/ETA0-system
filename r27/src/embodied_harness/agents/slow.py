import asyncio
import time
import re
import copy
from collections import deque
from contextlib import asynccontextmanager
from embodied_harness.domain.models import Advice, StrategicAdvice, Milestone, uid
from .prompts import SLOW_DELEGATE, SLOW_REVIEW


class SlowAgent:
    def __init__(self, model, context, coordinator, store, *, priority_foreground=False, review_economy=False,
                 retain_diagnostics=False, urgent_completion_review=False, milestone_advice_delivery=False):
        self.model, self.context, self.coordinator, self.store = model, context, coordinator, store
        self.pending = asyncio.Event()
        self.reviews = 0
        self.review_durations = []
        self.errors = 0
        self.last_review_action = -1
        self.last_review_observation_id = None
        self.completed = 0
        self.changed = asyncio.Event()
        self.reasons = deque(maxlen=6)
        self.request_ticket = 0
        self.delivered_ticket = 0
        self.model_lock = asyncio.Lock()
        self.foreground_pending = 0
        self.foreground_done = asyncio.Event()
        self.foreground_done.set()
        self.milestones = None
        self.last_review_started_at = -float("inf")
        self.priority_foreground = priority_foreground
        self.review_economy = review_economy
        self.retain_diagnostics = retain_diagnostics
        self._review_task = None
        self._preempted_reviews = set()
        self.urgent_completion_review = urgent_completion_review
        self._review_scope = None
        self._urgent_preempted_reviews = set()
        self.milestone_advice_delivery = milestone_advice_delivery

    def strategy_message(self, text, milestone, review_basis):
        from embodied_harness.runtime.advice_delivery import unbound_strategy
        if (not self.milestone_advice_delivery or not self.milestones or not review_basis
                or not getattr(self.context, "defer_routine_advice", False) or not unbound_strategy(text)):
            return None
        # Preserve the full checked text up to the same compact-context bound.
        # Truncation is after the binding scan, so a late coordinate cannot vanish.
        basis = copy.deepcopy(review_basis)
        basis["published_at"] = time.monotonic()
        basis.update(mission=milestone.mission, goal=milestone.goal,
                     stage_index=milestone.stage_index)
        return StrategicAdvice(milestone.id, milestone.revision, milestone.generation,
            "advice", text.strip()[:1200], review_basis=basis)

    def publish_diagnostics(self, decision, milestone, observed_action_seq, observation_id, observed_at=None):
        """Archive typed hypotheses, never revive stale action authority."""
        if not self.retain_diagnostics or decision.get("kind") != "advice":
            return 0
        current = self.context.milestone
        if (current is None or milestone is None
                or (current.id, current.revision, current.generation) !=
                   (milestone.id, milestone.revision, milestone.generation)
                or not 0 <= self.context.action_seq - observed_action_seq <= 4):
            return 0
        now = time.monotonic()
        observed_at = now if observed_at is None else observed_at
        if now - observed_at >= 180:
            return 0
        rows = decision.get("diagnostic_hypotheses")
        if not isinstance(rows, list) or len(rows) > 2:
            return 0
        published = 0
        for row in rows:
            valid = (isinstance(row, dict)
                and set(row) == {"category", "subject", "hypothesis", "recheck"}
                and isinstance(row["category"], str)
                and row["category"] in {"identity", "task_interpretation", "failure_cause"}
                and all(isinstance(row[k], str) and 0 < len(row[k].strip()) <= 400
                        for k in ("subject", "hypothesis", "recheck")))
            text = " ".join(str(row.get(k, "")) for k in ("subject", "hypothesis", "recheck")) if isinstance(row, dict) else ""
            if (not valid or re.search(r"\bpoint[-_ ]\d+|\bp\d+[ob]-c\d+\b|\bc\d+\b|\bcandidate[_ ]id\b|\bpixels?\b|\bcoordinates?\b|\b[xyz]\s*[=:~]\s*-?\d|\[\s*-?\d+[.,\d ]*\]|\b(?:move_to|mark_point|locate_point|select_candidate|grasp_point|place_point|pull_handle|gripper_control)\s*\(", text, re.I)):
                self.store.emit("diagnostic.rejected", {"reason": "invalid_or_frame_bound_hypothesis"})
                continue
            entry = {**{k: v.strip() for k, v in row.items()}, "id": uid(),
                "milestone_id": milestone.id, "revision": milestone.revision,
                "generation": milestone.generation, "observed_action_seq": observed_action_seq,
                "observation_id": observation_id, "observed_at": observed_at,
                "expires_at": observed_at + 180, "source": "slow_review",
                "status": "historical_unverified", "authority": "diagnostic_only",
                "instruction": "Compare this historical hypothesis with CURRENT images and evidence. It cannot authorize an action, prove current hold/pose, satisfy a stage, or override a pause/constraint."}
            # Repeated wording replaces its old record; conflicting alternatives
            # remain explicitly uncertain until the executor checks the scene.
            self.context.diagnostic_hypotheses = deque((h for h in self.context.diagnostic_hypotheses
                if (h["category"], h["subject"], h["hypothesis"]) !=
                   (entry["category"], entry["subject"], entry["hypothesis"])), maxlen=4)
            self.context.diagnostic_hypotheses.append(entry)
            self.store.emit("diagnostic.published", dict(entry))
            published += 1
        return published

    @asynccontextmanager
    async def foreground(self, purpose):
        """Reserve the single Slow client; cancel only obsolete background work."""
        self.foreground_pending += 1
        self.foreground_done.clear()
        started = time.monotonic()
        task = self._review_task
        preempted = bool(self.priority_foreground and task and not task.done())
        try:
            if preempted:
                self._preempted_reviews.add(task)
                task.cancel()
                self.store.emit("slow.background_preempt_requested", {"purpose": purpose})
            self.store.emit("slow.foreground_requested", {"purpose": purpose, "preempted_review": preempted})
            async with self.model_lock:
                self.store.emit("slow.foreground_ready", {"purpose": purpose,
                    "waited_s": time.monotonic()-started, "preempted_review": preempted})
                yield
        finally:
            self.foreground_pending -= 1
            if not self.foreground_pending:
                self.foreground_done.set()

    def publish_semantic_hypothesis(self, decision, milestone, action_scope=None, review_basis=None):
        """Preserve an explicitly separate identity hypothesis, never old pixels.

        It is model-proposed evidence for reinspection, not a motion command or
        an assertion about the current hand/pose. Milestone fences still apply.
        """
        text = decision.get("semantic_advice")
        if not isinstance(text, str) or not text.strip():
            return False
        text = text.strip()[:1000]
        if re.search(r"\d|\bpixels?\b|\bpoint[-_ ]|\bcandidate[_ ]id\b|\bcoordinates?\b|\b[xy]\s*[=:~]", text, re.I):
            self.store.emit("slow.semantic_rejected", {"reason": "contains_frame_specific_reference"})
            return False
        scope = {}
        if self.milestones:
            # Free prose cannot be certified as identity-only by its field name.
            # Until identity is a separately verified entity record, preserve the
            # reviewed action boundary even for this explicitly tentative channel.
            if action_scope is None:
                self.store.emit("slow.semantic_rejected", {"reason": "missing_reviewed_action_scope"})
                return False
            scope = dict(zip(("expected_action_seq", "expected_action_id", "expected_action_phase"), action_scope))
        text = "IDENTITY HYPOTHESIS — recheck against CURRENT images, not a current-pose fact: " + text
        advice = self.strategy_message(text, milestone, review_basis)
        if advice is None:
            advice = Advice(milestone.id, milestone.revision, milestone.generation, "advice",
                            text, time.monotonic()+120, priority=40, **scope)
        accepted = self.coordinator.apply_advice(advice)
        self.store.emit("slow.semantic_published", {"advice_id": advice.id, "accepted": accepted,
                                                   "provenance": "model_identity_hypothesis"})
        return accepted

    async def delegate(self, mission):
        projection = self.context.projection("slow")
        projection["mission"] = mission
        async with self.model_lock:
            decision = await self.model.decide(SLOW_DELEGATE, projection, self.context.latest.images)
        if decision.get("kind") != "delegate" or not isinstance(decision.get("goal"), str):
            raise ValueError("Slow initial decision must delegate a goal")
        # Semantic guesses are revisable hypotheses, never a rewrite of the user's mission.
        self.context.target_plan={"visual_scope":self.context.latest.state.get("candidate_scope"),"strategy":decision["goal"],"targets":decision.get("targets",[]),"provenance":"slow_visual_hypothesis"}
        m = Milestone(uid(), 1, mission, mission, capabilities=tuple(k for k in self.coordinator.adapter.schemas if k not in {"observe","mark_point","locate_point","select_candidate"}))
        self.coordinator.delegate(m)

    def request_review(self, reason):
        passive = reason in {"periodic_progress", "streamed_execution_progress"}
        if self.review_economy and passive and self.milestones and (
                self.last_review_action == self.context.action_seq
                or time.monotonic()-self.last_review_started_at < 90):
            self.store.emit("slow.review_coalesced", {"reason": reason,
                "policy": "passive_review_economy", "action_seq": self.context.action_seq})
            return self.request_ticket
        if self.milestones and passive and (
                self.pending.is_set() or self.context.review_state["in_flight"]
                or time.monotonic() - self.last_review_started_at < 30
                or (self.last_review_action == self.context.action_seq
                    and self.last_review_observation_id == self.context.latest.id)):
            self.store.emit("slow.review_coalesced", {"reason": reason, "action_seq": self.context.action_seq})
            return self.request_ticket
        self.store.emit("slow.review_requested", {"reason": reason})
        self.reasons.append(reason)
        if hasattr(self.context,"review_state"):self.context.review_state["pending"]=True
        self.pending.set()  # bounded coalescing; no stale backlog of frames
        self.request_ticket+=1
        return self.request_ticket

    async def wait_for_review(self, reason, timeout=135):
        """Selective recovery rendezvous; normal successful execution remains asynchronous."""
        ticket=self.request_review(reason)
        end=time.monotonic()+timeout
        while self.delivered_ticket<ticket:
            self.changed.clear()
            if self.delivered_ticket >= ticket:
                return True
            remaining=end-time.monotonic()
            if remaining<=0:return False
            try:await asyncio.wait_for(self.changed.wait(),remaining)
            except asyncio.TimeoutError:return False
        return True

    def request_completion_review(self, reason):
        """Queue official-negative feedback; preempt only an older action review.

        Called by the executor after the official completion check rejects the
        current outcome. This never waits, changes a contract or accepts advice.
        """
        ticket = self.request_review(reason)
        if not self.urgent_completion_review:
            return ticket
        task, scope = self._review_task, self._review_scope
        current = self.context.action_seq
        eligible = (
            self.priority_foreground and self.milestones
            and getattr(self.milestones, "pipelined", False)
            and not self.foreground_pending
            and not (self.coordinator.running or self.coordinator.unknown
                     or self.coordinator.paused
                     or self.coordinator.contract_review_pending
                     or self.milestones.pending_reason)
            and task is not None and not task.done()
            and task not in self._preempted_reviews
            and scope is not None and scope[0] is task
            and scope[1] < current)
        if eligible:
            self._preempted_reviews.add(task)
            self._urgent_preempted_reviews.add(task)
            task.cancel()
            self.store.emit("slow.urgent_completion_preempt_requested", {
                "ticket": ticket, "interrupted_review_ticket": scope[2],
                "reviewed_action_seq": scope[1],
                "current_action_seq": current,
                "reason": "official_negative_after_newer_physical_action",
                "wait_for_review": False, "cancelled_provider_cost": "unknown"})
        else:
            self.store.emit("slow.urgent_completion_queued", {
                "ticket": ticket, "current_action_seq": current,
                "preempted": False, "wait_for_review": False})
        return ticket

    async def review(self):
        async with self.model_lock:
            if self.foreground_pending:
                self.pending.set()
                return
            task = asyncio.create_task(self._review())
            self._review_task = task
            try:
                await task
            except asyncio.CancelledError:
                # Cancelling the outer reviewer (episode shutdown) still propagates.
                # A foreground request cancels only its child inference, preserving
                # the persistent reviewer and refreshing its next projection.
                if asyncio.current_task().cancelling() or task not in self._preempted_reviews:
                    raise
                urgent = task in self._urgent_preempted_reviews
                self.reasons.append("urgent_completion_preempted_refresh_current_evidence"
                                    if urgent else "foreground_preempted_refresh_current_evidence")
                self.pending.set()
                self.store.emit("slow.urgent_completion_preempted" if urgent
                                else "slow.background_preempted", {"will_refresh": True})
            finally:
                self._preempted_reviews.discard(task)
                self._urgent_preempted_reviews.discard(task)
                if self._review_task is task:
                    self._review_task = None

    async def _review(self):
        projection = self.context.projection("slow")
        images = self.context.latest.images
        last_visual = getattr(self.context, "latest_visual", None)
        if not images and last_visual is not None:
            images = last_visual.images
            projection["attached_visual"] = {
                "observation_id": last_visual.id, "action_seq": last_visual.action_seq,
                "captured_at": last_visual.captured_at,
                "age_s": max(0, time.monotonic() - last_visual.captured_at),
                "views": copy.deepcopy(last_visual.state.get("views", [])),
                "status": "older_visual_reference_not_current_robot_pose",
            }
        ticket=self.request_ticket
        projection["review_reasons"]=list(self.reasons)
        self.reasons.clear()
        milestone = self.context.milestone
        review_stamp = self.milestones.stamp() if self.milestones else None
        action_seq = self.context.action_seq
        action = self.coordinator.running
        action_id = action.action_id if action else None
        action_phase = action.status if action else "idle"
        # Bind captured metadata to its owning Task. A newly created task must
        # never inherit a previous task's scope before taking its projection.
        scope_owner = asyncio.current_task()
        self._review_scope = (scope_owner, action_seq, ticket)
        started = time.monotonic()
        review_basis = None
        if self.milestone_advice_delivery:
            reviewed = projection.get("attached_visual") or projection["observation"]
            review_basis = {
                "observation_id": projection["observation"]["id"],
                "visual_observation_id": reviewed.get("observation_id", reviewed.get("id")),
                "observed_at": min(started, reviewed.get("captured_at", started)),
                "action_seq": action_seq, "action_id": action_id, "action_phase": action_phase,
                "control_version": projection["control_version"], "ticket": ticket,
            }
        self.last_review_started_at = started
        self.last_review_observation_id = self.context.latest.id
        self.context.review_state.update(in_flight=True,pending=False)
        outcome="cancelled"
        self.store.emit("slow.review_started", {"action_seq": action_seq,"ticket":ticket,"observation_id":self.context.latest.id})
        try:
            prompt = SLOW_REVIEW
            if self.milestones:
                from .milestone_prompts import BACKGROUND_REVIEW
                prompt = BACKGROUND_REVIEW
                if getattr(self.milestones, "pipelined", False):
                    prompt += "\nRoutine advice is considered at the next decision only if its evidence is still fresh; it does not cancel an ongoing decision. If current evidence shows a wrong target, constraint conflict, or an action that must not continue, use pause or replan instead of a routine advisory message.\n"
            if self.retain_diagnostics:
                from .milestone_prompts import DIAGNOSTIC_REVIEW
                prompt += DIAGNOSTIC_REVIEW
            decision = await self.model.decide(prompt, projection, images)
            kind = decision.get("kind")
            if kind=="delegate" and isinstance(decision.get("goal"),str):
                self.store.emit("slow.format_recovered",{"from":"delegate","to":"advice","mission_unchanged":True})
                decision={**decision,"kind":"advice","text":decision["goal"],"action_sensitive":False}
                kind="advice"
            if self.retain_diagnostics:
                visual = projection.get("attached_visual") or projection["observation"]
                self.publish_diagnostics(decision, milestone, visual.get("action_seq", action_seq),
                    visual.get("observation_id", visual.get("id")),
                    min(started, visual.get("captured_at", started)))
            if kind == "replan" and self.milestones:
                accepted = self.milestones.request_replan(decision.get("reason", "background review requests a revised contract"), milestone, expected_stamp=review_stamp)
                outcome = "replan" if accepted else "stale"
            elif kind in {"advice", "pause"}:
                # A model's flag cannot certify that prose contains no dynamic state.
                # In milestone mode, ordinary execution advice always belongs to the
                # reviewed action boundary. Opted-in strategic prose below is a
                # historical hypothesis, never an executable instruction.
                sensitive = bool(self.milestones) or bool(decision.get("action_sensitive", True)) or kind == "pause"
                visual_reference=bool(re.search(r"\bp\d+[ob]-c\d+\b|\bc\d+\b|\bpoint-\d+\b|\bpixels?\b|\b[xy]\s*[=:~]\s*-?\d",str(decision.get('text','')),re.I)) or any(
                    t.get('candidate_id') or t.get('pixel_xy') for t in decision.get('targets',[]) if isinstance(t,dict))
                visual_scope=(projection.get('observation',{}).get('state',{}).get('candidate_scope') if visual_reference else None)
                if visual_reference and visual_scope is None:
                    raise ValueError('Frame-specific review lacks candidate_scope; cannot publish an unbound visual instruction')
                advice = Advice(milestone.id, milestone.revision, milestone.generation, kind,
                                str(decision.get("text", "")), time.monotonic() + 120,
                                action_seq if sensitive else None,
                                action_id if sensitive else None, action_phase if sensitive else None,
                                expected_visual_scope=visual_scope)
                if kind == "advice" and decision.get("action_sensitive") is False and not visual_reference:
                    strategy = self.strategy_message(str(decision.get("text", "")), milestone, review_basis)
                    if strategy is not None:
                        advice = strategy
                accepted=self.coordinator.apply_advice(advice)
                if accepted and not self.milestones:
                    # Hierarchical advice has one scoped representation. Copying
                    # its prose into an unscoped target_plan would revive stale state.
                    if self.context.target_plan.get("visual_scope") != visual_scope:
                        self.context.target_plan["targets"]=[]
                    self.context.target_plan["visual_scope"]=visual_scope
                    self.context.target_plan["strategy"]=decision.get("text","")
                    if decision.get("invalidate_targets"):self.context.target_plan["targets"]=[]
                    if isinstance(decision.get("targets"),list) and decision["targets"]:
                        self.context.target_plan["targets"]=decision["targets"]
                semantic_accepted = kind == "advice" and self.publish_semantic_hypothesis(
                    decision, milestone, (action_seq, action_id, action_phase), review_basis)
                outcome=("strategy_advice" if accepted and isinstance(advice, StrategicAdvice)
                         else "advice" if accepted else "semantic_advice" if semantic_accepted else "stale")
            elif kind != "noop":
                raise ValueError("invalid Slow review decision")
            else:outcome="noop"
            self.reviews += 1
            self.last_review_action = action_seq
            if outcome in {"advice","strategy_advice","semantic_advice","noop","replan"}:self.delivered_ticket=max(self.delivered_ticket,ticket)
        except Exception as exc:
            outcome="error"
            self.errors += 1
            self.store.emit("slow.model_error", {"type": type(exc).__name__, "message": str(exc)[:400]})
        finally:
            if self._review_scope is not None and self._review_scope[0] is scope_owner:
                self._review_scope = None
            self.completed+=1
            self.context.review_state.update(in_flight=False,completed=self.completed,last_outcome=outcome,delivered_ticket=self.delivered_ticket)
            self.changed.set()
            elapsed = time.monotonic() - started
            self.review_durations.append(elapsed)
            self.store.emit("slow.review_finished", {"based_on_action_seq": action_seq,
                                                      "current_action_seq": self.context.action_seq,
                                                      "elapsed_s": elapsed,"outcome":outcome,"ticket":ticket})

    async def run(self):
        while True:
            await self.pending.wait()
            await self.foreground_done.wait()
            self.pending.clear()
            await self.review()

    async def watch_execution(self, minimum_interval=30.):
        """Coalesced background review triggers from real in-action sensor samples."""
        last = -float("inf")
        async for event in self.store.stream("slow-execution-monitor"):
            if event["kind"] != "sensor.observed" or not self.coordinator.running:
                continue
            if event["payload"]["id"] != self.context.latest.id:
                continue
            now = time.monotonic()
            if now-last >= minimum_interval:
                last = now
                self.request_review("streamed_execution_progress")
