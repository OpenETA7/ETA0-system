import argparse
import asyncio
import contextlib
import json
import os
import time
import hashlib
from pathlib import Path

from embodied_harness.adapters.openeta import OpenETAAdapter
from embodied_harness.agents.model import ModelClient
from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone, uid
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.events import EventStore
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.topology import ModelTopology, validate_two_role_options, allowed_roles


def load_env(path):
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def role_base_url(role):
    if role not in {"fast", "slow"}:
        raise ValueError("model endpoint role must be fast or slow")
    return os.environ.get(role.upper()+"_MODEL_BASE_URL") or os.environ["MODEL_BASE_URL"]


async def episode(args):
    validate_two_role_options(args)
    if getattr(args, 'milestone_advice_delivery', False) and (
            args.mode != 'dual_async' or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'two_role_runtime', False)
            or getattr(args, 'hierarchy', 'milestones') != 'milestones'):
        raise ValueError('milestone_advice_delivery requires two-role dual_async pipelined milestones')
    if getattr(args, 'scout_prompt_parity', False) and (
            args.mode != 'dual_async' or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'two_role_runtime', False)):
        raise ValueError('scout_prompt_parity requires two-role dual_async pipelined preparation')
    if getattr(args, 'physical_failure_executor', False) and (
            not getattr(args, 'strong_recovery_executor', False)
            or not getattr(args, 'progress_recovery', False)):
        raise ValueError('physical_failure_executor requires Strong recovery routing and physical progress feedback')
    if getattr(args, 'strong_recovery_executor', False) and (
            args.mode != 'dual_async' or args.backend != 'direct_libero'
            or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'async_completion_recovery', False)):
        raise ValueError('strong_recovery_executor requires direct dual_async pipeline and completion recovery')
    if getattr(args, 'strong_recovery_executor', False) and any(getattr(args, name, False) for name in (
            'completion_schema_guard', 'completion_contract_repair', 'urgent_completion_review',
            'completion_rendezvous', 'recovery_reasoning', 'final_stage_prompt')):
        raise ValueError('strong_recovery_executor must be tested separately from completion ablations')
    if getattr(args, 'completion_schema_guard', False) and (
            args.mode != 'dual_async' or args.backend != 'direct_libero'
            or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'async_completion_recovery', False)):
        raise ValueError('completion_schema_guard requires direct dual_async pipeline and completion recovery')
    if getattr(args, 'completion_schema_guard', False) and any(getattr(args, name, False) for name in (
            'completion_contract_repair', 'urgent_completion_review', 'completion_rendezvous',
            'recovery_reasoning', 'final_stage_prompt')):
        raise ValueError('completion_schema_guard is a separate completion ablation')
    if getattr(args, 'completion_contract_repair', False) and (
            args.mode != 'dual_async' or args.backend != 'direct_libero'
            or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'priority_foreground', False)
            or not getattr(args, 'async_completion_recovery', False)):
        raise ValueError('completion_contract_repair requires direct dual_async pipeline, foreground priority and completion recovery')
    if getattr(args, 'completion_contract_repair', False) and (
            getattr(args, 'completion_rendezvous', False)
            or getattr(args, 'urgent_completion_review', False)):
        raise ValueError('completion_contract_repair is a separate ablation from completion_rendezvous and urgent_completion_review')
    if getattr(args, 'urgent_completion_review', False) and getattr(args, 'completion_rendezvous', False):
        raise ValueError('urgent_completion_review is a no-wait ablation and cannot be combined with completion_rendezvous')
    if getattr(args, 'urgent_completion_review', False) and (
            args.mode != 'dual_async' or args.backend != 'direct_libero'
            or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'priority_foreground', False)
            or not getattr(args, 'async_completion_recovery', False)):
        raise ValueError('urgent_completion_review requires direct dual_async pipeline, foreground priority and completion recovery')
    if getattr(args, 'view_recovery_guidance', False) and args.backend != 'direct_libero':
        raise ValueError('view_recovery_guidance describes direct_libero robot state and relative motion')
    if getattr(args, 'rim_grasp_guidance', False) and args.backend != 'direct_libero':
        raise ValueError('rim_grasp_guidance describes direct_libero measured candidate geometry')
    if getattr(args, 'phase_aware_grasp_feedback', False) and args.backend != 'direct_libero':
        raise ValueError('phase_aware_grasp_feedback requires direct_libero primitive receipts')
    if getattr(args, 'strong_destination_judge', False) and (
            args.mode != 'dual_async' or args.backend != 'direct_libero'
            or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'strong_target_judge', False)):
        raise ValueError('strong_destination_judge requires dual_async direct pipeline with Strong target judge')
    if getattr(args, 'target_judge_overlap', False) and (
            args.mode != 'dual_async' or not getattr(args, 'pipeline_supervision', False)
            or not getattr(args, 'strong_target_judge', False)):
        raise ValueError('target_judge_overlap requires dual_async pipeline and Strong target judge')
    if getattr(args, "faithful_initial_referents", False) and (
            args.mode != "dual_async" or not getattr(args, "pipeline_supervision", False)):
        raise ValueError("faithful_initial_referents requires dual_async pipelined milestones")
    if getattr(args, "recovery_reasoning", False) and (
            args.mode != "dual_async" or not getattr(args, "pipeline_supervision", False)
            or not getattr(args, "async_completion_recovery", False)):
        raise ValueError("recovery_reasoning requires dual_async pipelined final completion recovery")
    if getattr(args, "final_stage_prompt", False) and (
            args.mode != "dual_async" or not getattr(args, "pipeline_supervision", False)
            or not getattr(args, "async_completion_recovery", False)):
        raise ValueError("final_stage_prompt requires dual_async pipelined final completion recovery")
    if getattr(args, "completion_rendezvous", False) and (
            args.mode != "dual_async" or not getattr(args, "pipeline_supervision", False)
            or not getattr(args, "async_completion_recovery", False)):
        raise ValueError("completion_rendezvous requires dual_async pipelined final completion recovery")
    if getattr(args, "strong_target_judge", False) and (
            args.mode != "dual_async" or args.backend != "direct_libero" or not args.identity_check):
        raise ValueError("strong_target_judge requires dual_async direct_libero with identity_check")
    if getattr(args, "async_completion_recovery", False) and (
            args.mode != "dual_async" or not getattr(args, "pipeline_supervision", False)):
        raise ValueError("async_completion_recovery requires dual_async pipelined supervision")
    if getattr(args, "reliable_surfaces", False) and args.backend != "direct_libero":
        raise ValueError("reliable_surfaces requires the direct_libero RGB-D geometry adapter")
    if getattr(args, "pipeline_supervision", False) and (
            args.mode != "dual_async" or args.backend != "direct_libero"
            or getattr(args, "hierarchy", "milestones") != "milestones"):
        raise ValueError("pipeline_supervision currently requires dual_async milestones with direct_libero physics-clock evidence")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "events.sqlite").exists():
        raise RuntimeError("Run directory already contains state; use a new directory. Automatic physical replay is prohibited.")
    load_env(args.model_env)
    settings = vars(args).copy()
    if getattr(args, 'two_role_runtime', False):
        settings['agent_topology'] = {
            'name': 'slow_fast_only', 'roles': list(allowed_roles(args.mode)),
            'independent_checkers': False, 'strong_executor_handoff': False,
            'slow_execution_tools': False,
            'semantics': 'existing actor and tool-free planner; normal action and success gates unchanged',
        }
    from embodied_harness.runtime.memory import retrieve
    memory_options = {'appearance_reference_memory':True} if getattr(args, 'appearance_reference_memory', False) else {}
    memory=retrieve(getattr(args,'memory',None),args.mission, **memory_options)
    settings['appearance_memory']=[{k:v for k,v in entry.items() if k!='caption'} for entry in memory]
    settings.pop("model_env", None)
    settings.update({"slow_model": os.environ["SLOW_MODEL"], "fast_model": os.environ["FAST_MODEL"],
                     "model_service_fingerprints": {role: hashlib.sha256(role_base_url(role).rstrip("/").encode()).hexdigest()
                                                    for role in ("fast", "slow")},
                     "fast_model_options": json.loads(os.environ.get("FAST_MODEL_OPTIONS", "{}")),
                     "slow_model_options": json.loads(os.environ.get("SLOW_MODEL_OPTIONS", "{}")),
                     "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*.py")))).hexdigest(),
                     "policy": "RGB-D visual model with ETA atomic tools; no privileged object-state policy",
                     "initial_state_protocol": "official init_state index selected by dedicated server; seed selects matching port in paired runner",
                     "cancellation": "safe_boundary_only", "started_at": time.time()})
    if getattr(args, "recovery_reasoning", False):
        if (settings["fast_model"] != "Qwen3.6-35B-A3B"
                or (settings["fast_model_options"].get("chat_template_kwargs") or {}).get("enable_thinking") is not False):
            raise ValueError("recovery_reasoning requires the non-thinking Qwen Fast profile")
        token_limit = settings["fast_model_options"].get("max_tokens", 4096)
        if type(token_limit) is not int or token_limit != 4096:
            raise ValueError("recovery_reasoning preserves the 4096-token output budget")
        settings["recovery_reasoning_policy"] = {
            "trigger": "official_final_completion_rejected_at_current_scope",
            "max_requests_per_episode": 4, "model": settings["fast_model"],
            "option_override": {"chat_template_kwargs": {"enable_thinking": True}},
            "changes_output_budget": False,
            "cost_semantics": "extra reasoning within existing decision/output/wall budgets; billed explicitly"}
    eta_worker = Path("vendor/Stage2/sim/mcp_server/worker_mgr.py")
    if getattr(args, "completion_rendezvous", False):
        settings["completion_review_timeout_s"] = 30.0
    if getattr(args, "strong_target_judge", False):
        settings.update(target_judge_model=settings["slow_model"],
                        target_judge_options=settings["slow_model_options"],
                        target_judge_trigger="every_pregrasp_with_same_scope_cache",
                        target_judge_memory=("verified_prior_appearance_if_matched_plus_original_mission_and_public_images"
                            if getattr(args, 'appearance_reference_memory', False)
                            else "none_original_mission_and_public_images_only"))
    if getattr(args, 'target_judge_overlap', False):
        settings['target_judge_scheduling'] = 'one_speculative_same_snapshot_check_overlapped_with_actor'
    if getattr(args, 'strong_destination_judge', False):
        settings.update(destination_judge_model=settings['slow_model'],
            destination_judge_options=settings['slow_model_options'],
            destination_judge_trigger='every_place_with_exact_point_and_scope_cache',
            destination_judge_memory='original_mission_task_start_and_current_images_only')
    if getattr(args, 'strong_recovery_executor', False):
        settings['recovery_executor_policy'] = {
            'model': settings['slow_model'], 'options': settings['slow_model_options'],
            'role': 'recovery_executor', 'max_requests': 4,
            'trigger': ('first_official_negative_final_claim_or_two_same_key_physical_failures'
                if getattr(args, 'physical_failure_executor', False)
                else 'first_official_negative_final_claim_with_two_future_decisions'),
            'scheduling': 'replace_actor_calls_within_existing_decision_budget',
            'scope': 'current_final_contract_id_revision_generation_mission',
            'history': 'copy_primary_executed_history_and_commit_only_returned_decisions',
            'cost': 'separate_strong_executor_accounting_no_planner_tool_access'}
    settings["eta_worker_manager_sha256"] = hashlib.sha256(eta_worker.read_bytes()).hexdigest() if eta_worker.exists() else None
    (output / "config.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    store = EventStore(output / "events.sqlite")
    if getattr(args, 'appearance_reference_memory', False):
        store.emit('memory.appearance_recalled', {
            'reference_count':len(memory),
            'new_experience_count':sum(e.get('memory_class') == 'verified_single_acquisition_v1' for e in memory),
            'references':[{k:e[k] for k in ('match_text','image_sha256','manifest_sha256')} for e in memory],
            'semantics':'prior appearance only; no current identity, pose, retention, or success certification'})
    context = Context(store)
    context.phase_aware_grasp_feedback = getattr(args, "phase_aware_grasp_feedback", False)
    adapter_type = OpenETAAdapter
    if args.backend == "direct_libero":
        settings.pop("eta_worker_manager_sha256", None)
        from embodied_harness.adapters.libero_direct import DirectLiberoAdapter
        adapter_type = DirectLiberoAdapter
    adapter = adapter_type(output, args.sim_url, args.env_id, args.mission, args.seed, args.timeout + 120, args.max_decisions * 12)
    adapter.reliable_surfaces = getattr(args, "reliable_surfaces", False)
    settings["cancellation"] = adapter.cancellation
    if args.backend == "direct_libero":
        settings.update(policy="RGB-D model with bounded simulator-only OSC tools; no hidden object-state policy",
                        initial_state_protocol="official LIBERO initial-state index equals seed; ten settling steps",
                        controller="OSC_POSE", collision_coverage="simulated_contacts_only_no_path_certificate")
    (output / "config.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    coordinator = Coordinator(adapter, context, store)
    models = []
    reviewer = None
    sensor_watcher = None
    fast = slow = None
    milestones = None
    started = time.monotonic()
    status = "initialization_error"
    error = None
    evidence = None
    topology = ModelTopology(args.mode, getattr(args, 'two_role_runtime', False), store)
    execution_elapsed_s = None
    try:
        await adapter.start()
        await coordinator.observe()
        mission_start_observation = context.latest
        use_strong = args.mode == "single_strong"
        main = topology.create(ModelClient, role_base_url("slow" if use_strong else "fast"), os.environ["SLOW_API_KEY" if use_strong else "FAST_API_KEY"],
                           os.environ["SLOW_MODEL" if use_strong else "FAST_MODEL"], "fast", output,
                           options=settings["slow_model_options" if use_strong else "fast_model_options"])
        models.append(main)
        main.bind_tools(adapter.schemas)
        main.bind_memory(memory)
        main.bind_mission_reference(mission_start_observation)
        if args.mode.startswith("dual"):
            sm = topology.create(ModelClient, role_base_url("slow"), os.environ["SLOW_API_KEY"], os.environ["SLOW_MODEL"], "slow", output,
                             options=settings["slow_model_options"])
            models.append(sm)
            sm.bind_memory(memory)
            sm.bind_mission_reference(mission_start_observation)
            slow = SlowAgent(sm, context, coordinator, store,
                priority_foreground=getattr(args, "priority_foreground", False),
                review_economy=getattr(args, "review_economy", False),
                retain_diagnostics=getattr(args, "diagnostic_memory", False),
                urgent_completion_review=getattr(args, "urgent_completion_review", False),
                milestone_advice_delivery=getattr(args, "milestone_advice_delivery", False))
            if getattr(args, "hierarchy", "milestones") == "milestones":
                from embodied_harness.runtime.milestones import MilestoneManager
                manager_type = MilestoneManager
                if getattr(args, "pipeline_supervision", False):
                    if args.mode != "dual_async":
                        raise ValueError("pipeline_supervision requires dual_async")
                    from embodied_harness.runtime.pipeline import PipelinedMilestoneManager
                    manager_type = PipelinedMilestoneManager
                coordinator.authorize_mission(args.mission,
                    tuple(k for k in adapter.schemas if k not in {"observe", "mark_point", "locate_point", "select_candidate"}))
                milestones = manager_type(slow, context, coordinator, store, args.max_decisions,
                                               started + args.timeout)
                milestones.faithful_initial_referents = getattr(args, "faithful_initial_referents", False)
                if getattr(milestones, "pipelined", False):
                    milestones.scout_prompt_parity = getattr(args, "scout_prompt_parity", False)
                    milestones.rim_grasp_guidance = getattr(args, "rim_grasp_guidance", False)
                    milestones.view_recovery_guidance = getattr(args, "view_recovery_guidance", False)
                    await milestones.initialize_with_scout(main)
                else:
                    await milestones.initialize()
                if args.mode == "dual_async":
                    reviewer = asyncio.create_task(slow.run())
                    if args.backend == "direct_libero":
                        sensor_watcher = asyncio.create_task(slow.watch_execution())
            elif args.mode == "dual_async":
                # The user already authorized the mission. Start local execution
                # immediately; the strong model reviews rather than blocking startup
                # or turning an uncertain visual guess into the mission contract.
                coordinator.delegate(Milestone(uid(),1,args.mission,args.mission,
                    capabilities=tuple(k for k in adapter.schemas if k not in {"observe","mark_point","locate_point","select_candidate"})))
                slow.request_review("initial_background_grounding: independently check target identity and local strategy")
                reviewer = asyncio.create_task(slow.run())
                if args.backend == "direct_libero":
                    sensor_watcher = asyncio.create_task(slow.watch_execution())
            else:await slow.delegate(args.mission)
        else:
            coordinator.delegate(Milestone(uid(), 1, args.mission, args.mission,
                                           capabilities=tuple(k for k in adapter.schemas if k not in {"observe","mark_point","locate_point","select_candidate"})))
        fast = FastAgent(main, context, coordinator, store, slow, args.mode, args.review_every,
            progress_recovery=getattr(args, "progress_recovery", False),
            async_completion_recovery=getattr(args, "async_completion_recovery", False),
            completion_rendezvous=getattr(args, "completion_rendezvous", False),
            final_stage_prompt=getattr(args, "final_stage_prompt", False),
            recovery_reasoning=getattr(args, "recovery_reasoning", False),
            rim_grasp_guidance=getattr(args, "rim_grasp_guidance", False),
            view_recovery_guidance=getattr(args, "view_recovery_guidance", False),
            completion_contract_repair=getattr(args, "completion_contract_repair", False),
            completion_schema_guard=getattr(args, "completion_schema_guard", False),
            physical_failure_executor=getattr(args, "physical_failure_executor", False),
            scout_prompt_parity=getattr(args, "scout_prompt_parity", False))
        fast.milestones = milestones
        if milestones:
            main.control_context = context
        if getattr(args, 'strong_recovery_executor', False):
            recovery_executor = topology.create(ModelClient, role_base_url('slow'), os.environ['SLOW_API_KEY'],
                os.environ['SLOW_MODEL'], 'recovery_executor', output,
                options=settings['slow_model_options'])
            models.append(recovery_executor)
            recovery_executor.bind_tools(adapter.schemas)
            recovery_executor.bind_memory(memory)
            recovery_executor.bind_mission_reference(mission_start_observation)
            recovery_executor.control_context = context
            fast.recovery_executor_model = recovery_executor
        if getattr(args, "strong_target_judge", False):
            from embodied_harness.agents.grounding import MissionTargetJudge
            judge = topology.create(ModelClient, role_base_url("slow"), os.environ["SLOW_API_KEY"],
                os.environ["SLOW_MODEL"], "target_judge", output, options=settings["slow_model_options"])
            if getattr(args, 'appearance_reference_memory', False):
                judge.bind_memory(memory)
            judge.bind_mission_reference(mission_start_observation)
            judge.control_context = context
            models.append(judge)
            judge_type = MissionTargetJudge
            if getattr(args, 'target_judge_overlap', False):
                from embodied_harness.agents.overlap_grounding import OverlappedTargetJudge
                judge_type = OverlappedTargetJudge
            fast.identity_checker = judge_type(judge, context, coordinator, store)
        elif memory and args.backend=="direct_libero" and getattr(args,'identity_check',True):
            from embodied_harness.agents.grounding import IdentityChecker
            # A single-strong baseline uses the strong model for EVERY policy
            # judgement, including this independent appearance verification.
            judge=topology.create(ModelClient, role_base_url("slow" if use_strong else "fast"),os.environ["SLOW_API_KEY" if use_strong else "FAST_API_KEY"],
                os.environ["SLOW_MODEL" if use_strong else "FAST_MODEL"],"grounding",output,
                options={**settings["slow_model_options" if use_strong else "fast_model_options"],"max_tokens":1024})
            judge.bind_memory(memory)
            judge.bind_mission_reference(mission_start_observation)
            models.append(judge)
            if milestones:
                judge.control_context = context
            fast.identity_checker=IdentityChecker(judge,context,coordinator,store)
        if getattr(args, 'strong_destination_judge', False):
            from embodied_harness.agents.destination_grounding import DestinationJudge
            destination_model = topology.create(ModelClient, role_base_url('slow'), os.environ['SLOW_API_KEY'],
                os.environ['SLOW_MODEL'], 'destination_judge', output, options=settings['slow_model_options'])
            destination_model.bind_mission_reference(mission_start_observation)
            destination_model.control_context = context
            models.append(destination_model)
            fast.destination_checker = DestinationJudge(destination_model, context, coordinator, store)
        topology.verify_complete()
        # Deadline checked between decisions, rather than abandoning a possibly running physical call.
        async def run():
            return await fast.run(args.max_decisions-getattr(milestones, "scout_decisions", 0), deadline=started + args.timeout)
        status = await run()
        evidence = await coordinator.verify()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        store.emit("episode.error", error)
    finally:
        execution_elapsed_s = time.monotonic() - started
        if getattr(milestones, "pipelined", False):
            await milestones.close()
        if sensor_watcher:
            sensor_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sensor_watcher
        if reviewer:
            reviewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reviewer
        with contextlib.suppress(Exception):
            await adapter.close()
        rows = store.since(0)
        result = {**settings, "status": status, "success": bool(evidence and evidence["status"] == "verified"),
                  "elapsed_s": time.monotonic() - started, "execution_elapsed_s": execution_elapsed_s,
                  "decisions": (fast.steps if fast else 0) + getattr(milestones, "scout_decisions", 0),
                  "scout_decisions": getattr(milestones, "scout_decisions", 0),
                  "action_count": context.action_seq, "false_completion_claims": fast.false_claims if fast else 0,
                  "action_count_semantics": "accepted Coordinator mutation requests; excludes read-only calls/admission rejects; includes Host preflight rejects; skills can dispatch multiple primitives",
                  "execution_accounting": dict(adapter.host.execution_accounting) if adapter.host else {},
                  "model_errors": fast.errors if fast else 0, "slow_reviews": slow.reviews if slow else 0,
                  "milestone_metrics": {"planned": milestones.plan_id is not None, "plans": 1 + milestones.replans,
                      "completed_stages": len(milestones.completed), "checkpoints": milestones.checkpoints,
                      "completed_stages_semantics": "intermediate outcomes assessed at checkpoints; environment termination recorded separately",
                      "official_terminal": milestones.official_terminal,
                      "recovery_requests": milestones.recoveries, "active_stage": milestones.index,
                      "final_stage_active": bool(context.milestone and context.milestone.final_stage)} if milestones else None,
                  "advice_considered_decisions": sum(r["kind"] == "advice.considered" for r in rows),
                  "inference_preemptions": sum(r["kind"] == "control.inference_preempted" for r in rows),
                  "stale_advice": sum(r["kind"] == "advice.stale" for r in rows),
                  "accepted_advice": sum(r["kind"] == "advice.accepted" for r in rows),
                  "reviews_overlapping_actions": sum(r["kind"] == "slow.review_finished" and
                        r["payload"]["based_on_action_seq"] != r["payload"]["current_action_seq"] for r in rows),
                  "models": [{"role": m.role, "model": m.model, "calls": m.calls, "usage": m.usage,
                               "requests_without_reported_usage": max(0, m.calls - m.accounting["responses_with_usage"]),
                               "request_accounting": m.accounting,
                               "usage_semantics": "reported usage only; transport failures and cancellations have unknown provider cost",
                               "latencies_s": m.latencies} for m in models], "evidence": evidence, "error": error}
        (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        for m in models:
            await m.close()
        store.close()
        print(json.dumps({k: result[k] for k in ("status", "success", "elapsed_s", "decisions", "action_count", "error")}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-url", default="http://127.0.0.1:18769/sse")
    p.add_argument("--backend", choices=["eta", "direct_libero"], default="eta")
    p.add_argument("--env-id", default="openeta/libero_libero_spatial_task0-v0")
    p.add_argument("--mission", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["single_fast", "single_strong", "dual_sync", "dual_async"], default="dual_async")
    p.add_argument("--max-decisions", type=int, default=40)
    p.add_argument("--review-every", type=int, default=3)
    p.add_argument('--two-role-runtime', action='store_true', help='Only one actor and, in dual mode, one tool-free Slow client; requires --no-identity-check and no independent judge/recovery routes')
    p.add_argument('--scout-prompt-parity', action='store_true', help='Reuse the actor base prompt in initial and checkpoint read-only preparation; requires two-role pipeline')
    p.add_argument("--priority-foreground", action="store_true", help="Preempt background inference for foreground milestone work")
    p.add_argument("--review-economy", action="store_true", help="Outcome-based stage granularity and coalesced redundant review")
    p.add_argument("--diagnostic-memory", action="store_true", help="Retain typed historical Slow hypotheses for current-evidence rechecks")
    p.add_argument("--reliable-surfaces", action="store_true", help="Choose candidate surface samples only from depth-continuous visible patches")
    p.add_argument("--async-completion-recovery", action="store_true", help="Recover rejected final claims immediately while Slow reviews in background")
    p.add_argument("--strong-target-judge", action="store_true", help="Independent Strong mission-to-current-crop judgment before grasps; separately accounted")
    p.add_argument('--target-judge-overlap', action='store_true', help='Speculate at most one same-snapshot target check alongside Fast inference; discard unused work and retain all execution gates')
    p.add_argument('--strong-destination-judge', action='store_true', help='Independent Strong check of actual current placement anchor against mission destination, separately accounted')
    p.add_argument("--completion-rendezvous", action="store_true", help="After repeated rejected final claims, join the already requested Slow review once with a 30s bound")
    p.add_argument("--faithful-initial-referents", action="store_true", help="Keep inferred appearance out of persistent initial mission contracts; prompt ablation")
    p.add_argument("--recovery-reasoning", action="store_true", help="At most four Qwen thinking requests after scoped official final rejection; explicit compute ablation")
    p.add_argument("--final-stage-prompt", action="store_true", help="Use an explicit final-mission completion contract instead of intermediate-stage handoff wording")
    p.add_argument("--phase-aware-grasp-feedback", action="store_true", help="Expose bounded historical grasp phase outcomes in failure receipts; no new decisions or permissions")
    p.add_argument("--rim-grasp-guidance", action="store_true", help="Clarify executor rim selection and measured orientation defaults; prompt-only ablation")
    p.add_argument("--view-recovery-guidance", action="store_true", help="Suggest conditional measured retreat for a robot-occluded failed-grasp view; prompt-only ablation")
    p.add_argument("--urgent-completion-review", action="store_true", help="Prioritize official-negative review by cancelling only background work from an older physical action; no Fast wait")
    p.add_argument("--completion-contract-repair", action="store_true", help="Once per physical revision, reassess remaining outcomes in the foreground after two rejected final claims; existing budgets apply")
    p.add_argument("--completion-schema-guard", action="store_true", help="Request nonterminal Fast JSON decisions after official rejection at an unchanged physics/control revision; no host action choice")
    p.add_argument("--strong-recovery-executor", action="store_true", help="Replace at most four actor calls after the first official negative final claim with a separately accounted Strong executor; Slow remains tool-free")
    p.add_argument("--physical-failure-executor", action="store_true", help="Allow the same one-burst Strong executor after two actual same-key physical failures within the current final contract; no added decisions or tools")
    p.add_argument("--progress-recovery", action="store_true", help="Retain measured physical failures and block equivalent stagnant retries")
    p.add_argument("--pipeline-supervision", action="store_true", help="Overlap planning/checkpoints with read-only preparation and route automatic recovery to background review")
    p.add_argument("--milestone-advice-delivery", action="store_true", help="Keep timestamped high-level Slow hypotheses in current-episode history without time expiry; Fast assesses relevance against current evidence")
    p.add_argument("--hierarchy", choices=["milestones", "legacy_review"], default="milestones",
                   help="Dual modes only: Slow outcome contracts, or the frozen mission/advisory loop")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--output", required=True)
    p.add_argument("--model-env", default=".model-env")
    p.add_argument("--identity-check",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--memory",default=None,help="Optional verified appearance manifest; disclose source states in evaluation")
    p.add_argument("--appearance-reference-memory", action="store_true", help="Enable frozen single-acquisition appearance experience for actors, planner and independent identity judge; prior pixels never ground a current action")
    args = p.parse_args()
    asyncio.run(episode(args))


if __name__ == "__main__":
    main()
