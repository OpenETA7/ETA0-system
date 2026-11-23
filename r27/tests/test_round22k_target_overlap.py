"""Offline regressions for speculative, same-snapshot target judging."""
import asyncio
import copy
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.grounding import MissionTargetJudge
from embodied_harness.agents.overlap_grounding import OverlappedTargetJudge
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator, GateError


MISSION = "pick up the blue jar and put it inside the bowl"
SENSOR = {"sim_time": 1.0, "control_steps": 12, "captured_at": 100.0,
          "visual_sample": 9}
ROBOT = {"grip_xyz_m": [0.1, 0.2, 0.3], "joint_positions": [0.0, 0.1]}
TACTILE = {"left_contact": False, "right_contact": False}


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


def target_point(point_id="p1", candidate_id="candidate-1", *, packet="packet-1",
                 camera="agentview", bbox=(20, 30, 90, 110)):
    return {
        "point_id": point_id,
        "source_packet_id": packet,
        "camera_frame_id": camera,
        "pixel_xy": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
        "xyz_m": [0.1, 0.2, 0.3],
        "visual_candidate": {"candidate_id": candidate_id, "bbox_xyxy": list(bbox)},
    }


def actor_observation(point, *, version=1, observation_id="actor-observation", full_image="full.png"):
    return Observation(
        observation_id, version, 1, time.monotonic(), 1.0, 0,
        {"candidate_scope": "packet-1/objects", "sensor": copy.deepcopy(SENSOR),
         "robot": copy.deepcopy(ROBOT), "tactile": copy.deepcopy(TACTILE),
         "feedback": {"point": copy.deepcopy(point)},
         "views": [{"source_packet_id": "packet-1", "camera_frame_id": "agentview",
                    "is_current_observation": True}]},
        (full_image,),
    )


class Adapter:
    generation = 1
    terminal = False
    possible_hold = False
    schemas = {
        "observe": {"parameters": {"properties": {"camera": {"type": "string"}}}},
        "grasp_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
    }

    def __init__(self, points, *, scope="packet-1/objects", capture_mutation=None,
                 full_image="full.png", crop_image="crop.png"):
        self.points = points
        self.scope = scope
        self.capture_mutation = capture_mutation
        self.full_image, self.crop_image = full_image, crop_image
        self.version = 1
        self.calls = []
        self.executed = []
        self.capture_action_seq = None
        self.capture_generation = None

    def validate_args(self, _tool, _args):
        return None

    def _observation(self, action_seq, *, camera="agentview", crop=None):
        self.version += 1
        point = next(iter(self.points.values()))
        packet = point["source_packet_id"]
        views = [{"source_packet_id": packet, "camera_frame_id": "agentview",
                  "is_current_observation": True}]
        images = (self.full_image,)
        if crop is not None:
            views.append({"source_packet_id": packet, "camera_frame_id": "zoom_preview",
                          "source_camera": camera, "original_crop_xyxy": list(crop)})
            images = (self.full_image, self.crop_image)
        state = {"candidate_scope": self.scope, "sensor": copy.deepcopy(SENSOR),
                 "robot": copy.deepcopy(ROBOT), "tactile": copy.deepcopy(TACTILE),
                 "views": views}
        if crop is not None and self.capture_mutation:
            self.capture_mutation(state)
        return Observation(
            f"adapter-{self.version}", self.version,
            self.generation if self.capture_generation is None else self.capture_generation,
            time.monotonic(), 1.0,
            action_seq if self.capture_action_seq is None else self.capture_action_seq,
            state, images,
        )

    async def inspect(self, tool, args, action_seq):
        self.calls.append((tool, copy.deepcopy(args), action_seq))
        return self._observation(action_seq, camera=args.get("camera", "agentview"),
                                 crop=args.get("crop_xyxy"))

    async def observe(self, action_seq):
        return self._observation(action_seq)

    async def execute(self, capability, args, *_args):
        self.executed.append((capability, copy.deepcopy(args)))
        raise AssertionError("offline tests must never execute a physical action")

    async def verify(self, _observation_id):
        return {"status": "not_verified"}


class JudgeModel:
    def __init__(self, verdicts=None):
        self.verdicts = deque(verdicts or ["match"])
        self.history = deque([{"role": "assistant", "content": "actor says p1 is the jar"}], maxlen=6)
        self.mission_reference_metadata = {"image_sha256": "task-start-sha"}
        self.mission_reference_messages = [{"role": "user", "content": "bound task-start image"}]
        self.reference_metadata = []
        self.reference_messages = []
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.cancelled = asyncio.Event()
        self.cancel_cleanup = asyncio.Event()
        self.hold_cancel_cleanup = False

    async def decide(self, system, query, images):
        self.calls.append({"system": system, "query": copy.deepcopy(query), "images": tuple(images)})
        self.started.set()
        if self.block:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                if self.hold_cancel_cleanup:
                    await self.cancel_cleanup.wait()
                raise
        verdict = self.verdicts.popleft()
        if isinstance(verdict, dict):
            return verdict
        return {"verdict": verdict, "selected_appearance": "blue glass jar",
                "reason": "Visible color and shape match the task-start reference."}


def setup(tmp_path, *, verdicts=None, camera="agentview", capture_mutation=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, payload in (("full.png", b"current-full"), ("crop.png", b"selected-crop")):
        (tmp_path / name).write_bytes(payload)
    point = target_point(camera=camera)
    full_image, crop_image = str(tmp_path / "full.png"), str(tmp_path / "crop.png")
    store = Store()
    context = Context(store)
    context.milestone = Milestone("stage-1", 1, MISSION, "intermediate lift goal",
                                  capabilities=("grasp_point", "observe"))
    context.ingest(actor_observation(point, full_image=full_image))
    adapter = Adapter({"p1": point}, capture_mutation=capture_mutation,
                      full_image=full_image, crop_image=crop_image)
    coordinator = SimpleNamespace(
        adapter=adapter, lock=asyncio.Lock(), paused=False, unknown=False, running=None,
        contract_review_pending=False, pause_epoch=0, contract_review_epoch=0,
    )
    model = JudgeModel(verdicts)
    judge = OverlappedTargetJudge(model, context, coordinator, store)
    return store, context, adapter, model, judge


def events(store, kind):
    return [row["payload"] for row in store.events if row["kind"] == kind]


def test_prefetch_overlaps_actor_decision_without_mutating_actor_observation(tmp_path):
    async def run():
        store, context, adapter, judge_model, judge = setup(tmp_path)
        judge_model.block = True
        coordinator = Coordinator(adapter, context, store)
        actor_observation_before = context.latest
        frames_before = tuple(context.frames)
        action_seq_before = context.action_seq

        class Actor:
            history = deque(maxlen=6)

            async def decide(self, _system, projection, _images):
                await judge_model.started.wait()
                self.saw_judge_pending = not judge_model.release.is_set()
                self.actor_context_unchanged = (
                    context.latest is actor_observation_before
                    and len(context.frames) == len(frames_before))
                self.projected_observation = projection["observation"]["id"]
                return {"kind": "tool", "tool": "observe", "args": {"camera": "agentview"}}

        actor = Actor()
        fast = FastAgent(actor, context, coordinator, store)
        fast.identity_checker = judge
        status = await fast.run(1)

        assert status == "decision_budget"
        assert actor.saw_judge_pending is True
        assert actor.actor_context_unchanged is True
        assert actor.projected_observation == actor_observation_before.id
        assert context.action_seq == action_seq_before
        assert context.frames[0] is frames_before[0]
        assert context.latest.id != actor_observation_before.id  # Actor's own observe was ingested afterward.
        assert judge.pending is None
        assert judge.cache == {}
        assert events(store, "target_identity.prefetch_discarded")[0]["outcome"] == "cancelled"
        assert not events(store, "target_identity.checked")
        assert not adapter.executed

    asyncio.run(run())


@pytest.mark.parametrize("hold_state", ["possibly_holding", "unknown"])
def test_prefetch_is_skipped_unless_adapter_and_actor_both_know_gripper_is_empty(tmp_path, hold_state):
    async def run():
        store, _context, adapter, model, judge = setup(tmp_path)
        if hold_state == "possibly_holding":
            judge.context.local_state["possible_hold"] = True
        else:
            adapter.possible_hold = None

        await judge.prefetch_current()

        assert judge.pending is None
        assert adapter.calls == []
        assert model.calls == []
        assert not events(store, "target_identity.prefetch_started")

    asyncio.run(run())


@pytest.mark.parametrize("change", ["control", "action", "observation", "pause", "unknown",
                                     "generation", "contract"])
def test_pending_result_is_rejected_when_its_authority_or_observation_scope_changes(tmp_path, change):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path)
        model.block = True
        await judge.prefetch_current()
        pending_task = judge.pending["task"]
        await asyncio.wait_for(model.started.wait(), timeout=1)

        if change == "control":
            context.control_version += 1
        elif change == "action":
            context.action_seq += 1
        elif change == "observation":
            context.ingest(actor_observation(adapter.points["p1"], version=2,
                                             observation_id="replacement-observation",
                                             full_image=adapter.full_image))
        elif change == "pause":
            context.control_version += 1
            judge.coordinator.paused = True
            judge.coordinator.pause_epoch += 1
        elif change == "unknown":
            judge.coordinator.unknown = True
        elif change == "generation":
            adapter.generation += 1
        elif change == "contract":
            context.milestone = Milestone("stage-1", 2, MISSION, "revised lift goal",
                                          capabilities=("grasp_point", "observe"))
        model.release.set()
        with pytest.raises(GateError):
            await judge.check("p1")

        assert pending_task.done()
        assert judge.pending is None
        assert judge.cache == {}
        assert "target_identity_check" not in context.local_state
        assert not events(store, "target_identity.checked")
        assert not events(store, "target_identity.prefetch_consumed")

    asyncio.run(run())


def test_changed_point_cannot_consume_an_already_returned_prefetch(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path)
        model.block = True
        await judge.prefetch_current()
        pending = judge.pending
        await asyncio.wait_for(model.started.wait(), timeout=1)
        model.release.set()
        await asyncio.wait_for(pending["task"], timeout=1)
        adapter.points["p1"] = target_point("p1", "replacement-candidate")

        with pytest.raises(GateError, match="prefetched_point_changed"):
            await judge.check("p1")
        assert judge.pending is None
        assert judge.cache == {}
        assert not events(store, "target_identity.checked")
        assert events(store, "target_identity.prefetch_returned")[0]["published_to_actor"] is False
        assert events(store, "target_identity.prefetch_discarded")[0]["outcome"] == "returned_but_unused"
        assert "target_identity_check" not in context.local_state
        assert pending["task"].done()

    asyncio.run(run())


def test_physical_drift_skips_prefetch_but_visual_sample_fields_may_advance(tmp_path):
    async def run():
        def visual_only(state):
            state["sensor"]["captured_at"] += 1.0
            state["sensor"]["visual_sample"] += 1

        store, context, _adapter, model, judge = setup(tmp_path, capture_mutation=visual_only)
        latest_before, frame_count = context.latest, len(context.frames)
        await judge.prefetch_current()
        assert judge.pending is not None
        assert context.latest is latest_before
        assert len(context.frames) == frame_count
        await judge.discard("test_cleanup")

        def physics_changed(state):
            state["sensor"]["sim_time"] += 0.01

        store2, context2, _adapter2, model2, judge2 = setup(tmp_path / "drift", capture_mutation=physics_changed)
        latest_before2, frame_count2 = context2.latest, len(context2.frames)
        await judge2.prefetch_current()
        assert judge2.pending is None
        assert model2.calls == []
        assert events(store2, "target_identity.prefetch_skipped")[0]["mandatory_check_unchanged"] is True
        assert context2.latest is latest_before2
        assert len(context2.frames) == frame_count2

    asyncio.run(run())


def test_unused_returned_result_reports_uncertain_cost_and_is_not_cached_or_published(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path, verdicts=["match", "match"])
        await judge.prefetch_current()
        await asyncio.wait_for(judge.pending["task"], timeout=1)
        returned = events(store, "target_identity.prefetch_returned")[0]
        assert returned["published_to_actor"] is False
        await judge.discard("actor_chose_other_operation")

        discarded = events(store, "target_identity.prefetch_discarded")[0]
        assert discarded["outcome"] == "returned_but_unused"
        assert "reported usage only" in discarded["provider_cost"]
        assert "does not prove unbilled" in discarded["provider_cost"]
        assert judge.pending is None
        assert judge.cache == {}
        assert not events(store, "target_identity.checked")
        assert "target_identity_check" not in context.local_state

        assert await judge.check("p1") is True  # Mandatory baseline path asks again.
        assert len(model.calls) == 2
        assert len(adapter.calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize("verdict", ["match", "mismatch", "uncertain"])
def test_consumed_verdict_uses_one_api_call_and_uncertain_is_not_persisted(tmp_path, verdict):
    async def run():
        store, _context, adapter, model, judge = setup(tmp_path, verdicts=[verdict, verdict])
        assert await judge.check("p1") is (verdict == "match")
        assert len(model.calls) == len(adapter.calls) == 1
        assert not judge.pending
        if verdict == "uncertain":
            assert judge.cache == {}
            assert not events(store, "target_identity.cached")
            assert await judge.check("p1") is False
            assert len(model.calls) == len(adapter.calls) == 2
            assert judge.cache == {}
        elif verdict == "match":
            assert len(judge.cache) == 1
        else:
            assert len(judge.cache) == 1  # A scoped mismatch remains reusable as a veto.

    asyncio.run(run())


@pytest.mark.parametrize("invalid", [
    {"verdict": "maybe", "selected_appearance": "blue jar", "reason": "unclear"},
    {"verdict": "match", "selected_appearance": "blue jar"},
    {"verdict": "match", "selected_appearance": "blue jar", "reason": "ok", "extra": True},
])
def test_invalid_prefetched_verdict_never_publishes_or_authorizes(tmp_path, invalid):
    async def run():
        store, context, _adapter, _model, judge = setup(tmp_path, verdicts=[invalid])
        with pytest.raises(ValueError, match="target judge"):
            await judge.check("p1")
        assert judge.pending is None
        assert judge.cache == {}
        assert "target_identity_check" not in context.local_state
        assert not events(store, "target_identity.checked")

    asyncio.run(run())


def test_cached_match_skips_new_prefetch_and_extra_judge_or_capture_calls(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path, verdicts=["match"])
        assert await judge.check("p1") is True
        assert len(model.calls) == len(adapter.calls) == 1
        context.ingest(actor_observation(adapter.points["p1"], version=3,
                                         observation_id="same-physical-snapshot-new-view",
                                         full_image=adapter.full_image))
        await judge.prefetch_current()
        assert judge.pending is None
        assert await judge.check("p1") is True
        assert len(model.calls) == len(adapter.calls) == 1
        assert len(events(store, "target_identity.cached")) == 1

    asyncio.run(run())


def test_manual_wrist_point_keeps_mandatory_base_checker_fallback(tmp_path):
    async def run():
        store, _context, adapter, model, judge = setup(tmp_path, camera="robot0_wrist")
        await judge.prefetch_current()
        assert judge.pending is None
        assert model.calls == []

        assert await judge.check("p1") is True
        assert len(model.calls) == 1
        assert [call[1].get("camera") for call in adapter.calls] == ["agentview", "robot0_wrist"]
        assert len(events(store, "target_identity.prefetch_fallback")) == 1

    asyncio.run(run())


def test_mismatch_veto_in_fast_prevents_physical_dispatch(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path, verdicts=["mismatch"])
        coordinator = Coordinator(adapter, context, store)

        class Actor:
            history = deque([{"role": "assistant", "content": "p1 is the jar"}], maxlen=6)

            async def decide(self, *_args):
                return {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "p1"}}

        fast = FastAgent(Actor(), context, coordinator, store)
        fast.identity_checker = judge
        assert await fast.run(1) == "decision_budget"
        assert not adapter.executed
        assert not any(row["kind"] == "action.accepted" for row in store.events)
        assert context.action_seq == 0
        assert context.local_state["target_identity_check"]["verdict"] == "mismatch"
        assert model.calls and model.calls[0]["query"]["mission"] == MISSION

    asyncio.run(run())


@pytest.mark.parametrize(("mode", "pipeline", "strong"), [
    ("single_fast", True, True),
    ("dual_async", False, True),
    ("dual_async", True, False),
])
def test_overlap_flag_rejects_unsupported_runtime_combinations(tmp_path, mode, pipeline, strong):
    from embodied_harness import app

    args = SimpleNamespace(output=str(tmp_path / "invalid"), mode=mode, backend="direct_libero",
                           identity_check=True, target_judge_overlap=True,
                           pipeline_supervision=pipeline, strong_target_judge=strong)
    with pytest.raises(ValueError, match="target_judge_overlap requires dual_async pipeline and Strong target judge"):
        asyncio.run(app.episode(args))


def test_overlap_opt_in_is_absent_by_default_and_runner_requires_scheduling_metadata():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner
    from test_round22d_target_judge import _strong_target_runner_job
    from test_round22_runner import load_fixture_catalog

    _runner, _config, _case, default_job = _strong_target_runner_job()
    assert not default_job["feature_flags"].get("target_judge_overlap", False)
    assert "--target-judge-overlap" not in runner.app_arguments(default_job, Path("/tmp/no-memory-manifest.json"))

    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"]["strong_target_judge"] = True
    flags["dual_async"]["target_judge_overlap"] = True
    config = copy.deepcopy(_config)
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    case = next(row for row in config["cases"] if row["variant"] == "patched22"
                and row["mode"] == "dual_async" and row["suite"] != "libero_10")
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    args = runner.app_arguments(job, Path("/tmp/no-memory-manifest.json"))
    assert "--target-judge-overlap" in args

    result = {"target_judge_scheduling": None}
    assert "mismatch_target_judge_scheduling" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})


def test_default_strong_target_judge_wiring_remains_non_overlapped(monkeypatch, tmp_path):
    from embodied_harness import app
    from embodied_harness.agents.grounding import MissionTargetJudge
    from test_round20c_model_role_routing import _episode_args, _install_offline_app

    endpoints = {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"}
    constructed = _install_offline_app(monkeypatch, endpoints)
    agents = []

    class CapturingFast:
        def __init__(self, *_args, **_kwargs):
            self.identity_checker = None
            self.steps = self.false_claims = self.errors = 0
            agents.append(self)

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "FastAgent", CapturingFast)
    args = _episode_args(tmp_path, "dual_async")
    args.strong_target_judge = True
    # Deliberately omit target_judge_overlap: its feature remains opt-in.
    result = asyncio.run(app.episode(args))
    assert isinstance(agents[0].identity_checker, MissionTargetJudge)
    assert type(agents[0].identity_checker) is MissionTargetJudge
    assert result["error"] is None
    assert [model.role for model in constructed] == ["fast", "slow", "target_judge"]


def test_executor_cancellation_is_not_swallowed_by_prefetch_cleanup(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path)
        model.block = True
        model.hold_cancel_cleanup = True
        coordinator = Coordinator(adapter, context, store)

        class Actor:
            async def decide(self, *_args):
                await model.started.wait()
                await asyncio.Future()

        fast = FastAgent(Actor(), context, coordinator, store)
        fast.identity_checker = judge
        execution = asyncio.create_task(fast.run(1))
        await asyncio.wait_for(model.started.wait(), timeout=1)
        execution.cancel()
        await asyncio.wait_for(model.cancelled.wait(), timeout=1)
        execution.cancel()
        model.cancel_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert judge.pending is None
        assert judge.cache == {}
        assert not events(store, "target_identity.checked")
        assert events(store, "target_identity.prefetch_discarded")

    asyncio.run(run())
