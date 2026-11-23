"""Offline checks for semantic destination grounding at an exact marked point."""
import asyncio
import copy
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.destination_grounding import DestinationJudge
from embodied_harness.agents.fast import FastAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator, GateError


MISSION = "put the blue jar inside the red bowl"
SCOPE = "packet-1/objects"


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


def point(point_id="p1", *, pixel=(55, 70), xyz=(0.2, 0.1, 0.03), candidate="candidate-1",
          camera="agentview", packet="packet-1", bbox=(20, 30, 90, 110)):
    return {"point_id": point_id, "source_packet_id": packet, "camera_frame_id": camera,
            "pixel_xy": list(pixel), "xyz_m": list(xyz), "placement_anchor": True,
            "visual_candidate": {"candidate_id": candidate, "bbox_xyxy": list(bbox)}}


class Adapter:
    generation = 1
    terminal = False
    schemas = {"place_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}}}

    def __init__(self, points, full_image, crop_image):
        self.points, self.full_image, self.crop_image = points, full_image, crop_image
        self.version, self.calls, self.executed = 1, [], []
        self.marked_point = copy.deepcopy(next(iter(points.values())))
        self.selected_point = copy.deepcopy(self.marked_point)
        self.bad_crop = False
        self.marker_override = None
        self.on_inspect = None
        self.scene_packet = "packet-1"

    def validate_args(self, _tool, _args):
        return None

    async def inspect(self, tool, args, action_seq):
        self.calls.append((tool, copy.deepcopy(args), action_seq))
        crop = list(args["crop_xyxy"])
        marked = copy.deepcopy(self.marker_override if self.marker_override is not None else self.marked_point)
        zoom_crop = list(crop)
        if self.bad_crop:
            zoom_crop[0] += 1
        views = [
            {"source_packet_id": self.scene_packet, "camera_frame_id": "agentview",
             "is_current_observation": True},
            {"source_packet_id": self.selected_point["source_packet_id"], "camera_frame_id": "zoom_preview",
             "source_camera": args["camera"], "original_crop_xyxy": zoom_crop},
        ]
        self.version += 1
        obs = Observation(f"destination-obs-{self.version}", self.version, self.generation,
            time.monotonic(), 1.0, action_seq,
            {"candidate_scope": SCOPE, "views": views, "feedback": {"point": marked}},
            (self.full_image, self.crop_image))
        if self.on_inspect:
            await self.on_inspect(args)
        return obs

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"actor-observe-{self.version}", self.version, self.generation,
            time.monotonic(), 1.0, action_seq, {"candidate_scope": SCOPE, "views": []},
            (self.full_image,))

    async def execute(self, capability, args, *_args):
        self.executed.append((capability, copy.deepcopy(args)))
        raise AssertionError("offline tests must not dispatch physical actions")

    async def verify(self, _observation_id):
        return {"status": "not_verified"}


class Model:
    def __init__(self, verdicts=None):
        self.verdicts = deque(verdicts or ["match"])
        self.history = deque([{"role": "assistant", "content": "actor claims this is the bowl"}], maxlen=6)
        self.mission_reference_metadata = {"image_sha256": "task-start-sha"}
        self.mission_reference_messages = [{"role": "user", "content": "original task-start image"}]
        self.reference_metadata, self.reference_messages = [], []
        self.calls = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.block = False

    async def decide(self, system, query, images):
        self.calls.append({"system": system, "query": copy.deepcopy(query), "images": tuple(images)})
        self.entered.set()
        if self.block:
            await self.release.wait()
        value = self.verdicts.popleft()
        if isinstance(value, dict):
            return value
        return {"verdict": value, "selected_appearance": "red ceramic bowl opening",
                "reason": "The marked opening is the red bowl requested by the mission."}


def setup(tmp_path, *, verdicts=None, camera="agentview"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    full_image, crop_image = str(tmp_path / "full.png"), str(tmp_path / "crop.png")
    Path(full_image).write_bytes(b"current-scene")
    Path(crop_image).write_bytes(b"current-anchor-crop")
    store, context = Store(), None
    context = Context(store)
    context.milestone = Milestone("mission", 4, MISSION, "intermediate placement: stabilize object",
                                  capabilities=("place_point",))
    context.ingest(Observation("actor-observation", 1, 1, time.monotonic(), 1.0, 0,
                               {"candidate_scope": SCOPE, "views": []}, (full_image,)))
    selected = point(camera=camera)
    adapter = Adapter({"p1": selected}, full_image, crop_image)
    coordinator = SimpleNamespace(adapter=adapter, lock=asyncio.Lock(), paused=False,
        unknown=False, running=None, contract_review_pending=False,
        pause_epoch=0, contract_review_epoch=0)
    model = Model(verdicts)
    judge = DestinationJudge(model, context, coordinator, store)
    return store, context, adapter, model, judge


def events(store, name):
    return [event["payload"] for event in store.events if event["kind"] == name]


def test_match_uses_original_mission_task_start_and_only_scene_plus_exact_anchor(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path)
        task_start_messages = copy.deepcopy(model.mission_reference_messages)
        assert await judge.check("p1") is True

        call = model.calls[0]
        assert call["query"]["mission"] == MISSION
        assert call["query"]["mission"] != context.milestone.goal
        assert call["images"] == (adapter.full_image, adapter.crop_image)
        assert call["query"]["evidence"]["pixel_xy"] == adapter.marked_point["pixel_xy"]
        assert call["query"]["evidence"]["placement_anchor"] is True
        assert call["query"]["judgment_scope"] == "semantic_destination_at_marked_anchor_not_physical_success"
        assert model.history.maxlen == 0 and not model.history
        assert model.reference_metadata == model.reference_messages == []
        assert model.mission_reference_messages == task_start_messages
        assert "actor claims" not in call["system"]
        assert call["query"].keys() == {"mission", "control_version", "evidence", "views", "judgment_scope"}
        assert context.local_state["destination_check"]["verdict"] == "match"
        assert len(adapter.calls) == 1
        assert len(events(store, "destination_identity.checked")) == 1

    asyncio.run(run())


@pytest.mark.parametrize(("verdict", "expected"), [("match", True), ("mismatch", False), ("uncertain", False)])
def test_only_match_clears_destination_veto_and_uncertain_is_not_cached(tmp_path, verdict, expected):
    async def run():
        store, _context, adapter, model, judge = setup(tmp_path, verdicts=[verdict, verdict])
        assert await judge.check("p1") is expected
        assert len(model.calls) == len(adapter.calls) == 1
        if verdict == "uncertain":
            assert judge.cache == {}
            assert await judge.check("p1") is False
            assert len(model.calls) == len(adapter.calls) == 2
        elif verdict == "mismatch":
            assert len(judge.cache) == 1  # A scoped negative remains reusable as a veto.
        else:
            assert len(judge.cache) == 1

    asyncio.run(run())


def test_same_crop_different_marked_point_cannot_reuse_semantics(tmp_path):
    async def run():
        _store, _context, adapter, model, judge = setup(tmp_path, verdicts=["match", "mismatch"])
        p1 = adapter.points["p1"]
        p2 = point("p2", pixel=(60, 75), xyz=(0.21, 0.1, 0.03), candidate="candidate-1",
                   bbox=tuple(p1["visual_candidate"]["bbox_xyxy"]))
        adapter.points["p2"] = p2
        adapter.marked_point = p1
        adapter.selected_point = p1
        assert await judge.check("p1") is True
        adapter.marked_point = p2
        adapter.selected_point = p2
        assert await judge.check("p2") is False
        assert len(model.calls) == len(adapter.calls) == 2
        assert len(judge.cache) == 2

    asyncio.run(run())


def test_renamed_point_at_same_pixel_and_xyz_reuses_exact_anchor_cache(tmp_path):
    async def run():
        store, _context, adapter, model, judge = setup(tmp_path)
        original = adapter.points["p1"]
        assert await judge.check("p1") is True
        renamed = copy.deepcopy(original)
        renamed["point_id"] = "renamed-p1"
        adapter.points["renamed-p1"] = renamed
        adapter.marked_point = renamed
        adapter.selected_point = renamed
        assert await judge.check("renamed-p1") is True
        assert len(model.calls) == len(adapter.calls) == 1
        assert len(events(store, "destination_identity.cached")) == 1

    asyncio.run(run())


@pytest.mark.parametrize("marker_field", ["point_id", "source_packet_id", "camera_frame_id", "pixel_xy", "xyz_m"])
def test_missing_or_mismatched_actual_marker_fails_closed_before_model(marker_field, tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path)
        marked = copy.deepcopy(adapter.marked_point)
        marked[marker_field] = "wrong" if marker_field not in {"pixel_xy", "xyz_m"} else [9, 9]
        adapter.marker_override = marked
        with pytest.raises(GateError, match="selected_anchor_marker"):
            await judge.check("p1")
        assert model.calls == []
        assert not judge.cache
        assert "destination_check" not in context.local_state
        assert not events(store, "destination_identity.checked")

    asyncio.run(run())


def test_non_agentview_or_wrong_crop_provenance_fails_closed(tmp_path):
    async def run():
        _store, _context, _adapter, model, judge = setup(tmp_path, camera="robot0_wrist")
        with pytest.raises(GateError, match="agentview_anchor"):
            await judge.check("p1")
        assert model.calls == []

        store2, _context2, adapter2, model2, judge2 = setup(tmp_path / "wrong-crop")
        adapter2.bad_crop = True
        with pytest.raises(GateError, match="exact_current_scene_and_crop"):
            await judge2.check("p1")
        assert model2.calls == []
        assert not events(store2, "destination_identity.checked")

    asyncio.run(run())


@pytest.mark.parametrize("change", ["control", "action", "pause", "contract", "observation"])
def test_stale_authority_or_observation_after_model_await_rejects_without_cache(tmp_path, change):
    async def run():
        store, context, _adapter, model, judge = setup(tmp_path)
        model.block = True
        task = asyncio.create_task(judge.check("p1"))
        await asyncio.wait_for(model.entered.wait(), timeout=1)
        if change == "control":
            context.control_version += 1
        elif change == "action":
            context.action_seq += 1
        elif change == "pause":
            context.control_version += 1
            judge.coordinator.paused = True
        elif change == "contract":
            context.milestone = replace(context.milestone, revision=context.milestone.revision + 1)
        else:
            context.ingest(Observation("replacement", 3, 1, time.monotonic(), 1.0, 0,
                                       {"candidate_scope": SCOPE, "views": []}))
        model.release.set()
        with pytest.raises(GateError):
            await asyncio.wait_for(task, timeout=1)
        assert judge.cache == {}
        assert "destination_check" not in context.local_state
        assert not events(store, "destination_identity.checked")

    asyncio.run(run())


def test_point_replacement_while_inference_is_pending_cannot_publish(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path)
        model.block = True
        task = asyncio.create_task(judge.check("p1"))
        await asyncio.wait_for(model.entered.wait(), timeout=1)
        adapter.points["p1"] = point("p1", pixel=(80, 80), xyz=(0.3, 0.2, 0.03))
        adapter.marked_point = adapter.points["p1"]
        adapter.selected_point = adapter.marked_point
        model.release.set()
        with pytest.raises(GateError, match="point_changed"):
            await asyncio.wait_for(task, timeout=1)
        assert judge.cache == {}
        assert "destination_check" not in context.local_state
        assert not events(store, "destination_identity.checked")

    asyncio.run(run())


@pytest.mark.parametrize("scope_change", ["action", "control", "contract", "visual_scope"])
def test_context_projection_expires_destination_judgment_when_scope_changes(tmp_path, scope_change):
    async def run():
        store, context, adapter, _model, judge = setup(tmp_path)
        assert await judge.check("p1") is True
        assert "destination_check" in context.projection("fast")["local_state"]
        if scope_change == "action":
            context.action_seq += 1
        elif scope_change == "control":
            context.signal_control("pause", 90)
        elif scope_change == "contract":
            context.milestone = replace(context.milestone, revision=context.milestone.revision + 1)
        else:
            context.ingest(Observation("new-scope", 3, 1, time.monotonic(), 1.0, 0,
                                       {"candidate_scope": "packet-2/objects", "views": []}))
        assert "destination_check" not in context.projection("fast")["local_state"]
        assert len(events(store, "destination_identity.expired")) == 1

    asyncio.run(run())


def test_fast_waits_for_veto_and_mismatch_never_dispatches_place(tmp_path):
    async def run():
        store, context, adapter, model, judge = setup(tmp_path, verdicts=["mismatch"])
        model.block = True
        coordinator = Coordinator(adapter, context, store)

        class Actor:
            history = deque([{"role": "assistant", "content": "actor says the mark is the bowl"}], maxlen=6)

            async def decide(self, *_args):
                return {"kind": "tool", "tool": "place_point", "args": {"point_id": "p1"}}

        fast = FastAgent(Actor(), context, coordinator, store)
        fast.destination_checker = judge
        run_task = asyncio.create_task(fast.run(1))
        await asyncio.wait_for(model.entered.wait(), timeout=1)
        assert not any(row["kind"] == "action.accepted" for row in store.events)
        assert not adapter.executed
        model.release.set()
        assert await asyncio.wait_for(run_task, timeout=1) == "decision_budget"
        assert not any(row["kind"] == "action.accepted" for row in store.events)
        assert not adapter.executed
        assert context.action_seq == 0
        assert context.local_state["destination_check"]["verdict"] == "mismatch"

    asyncio.run(run())


@pytest.mark.parametrize(("mode", "backend", "pipeline", "target"), [
    ("single_fast", "direct_libero", True, True),
    ("dual_async", "eta", True, True),
    ("dual_async", "direct_libero", False, True),
    ("dual_async", "direct_libero", True, False),
])
def test_destination_judge_flag_rejects_unsupported_app_combinations(tmp_path, mode, backend, pipeline, target):
    from embodied_harness import app

    args = SimpleNamespace(output=str(tmp_path / "invalid"), mode=mode, backend=backend,
        pipeline_supervision=pipeline, strong_target_judge=target, strong_destination_judge=True)
    with pytest.raises(ValueError, match="strong_destination_judge requires dual_async direct pipeline with Strong target judge"):
        asyncio.run(app.episode(args))


def test_runner_destination_role_options_and_flag_are_opt_in():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner
    from test_round22d_target_judge import _strong_target_runner_job
    from test_round22_runner import load_fixture_catalog

    _runner, config, _case, default_job = _strong_target_runner_job()
    assert not default_job["feature_flags"].get("strong_destination_judge", False)
    assert "--strong-destination-judge" not in runner.app_arguments(default_job, Path("/tmp/memory.json"))

    flags = copy.deepcopy(runner.R21_FLAGS)
    flags["dual_async"]["strong_target_judge"] = True
    flags["dual_async"]["strong_destination_judge"] = True
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert job["expected_active_models"]["destination_judge"] == runner.SLOW_MODEL
    assert "--strong-destination-judge" in runner.app_arguments(job, Path("/tmp/memory.json"))

    result = {"backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "max_decisions": job["budget"]["max_decisions"], "timeout": job["budget"]["timeout_s"],
        "review_every": 3, "identity_check": job["identity_check"],
        "fast_model": runner.FAST_MODEL, "fast_model_options": runner.FAST_OPTIONS,
        "slow_model": runner.SLOW_MODEL, "slow_model_options": runner.SLOW_OPTIONS,
        "source_sha256": "b" * 64, **job["feature_flags"],
        "models": [{"role": role, "model": model} for role, model in job["expected_active_models"].items()
                   if model is not None],
        "destination_judge_model": runner.SLOW_MODEL,
        "destination_judge_options": runner.SLOW_OPTIONS,
        "destination_judge_trigger": "every_place_with_exact_point_and_scope_cache",
        "destination_judge_memory": "original_mission_task_start_and_current_images_only"}
    issues = runner.check_result(result, job, {"package_sha256": "b" * 64})
    assert not any(issue.startswith("missing_destination_judge") for issue in issues)
    assert not any(issue.startswith("mismatch_destination_judge") for issue in issues)

    result["destination_judge_options"] = {"max_tokens": 512}
    assert "mismatch_destination_judge_options" in runner.check_result(
        result, job, {"package_sha256": "b" * 64})


def test_fast_default_has_no_destination_judge():
    from embodied_harness.agents.fast import FastAgent

    context = SimpleNamespace()
    agent = FastAgent(None, context, SimpleNamespace(adapter=None), Store())
    assert agent.destination_checker is None
