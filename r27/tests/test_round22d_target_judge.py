"""Offline tests for the strict, memory-independent mission target judge."""
import asyncio
import copy
import hashlib
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness.agents.grounding import MissionTargetJudge
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import GateError


MISSION = "pick up the blue jar and put it inside the bowl"


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


def _write_image(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


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


class Adapter:
    generation = 1
    terminal = False

    def __init__(self, points, scope="packet-1/objects", full_image=None, crop_image=None):
        self.points = points
        self.scope = scope
        self.full_image, self.crop_image = full_image, crop_image
        self.version = 1
        self.calls = []
        self.on_inspect = None
        self.scene_packet_override = None

    async def inspect(self, tool, args, action_seq):
        assert tool == "observe"
        self.calls.append((tool, copy.deepcopy(args), action_seq))
        point = next(iter(self.points.values()))
        packet = point["source_packet_id"]
        if "crop_xyxy" not in args:
            packet = self.scene_packet_override or packet
            views = [{"source_packet_id": packet, "camera_frame_id": args["camera"],
                      "is_current_observation": True}]
            images = (self.full_image,)
        else:
            crop = list(args["crop_xyxy"])
            point = next(p for p in self.points.values()
                         if p["camera_frame_id"] == args["camera"])
            packet = point["source_packet_id"]
            views = [
                {"source_packet_id": packet, "camera_frame_id": args["camera"],
                 "is_current_observation": True},
                {"source_packet_id": packet, "camera_frame_id": "zoom_preview",
                 "source_camera": args["camera"], "original_crop_xyxy": crop},
            ]
            images = (self.full_image, self.crop_image)
        self.version += 1
        observation = Observation(
            f"judge-obs-{self.version}", self.version, self.generation, time.monotonic(),
            0.0, action_seq, {"candidate_scope": self.scope, "views": views},
            images,
        )
        if self.on_inspect:
            await self.on_inspect(args)
        return observation


class Model:
    def __init__(self, verdicts=None, *, reference_sha="start-sha"):
        self.verdicts = deque(verdicts or ["match"])
        self.history = deque([{"role": "assistant", "content": "the actor says this is the jar"}], maxlen=6)
        self.mission_reference_metadata = {"image_sha256": reference_sha, "semantics": "task_start_reference"}
        self.mission_reference_messages = [{"role": "user", "content": "bound task-start image"}]
        # No prior-memory manifest is required by the semantic judge.
        self.reference_metadata = []
        self.reference_messages = []
        self.calls = []
        self.entered = None
        self.release = None

    async def decide(self, system, query, images):
        self.calls.append({"system": system, "query": copy.deepcopy(query), "images": tuple(images)})
        if self.entered is not None:
            self.entered.set()
            await self.release.wait()
        verdict = self.verdicts.popleft()
        if isinstance(verdict, dict):
            return verdict
        return {
            "verdict": verdict,
            "selected_appearance": "blue glass jar with a narrow neck",
            "reason": "The visible color and silhouette match the mission reference.",
        }


def setup(tmp_path, *, verdicts=None, scope="packet-1/objects", camera="agentview"):
    full_image = _write_image(tmp_path / "full.png", b"current-full-scene")
    crop_image = _write_image(tmp_path / "crop.png", b"selected-object-crop")
    start_image = _write_image(tmp_path / "task-start.png", b"task-start-reference")
    reference_sha = hashlib.sha256(Path(start_image).read_bytes()).hexdigest()
    store = Store()
    context = Context(store)
    # The planner goal intentionally differs: target identity comes from the original mission.
    context.milestone = Milestone(
        "stage-2", 4, MISSION, "intermediate goal: lift the object clear of the table",
        capabilities=("grasp_point",),
    )
    context.ingest(Observation(
        "initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": scope, "views": []}, (full_image,),
    ))
    points = {"p1": target_point(camera=camera)}
    adapter = Adapter(points, scope=scope, full_image=full_image, crop_image=crop_image)
    coordinator = SimpleNamespace(
        adapter=adapter, lock=asyncio.Lock(), paused=False, unknown=False, running=None,
        contract_review_pending=False, pause_epoch=0, contract_review_epoch=0,
    )
    model = Model(verdicts, reference_sha=reference_sha)
    task_start_messages = copy.deepcopy(model.mission_reference_messages)
    judge = MissionTargetJudge(model, context, coordinator, store)
    return store, context, adapter, model, judge, task_start_messages


def test_judge_uses_original_mission_two_current_images_and_no_actor_or_memory_history(tmp_path):
    async def run():
        store, context, adapter, model, judge, task_start_messages = setup(tmp_path)
        assert judge.pregrasp_only is True
        assert await judge.check("p1") is True

        assert model.history.maxlen == 0 and not model.history
        assert model.reference_metadata == [] and model.reference_messages == []
        assert model.mission_reference_messages == task_start_messages
        call = model.calls[0]
        assert call["query"]["mission"] == MISSION
        assert call["query"]["mission"] != context.milestone.goal
        assert call["images"] == (adapter.full_image, adapter.crop_image)
        assert call["query"]["evidence"]["task_start_image_sha256"] == model.mission_reference_metadata["image_sha256"]
        assert call["query"]["evidence"]["current_image_sha256"] == [
            hashlib.sha256(Path(adapter.full_image).read_bytes()).hexdigest(),
            hashlib.sha256(Path(adapter.crop_image).read_bytes()).hexdigest(),
        ]
        assert len(adapter.calls) == 1
        assert context.local_state["target_identity_check"]["verdict"] == "match"
        assert context.local_state["identity_candidates"]["checks"]["candidate-1"]["verdict"] == "match"
        assert sum(e["kind"] == "target_identity.checked" for e in store.events) == 1

    asyncio.run(run())


def test_wrist_camera_judgment_first_gets_current_agentview_then_exact_wrist_crop(tmp_path):
    async def run():
        _store, _context, adapter, model, judge, _messages = setup(
            tmp_path, camera="robot0_wrist")
        assert await judge.check("p1") is True

        assert len(adapter.calls) == 2
        assert adapter.calls[0][1] == {"camera": "agentview"}
        assert adapter.calls[1][1]["camera"] == "robot0_wrist"
        assert "crop_xyxy" in adapter.calls[1][1]
        assert model.calls[0]["images"] == (adapter.full_image, adapter.crop_image)
        evidence = model.calls[0]["query"]["evidence"]
        assert evidence["scene_observation_id"] == "judge-obs-2"
        assert evidence["observation_id"] == "judge-obs-3"
        assert model.calls[0]["query"]["views"][0]["camera_frame_id"] == "agentview"
        assert model.calls[0]["query"]["views"][0]["is_current_observation"] is True
        assert model.calls[0]["query"]["views"][1]["camera_frame_id"] == "zoom_preview"
        assert model.calls[0]["query"]["views"][1]["source_camera"] == "robot0_wrist"

    asyncio.run(run())


def test_wrist_camera_cut_in_between_scene_and_crop_stops_before_second_inspection(tmp_path):
    async def run():
        _store, context, adapter, model, judge, _messages = setup(
            tmp_path, camera="robot0_wrist")

        async def cut_in_after_scene(args):
            if "crop_xyxy" not in args:
                context.control_version += 1

        adapter.on_inspect = cut_in_after_scene
        with pytest.raises(GateError, match="evidence_or_authority_changed"):
            await judge.check("p1")

        assert len(adapter.calls) == 1
        assert adapter.calls[0][1] == {"camera": "agentview"}
        assert model.calls == []
        assert not judge.cache
        assert "target_identity_check" not in context.local_state

    asyncio.run(run())


def test_wrist_packet_mismatch_veto_clears_previous_candidate_verdict(tmp_path):
    async def run():
        store, context, adapter, model, judge, _messages = setup(
            tmp_path, camera="robot0_wrist", verdicts=["match", "match"])
        assert await judge.check("p1") is True
        assert context.local_state["target_identity_check"]["verdict"] == "match"

        # A new candidate forces a fresh check, and the scene packet no longer
        # corresponds to the measured wrist point. That prior match must vanish.
        adapter.points["p2"] = target_point(
            "p2", "candidate-2", camera="robot0_wrist")
        adapter.scene_packet_override = "wrong-current-packet"
        with pytest.raises(GateError, match="exact_current_scene_and_crop"):
            await judge.check("p2")

        assert len(adapter.calls) == 4
        assert len(model.calls) == 1
        assert "target_identity_check" not in context.local_state
        assert sum(e["kind"] == "target_identity.checked" for e in store.events) == 1
        assert not any(e["kind"] == "target_identity.cached" for e in store.events)

    asyncio.run(run())


@pytest.mark.parametrize(("verdict", "expected"), [("match", True), ("mismatch", False), ("uncertain", False)])
def test_judge_returns_only_match_as_clearance_and_does_not_cache_uncertain(tmp_path, verdict, expected):
    async def run():
        store, _context, adapter, model, judge, _messages = setup(tmp_path, verdicts=[verdict, verdict])
        assert await judge.check("p1") is expected
        if verdict == "uncertain":
            assert await judge.check("p1") is False
            assert len(model.calls) == len(adapter.calls) == 2
            assert len(judge.cache) == 0
            assert not any(e["kind"] == "target_identity.cached" for e in store.events)

    asyncio.run(run())


@pytest.mark.parametrize("invalid", [
    {"verdict": "maybe", "selected_appearance": "a jar", "reason": "unclear"},
    {"verdict": "match", "selected_appearance": "a jar"},
    {"verdict": "match", "selected_appearance": 12, "reason": "bad type"},
    {"verdict": "match", "selected_appearance": "", "reason": "empty evidence"},
    {"verdict": "match", "selected_appearance": "a jar", "reason": "ok", "extra": "claims"},
])
def test_judge_rejects_invalid_or_non_strict_verdict_payloads(tmp_path, invalid):
    async def run():
        _store, context, _adapter, _model, judge, _messages = setup(tmp_path, verdicts=[invalid])
        with pytest.raises(ValueError, match="target judge"):
            await judge.check("p1")
        assert "target_identity_check" not in context.local_state
        assert not judge.cache

    asyncio.run(run())


def test_match_cache_survives_same_readonly_scope_but_expires_when_control_changes(tmp_path):
    async def run():
        store, context, adapter, model, judge, _messages = setup(tmp_path, verdicts=["match", "match"])
        assert await judge.check("p1") is True
        # A new point binding for the same candidate and exact crop can reuse the
        # verdict despite the observation ID changing during the read-only crop.
        adapter.points["p2"] = target_point("p2", packet="packet-1")
        assert await judge.check("p2") is True
        assert len(adapter.calls) == 1
        assert len(model.calls) == 1
        assert sum(e["kind"] == "target_identity.cached" for e in store.events) == 1

        context.control_version += 1
        assert await judge.check("p2") is True
        assert len(adapter.calls) == 2
        assert len(model.calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize("change", ["control", "physical_action"])
def test_target_judgment_is_removed_from_future_projection_when_scope_expires(tmp_path, change):
    async def run():
        _store, context, _adapter, _model, judge, _messages = setup(tmp_path)
        assert await judge.check("p1") is True
        assert "target_identity_check" in context.projection("fast")["local_state"]

        if change == "control":
            context.signal_control("pause", priority=90)
        else:
            context.remember("action.accepted", {
                "request": {"capability": "move_to", "args": {"xyz_m": [0.2, 0.1, 0.8]}},
                "receipt": {},
            })

        projected = context.projection("fast")["local_state"]
        assert "target_identity_check" not in projected
        assert "identity_candidates" not in projected

        # Control changes must leave the legacy memory-identity ledger alone.
        context.local_state["identity_candidates"] = {
            "visual_scope": "packet-1/objects", "checks": {"candidate-old": {"verdict": "mismatch"}},
            "provenance": "independent_model_judgments_not_ground_truth",
        }
        if change == "control":
            context.signal_control("contract_review", priority=80)
        assert context.local_state["identity_candidates"]["checks"]["candidate-old"]["verdict"] == "mismatch"

    asyncio.run(run())


@pytest.mark.parametrize("stale_point", ["crop", "model"])
def test_authority_change_during_crop_or_model_await_rejects_without_publishing(tmp_path, stale_point):
    async def run():
        store, context, adapter, model, judge, _messages = setup(tmp_path)
        if stale_point == "crop":
            async def mutate_during_crop(_args):
                context.action_seq += 1
            adapter.on_inspect = mutate_during_crop
            with pytest.raises(GateError):
                await judge.check("p1")
            assert model.calls == []
        else:
            model.entered, model.release = asyncio.Event(), asyncio.Event()
            pending = asyncio.create_task(judge.check("p1"))
            await asyncio.wait_for(model.entered.wait(), timeout=1)
            context.control_version += 1
            model.release.set()
            with pytest.raises(GateError):
                await pending
            assert len(model.calls) == 1

        assert not judge.cache
        assert "target_identity_check" not in context.local_state
        assert not any(e["kind"] == "target_identity.checked" for e in store.events)

    asyncio.run(run())


def test_target_judge_requires_task_start_reference_and_exact_full_scene_crop(tmp_path):
    async def run():
        _store, _context, adapter, model, judge, _messages = setup(tmp_path)
        del model.mission_reference_metadata
        with pytest.raises(GateError, match="task_start_reference"):
            await judge.check("p1")

        _store2, _context2, adapter2, model2, judge2, _messages2 = setup(tmp_path / "second")
        # The task-start image binding is required, but a reusable prior-memory manifest is not.
        assert model2.reference_metadata == []
        original = adapter2.inspect

        async def incomplete_views(tool, args, action_seq):
            obs = await original(tool, args, action_seq)
            obs.state["views"] = [v for v in obs.state["views"] if v["camera_frame_id"] != "zoom_preview"]
            return obs

        adapter2.inspect = incomplete_views
        with pytest.raises(GateError, match="exact_current_scene_and_crop"):
            await judge2.check("p1")
        assert model2.calls == []

    asyncio.run(run())


@pytest.mark.parametrize(("mode", "backend", "identity_check"), [
    ("single_fast", "direct_libero", True),
    ("dual_async", "eta", True),
    ("dual_async", "direct_libero", False),
])
def test_strong_target_judge_flag_rejects_unsupported_runtime_gates(
        monkeypatch, tmp_path, mode, backend, identity_check):
    from embodied_harness import app

    args = SimpleNamespace(
        output=str(tmp_path / "invalid"), mode=mode, backend=backend,
        identity_check=identity_check, strong_target_judge=True,
    )
    with pytest.raises(ValueError, match="strong_target_judge requires dual_async direct_libero with identity_check"):
        asyncio.run(app.episode(args))


def _strong_target_runner_job():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner
    from test_round22_runner import fixture_config, load_fixture_catalog

    config = fixture_config()
    feature_flags = copy.deepcopy(runner.R21_FLAGS)
    feature_flags["dual_async"]["strong_target_judge"] = True
    config["variants"]["patched22"]["feature_flags_by_mode"] = feature_flags
    standard_case = next(case for case in config["cases"]
                         if case["variant"] == "patched22" and case["mode"] == "dual_async"
                         and case["suite"] != "libero_10")
    config["cases"] = [copy.deepcopy(standard_case)]
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    return runner, config, standard_case, job


def test_runner_routes_optional_flag_to_slow_target_judge_without_grounding_requirement():
    runner, config, _case, job = _strong_target_runner_job()
    assert job["identity_check"] is True
    assert job["expected_active_models"] == {
        "fast": runner.FAST_MODEL, "slow": runner.SLOW_MODEL,
        "grounding": None, "target_judge": runner.SLOW_MODEL,
    }
    args = runner.app_arguments(job, Path("/tmp/no-memory-manifest.json"))
    assert "--strong-target-judge" in args

    # An independent target judge is the grounding route in this configuration;
    # the prior-memory grounding model stays absent even when memory is recorded.
    result = {
        "backend": "direct_libero", "env_id": job["env_id"], "seed": job["seed"],
        "mode": job["mode"], "status": "decision_budget", "success": False,
        "evidence": {"status": "not_verified"},
        "max_decisions": job["budget"]["max_decisions"], "timeout": job["budget"]["timeout_s"],
        "review_every": 3, "identity_check": True,
        "fast_model": runner.FAST_MODEL, "fast_model_options": runner.FAST_OPTIONS,
        "slow_model": runner.SLOW_MODEL, "slow_model_options": runner.SLOW_OPTIONS,
        "models": [
            {"role": "fast", "model": runner.FAST_MODEL},
            {"role": "slow", "model": runner.SLOW_MODEL},
            {"role": "target_judge", "model": runner.SLOW_MODEL},
        ],
        "source_sha256": "b" * 64,
        "appearance_memory": [{"image_sha256": "prior-memory-is-not-the-target-route"}],
        "target_judge_model": runner.SLOW_MODEL,
        "target_judge_options": runner.SLOW_OPTIONS,
        "target_judge_trigger": "every_pregrasp_with_same_scope_cache",
        **job["feature_flags"],
    }
    source = {"package_sha256": "b" * 64}
    assert runner.check_result(result, job, source) == []

    misrouted = copy.deepcopy(result)
    misrouted["models"].append({"role": "grounding", "model": runner.FAST_MODEL})
    assert "unexpected_grounding_route" in runner.check_result(misrouted, job, source)
    missing_judge = copy.deepcopy(result)
    missing_judge["models"] = [m for m in result["models"] if m["role"] != "target_judge"]
    assert "missing_target_judge_route" in runner.check_result(missing_judge, job, source)
    wrong_options = copy.deepcopy(result)
    wrong_options["target_judge_options"] = {"max_tokens": 1024}
    assert "mismatch_target_judge_model_options" in runner.check_result(wrong_options, job, source)

    # The frozen long stratum disables identity checks, so this route cannot be
    # smuggled into a job without its required pre-grasp gate.
    long_case = next(case for case in config["cases"] if case["suite"] != "libero_10")
    long_case = {**long_case, "suite": "libero_10", "task_index": 5}
    config["cases"] = [long_case]
    with pytest.raises(ValueError, match="strong_target_judge requires identity_check"):
        runner.make_jobs(config, runner.load_catalog(config, Path(config["catalog"]["path"]))[0],
                         Path(config["output_root"]))


def test_mock_episode_constructs_slow_judge_role_without_memory_or_legacy_grounding(
        monkeypatch, tmp_path):
    from embodied_harness import app
    from embodied_harness.runtime import memory
    from test_round20c_model_role_routing import _episode_args, _install_offline_app

    endpoints = {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"}
    constructed = _install_offline_app(monkeypatch, endpoints)
    monkeypatch.setattr(memory, "retrieve", lambda *_args: [])
    fast_agents = []

    class CapturingFast:
        def __init__(self, *_args, **_kwargs):
            self.steps = self.false_claims = self.errors = 0
            self.identity_checker = None
            fast_agents.append(self)

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "FastAgent", CapturingFast)
    args = _episode_args(tmp_path, "dual_async")
    args.strong_target_judge = True
    result = asyncio.run(app.episode(args))

    assert [model.role for model in constructed] == ["fast", "slow", "target_judge"]
    judge = constructed[-1]
    assert (judge.base_url, judge.key, judge.model, judge.options) == (
        endpoints["slow"], "synthetic-slow-key", "slow-test", {"temperature": 0.1},
    )
    assert judge.mission_references and judge.mission_references[0].id == "mission-start"
    assert isinstance(fast_agents[0].identity_checker, MissionTargetJudge)
    assert result["error"] is None
    settings = __import__("json").loads((tmp_path / "dual_async" / "config.json").read_text())
    assert settings["appearance_memory"] == []
    assert settings["target_judge_model"] == "slow-test"
    assert settings["target_judge_options"] == {"temperature": 0.1}
    assert settings["target_judge_trigger"] == "every_pregrasp_with_same_scope_cache"
    assert settings["target_judge_memory"] == "none_original_mission_and_public_images_only"

