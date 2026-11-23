"""Offline coverage for opt-in verified appearance references in Round 22J."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

from embodied_harness import app  # noqa: E402
from embodied_harness.runtime.memory import retrieve  # noqa: E402
import run_round22_targeted as runner  # noqa: E402


MANIFEST = REPO / "memory" / "verified-experience.json"
REFERENCE_CLASS = "verified_single_acquisition_v1"
REFERENCE_TRANSFORM = {
    "method": "strip_debug_margins_v1",
    "source_size": [512, 512],
    "crop_xyxy": [48, 16, 512, 474],
}


def _experience_entries():
    payload = json.loads(MANIFEST.read_text())
    return [entry for entry in payload["entries"] if entry.get("memory_class") == REFERENCE_CLASS]


def _write_entry_manifest(tmp_path: Path, entry: dict, *, image_bytes: bytes | None = None) -> Path:
    entry = copy.deepcopy(entry)
    source_image = REPO / "memory" / entry["image"]
    if image_bytes is None:
        image_bytes = source_image.read_bytes()
    tmp_path.mkdir(parents=True, exist_ok=True)
    entry["image"] = "reference.png"
    (tmp_path / "reference.png").write_bytes(image_bytes)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [entry]}))
    return manifest


def test_manifest_contains_only_the_three_verified_new_experiences_and_exact_reference_bytes():
    entries = _experience_entries()
    assert [entry["match_text"] for entry in entries] == ["bbq sauce", "ketchup", "tomato sauce"]
    assert not any("chocolate" in entry["match_text"].casefold() for entry in entries)

    for entry in entries:
        provenance = entry["provenance"]
        assert provenance["official_success"] is True
        assert provenance["accepted_grasp_count"] == 1
        assert provenance["pregrasp_action_seq"] == 0
        assert provenance["receipt_status"] == "completed"
        assert provenance["receipt_macro_status"] == "lift_completed_verify_retention_visually"
        assert 0 < provenance["accepted_grasp_seq"] < provenance["receipt_seq"] < provenance["official_verification_seq"]
        assert provenance["official_verification"]["status"] == "verified"
        assert provenance["official_verification"]["source"] == "official_libero_check_success"
        assert provenance["reference_transform"] == REFERENCE_TRANSFORM
        for key in ("source_result_sha256", "source_events_sha256", "source_fast_trace_sha256",
                    "source_image_sha256", "reference_image_sha256"):
            assert re.fullmatch(r"[0-9a-f]{64}", provenance[key])
        assert entry["image_sha256"] == provenance["reference_image_sha256"]
        actual = hashlib.sha256((REPO / "memory" / entry["image"]).read_bytes()).hexdigest()
        assert actual == entry["image_sha256"]
        image_bytes = (REPO / "memory" / entry["image"]).read_bytes()
        assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", image_bytes[16:24]) == (464, 458)


def test_retrieval_defaults_off_preserves_legacy_matching_and_requires_explicit_phrase_match():
    assert retrieve(None, "pick up the bbq sauce") == []
    # New experience rows are invisible by default, but legacy rows retain their
    # original case-insensitive substring behavior.
    assert retrieve(MANIFEST, "pick up BBQ sauce") == []
    legacy = retrieve(MANIFEST, "please pick up ALPHABET SOUP")
    assert [entry["match_text"] for entry in legacy] == ["alphabet soup"]

    assert [entry["match_text"] for entry in retrieve(
        MANIFEST, "Please pick up the BBQ SAUCE", appearance_reference_memory=True)] == ["bbq sauce"]
    assert retrieve(MANIFEST, "pick up superbbq sauce", appearance_reference_memory=True) == []
    assert retrieve(MANIFEST, "pick up bbq sauces", appearance_reference_memory=True) == []
    assert retrieve(MANIFEST, "pick up the chocolate", appearance_reference_memory=True) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda e: e["provenance"].update(official_success=False), "verified source episode"),
        (lambda e: e["provenance"].update(accepted_grasp_count=2), "one completed acquisition"),
        (lambda e: e["provenance"].update(pregrasp_action_seq=1), "one completed acquisition"),
        (lambda e: e["provenance"].update(receipt_status="accepted"), "one completed acquisition"),
        (lambda e: e["provenance"].update(receipt_seq=e["provenance"]["accepted_grasp_seq"]), "evidence order"),
        (lambda e: e["provenance"].update(source_events_sha256="not-a-sha256"), "immutable source hashes"),
        (lambda e: e["provenance"].update(reference_image_sha256="0" * 64), "differs from its recorded source"),
        (lambda e: e["provenance"].update(reference_transform={"method": "wrong"}), "remove historical"),
    ],
)
def test_enabled_retrieval_rejects_invalid_acquisition_provenance_and_hashes(tmp_path, mutation, message):
    entry = copy.deepcopy(_experience_entries()[0])
    mutation(entry)
    path = _write_entry_manifest(tmp_path, entry)

    with pytest.raises(ValueError, match=message):
        retrieve(path, "pick up the bbq sauce", appearance_reference_memory=True)


def test_retrieval_checks_actual_reference_bytes_and_rejects_path_traversal(tmp_path):
    entry = copy.deepcopy(_experience_entries()[0])
    manifest = _write_entry_manifest(tmp_path / "bytes", entry, image_bytes=b"different reference bytes")
    with pytest.raises(ValueError, match="differs from its recorded source"):
        retrieve(manifest, "bbq sauce", appearance_reference_memory=True)

    outside = tmp_path / "outside.png"
    outside.write_bytes((REPO / "memory" / entry["image"]).read_bytes())
    entry["image"] = "../outside.png"
    traversal_manifest = tmp_path / "traversal" / "manifest.json"
    traversal_manifest.parent.mkdir()
    traversal_manifest.write_text(json.dumps({"entries": [entry]}))
    with pytest.raises(ValueError, match="within manifest directory"):
        retrieve(traversal_manifest, "bbq sauce", appearance_reference_memory=True)


def test_missing_matching_image_fails_only_when_new_memory_is_enabled(tmp_path):
    entry = copy.deepcopy(_experience_entries()[0])
    entry["image"] = "missing.png"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": [entry]}))

    assert retrieve(path, "bbq sauce") == []
    with pytest.raises(FileNotFoundError):
        retrieve(path, "bbq sauce", appearance_reference_memory=True)


def test_reference_cap_is_two_and_legacy_rows_coexist_with_new_experiences(tmp_path):
    base = copy.deepcopy(_experience_entries()[0])
    reference_bytes = (REPO / "memory" / base["image"]).read_bytes()
    (tmp_path / "reference.png").write_bytes(reference_bytes)
    entries = []
    legacy = {
        "match_text": "sauce", "image": "reference.png", "caption": "legacy ref",
        "provenance": {"official_success": True},
    }
    entries.append(legacy)
    for phrase in ("sauce", "sauce", "sauce"):
        item = copy.deepcopy(base)
        item["match_text"] = phrase
        item["image"] = "reference.png"
        entries.append(item)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": entries}))

    enabled = retrieve(manifest, "pick up the sauce", appearance_reference_memory=True)
    assert len(enabled) == 2
    assert enabled[0]["caption"] == "legacy ref"
    assert enabled[1]["memory_class"] == REFERENCE_CLASS
    disabled = retrieve(manifest, "pick up the sauce")
    assert len(disabled) == 1 and "memory_class" not in disabled[0]


def test_appearance_memory_flag_is_optional_and_runner_serializes_only_when_enabled():
    assert "appearance_reference_memory" in runner.OPTIONAL_FEATURE_FLAGS
    assert "appearance_reference_memory" not in runner.R21_FLAGS["dual_async"]

    task = {
        "suite": "libero_object", "task_index": 3,
        "env_id": "openeta/libero_libero_object_task3-v0", "mission": "pick up the bbq sauce",
        "official_state_count": 1, "state_hashes": {"0": "a" * 64},
    }
    config = {
        "variants": {"patched22": {
            "kind": "patched22",
            "feature_flags_by_mode": {"dual_async": {**runner.R21_FLAGS["dual_async"],
                                                         "appearance_reference_memory": True}},
        }},
        "cases": [{"suite": "libero_object", "task_index": 3, "seed": 0,
                   "variant": "patched22", "mode": "dual_async"}],
    }
    job = runner.make_jobs(config, {"_by_key": {("libero_object", 3): task}}, Path("/tmp/memory-test"))[0]
    assert job["feature_flags"]["appearance_reference_memory"] is True
    enabled_args = runner.app_arguments(job, Path("/tmp/manifest.json"))
    assert "--appearance-reference-memory" in enabled_args

    job["feature_flags"]["appearance_reference_memory"] = False
    disabled_args = runner.app_arguments(job, Path("/tmp/manifest.json"))
    assert "--appearance-reference-memory" not in disabled_args
    assert [arg for arg in enabled_args if arg not in disabled_args] == ["--appearance-reference-memory"]


def test_runner_requires_strong_judge_memory_policy_only_for_new_feature_flag():
    flags = {"strong_target_judge": True, "appearance_reference_memory": False}
    job = {
        "feature_flags": flags, "env_id": "openeta/libero_libero_object_task3-v0", "seed": 0,
        "mode": "dual_async", "budget": {"max_decisions": 24, "timeout_s": 1200},
        "identity_check": True,
        "expected_recorded_model_options": runner.expected_recorded_options(),
        "expected_active_models": {"fast": runner.FAST_MODEL, "slow": runner.SLOW_MODEL,
                                    "grounding": None, "target_judge": runner.SLOW_MODEL},
    }
    source = {"package_sha256": "c" * 64}
    result = {
        "backend": "direct_libero", "env_id": job["env_id"], "seed": 0, "mode": "dual_async",
        "status": "decision_budget", "success": False, "max_decisions": 24, "timeout": 1200,
        "review_every": 3, "identity_check": True, "source_sha256": source["package_sha256"],
        "fast_model": runner.FAST_MODEL, "fast_model_options": runner.FAST_OPTIONS,
        "slow_model": runner.SLOW_MODEL, "slow_model_options": runner.SLOW_OPTIONS,
        "models": [{"role": role, "model": model} for role, model in
                   (("fast", runner.FAST_MODEL), ("slow", runner.SLOW_MODEL),
                    ("target_judge", runner.SLOW_MODEL))],
        "target_judge_model": runner.SLOW_MODEL, "target_judge_options": runner.SLOW_OPTIONS,
        "target_judge_trigger": "every_pregrasp_with_same_scope_cache",
        **flags,
    }
    assert runner.check_result(result, job, source) == []

    flags["appearance_reference_memory"] = True
    result["appearance_reference_memory"] = True
    assert "mismatch_target_judge_memory_policy" in runner.check_result(result, job, source)
    result["target_judge_memory"] = "verified_prior_appearance_if_matched_plus_original_mission_and_public_images"
    assert runner.check_result(result, job, source) == []


def test_app_cli_defaults_reference_memory_off_and_accepts_explicit_flag(monkeypatch):
    parsed = []

    def capture(coroutine):
        parsed.append(coroutine.cr_frame.f_locals["args"])
        coroutine.close()

    monkeypatch.setattr(app.asyncio, "run", capture)
    monkeypatch.setattr(sys, "argv", ["embodied_harness.app", "--mission", "pick up sauce", "--output", "/tmp/out"])
    app.main()
    assert parsed[-1].appearance_reference_memory is False

    monkeypatch.setattr(sys, "argv", ["embodied_harness.app", "--mission", "pick up sauce", "--output", "/tmp/out",
                                       "--appearance-reference-memory"])
    app.main()
    assert parsed[-1].appearance_reference_memory is True


def test_enabled_memory_is_bound_to_fast_slow_and_strong_judge_and_emits_hash_audit(
        monkeypatch, tmp_path):
    from test_round20c_model_role_routing import _episode_args, _install_offline_app
    from embodied_harness.runtime import memory

    constructed = _install_offline_app(monkeypatch, {
        "fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/",
    })
    bound = {}
    model_class = app.ModelClient

    def capture_memory(self, entries):
        bound[self.role] = list(entries)

    monkeypatch.setattr(model_class, "bind_memory", capture_memory)
    reference = {
        "memory_class": REFERENCE_CLASS, "match_text": "bbq sauce", "image": "/tmp/reference.png",
        "image_sha256": "a" * 64, "manifest_sha256": "b" * 64,
        "caption": "Prior successful appearance only.",
    }
    monkeypatch.setattr(memory, "retrieve", lambda *_args, **_kwargs: [reference])

    events = []

    class Store:
        def __init__(self, _path):
            pass

        def emit(self, kind, payload):
            events.append((kind, payload))

        def since(self, _cursor):
            return []

        def close(self):
            pass

    monkeypatch.setattr(app, "EventStore", Store)
    args = _episode_args(tmp_path, "dual_async")
    args.strong_target_judge = True
    args.appearance_reference_memory = True
    args.memory = "/tmp/synthetic-manifest.json"
    result = asyncio.run(app.episode(args))

    assert result["error"] is None
    assert [model.role for model in constructed] == ["fast", "slow", "target_judge"]
    assert {"fast", "slow", "target_judge"}.issubset(bound)
    assert all(entries == [reference] for role, entries in bound.items()
               if role in {"fast", "slow", "target_judge"})
    assert result["target_judge_memory"] == "verified_prior_appearance_if_matched_plus_original_mission_and_public_images"
    recalled = [payload for kind, payload in events if kind == "memory.appearance_recalled"]
    assert recalled == [{
        "reference_count": 1, "new_experience_count": 1,
        "references": [{"match_text": "bbq sauce", "image_sha256": "a" * 64,
                        "manifest_sha256": "b" * 64}],
        "semantics": "prior appearance only; no current identity, pose, retention, or success certification",
    }]


def test_absent_app_flag_keeps_target_judge_unbound_and_preserves_legacy_memory_policy(
        monkeypatch, tmp_path):
    from test_round20c_model_role_routing import _episode_args, _install_offline_app
    from embodied_harness.runtime import memory

    constructed = _install_offline_app(monkeypatch, {
        "fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/",
    })
    bound_roles = []
    model_class = app.ModelClient
    monkeypatch.setattr(model_class, "bind_memory", lambda self, entries: bound_roles.append((self.role, list(entries))))
    real_retrieve = retrieve
    retrieval_calls = []

    def tracked_retrieve(path, mission, **kwargs):
        retrieval_calls.append((path, mission, dict(kwargs)))
        return real_retrieve(path, mission, **kwargs)

    monkeypatch.setattr(memory, "retrieve", tracked_retrieve)
    events = []

    class Store:
        def __init__(self, _path):
            pass

        def emit(self, kind, payload):
            events.append((kind, payload))

        def since(self, _cursor):
            return []

        def close(self):
            pass

    monkeypatch.setattr(app, "EventStore", Store)
    args = _episode_args(tmp_path, "dual_async")
    args.strong_target_judge = True
    args.memory = str(MANIFEST)
    args.mission = "pick up the alphabet soup"
    result = asyncio.run(app.episode(args))

    assert result["error"] is None
    assert retrieval_calls == [(str(MANIFEST), "pick up the alphabet soup", {})]
    by_role = {role: entries for role, entries in bound_roles}
    assert [entry["match_text"] for entry in by_role["fast"]] == ["alphabet soup"]
    assert [entry["match_text"] for entry in by_role["slow"]] == ["alphabet soup"]
    assert all("memory_class" not in entry for role in ("fast", "slow") for entry in by_role[role])
    assert not any(role == "target_judge" for role, _entries in bound_roles)
    assert result["target_judge_memory"] == "none_original_mission_and_public_images_only"
    assert not any(kind == "memory.appearance_recalled" for kind, _payload in events)
    assert any(model.role == "target_judge" for model in constructed)
