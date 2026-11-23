"""Semantic supervisor hypotheses remain separate from frame-bound advice."""
import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace

from PIL import Image

from embodied_harness.adapters.openeta import OpenETAAdapter
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})

    def since(self, cursor):
        return self.events[cursor:]


def image_bytes(color="#345678"):
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(stream, format="PNG")
    return stream.getvalue()


def host_result(scope, packet="direct-7"):
    body = {
        "robot": {"grip_xyz_m": [0.0, 0.0, 0.5]},
        "feedback": {"status": "observed tool result"},
        "views": [{"source_packet_id": packet, "camera_frame_id": "agentview"}],
        "visible_candidates": [{"candidate_id": "p7o-c2", "bbox_xyxy": [20, 30, 80, 90]}],
        "candidate_scope": scope,
    }
    return SimpleNamespace(content=[
        SimpleNamespace(type="text", text=json.dumps(body)),
        SimpleNamespace(type="image", data=base64.b64encode(image_bytes()).decode("ascii")),
    ])


def review_fixture(tmp_path, scope="direct-7/objects"):
    adapter = object.__new__(OpenETAAdapter)
    adapter.images = tmp_path / "frames"
    adapter.images.mkdir(parents=True, exist_ok=True)
    adapter.version = 0
    adapter.generation = 1
    adapter.action_seq = 0
    adapter.schemas = {"observe": {"parameters": {}}}
    adapter.points = {}
    adapter.cached = None
    adapter.last = None

    store = MemoryStore()
    context = Context(store)
    coordinator = Coordinator(adapter, context, store)
    milestone = Milestone("mission", 1, "place the blue ramekin in the bowl",
                          "place the blue ramekin in the bowl", capabilities=())
    coordinator.delegate(milestone)
    context.ingest(adapter._unpack(host_result(scope)))
    return adapter, context, coordinator, store, milestone


class WaitingReviewModel:
    def __init__(self, decision):
        self.decision = decision
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.projection = None

    async def decide(self, _system, projection, _images):
        self.projection = projection
        self.entered.set()
        await self.release.wait()
        return self.decision


async def run_review_while(model, slow, mutate):
    ticket = slow.request_review("review ticket")
    task = asyncio.create_task(slow.review())
    await asyncio.wait_for(model.entered.wait(), timeout=1)
    mutate()
    model.release.set()
    await task
    return ticket


def test_stale_id_advice_is_dropped_but_separate_hypothesis_survives_and_delivers_ticket(tmp_path):
    async def run():
        adapter, context, coordinator, store, milestone = review_fixture(tmp_path)
        stale_text = "Use region p7o-c2 to identify the ramekin beside the bowl."
        hypothesis = (
            "The small blue ramekin has a pale rim; compare its shape with the current images "
            "and recheck before choosing it."
        )
        model = WaitingReviewModel({
            "kind": "advice", "text": stale_text, "action_sensitive": False,
            "targets": [{"candidate_id": "p7o-c2"}],
            "semantic_advice": hypothesis,
        })
        slow = SlowAgent(model, context, coordinator, store)

        def change_scope_during_inference():
            context.ingest(adapter._unpack(host_result("direct-8/objects", packet="direct-8")))

        ticket = await run_review_while(model, slow, change_scope_during_inference)

        assert model.projection["observation"]["state"]["candidate_scope"] == "direct-7/objects"
        assert context.latest.state["candidate_scope"] == "direct-8/objects"
        assert context.advice and len(context.advice) == 1
        published = context.advice[0]
        assert published["expected_visual_scope"] is None
        assert published["expected_action_seq"] is None
        assert published["expected_action_phase"] is None
        assert "IDENTITY HYPOTHESIS" in published["text"]
        assert "recheck against CURRENT images, not a current-pose fact" in published["text"]
        assert "p7o-c2" not in published["text"]
        assert not any(char.isdigit() for char in published["text"])
        assert slow.delivered_ticket == ticket == slow.request_ticket
        assert context.review_state["last_outcome"] == "semantic_advice"
        assert not context.target_plan.get("targets")
        stale = [event for event in store.events if event["kind"] == "advice.stale"]
        assert len(stale) == 1
        assert any(event["kind"] == "slow.semantic_published" and event["payload"]["accepted"]
                   for event in store.events)
        assert not any(event["kind"] == "slow.model_error" for event in store.events)

    asyncio.run(run())


def test_digit_or_pixel_semantic_hypotheses_are_rejected_and_do_not_fulfill_stale_ticket(tmp_path):
    async def run():
        rejected_texts = [
            "The ramekin is candidate p7o-c2.",
            "The ramekin is near the pixel cluster on the left.",
            "The ramekin is at coordinates x=120, y=220.",
        ]
        for index, semantic_text in enumerate(rejected_texts):
            adapter, context, coordinator, store, _milestone = review_fixture(
                tmp_path / str(index), scope="direct-7/objects")
            model = WaitingReviewModel({
                "kind": "advice", "text": "Look at region p7o-c2.", "action_sensitive": False,
                "semantic_advice": semantic_text,
            })
            slow = SlowAgent(model, context, coordinator, store)

            def change_scope_during_inference():
                context.ingest(adapter._unpack(host_result("direct-8/objects", packet="direct-8")))

            ticket = await run_review_while(model, slow, change_scope_during_inference)

            assert not context.advice
            assert slow.delivered_ticket == 0
            assert context.review_state["last_outcome"] == "stale"
            assert any(event["kind"] == "slow.semantic_rejected"
                       and event["payload"]["reason"] == "contains_frame_specific_reference"
                       for event in store.events)
            assert not any(event["kind"] == "slow.semantic_published" for event in store.events)
            assert slow.request_ticket == ticket

    asyncio.run(run())


def test_semantic_hypothesis_keeps_milestone_revision_and_generation_fences(tmp_path):
    async def run():
        for index, drift in enumerate(("milestone_revision", "environment_generation")):
            adapter, context, coordinator, store, _milestone = review_fixture(
                tmp_path / drift, scope="direct-7/objects")
            model = WaitingReviewModel({
                "kind": "advice", "text": "Use a different approach to the bowl.",
                "action_sensitive": False,
                "semantic_advice": "The small blue ramekin has a pale rim; recheck its appearance in current images.",
            })
            slow = SlowAgent(model, context, coordinator, store)

            def invalidate_fence():
                if drift == "milestone_revision":
                    coordinator.delegate(Milestone(
                        "mission", 2, "place the blue ramekin in the bowl",
                        "place the blue ramekin in the bowl", capabilities=()), expected_revision=1)
                else:
                    adapter.generation = 2

            ticket = await run_review_while(model, slow, invalidate_fence)

            assert not context.advice
            assert slow.delivered_ticket == 0
            assert slow.request_ticket == ticket
            assert context.review_state["last_outcome"] == "stale"
            published = [event["payload"] for event in store.events
                         if event["kind"] == "slow.semantic_published"]
            assert len(published) == 1
            assert published[0]["accepted"] is False
            assert published[0]["provenance"] == "model_identity_hypothesis"
            assert any(event["kind"] == "advice.stale" for event in store.events)

    asyncio.run(run())


def test_review_without_semantic_field_keeps_existing_main_advice_path(tmp_path):
    async def run():
        _adapter, context, coordinator, store, _milestone = review_fixture(tmp_path)
        main_text = "Approach the bowl from the open side."
        model = WaitingReviewModel({"kind": "advice", "text": main_text, "action_sensitive": False})
        slow = SlowAgent(model, context, coordinator, store)

        ticket = await run_review_while(model, slow, lambda: None)

        assert len(context.advice) == 1
        assert context.advice[0]["text"] == main_text
        assert context.advice[0]["expected_visual_scope"] is None
        assert slow.delivered_ticket == ticket == slow.request_ticket
        assert context.review_state["last_outcome"] == "advice"
        assert not any(event["kind"].startswith("slow.semantic_") for event in store.events)

    asyncio.run(run())

