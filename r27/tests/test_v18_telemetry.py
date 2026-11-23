import asyncio
import json
import time
from types import SimpleNamespace

from embodied_harness.adapters.libero_direct import LiberoHost
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class Store:
    def emit(self, *args, **kwargs):
        pass

    def since(self, _cursor):
        return []


def test_motion_telemetry_does_not_build_visual_proposals_or_encode_images():
    host = object.__new__(LiberoHost)
    host.state = lambda: {"gripper": {"openness": 0.25}}
    host.contact = lambda: {"left_fingerpad_contact": True, "right_fingerpad_contact": False}
    host.sim = SimpleNamespace(data=SimpleNamespace(time=1.25))
    host.execution_accounting = {"control_steps": 12}
    host.feedback = {"motion": {"execution_state": "running"}}
    host.binding = False
    host.generation = 1

    def forbidden():
        raise AssertionError("visual proposals must not run on the motion telemetry path")

    host.candidates = forbidden
    result = host.result(telemetry_only=True)
    assert len(result.content) == 1
    body = json.loads(result.content[0].text)
    assert body["sensor"]["visual_sample"] is False
    assert body["sensor"]["control_steps"] == 12
    assert body["tactile"]["left_fingerpad_contact"]
    assert body["views"] == []


def test_slow_receives_live_telemetry_and_explicitly_older_visual_reference():
    async def run():
        store = Store()
        context = Context(store)
        coordinator = Coordinator(SimpleNamespace(generation=1), context, store)
        coordinator.delegate(Milestone("m", 1, "mission", "current outcome"))
        visual = Observation("image-frame", 1, 1, time.monotonic() - 2, 0, 0,
                             {"views": [{"camera_frame_id": "agentview"}]}, ("image.png",))
        context.ingest(visual)
        context.ingest(Observation("telemetry", 2, 1, time.monotonic(), 1, 1,
                                   {"tactile": {"left_fingerpad_contact": False}, "views": []}))
        calls = []

        class Model:
            async def decide(self, prompt, projection, images):
                calls.append((projection, images))
                return {"kind": "noop"}

        slow = SlowAgent(Model(), context, coordinator, store)
        await slow.review()
        projection, images = calls[-1]
        assert projection["observation"]["id"] == "telemetry"
        assert not projection["observation"]["state"]["tactile"]["left_fingerpad_contact"]
        assert projection["attached_visual"]["observation_id"] == "image-frame"
        assert projection["attached_visual"]["action_seq"] == 0
        assert projection["attached_visual"]["age_s"] >= 2
        assert projection["attached_visual"]["status"] == "older_visual_reference_not_current_robot_pose"
        assert images == ("image.png",)
        context.ingest(Observation("new-visual", 3, 1, time.monotonic(), 2, 2,
                                   {"views": [{"camera_frame_id": "agentview"}]}, ("new.png",)))
        await slow.review()
        assert "attached_visual" not in calls[-1][0]
        assert calls[-1][1] == ("new.png",)
    asyncio.run(run())
