"""Focused regressions for coalescing repeated in-motion Slow reviews."""
import time
from types import SimpleNamespace

from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Observation
from embodied_harness.runtime.context import Context


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})


class UnusedModel:
    pass


def make_slow():
    store = MemoryStore()
    context = Context(store)
    context.action_seq = 7
    context.ingest(Observation("telemetry-1", 1, 1, time.monotonic(), 1.0, 7,
                               {"tactile": {"left_fingerpad_contact": True}, "views": []}))
    slow = SlowAgent(UnusedModel(), context, SimpleNamespace(), store)
    slow.milestones = object()
    slow.last_review_action = context.action_seq
    slow.last_review_observation_id = context.latest.id
    # Keep these cases independent of the 30-second throttle itself.
    slow.last_review_started_at = time.monotonic() - 31
    return slow, context, store


def test_new_telemetry_observation_allows_another_same_action_review():
    slow, context, store = make_slow()
    context.ingest(Observation("telemetry-2", 2, 1, time.monotonic(), 1.5, 7,
                               {"tactile": {"left_fingerpad_contact": False}, "views": []}))

    ticket = slow.request_review("streamed_execution_progress")

    assert ticket == 1
    assert slow.pending.is_set()
    assert store.events[-1]["kind"] == "slow.review_requested"


def test_identical_observation_still_coalesces_after_throttle():
    slow, _context, store = make_slow()

    ticket = slow.request_review("streamed_execution_progress")

    assert ticket == 0
    assert not slow.pending.is_set()
    assert store.events[-1]["kind"] == "slow.review_coalesced"


def test_new_same_action_observation_coalesces_while_review_is_in_flight():
    slow, context, store = make_slow()
    context.review_state["in_flight"] = True
    context.ingest(Observation("telemetry-2", 2, 1, time.monotonic(), 1.5, 7,
                               {"tactile": {"left_fingerpad_contact": False}, "views": []}))

    ticket = slow.request_review("streamed_execution_progress")

    assert ticket == 0
    assert not slow.pending.is_set()
    assert store.events[-1]["kind"] == "slow.review_coalesced"
