import copy
import importlib.util
import json
import signal
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "rollout_context.py"
spec = importlib.util.spec_from_file_location("rollout_context", path)
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def history(start=0, stop=10):
    messages = [dict(role="system", content="Original robot policy")]
    for i in range(start, stop):
        messages.extend(
            [
                dict(
                    role="user",
                    content=[
                        dict(type="image_url", image_url={"url": "image" + str(i)}),
                        dict(type="text", text=json.dumps({"observation_id": "obs" + str(i)})),
                    ],
                ),
                dict(
                    role="assistant",
                    content="action-" + str(i),
                    reasoning="historical reasoning " + ("x" * 1000),
                ),
                dict(role="user", content=[dict(type="text", text="measured outcome " + str(i))]),
            ]
        )
    return messages


def memory(evidence="obs0"):
    return dict(
        schema_version=context.SCHEMA,
        task="put object in basket",
        facts=[dict(statement="An earlier observation exists.", evidence=[evidence])],
        attempts=[],
        uncertainties=["Grasp remains unverified."],
        remaining=["Place the object."],
        constraints=["Positions use millimetres."],
    )


def response(value=None, finish="stop"):
    return {
        "choices": [
            {"finish_reason": finish, "message": {"content": json.dumps(value or memory())}}
        ]
    }


def rejected_turns(count=12, label="failed"):
    turns = []
    for i in range(count):
        turns.extend(
            [
                dict(role="assistant", content=None, reasoning=label + str(i) + "x" * 1000),
                dict(
                    role="user", content=[dict(type="text", text="Rejected: incomplete generation")]
                ),
            ]
        )
    return turns


class HistoryCompactionTests(unittest.TestCase):
    def create(self, summarize, count=None, record=lambda x: None):
        return context.HistoryCompactor(
            task="put object in basket",
            count_tokens=count or (lambda m: len(json.dumps(m))),
            summarize=summarize,
            record=record,
            threshold=8000,
            max_model_len=50000,
            max_output_tokens=128,
            memory_format="json",
        )

    def test_compact_preserves_policy_latest_observations_and_original_audit(self):
        original = history()
        before = copy.deepcopy(original)
        requests = []
        events = []

        def summarize(messages, attempt, schema):
            requests.append(messages)
            return response()

        c = self.create(summarize, record=events.append)
        wire = c.prepare(original)
        self.assertEqual(original, before)
        self.assertEqual(wire[0], original[0])
        self.assertEqual(context.observation_ids(wire), ["obs8", "obs9"])
        self.assertEqual(wire[-6:], original[-6:])
        self.assertNotIn("obs8", context.observation_ids(requests[0]))
        self.assertNotIn("obs9", context.observation_ids(requests[0]))
        self.assertEqual(c.compactions, 1)
        self.assertLess(events[-1]["after_tokens"], events[-1]["before_tokens"])
        self.assertEqual(events[-1]["event"], "compact_committed")

    def test_later_compaction_can_summarize_previous_memory(self):
        requests = []

        def summarize(messages, attempt, schema):
            requests.append(messages)
            return response()

        c = self.create(summarize)
        original = history()
        c.prepare(original)
        extended = original + history(10, 20)[1:]
        wire = c.prepare(extended)
        self.assertEqual(c.compactions, 2)
        self.assertTrue(
            any(
                isinstance(m.get("content"), list)
                and any(p.get("text", "").startswith(context.MEMORY_NOTICE) for p in m["content"])
                for m in requests[1]
            )
        )
        self.assertEqual(context.observation_ids(wire), ["obs18", "obs19"])

    def test_query_images_do_not_replace_recent_physical_observation_groups(self):
        original = history()
        for i in reversed(range(10)):
            index = 1 + i * 3
            queries = []
            for q in range(4):
                queries.extend(
                    [
                        dict(role="assistant", content="query"),
                        dict(
                            role="user",
                            content=[
                                dict(
                                    type="text",
                                    text=json.dumps(
                                        dict(stage="mark_point", observation_id="obs" + str(i))
                                    ),
                                ),
                                dict(type="image_url", image_url={"url": f"query-{i}-{q}"}),
                            ],
                        ),
                    ]
                )
            original[index + 1 : index + 1] = queries
        before = copy.deepcopy(original)
        c = self.create(lambda *args: response())
        wire = c.prepare(original)
        self.assertEqual(context.observation_ids(wire), ["obs8", "obs9"])
        self.assertEqual(original, before)
        # The physical image and all four annotated replies from BOTH recent
        # observations survive, together with their text and execution feedback.
        images = [
            p["image_url"]["url"]
            for m in wire
            if isinstance(m.get("content"), list)
            for p in m["content"]
            if p.get("type") == "image_url"
        ]
        self.assertEqual(
            images,
            ["image8"]
            + [f"query-8-{q}" for q in range(4)]
            + ["image9"]
            + [f"query-9-{q}" for q in range(4)],
        )

    def test_unseen_citations_never_commit_or_drop_history(self):
        c = self.create(lambda messages, attempt, schema: response(memory("obs9")))
        original = history()
        with self.assertRaises(context.CompactionError):
            c.prepare(original)
        self.assertIsNone(c.memory)
        self.assertEqual(c.cutoff, 1)
        self.assertEqual(c.compose(original), original)

    def test_truncated_summary_never_commits(self):
        c = self.create(lambda messages, attempt, schema: response(finish="length"))
        with self.assertRaises(context.CompactionError):
            c.prepare(history())
        self.assertEqual(c.compactions, 0)
        self.assertIsNone(c.memory)

    def test_retry_explains_schema_error_and_lists_only_available_evidence(self):
        requests = []

        def summarize(messages, attempt, schema):
            requests.append(messages)
            return response(memory("obs9")) if attempt == 0 else response()

        c = self.create(summarize)
        c.prepare(history())
        self.assertEqual(c.compactions, 1)
        self.assertEqual(len(requests), 2)
        self.assertIn("only allowed evidence observation IDs", requests[0][-1]["content"])
        self.assertIn("previous summary was rejected", requests[1][-1]["content"])
        self.assertIn("outside the summary evidence", requests[1][-1]["content"])

    def test_nonshrinking_summary_is_rejected_transactionally(self):
        def count(messages):
            has_memory = any(
                isinstance(m.get("content"), list)
                and any(p.get("text", "").startswith(context.MEMORY_NOTICE) for p in m["content"])
                for m in messages
            )
            return 9000 if has_memory else len(json.dumps(messages))

        c = self.create(lambda messages, attempt, schema: response(), count=count)
        with self.assertRaises(context.CompactionError):
            c.prepare(history())
        self.assertIsNone(c.memory)

    def test_under_threshold_is_unchanged_and_does_not_summarize(self):
        def summarize(*args):
            raise AssertionError("unexpected model summary")

        c = self.create(summarize)
        original = history(0, 2)
        self.assertEqual(c.prepare(original), original)
        self.assertEqual(c.compactions, 0)

    def test_stalled_single_observation_compacts_retries_and_keeps_raw_evidence(self):
        original = history(0, 1)[:2]
        query = [
            dict(role="assistant", content="point query"),
            dict(
                role="user",
                content=[
                    dict(
                        type="text",
                        text=json.dumps(
                            dict(stage="mark_point", observation_id="obs0", measured_depth_mm=125)
                        ),
                    ),
                    dict(type="image_url", image_url={"url": "annotated-image0"}),
                ],
            ),
        ]
        original += query + rejected_turns()
        before = copy.deepcopy(original)
        requests = []
        events = []

        def summarize(messages, attempt, schema):
            requests.append(messages)
            return response()

        c = self.create(summarize, record=events.append)
        wire = c.prepare(original)
        self.assertEqual(original, before)
        self.assertEqual(wire[2:5], original[1:4])
        self.assertEqual(wire[-2:], original[-2:])
        self.assertEqual(events[-1]["mode"], "rejected_policy_turns")
        self.assertEqual(context.observation_ids(wire), ["obs0"])
        self.assertIn("failed0", json.dumps(requests[0]))
        self.assertNotIn("failed0", json.dumps(wire))
        self.assertLess(c.last_input_tokens, c.threshold)
        self.assertEqual(c.prepare(original), wire)

    def test_stalled_two_observations_keep_actions_outcomes_and_both_images(self):
        original = history(0, 2) + rejected_turns()
        before = copy.deepcopy(original)
        c = self.create(lambda *args: response())
        wire = c.prepare(original)
        self.assertEqual(wire[2:8], original[1:7])
        self.assertEqual(context.observation_ids(wire), ["obs0", "obs1"])
        self.assertEqual(wire[-2:], original[-2:])
        self.assertEqual(original, before)

    def test_large_recent_group_uses_rejected_turn_fallback_despite_older_prefix(self):
        original = history(0, 3) + rejected_turns()
        c = self.create(lambda *args: response())
        wire = c.prepare(original)
        self.assertEqual(context.observation_ids(wire), ["obs0", "obs1", "obs2"])
        self.assertEqual(wire[2:11], original[1:10])
        self.assertEqual(c.cutoff, 1)

    def test_stalled_retries_can_compact_again_after_a_previous_memory(self):
        c = self.create(lambda *args: response())
        original = history()
        c.prepare(original)
        original += rejected_turns(label="later")
        wire = c.prepare(original)
        self.assertEqual(c.compactions, 2)
        self.assertEqual(context.observation_ids(wire), ["obs8", "obs9"])
        self.assertEqual(wire[-2:], original[-2:])
        self.assertNotIn("later0", json.dumps(wire))
        extended = original + history(10, 20)[1:]
        wire = c.prepare(extended)
        self.assertEqual(c.compactions, 3)
        self.assertEqual(context.observation_ids(wire), ["obs18", "obs19"])

    def test_failed_stalled_summary_is_transactional(self):
        original = history(0, 1) + rejected_turns()
        before = copy.deepcopy(original)
        c = self.create(lambda *args: response(memory("unseen")))
        with self.assertRaises(context.CompactionError):
            c.prepare(original)
        self.assertIsNone(c.memory)
        self.assertEqual(c.removed_indices, set())
        self.assertEqual(c.cutoff, 1)
        self.assertEqual(c.compose(original), before)

    def test_service_must_leave_summary_and_output_headroom(self):
        with self.assertRaises(ValueError):
            context.HistoryCompactor(
                task="x",
                count_tokens=len,
                summarize=lambda *x: None,
                threshold=200000,
                max_model_len=131072,
            )

    def test_original_task_cannot_be_changed_by_summary(self):
        value = memory()
        value["task"] = "a different task"
        with self.assertRaises(context.CompactionError):
            context.validate_memory(json.dumps(value), "put object in basket", {"obs0"})

    def test_schema_requires_evidence_and_restricts_it_to_summarized_prefix(self):
        schema = context.memory_schema("task", {"obs0", "obs1"})
        fact = schema["properties"]["facts"]["items"]
        self.assertIn("evidence", fact["required"])
        self.assertEqual(fact["properties"]["evidence"]["items"]["enum"], ["obs0", "obs1"])
        self.assertEqual(schema["properties"]["task"]["const"], "task")
        self.assertFalse(schema["additionalProperties"])

    def test_gross_length_unit_drift_is_rejected(self):
        value = memory()
        value["facts"][0]["statement"] = "Several multi-meter reposition sweeps."
        with self.assertRaises(context.CompactionError):
            context.validate_memory(json.dumps(value), "put object in basket", {"obs0"})


class TextMemoryTests(unittest.TestCase):
    def create(self, summarize):
        return context.HistoryCompactor(
            task="put object in basket",
            count_tokens=lambda m: len(json.dumps(m)),
            summarize=summarize,
            threshold=8000,
            max_model_len=50000,
            max_output_tokens=128,
        )

    def test_model_prose_needs_no_json_and_program_owns_the_metadata(self):
        requests = []
        body = "Attempted a lift; grasp unverified. Check fresh images. }"

        def summarize(messages, attempt, schema):
            requests.append((messages, schema))
            return {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}

        c = self.create(summarize)
        original = history()
        before = copy.deepcopy(original)
        wire = c.prepare(original)
        self.assertEqual(original, before)
        self.assertIsNone(requests[0][1])
        self.assertIn("natural language", requests[0][0][-1]["content"])
        self.assertNotIn("Required JSON schema", requests[0][0][-1]["content"])
        self.assertEqual(c.memory["summary_text"], body)
        self.assertEqual(c.memory["task"], "put object in basket")
        self.assertEqual(c.memory["source_observation_ids"], ["obs" + str(i) for i in range(10)])
        self.assertTrue(c.memory["untrusted_model_summary"])
        self.assertEqual(wire[-1], original[-1])
        self.assertEqual(wire[-2]["content"], original[-2]["content"])
        self.assertEqual(context.observation_ids(wire), ["obs9"])

    def test_model_json_cannot_override_the_envelope_or_cause_a_syntax_failure(self):
        body = '{"task":"wrong task","schema_version":"wrong", "facts": [ }'
        c = self.create(
            lambda *args: {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
        )
        c.prepare(history())
        self.assertEqual(c.memory["task"], "put object in basket")
        self.assertEqual(c.memory["schema_version"], context.TEXT_SCHEMA)
        self.assertEqual(c.memory["summary_text"], body)

    def test_nonempty_length_limited_prose_is_retained_and_marked_incomplete(self):
        c = self.create(
            lambda *args: {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Lift was attempted; verify"},
                    }
                ]
            }
        )
        c.prepare(history())
        self.assertTrue(c.memory["may_be_incomplete"])
        self.assertEqual(c.compactions, 1)

    def test_empty_summary_still_retries_and_preserves_all_original_history(self):
        calls = []

        def summarize(*args):
            calls.append(args)
            return {"choices": [{"finish_reason": "stop", "message": {"content": " "}}]}

        c = self.create(summarize)
        original = history()
        before = copy.deepcopy(original)
        with self.assertRaises(context.CompactionError):
            c.prepare(original)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(c.memory)
        self.assertEqual(c.compose(original), before)

    def test_rejected_thinking_compacts_after_an_existing_text_memory(self):
        def summarize(*args):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Historical attempts remain unverified."},
                    }
                ]
            }

        c = self.create(summarize)
        original = history()
        c.prepare(original)
        original += rejected_turns()
        wire = c.prepare(original)
        self.assertEqual(c.compactions, 2)
        self.assertEqual(c.memory["source_observation_ids"], ["obs" + str(i) for i in range(10)])
        self.assertEqual(context.observation_ids(wire), ["obs9"])
        self.assertEqual(wire[-1], original[-1])

    def test_service_error_is_not_accepted_as_a_memory(self):
        with self.assertRaises(context.CompactionError):
            context.text_memory("partial", "task", {"obs1"}, "error")
        memory = context.text_memory("a" * 25000, "task", {"obs1"}, "stop")
        self.assertEqual(len(memory["summary_text"]), 24576)
        self.assertTrue(memory["may_be_incomplete"])

    def test_latest_observation_keeps_action_and_feedback_without_reintroducing_120k_cot(self):
        original = history(0, 3)
        original[-2]["content"] = '{"stage":"move_to","observation_id":"obs2"}'
        original[-2]["reasoning"] = "OLD_COT_" + "x" * 120000
        before = copy.deepcopy(original)
        requests = []

        def summarize(messages, *args):
            requests.append(messages)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Movement attempted; grasp unverified."},
                    }
                ]
            }

        c = context.HistoryCompactor(
            task="put object in basket",
            count_tokens=lambda m: len(json.dumps(m)),
            summarize=summarize,
            threshold=8000,
            max_model_len=500000,
            max_output_tokens=128,
        )
        wire = c.prepare(original)
        self.assertIn("OLD_COT_", json.dumps(requests[0]))
        self.assertNotIn("OLD_COT_", json.dumps(wire))
        self.assertEqual(context.observation_ids(wire), ["obs2"])
        self.assertEqual(wire[-2]["content"], original[-2]["content"])
        self.assertEqual(wire[-1], original[-1])
        self.assertEqual(original, before)
        self.assertEqual(c.compose(original), wire)
        self.assertLess(len(json.dumps(wire)), 8000)
        original.append(
            dict(role="assistant", content="next action", reasoning="NEW_COT_not_yet_summarized")
        )
        next_wire = c.prepare(original)
        self.assertIn("NEW_COT_not_yet_summarized", json.dumps(next_wire))
        self.assertNotIn("OLD_COT_", json.dumps(next_wire))


class PolicyExampleTests(unittest.TestCase):
    def test_actual_current_id_is_bound_without_rewriting_actions_or_history(self):
        original = history(0, 2)
        original[0]["content"] = '{"observation_id":"CURRENT"}\n{"observation_id":"COPY_CURRENT"}'
        original[-2]["content"] = '{"observation_id":"CURRENT"}'
        before = copy.deepcopy(original)
        wire = context.bind_observation_examples(original)
        self.assertEqual(wire[0]["content"].count('"observation_id":"obs1"'), 2)
        self.assertNotIn("COPY_CURRENT", wire[0]["content"])
        self.assertEqual(wire[1:], original[1:])
        self.assertEqual(original, before)

    def test_all_query_notices_and_feedback_examples_are_consistent(self):
        notice = '{"stage":"mark_point","observation_id":"CURRENT"}'
        query = SimpleNamespace(BBOX_NOTICE=notice, POINT_NOTICE=notice, QUERY_NOTICE=notice)
        harness = SimpleNamespace(QUERY_NOTICE=notice)
        context.configure_query_notice(query, harness)
        context.configure_query_notice(query, harness)
        self.assertEqual(harness.QUERY_NOTICE, query.QUERY_NOTICE)
        for name in ("BBOX_NOTICE", "POINT_NOTICE", "QUERY_NOTICE"):
            value = getattr(query, name)
            self.assertNotIn('"observation_id":"CURRENT"', value)
            self.assertEqual(value.count(context.ID_NOTICE), 1)
            self.assertIn("never a literal ID", value)


class CleanupTests(unittest.TestCase):
    def test_hung_cleanup_is_bounded_and_restores_signal_handler(self):
        handler = signal.getsignal(signal.SIGALRM)
        report = context.bounded_cleanup(lambda: time.sleep(5), seconds=0.05)
        self.assertEqual(report["status"], "timeout")
        self.assertLess(report["seconds"], 1)
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_close_error_is_recorded_without_escaping_into_task_score(self):
        def close():
            raise RuntimeError("private detail should not be copied")

        report = context.bounded_cleanup(close)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertNotIn("private detail", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
