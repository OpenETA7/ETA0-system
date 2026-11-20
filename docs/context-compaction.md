# Eval context compaction

This ports the reusable context logic from ETA0-training commit
`658ec3ec975be54411c604c011dd464bd521d565` into the `bbox-mark-point` harness.
It does not require the training repository, a particular model, Ray, vLLM,
credentials, or an installed simulator. The live evaluation remains pinned to
its original release; publishing this module does not modify a running job.

## What changes

`HistoryCompactor` defaults to `robot-history-memory-text-v2`. The model writes
natural-language history; code supplies the exact task, mm/deg unit metadata,
source observation IDs, and an untrusted-summary marker. A nonempty length-stop
summary is usable if it fits and reduces context, with `may_be_incomplete=true`.
Empty or abnormal responses remain errors. `memory_format="json"` retains the
legacy strict schema and citation validation for callers that need it.

At compact time the summarizer sees all active history, including the reasoning
about to be removed. The next action request contains the summary and one latest
physical observation group: original RGB, measured state, action content, and
point/bbox query feedback. Already summarized assistant `reasoning` and
`reasoning_content` are removed from that inference tail, including leading
`<think>...</think>` content. Previously summarized indices stay stripped on
future requests. Fresh reasoning remains active until the next compact.

The audit messages are not mutated. Full action responses/CoT remain in
`events.jsonl`, original images remain in observation folders, and the caller
can archive every compaction event using `record`. Also archive the complete
summary requests/responses in the client if full generation provenance is
required; the compaction events alone are not a summary transport transcript.

Only older rejected assistant/feedback pairs may be removed separately from
the historical prefix; the latest rejection is retained for repair. Query images
belong to their physical observation, not an additional independent observation.
No action JSON or measured state is rewritten to repair model output.

## Wiring

```python
import json
from harness import run
from rollout_context import HistoryCompactor

task = "...exact benchmark task..."

def record_compact(event):
    # Use a separate audit file, outside run()'s not-yet-created episode folder.
    with open("compact-events.jsonl", "a") as stream:
        stream.write(json.dumps(event) + "\n")

def summarize(messages, attempt, schema):
    # Implement with your client. Archive request/response here as needed.
    # For text mode schema is None: no JSON output grammar is required.
    return client.chat(messages, max_tokens=4096, enable_thinking=False)

compactor = HistoryCompactor(
    task=task,
    count_tokens=client.count_message_tokens,
    summarize=summarize,
    record=record_compact,
    threshold=200000,
    max_model_len=262144,
    max_output_tokens=16384,   # reserve for the action reply
    memory_format="text",
    keep_observations=1,
)
result = run(
    benchmark, policy,
    instruction=task, action_prompt=action_prompt, output_dir="outputs/new-episode",
    context_compactor=compactor,
    bind_observation_ids=True,
    native_finish_feedback=True,
)
```

`client` above is an injected application client, not an API supplied by this
package. Its token counter must count the actual served model's chat template
and image tokens, not estimate characters or just sum text tokens. In text mode
keep the summarizer's 4K output within the reserved `max_output_tokens` budget.
Generation timeout and episode wall-time limits still belong to the client and
caller. `run()` keeps its call/action bounds; summary calls have their own bounded
two-attempt validation and must also obey the caller's remaining wall budget.

The module's current summary protocol assumes model-facing mm and degrees.
Adapters using other units need an explicitly adapted protocol; do not relabel
their state. One compactor belongs to one episode and exact task: never share
its committed memory/indices across parallel episodes or reuse it after reset.

`run()` defaults to no compaction and preserves its existing finish behavior.
The options above opt into the eval behavior:

- `bind_observation_ids=True` replaces CURRENT/COPY_CURRENT **system example**
  IDs with the current measured ID on each request. It does not fix stale IDs
  returned by a model or bypass the action/query validators. Query cameras are
  `external` and `wrist`, plus adapter-provided named views.
- `native_finish_feedback=True` treats model finish as a request to check native
  success. Failure returns explicit feedback and continues within the original
  budgets; no physics is executed by the finish request. Default False retains
  the previous early-stop behavior for compatibility.
- Compaction failures remain infrastructure exceptions and execute no action.
  An evaluation aggregator must list them separately from native task failures.

## Evidence and limits

The original fullmix2500 LIBERO-10 test pinned task wording, all 100 initial
states, 512px images, sampling, native success checks, and TP2/DFlash serving.
Only the context intervention changed. This library port has offline regression
and fake-robot integration coverage; it has not rerun those 100 physical cases.
The standalone robot adapter and deployment/scheduling remain caller-owned.

Tests cover 120K-character retained CoT, no later CoT reinsertion, fresh reasoning,
point/bbox observation grouping, failed-summary rollback, latest-ID binding,
premature finish feedback, and preservation of full audit responses.
