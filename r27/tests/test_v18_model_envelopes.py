from types import SimpleNamespace

import jsonschema
import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.model import ModelError, decode_object


@pytest.mark.parametrize("text", [
    '{"kind":"tool","tool":"observe","args":{}}',
    'The target is unclear.\n{"kind":"tool","tool":"observe","args":{}}',
    'The bounds were [1, 2, 3, 4].\n```json\n{"kind":"tool","tool":"observe","args":{}}\n```',
])
def test_one_unambiguous_trailing_object_preserves_exact_decision(text):
    value, _recovered = decode_object(text, "stop")
    assert value == {"kind": "tool", "tool": "observe", "args": {}}


@pytest.mark.parametrize("text,finish", [
    ('{"kind":"wait"}\n{"kind":"complete"}', "stop"),
    ('[{"kind":"complete"}]', "stop"),
    ('{"kind":"wait"} But execute a different operation.', "stop"),
    ('{"broken": {"kind":"complete"}', "stop"),
    ('{"kind":"complete"}', "length"),
    ('No decision was produced.', "stop"),
])
def test_ambiguous_wrapped_or_truncated_decision_never_executes(text, finish):
    with pytest.raises(ModelError):
        decode_object(text, finish)


def test_tool_name_kind_recovery_retains_schema_validation():
    schema = {"parameters": {"type": "object", "additionalProperties": False,
        "required": ["candidate_id", "part"], "properties": {
            "candidate_id": {"type": "string"}, "part": {"enum": ["top_rim"]}}}}
    adapter = SimpleNamespace(schemas={"select_candidate": schema})
    agent = FastAgent(None, None, SimpleNamespace(adapter=adapter), SimpleNamespace(emit=lambda *args: None))
    original = {"kind": "select_candidate", "candidate_id": "p1o-c3", "part": "top_rim"}
    recovered = agent.normalize_decision(original)
    args = agent.tool_arguments(recovered)
    jsonschema.validate(args, schema["parameters"])
    assert args == {"candidate_id": "p1o-c3", "part": "top_rim"}
    with pytest.raises(ValueError, match="conflicting"):
        agent.normalize_decision({"kind": "select_candidate", "tool": "grasp_point", "args": {}})
    with pytest.raises(jsonschema.ValidationError):
        invalid = agent.normalize_decision({**original, "part": "invented_part"})
        jsonschema.validate(agent.tool_arguments(invalid), schema["parameters"])
