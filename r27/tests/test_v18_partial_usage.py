import asyncio

import httpx
import pytest

from embodied_harness.agents.model import ModelClient


@pytest.mark.parametrize("usage,expected", [
    ({"prompt_tokens": 12}, {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 0}),
    ({"total_tokens": 20, "prompt_tokens": None},
     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 20}),
])
def test_partial_usage_remains_unknown_cost_in_accounting(tmp_path, usage, expected):
    async def run():
        model = ModelClient("http://test.invalid", "synthetic", "test", "fast", tmp_path)
        await model.client.aclose()
        model.client = httpx.AsyncClient(base_url="http://test.invalid/", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"choices": [{"message": {"content": '{"kind":"noop"}'}}],
                                                 "usage": usage})))
        try:
            assert await model.decide("system", {}) == {"kind": "noop"}
        finally:
            await model.close()
        assert model.accounting["responses_with_usage"] == 0
        assert model.accounting["responses_without_usage"] == 1
        assert model.accounting["attempts_started"] - model.accounting["responses_with_usage"] == 1
        assert model.usage == expected
    asyncio.run(run())
