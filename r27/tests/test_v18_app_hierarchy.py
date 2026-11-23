import asyncio
import time
from types import SimpleNamespace

from embodied_harness import app
from embodied_harness.domain.models import Observation


def test_hierarchical_startup_waits_for_slow_contract_before_fast(monkeypatch, tmp_path):
    entered, release, fast_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tool_roles = []

    class Adapter:
        generation = 1
        cancellation = "test_boundary"
        schemas = {"observe": {}, "move_to": {}}

        def __init__(self, *args):
            self.host = None

        async def start(self):
            self.host = SimpleNamespace(execution_accounting={})

        async def observe(self, action_seq):
            return Observation("obs", 1, 1, time.monotonic(), 0, action_seq, {})

        async def verify(self, observation_id):
            return {"status": "not_verified"}

        async def close(self):
            pass

    class Model:
        def __init__(self, base_url, key, model, role, output, options=None):
            self.role, self.model = role, model
            self.calls, self.usage, self.latencies = 0, {}, []
            self.accounting = {"responses_with_usage": 0}

        def bind_tools(self, schemas):
            assert self.role != "slow"
            tool_roles.append(self.role)

        def bind_memory(self, entries):
            pass

        def bind_mission_reference(self, _observation):
            pass

        async def decide(self, prompt, projection, images):
            assert self.role == "slow"
            assert projection["purpose"] == "initial_plan"
            entered.set()
            await release.wait()
            return {"kind": "plan", "fundamental_constraints": ["preserve target identity"], "milestones": [
                {"goal": "hold the requested cup", "success_conditions": [
                    {"predicate": "visual", "description": "The requested cup is visibly held above the table."}]},
                {"goal": "place the cup in the bowl", "success_conditions": [
                    {"predicate": "visual", "description": "The requested cup rests inside the bowl."}]},
            ]}

        async def close(self):
            pass

    class Fast:
        def __init__(self, model, context, coordinator, store, *args, **kwargs):
            self.context = context
            self.steps = self.errors = self.false_claims = 0

        async def run(self, max_decisions, deadline):
            assert self.milestones is not None
            assert self.context.milestone.goal == "hold the requested cup"
            assert "preserve target identity" in self.context.milestone.constraints
            fast_started.set()
            return "decision_budget"

    monkeypatch.setattr(app, "OpenETAAdapter", Adapter)
    monkeypatch.setattr(app, "ModelClient", Model)
    monkeypatch.setattr(app, "FastAgent", Fast)
    monkeypatch.setattr(app, "load_env", lambda _: None)
    for key in ("MODEL_BASE_URL", "FAST_API_KEY", "SLOW_API_KEY", "FAST_MODEL", "SLOW_MODEL"):
        monkeypatch.setenv(key, "offline-test")
    args = SimpleNamespace(output=str(tmp_path / "episode"), model_env="unused", memory=None,
                           mission="put the requested cup inside the bowl", backend="eta", sim_url="unused",
                           env_id="unused", seed=0, timeout=5, max_decisions=8,
                           mode="dual_async", hierarchy="milestones", review_every=3)

    async def run():
        episode = asyncio.create_task(app.episode(args))
        await asyncio.wait_for(entered.wait(), 1)
        assert not fast_started.is_set()
        release.set()
        result = await episode
        assert result["error"] is None
        assert result["milestone_metrics"]["planned"]
        assert not result["success"]  # A plan and a Fast return value do not certify the task.

    asyncio.run(run())
    assert fast_started.is_set()
    assert tool_roles == ["fast"]
