import asyncio
from types import SimpleNamespace
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.runtime.events import EventStore


def test_execution_monitor_ignores_idle_and_coalesces_samples(tmp_path):
    async def run():
        store=EventStore(tmp_path/"events.sqlite")
        context=SimpleNamespace(latest=SimpleNamespace(id="current"))
        coordinator=SimpleNamespace(running=None)
        slow=SlowAgent(None,context,coordinator,store)
        task=asyncio.create_task(slow.watch_execution(minimum_interval=999))
        store.emit("sensor.observed",{"id":"current"})
        await asyncio.sleep(.01)
        assert not slow.pending.is_set()
        coordinator.running=object()
        for _ in range(20):store.emit("sensor.observed",{"id":"current"})
        await asyncio.sleep(.01)
        assert slow.pending.is_set()
        assert sum(e["kind"]=="slow.review_requested" for e in store.since(0))==1
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
        store.close()
    asyncio.run(run())
