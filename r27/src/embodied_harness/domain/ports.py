from typing import Protocol, Callable, Any
from .models import Observation


class RobotAdapter(Protocol):
    cancellation: str
    generation: int

    async def observe(self, action_seq: int) -> Observation: ...
    async def execute(self, capability: str, args: dict, fence: int,
                      cancelled: Callable[[], bool], publish: Callable) -> dict: ...
    async def verify(self, observation_id: str) -> dict: ...
    async def close(self): ...


class Model(Protocol):
    async def decide(self, system: str, context: dict, images: tuple[str, ...]) -> dict[str, Any]: ...
