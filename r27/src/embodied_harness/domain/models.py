from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import time
import uuid


def uid() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Milestone:
    id: str
    revision: int
    mission: str
    goal: str
    capabilities: tuple[str, ...] = ("move_to", "gripper", "delta")
    resources: tuple[str, ...] = ("arm", "gripper")
    max_retries: int = 2
    generation: int = 1
    constraints: tuple[str, ...] = ()
    # v0.1 delegates the task-level predicate; intermediate predicates need a verifier plugin.
    success_predicate: str = "environment_success"
    preconditions: tuple[dict[str, Any], ...] = ()
    success_conditions: tuple[dict[str, Any], ...] = ()
    decision_budget: int = 0
    plan_id: str | None = None
    stage_index: int = 0
    final_stage: bool = True
    guards: tuple[dict[str, Any], ...] = ()
    constraint_scopes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    id: str
    version: int
    generation: int
    captured_at: float
    sim_time: float
    action_seq: int
    state: dict[str, Any]
    images: tuple[str, ...] = ()
    provenance: str = "simulator_sensor"


@dataclass(frozen=True)
class ActionRequest:
    request_id: str
    milestone_id: str
    revision: int
    generation: int
    observation_id: str
    capability: str
    args: dict[str, Any]
    resources: tuple[str, ...] = ("arm", "gripper")
    created_at: float = field(default_factory=time.monotonic)
    ttl: float = 180.0
    control_version: int | None = None


@dataclass
class Receipt:
    action_id: str
    request_id: str
    status: str
    fence: int
    observation_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Advice:
    milestone_id: str
    revision: int
    generation: int
    kind: str
    text: str
    expires_at: float
    # Only action-sensitive advice binds to the action counter. Extra sensor frames are harmless.
    expected_action_seq: int | None = None
    expected_action_id: str | None = None
    expected_action_phase: str | None = None
    id: str = field(default_factory=uid)
    expected_visual_scope: str | None = None
    priority: int = 50


@dataclass(frozen=True)
class StrategicAdvice(Advice):
    """Historical model evidence, never current action or motion authority."""
    expires_at: float | None = None
    review_basis: dict[str, Any] = field(default_factory=dict)
    scope: str = field(default="milestone_strategy", init=False)
    authority: str = field(default="hypothesis_requiring_current_recheck", init=False)


def record(value):
    return asdict(value)
