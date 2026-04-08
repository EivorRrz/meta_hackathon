from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator

ActionType = Literal["classify", "respond", "resolve", "escalate"]


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"


class ExpectedSupportAction(str, Enum):
    REFUND = "refund"
    TROUBLESHOOT = "troubleshoot"
    ESCALATE = "escalate"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Ticket(BaseModel):
    model_config = {"frozen": True}

    id: str
    message: str
    history: tuple[str, ...] = ()
    expected_category: Category
    expected_action: ExpectedSupportAction
    expected_resolution: str
    expected_response: str
    optimal_steps: int = Field(ge=1, default=4)
    history_ack_substrings: tuple[str, ...] = ()


class SupportReward(BaseModel):
    """Structured reward breakdown (also embedded in observation.metadata)."""

    step_reward: float
    cumulative_raw: float
    normalized_episode: float | None = None


class SupportAction(Action):
    action_type: ActionType
    content: str = ""

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        return v.strip()


class SupportObservation(Observation):
    message: str = ""
    history: list[str] = Field(default_factory=list)
    ticket_status: str = "open"
    task_name: str = ""
    difficulty: str = "easy"
    step_count: int = 0
    max_steps: int = 6
    actions_taken: list[str] = Field(default_factory=list)
    last_info: dict[str, Any] = Field(default_factory=dict)

    @property
    def echoed_message(self) -> str:
        """Alias for sample clients that read `observation.echoed_message`."""

        return self.message


class SupportState(State):
    ticket_id: str = ""
    task_name: str = ""
    cumulative_raw: float = 0.0
