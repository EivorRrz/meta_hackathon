from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from support_env.engine import TicketEpisodeEngine
from support_env.models import SupportAction, SupportObservation, SupportState


class CustomerSupportEnvironment(Environment[SupportAction, SupportObservation, SupportState]):
    """Customer-support simulator with deterministic ticket grading."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, max_steps: int = 6) -> None:
        super().__init__()
        self._engine = TicketEpisodeEngine(max_steps=max_steps)
        self._state = SupportState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        task = str(kwargs.get("task", "support_classify"))
        ticket_id = kwargs.get("ticket_id")
        tid = str(ticket_id) if ticket_id is not None else None
        self._engine.reset(task=task, ticket_id=tid, episode_id=episode_id, seed=seed)
        self._sync_state(task=task)
        return self._engine.build_initial_observation()

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        del timeout_s, kwargs
        observation = self._engine.step(action)
        self._sync_state(task=self._engine.task_name)
        return observation

    def _sync_state(self, *, task: str) -> None:
        self._state = SupportState(
            episode_id=self._engine.episode_id,
            step_count=self._engine.step_count,
            ticket_id=self._engine.ticket.id,
            task_name=task,
            cumulative_raw=self._engine.cumulative_raw,
        )

    @property
    def state(self) -> SupportState:
        return self._state
