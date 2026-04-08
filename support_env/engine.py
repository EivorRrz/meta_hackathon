from __future__ import annotations

import re
import unicodedata
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from support_env.models import (
    Category,
    ExpectedSupportAction,
    SupportAction,
    SupportObservation,
    SupportReward,
    TaskDifficulty,
    Ticket,
)
from support_env.tasks import TASK_SUPPORT_CLASSIFY, task_to_internal_difficulty
from support_env.tickets import ALL_TICKETS, ticket_by_id

INVALID_ACTION_PENALTY = -0.15
WRONG_ACTION_PENALTY = -0.2
REPEATED_ACTION_PENALTY = -0.3
IGNORE_HISTORY_PENALTY = -0.2
EFFICIENCY_PENALTY = -0.05

CLASSIFICATION_BONUS = 0.2
ACTION_BONUS = 0.3
RESPONSE_BONUS = 0.3
RESOLUTION_BONUS = 0.2

RAW_MIN = -3.0
RAW_MAX = 1.2

TICKET_STATUS_OPEN = "open"
TICKET_STATUS_IN_PROGRESS = "in_progress"
TICKET_STATUS_RESOLVED = "resolved"


def normalize_text(value: str) -> str:
    s = unicodedata.normalize("NFKC", value)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_category_from_classify_content(content: str) -> Category | None:
    n = normalize_text(content)
    for c in Category:
        token = c.value
        if n == token or token in n.split():
            return c
    if any(k in n for k in ("billing", "payment", "charge", "invoice", "refund")):
        return Category.BILLING
    if any(k in n for k in ("technical", "login", "crash", "slow", "performance", "app")):
        return Category.TECHNICAL
    if "general" in n:
        return Category.GENERAL
    return None


def normalized_episode_score(cumulative_raw: float) -> float:
    if RAW_MAX == RAW_MIN:
        return 0.0
    x = (cumulative_raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    return max(0.0, min(1.0, x))


def _action_fingerprint(action: SupportAction) -> str:
    return f"{action.action_type}|{normalize_text(action.content)}"


class TicketEpisodeEngine:
    """Deterministic transition model for one ticket episode."""

    def __init__(self, *, max_steps: int = 6) -> None:
        self._max_steps = max_steps
        self._run_id = uuid4().hex
        self.episode_id = ""
        self.task_name = TASK_SUPPORT_CLASSIFY
        self.ticket: Ticket = ALL_TICKETS[0]
        self.difficulty = task_to_internal_difficulty(TASK_SUPPORT_CLASSIFY)
        self.status = TICKET_STATUS_OPEN
        self.step_count = 0
        self.actions_log: list[str] = []
        self.last_action_fingerprint: str | None = None
        self.classified_category: Category | None = None
        self.gave_classification_reward = False
        self.gave_action_reward = False
        self.gave_response_reward = False
        self.gave_resolution_reward = False
        self.gave_history_penalty = False
        self.cumulative_raw = 0.0
        self.last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        task: str,
        ticket_id: str | None,
        episode_id: str | None,
        seed: int | None = None,
    ) -> None:
        del seed  # deterministic environment; reserved for API compatibility
        if episode_id is None:
            episode_id = str(uuid4())
        self._run_id = uuid4().hex
        self.episode_id = episode_id
        self.task_name = task
        self.difficulty = task_to_internal_difficulty(task)
        if ticket_id is None:
            ticket_id = ALL_TICKETS[hash(episode_id) % len(ALL_TICKETS)].id
        self.ticket = ticket_by_id(ticket_id)

        self.status = TICKET_STATUS_OPEN
        self.step_count = 0
        self.actions_log = []
        self.last_action_fingerprint = None
        self.classified_category = None
        self.gave_classification_reward = False
        self.gave_action_reward = False
        self.gave_response_reward = False
        self.gave_resolution_reward = False
        self.gave_history_penalty = False
        self.cumulative_raw = 0.0
        self.last_info = {}

    def build_initial_observation(self) -> SupportObservation:
        return self._observe(reward=0.0, done=False, info={"status": "ready", "run_id": self._run_id})

    def step(self, action: SupportAction | dict[str, Any]) -> SupportObservation:
        if isinstance(action, dict):
            try:
                action = SupportAction.model_validate(action)
            except ValidationError as exc:
                return self._invalid_step(exc)

        step_reward = 0.0
        info: dict[str, Any] = {
            "run_id": self._run_id,
            "ticket_id": self.ticket.id,
            "task": self.task_name,
        }

        fp = _action_fingerprint(action)
        if self.last_action_fingerprint == fp:
            step_reward += REPEATED_ACTION_PENALTY
            info["repeated_action"] = True
        self.last_action_fingerprint = fp

        if self.status == TICKET_STATUS_RESOLVED:
            self.step_count += 1
            if self.step_count > self.ticket.optimal_steps:
                step_reward += EFFICIENCY_PENALTY
            self.cumulative_raw += step_reward
            self.last_info = info
            return self._finalize(step_reward, done=True, info=info)

        if self._should_check_history_ack(action) and not self._hard_history_acknowledged(action):
            if not self.gave_history_penalty:
                step_reward += IGNORE_HISTORY_PENALTY
                self.gave_history_penalty = True
                info["ignored_history"] = True

        if action.action_type == "classify":
            parsed = parse_category_from_classify_content(action.content)
            if parsed is None:
                step_reward += WRONG_ACTION_PENALTY
                info["classification_parse_failed"] = True
            elif parsed != self.ticket.expected_category:
                step_reward += WRONG_ACTION_PENALTY
                info["wrong_category"] = True
            elif not self.gave_classification_reward:
                step_reward += CLASSIFICATION_BONUS
                self.gave_classification_reward = True
                self.classified_category = parsed
                info["classification_correct"] = True
            else:
                info["classification_duplicate"] = True

        elif action.action_type == "respond":
            if not self.gave_response_reward and self._response_matches(action.content):
                step_reward += RESPONSE_BONUS
                self.gave_response_reward = True
                info["response_correct"] = True
            elif not self.gave_response_reward:
                step_reward += WRONG_ACTION_PENALTY
                info["response_incorrect"] = True
            else:
                info["response_duplicate"] = True

        elif action.action_type == "escalate":
            if self._escalate_commit_correct(action):
                if not self.gave_action_reward:
                    step_reward += ACTION_BONUS
                    self.gave_action_reward = True
                    info["action_correct"] = True
                if not self.gave_resolution_reward:
                    step_reward += RESOLUTION_BONUS
                    self.gave_resolution_reward = True
                    info["resolution_correct"] = True
            else:
                step_reward += WRONG_ACTION_PENALTY
                info["escalate_incorrect_or_unexpected"] = True

        elif action.action_type == "resolve":
            if self._resolve_commit_correct(action):
                if not self.gave_action_reward:
                    step_reward += ACTION_BONUS
                    self.gave_action_reward = True
                    info["action_correct"] = True
                if not self.gave_resolution_reward:
                    step_reward += RESOLUTION_BONUS
                    self.gave_resolution_reward = True
                    info["resolution_correct"] = True
            else:
                step_reward += WRONG_ACTION_PENALTY
                info["resolve_incorrect"] = True
        else:
            return self._invalid_step(ValueError("unknown_action_type"))

        self.actions_log.append(f"{action.action_type}:{action.content}")
        self.step_count += 1
        if self.step_count > self.ticket.optimal_steps:
            step_reward += EFFICIENCY_PENALTY
        self.cumulative_raw += step_reward

        if self.status == TICKET_STATUS_OPEN and self.step_count >= 1:
            self.status = TICKET_STATUS_IN_PROGRESS

        if self._resolved_criteria_met():
            self.status = TICKET_STATUS_RESOLVED
            info["resolved"] = True

        done = self.status == TICKET_STATUS_RESOLVED or self.step_count >= self._max_steps
        if done and self.status != TICKET_STATUS_RESOLVED:
            self.status = TICKET_STATUS_IN_PROGRESS

        return self._finalize(step_reward, done=done, info=info)

    def _response_matches(self, content: str) -> bool:
        return normalize_text(content) == normalize_text(self.ticket.expected_response)

    def _resolution_matches(self, content: str) -> bool:
        return normalize_text(content) == normalize_text(self.ticket.expected_resolution)

    def _escalate_shape_ok(self, action: SupportAction) -> bool:
        return len(normalize_text(action.content)) >= 8

    def _resolve_commit_correct(self, action: SupportAction) -> bool:
        if action.action_type != "resolve":
            return False
        if self.ticket.expected_action not in (
            ExpectedSupportAction.REFUND,
            ExpectedSupportAction.TROUBLESHOOT,
        ):
            return False
        return self._resolution_matches(action.content)

    def _escalate_commit_correct(self, action: SupportAction) -> bool:
        if self.ticket.expected_action != ExpectedSupportAction.ESCALATE:
            return False
        if action.action_type != "escalate":
            return False
        if not self._escalate_shape_ok(action):
            return False
        return self._resolution_matches(action.content)

    def _should_check_history_ack(self, action: SupportAction) -> bool:
        return (
            self.difficulty == TaskDifficulty.HARD
            and bool(self.ticket.history_ack_substrings)
            and action.action_type == "respond"
            and not self.gave_response_reward
        )

    def _hard_history_acknowledged(self, action: SupportAction) -> bool:
        if self.difficulty != TaskDifficulty.HARD:
            return True
        if not self.ticket.history_ack_substrings:
            return True
        blob = normalize_text(action.content)
        return any(normalize_text(s) in blob for s in self.ticket.history_ack_substrings)

    def _resolved_criteria_met(self) -> bool:
        if not self.gave_classification_reward:
            return False
        if self.difficulty == TaskDifficulty.EASY:
            return True
        if not self.gave_action_reward:
            return False
        if self.difficulty == TaskDifficulty.MEDIUM:
            return True
        return self.gave_response_reward and self.gave_resolution_reward

    def _observe(self, *, reward: float, done: bool, info: dict[str, Any]) -> SupportObservation:
        norm: float | None = None
        if done:
            norm = normalized_episode_score(self.cumulative_raw)
        reward_model = SupportReward(
            step_reward=float(reward),
            cumulative_raw=float(self.cumulative_raw),
            normalized_episode=norm,
        )
        meta = dict(info)
        meta["reward_breakdown"] = reward_model.model_dump()
        meta["normalized_episode"] = norm
        meta["resolved_flag"] = self.status == TICKET_STATUS_RESOLVED

        return SupportObservation(
            done=done,
            reward=reward,
            metadata=meta,
            message=self.ticket.message,
            history=list(self.ticket.history),
            ticket_status=self.status,
            task_name=self.task_name,
            difficulty=self.difficulty.value,
            step_count=self.step_count,
            max_steps=self._max_steps,
            actions_taken=list(self.actions_log),
            last_info=dict(info),
        )

    def _finalize(self, step_reward: float, *, done: bool, info: dict[str, Any]) -> SupportObservation:
        self.last_info = dict(info)
        obs = self._observe(reward=step_reward, done=done, info=info)
        return obs

    def _invalid_step(self, exc: Exception) -> SupportObservation:
        inv_fp = "__invalid__"
        step_reward = INVALID_ACTION_PENALTY
        if self.last_action_fingerprint == inv_fp:
            step_reward += REPEATED_ACTION_PENALTY
        self.last_action_fingerprint = inv_fp
        self.step_count += 1
        if self.step_count > self.ticket.optimal_steps:
            step_reward += EFFICIENCY_PENALTY
        self.cumulative_raw += step_reward
        if self.status == TICKET_STATUS_OPEN:
            self.status = TICKET_STATUS_IN_PROGRESS
        info = {
            "run_id": self._run_id,
            "ticket_id": self.ticket.id,
            "task": self.task_name,
            "invalid_action": True,
            "error": str(exc),
        }
        done = self.step_count >= self._max_steps
        return self._finalize(step_reward, done=done, info=info)
