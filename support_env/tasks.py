from __future__ import annotations

from dataclasses import dataclass

from support_env.models import TaskDifficulty

TASK_SUPPORT_CLASSIFY = "support_classify"
TASK_SUPPORT_ROUTING = "support_routing"
TASK_SUPPORT_RESOLUTION = "support_resolution"

# Default tickets give stable baselines for inference / grading.
DEFAULT_TICKET_BY_TASK: dict[str, str] = {
    TASK_SUPPORT_CLASSIFY: "t_vague_complaint",
    TASK_SUPPORT_ROUTING: "t_duplicate_charge",
    TASK_SUPPORT_RESOLUTION: "t_angry_duplicate",
}

TASK_ORDER: tuple[str, ...] = (
    TASK_SUPPORT_CLASSIFY,
    TASK_SUPPORT_ROUTING,
    TASK_SUPPORT_RESOLUTION,
)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    difficulty: TaskDifficulty
    description: str


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        id=TASK_SUPPORT_CLASSIFY,
        title="Classify the ticket",
        difficulty=TaskDifficulty.EASY,
        description=(
            "Determine the single best category (billing / technical / general) for the "
            "customer message using a classify action."
        ),
    ),
    TaskSpec(
        id=TASK_SUPPORT_ROUTING,
        title="Route with the right action",
        difficulty=TaskDifficulty.MEDIUM,
        description=(
            "Classify correctly, then commit the correct operational action (refund path, "
            "troubleshoot resolution, or leadership escalation)."
        ),
    ),
    TaskSpec(
        id=TASK_SUPPORT_RESOLUTION,
        title="Full resolution with customer reply",
        difficulty=TaskDifficulty.HARD,
        description=(
            "Classify, send the canonical empathetic reply, then finish with the exact "
            "approved resolution text. Penalizes ignoring prior chat history when required."
        ),
    ),
)


def task_to_internal_difficulty(task_id: str) -> TaskDifficulty:
    mapping = {
        TASK_SUPPORT_CLASSIFY: TaskDifficulty.EASY,
        TASK_SUPPORT_ROUTING: TaskDifficulty.MEDIUM,
        TASK_SUPPORT_RESOLUTION: TaskDifficulty.HARD,
    }
    if task_id not in mapping:
        raise ValueError(f"Unknown task id: {task_id}")
    return mapping[task_id]


def grader_score(normalized_episode: float | None, *, resolved: bool) -> float:
    """Deterministic scalar in [0, 1] for validators."""

    del resolved  # reserved for future weighting; keep signature stable for harnesses
    if normalized_episode is None:
        return 0.0
    base = float(normalized_episode)
    if base < 0.0:
        return 0.0
    if base > 1.0:
        return 1.0
    return base
