import pytest

from support_env.engine import TicketEpisodeEngine
from support_env.models import SupportAction
from support_env.tasks import (
    TASK_SUPPORT_CLASSIFY,
    TASK_SUPPORT_RESOLUTION,
    TASK_SUPPORT_ROUTING,
    DEFAULT_TICKET_BY_TASK,
)


def gold_actions(ticket: str, task: str):
    from support_env.tickets import ticket_by_id

    t = ticket_by_id(ticket)
    if task == TASK_SUPPORT_CLASSIFY:
        return [SupportAction(action_type="classify", content=t.expected_category.value)]
    if task == TASK_SUPPORT_ROUTING:
        if t.expected_action.value == "escalate":
            tail = SupportAction(action_type="escalate", content=t.expected_resolution)
        else:
            tail = SupportAction(action_type="resolve", content=t.expected_resolution)
        return [
            SupportAction(action_type="classify", content=t.expected_category.value),
            tail,
        ]
    # HARD
    if t.expected_action.value == "escalate":
        tail = SupportAction(action_type="escalate", content=t.expected_resolution)
    else:
        tail = SupportAction(action_type="resolve", content=t.expected_resolution)
    return [
        SupportAction(action_type="classify", content=t.expected_category.value),
        SupportAction(action_type="respond", content=t.expected_response),
        tail,
    ]


@pytest.mark.parametrize("task", [TASK_SUPPORT_CLASSIFY, TASK_SUPPORT_ROUTING, TASK_SUPPORT_RESOLUTION])
def test_deterministic_golden_path(task):
    ticket = DEFAULT_TICKET_BY_TASK[task]
    eng = TicketEpisodeEngine()
    eng.reset(task=task, ticket_id=ticket, episode_id=f"unit-{task}")
    last = None
    for act in gold_actions(ticket, task):
        last = eng.step(act)
        if last.done:
            break
    assert last is not None
    assert last.done is True
    assert last.metadata.get("resolved") is True
    assert last.metadata.get("normalized_episode", 0) >= 0.75


def test_invalid_dict_action_penalizes():
    eng = TicketEpisodeEngine()
    eng.reset(
        task=TASK_SUPPORT_CLASSIFY,
        ticket_id="t_vague_complaint",
        episode_id="unit-invalid",
    )
    out = eng.step({"action_type": "this_is_invalid", "content": "x", "metadata": {}})
    assert float(out.reward or 0) < 0
