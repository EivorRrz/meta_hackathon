import pytest

from support_env.engine import TicketEpisodeEngine
from support_env.scoring import normalized_episode_score
from support_env.models import SupportAction
from support_env.tasks import (
    TASK_SUPPORT_CLASSIFY,
    TASK_SUPPORT_RESOLUTION,
    TASK_SUPPORT_ROUTING,
    DEFAULT_TICKET_BY_TASK,
    grader_score,
)
from support_env.tickets import ALL_TICKETS


def gold_actions(ticket_id: str, task: str):
    from support_env.tickets import ticket_by_id

    t = ticket_by_id(ticket_id)
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
    if t.expected_action.value == "escalate":
        tail = SupportAction(action_type="escalate", content=t.expected_resolution)
    else:
        tail = SupportAction(action_type="resolve", content=t.expected_resolution)
    return [
        SupportAction(action_type="classify", content=t.expected_category.value),
        SupportAction(action_type="respond", content=t.expected_response),
        tail,
    ]


# ---------------------------------------------------------------------------
# Golden-path tests (all 3 difficulty levels)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EASY: single classify resolves the episode
# ---------------------------------------------------------------------------

def test_easy_resolves_after_classify():
    eng = TicketEpisodeEngine()
    eng.reset(task=TASK_SUPPORT_CLASSIFY, ticket_id="t_duplicate_charge", episode_id="easy-1")
    out = eng.step(SupportAction(action_type="classify", content="billing"))
    assert out.done is True
    assert out.metadata.get("resolved") is True
    assert float(out.reward or 0) > 0


# ---------------------------------------------------------------------------
# MEDIUM: classify alone does NOT resolve; classify + resolve does
# ---------------------------------------------------------------------------

def test_medium_needs_action_after_classify():
    eng = TicketEpisodeEngine()
    eng.reset(task=TASK_SUPPORT_ROUTING, ticket_id="t_duplicate_charge", episode_id="med-1")
    out = eng.step(SupportAction(action_type="classify", content="billing"))
    assert out.done is False

    from support_env.tickets import ticket_by_id
    t = ticket_by_id("t_duplicate_charge")
    out2 = eng.step(SupportAction(action_type="resolve", content=t.expected_resolution))
    assert out2.done is True
    assert out2.metadata.get("resolved") is True


# ---------------------------------------------------------------------------
# Max-steps timeout: episode ends even without resolution
# ---------------------------------------------------------------------------

def test_max_steps_timeout():
    eng = TicketEpisodeEngine(max_steps=3)
    eng.reset(task=TASK_SUPPORT_RESOLUTION, ticket_id="t_angry_duplicate", episode_id="timeout-1")
    last = None
    for _ in range(4):
        last = eng.step(SupportAction(action_type="classify", content="wrong_category"))
        if last.done:
            break
    assert last is not None
    assert last.done is True
    assert last.metadata.get("resolved") is not True


# ---------------------------------------------------------------------------
# Repeated action penalty
# ---------------------------------------------------------------------------

def test_repeated_action_penalty():
    eng = TicketEpisodeEngine()
    eng.reset(task=TASK_SUPPORT_CLASSIFY, ticket_id="t_payment_failed", episode_id="repeat-1")
    same = SupportAction(action_type="classify", content="billing")
    out1 = eng.step(same)
    assert out1.done is True
    eng.reset(task=TASK_SUPPORT_ROUTING, ticket_id="t_payment_failed", episode_id="repeat-2")
    eng.step(same)
    out2 = eng.step(same)
    assert out2.metadata.get("repeated_action") is True
    assert float(out2.reward or 0) < 0


# ---------------------------------------------------------------------------
# Invalid dict action penalizes but doesn't crash
# ---------------------------------------------------------------------------

def test_invalid_dict_action_penalizes():
    eng = TicketEpisodeEngine()
    eng.reset(task=TASK_SUPPORT_CLASSIFY, ticket_id="t_vague_complaint", episode_id="unit-invalid")
    out = eng.step({"action_type": "this_is_invalid", "content": "x", "metadata": {}})
    assert float(out.reward or 0) < 0


# ---------------------------------------------------------------------------
# Grader score stays in [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [-5.0, -1.0, 0.0, 0.5, 1.0, 1.2, 3.0])
def test_grader_score_bounded(raw):
    norm = normalized_episode_score(raw)
    score = grader_score(norm, resolved=True)
    assert 0.0 <= score <= 1.0


def test_grader_score_none():
    assert grader_score(None, resolved=False) == 0.0


# ---------------------------------------------------------------------------
# Every ticket in the dataset can be run through all 3 tasks without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticket_id", [t.id for t in ALL_TICKETS])
@pytest.mark.parametrize("task", [TASK_SUPPORT_CLASSIFY, TASK_SUPPORT_ROUTING, TASK_SUPPORT_RESOLUTION])
def test_all_tickets_all_tasks_no_crash(ticket_id, task):
    eng = TicketEpisodeEngine()
    eng.reset(task=task, ticket_id=ticket_id, episode_id=f"smoke-{task}-{ticket_id}")
    for act in gold_actions(ticket_id, task):
        out = eng.step(act)
        if out.done:
            break
    assert out.done is True
