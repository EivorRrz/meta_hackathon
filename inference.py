"""
Baseline inference for the OpenEnv customer-support environment.

Loads optional `.env` next to this file (python-dotenv).

Environment variables (OpenAI-compatible):
  HF_TOKEN / OPENAI_API_KEY / API_KEY   API key for the LLM endpoint
  API_BASE_URL                         LLM base URL
  MODEL_NAME                           Model id

Connection:
  OPENENV_BASE_URL                  Running env (default http://127.0.0.1:8000)
  IMAGE_NAME / LOCAL_IMAGE_NAME     Optional: GenericEnvClient.from_docker_image

Tasks:
  (empty)                           Run all three tasks in order
  SUPPORT_ENV_TASK or TASK_NAME     Run a single task id (e.g. support_classify)

Stdout (strict):
  [START] task=... env=... model=...
  [STEP]  step=... (two spaces after [STEP] per sample spec)
  [END] success=... steps=... score=... rewards=...
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, List

from openai import OpenAI
from openenv.core.client_types import StepResult
from openenv.core.generic_client import GenericEnvClient

from support_env.tasks import DEFAULT_TICKET_BY_TASK, TASK_ORDER, TASK_SPECS, grader_score

_ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ROOT / ".env")


_load_dotenv()

API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("SUPPORT_ENV_BENCHMARK", "customer_support_env")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")
IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS = int(os.getenv("SUPPORT_ENV_MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("SUPPORT_ENV_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("SUPPORT_ENV_MAX_TOKENS", "240"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUPPORT_ENV_SUCCESS_THRESHOLD", "0.25"))


def selected_tasks() -> tuple[str, ...]:
    """All tasks, or one task if SUPPORT_ENV_TASK / TASK_NAME is set."""

    raw = (os.getenv("SUPPORT_ENV_TASK") or os.getenv("TASK_NAME") or "").strip()
    if not raw:
        return TASK_ORDER
    if raw not in DEFAULT_TICKET_BY_TASK:
        choices = ", ".join(sorted(DEFAULT_TICKET_BY_TASK))
        raise SystemExit(f"Unknown task {raw!r}. Use one of: {choices}")
    return (raw,)


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *,
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    err = "null" if error is None else error
    # Two spaces after [STEP] to match official sample formatting.
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    body = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={body}",
        flush=True,
    )


def _task_blurb(task_id: str) -> str:
    for spec in TASK_SPECS:
        if spec.id == task_id:
            return spec.description
    return ""


def build_system_prompt(task_id: str) -> str:
    return (
        "You output ONLY a single minified JSON object, no prose, no markdown.\n"
        "Schema: {\"action_type\":\"classify|respond|resolve|escalate\",\"content\":string}\n"
        "Optional key metadata may be an object (can be empty).\n"
        f"Objective: {_task_blurb(task_id)}\n"
        "Heuristics: duplicate charge → billing + refund resolution template; login/slow/crash → technical; "
        "vague rage → general + escalate template; acknowledge prior wait/reboot when history says so.\n"
    )


def extract_action_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    blob = m.group(0) if m else cleaned
    data = json.loads(blob)
    if "metadata" not in data:
        data["metadata"] = {}
    return data


def get_model_action(
    client: OpenAI,
    task_id: str,
    observation: dict[str, Any],
    history: list[str],
) -> dict[str, Any]:
    user_lines = [
        f"message: {observation.get('message', '')}",
        f"history: {observation.get('history', [])}",
        f"status: {observation.get('ticket_status', '')}",
        f"internal_difficulty: {observation.get('difficulty', '')}",
        f"step_count: {observation.get('step_count', 0)}",
        f"transcript:\n" + "\n".join(history[-16:]),
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt(task_id)},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return extract_action_json(text)
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "classify", "content": "general", "metadata": {}}


def _as_obs_dict(result: StepResult[dict[str, Any]]) -> dict[str, Any]:
    obs = result.observation
    if isinstance(obs, dict):
        return obs
    return {}


def _metadata(obs: dict[str, Any]) -> dict[str, Any]:
    meta = obs.get("metadata")
    if isinstance(meta, dict):
        return meta
    meta = obs.get("last_info")
    if isinstance(meta, dict):
        return meta
    return {}


def _reward_done(result: StepResult[dict[str, Any]], obs: dict[str, Any]) -> tuple[float, bool]:
    reward = result.reward
    if reward is None:
        reward = obs.get("reward", 0.0)
    done = result.done
    if done is None:
        done = bool(obs.get("done", False))
    try:
        rw = float(reward)
    except (TypeError, ValueError):
        rw = 0.0
    return rw, bool(done)


async def run_one_task(client: OpenAI, env: GenericEnvClient, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    transcript: list[str] = []
    result: StepResult[dict[str, Any]] | None = None

    ticket_id = DEFAULT_TICKET_BY_TASK[task_id]

    try:
        result = await env.reset(
            task=task_id,
            ticket_id=ticket_id,
            episode_id=f"baseline-{task_id}",
        )

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = _as_obs_dict(result)
            action = get_model_action(client, task_id, obs, transcript)
            action_repr = json.dumps(action, ensure_ascii=False)
            result = await env.step(action)
            obs = _as_obs_dict(result)
            rw, done = _reward_done(result, obs)
            meta = _metadata(obs)
            err = meta.get("error")
            err_s = None if err is None else str(err)

            rewards.append(rw)
            steps_taken = step
            transcript.append(f"step {step}: action={action_repr} reward={rw:+.2f}")

            log_step(step=step, action=action_repr, reward=rw, done=done, error=err_s)

            if done:
                norm = meta.get("normalized_episode")
                resolved = bool(meta.get("resolved", meta.get("resolved_flag", False)))
                score = grader_score(norm, resolved=resolved)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if not success and result is not None:
            obs = _as_obs_dict(result)
            meta = _metadata(obs)
            norm = meta.get("normalized_episode")
            resolved = bool(meta.get("resolved", meta.get("resolved_flag", False)))
            score = grader_score(norm, resolved=resolved)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] task failed: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    tasks = selected_tasks()
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    image = LOCAL_IMAGE_NAME or IMAGE_NAME

    if image:
        env = await GenericEnvClient.from_docker_image(image)
    else:
        env = GenericEnvClient(base_url=OPENENV_BASE_URL)

    try:
        async with env:
            for task_id in tasks:
                await run_one_task(llm, env, task_id)
    finally:
        try:
            await env.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
