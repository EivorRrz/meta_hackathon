---
title: Customer Support OpenEnv
emoji: 🎧
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Customer Support Tickets — OpenEnv

Realistic SaaS customer-support **triage and resolution** simulator. Agents choose structured
actions (`classify`, `respond`, `resolve`, `escalate`) to progress tickets with **deterministic**
grading (no LLM judges).

## Why this environment

Support queues are a common real-world agent task: interpret messy language, pick categories,
decide operational paths (refund vs troubleshoot vs escalate), acknowledge prior context, and
close with approved wording. This repo turns that into a reproducible RL / evaluation loop via
the OpenEnv HTTP + WebSocket runtime.

## Action space (`SupportAction`)

| Field | Type | Notes |
| --- | --- | --- |
| `action_type` | `"classify" \| "respond" \| "resolve" \| "escalate"` | Required |
| `content` | `string` | Category text, customer-facing reply, or final resolution text |
| `metadata` | `object` | Optional trace (OpenEnv base field) |

## Observation space (`SupportObservation`)

Core OpenEnv fields: `reward`, `done`, `metadata`.

Additional fields:

| Field | Meaning |
| --- | --- |
| `message` | Current customer message |
| `history` | Prior chatter / system notes (may be empty) |
| `ticket_status` | `open`, `in_progress`, `resolved` |
| `task_name` | Which benchmark task is active |
| `difficulty` | `easy`, `medium`, `hard` internal hint |
| `step_count` / `max_steps` | Efficiency diagnostics |
| `actions_taken` | Ordered log of actions |

`metadata` always includes structured `reward_breakdown`, `normalized_episode` (when `done`),
and booleans like `resolved`.

### `SupportReward` (Pydantic)

Emitted inside `metadata["reward_breakdown"]` every turn:

- `step_reward`
- `cumulative_raw`
- `normalized_episode` (`null` until episode completes)

## Tasks & graders (0.0 – 1.0)

| Task id | Difficulty | Objective |
| --- | --- | --- |
| `support_classify` | Easy | Correct category only |
| `support_routing` | Medium | Category + correct operational commitment |
| `support_resolution` | Hard | Category + canonical empathy reply + exact resolution; history-aware checks |

`support_env.tasks.grader_score` clamps the terminal `normalized_episode` to **[0, 1]** for harnesses.

Baseline ticket IDs (deterministic defaults) live in `DEFAULT_TICKET_BY_TASK`.

## Local quickstart

### Linux / macOS

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                           # unit tests (should all pass)
uvicorn support_env.server.app:app --host 0.0.0.0 --port 8000
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # or: & .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python -m uvicorn support_env.server.app:app --host 0.0.0.0 --port 8000
```

## Environment file (`.env`)

1. Copy **`.env.example`** → **`.env`** in the repo root (same folder as `inference.py`).
2. Fill in secrets; **`inference.py` loads `.env` automatically** when `python-dotenv` is installed
   (included in `pyproject.toml` dependencies).

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` / `OPENAI_API_KEY` / `API_KEY` | LLM API key (first non-empty wins) |
| `API_BASE_URL` | OpenAI-compatible base URL |
| `MODEL_NAME` | Model id |
| `OPENENV_BASE_URL` | Running OpenEnv server (local or HF Space) |
| `SUPPORT_ENV_TASK` or `TASK_NAME` | Optional: single task id instead of all three |
| See `.env.example` | Optional tuning (`MAX_STEPS`, temperature, etc.) |

**.env is gitignored** — never commit real keys.

## Docker

```bash
docker build -t support-openenv:latest .
docker run --rm -p 8000:8000 support-openenv:latest
```

Health probe: `GET http://127.0.0.1:8000/health`

## Baseline inference

Start the server first (Docker or local uvicorn), then in another terminal:

```bash
# Set variables (or use .env file)
export OPENENV_BASE_URL=http://127.0.0.1:8000
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Windows equivalent:

```powershell
$env:OPENENV_BASE_URL = "http://127.0.0.1:8000"
$env:HF_TOKEN = "hf_..."
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

Or use `.env` (see **Environment file** above). API key resolution order: **`HF_TOKEN`** →
**`OPENAI_API_KEY`** → **`API_KEY`**.

The script runs every task in `TASK_ORDER` unless **`SUPPORT_ENV_TASK`** / **`TASK_NAME`** is set.
Logs use `[START]`, `[STEP]  ...` (two spaces after `[STEP]` per sample), and `[END]`.

Point `OPENENV_BASE_URL` at your Hugging Face Space after deployment, or set `IMAGE_NAME` /
`LOCAL_IMAGE_NAME` for `GenericEnvClient.from_docker_image`.

### Pre-submission check script

From Git Bash / WSL / Linux (with Docker + `openenv` CLI installed):

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://YOUR_SPACE.hf.space /path/to/Yo-Codes
```

## OpenEnv validation

```bash
pip install openenv-core
openenv validate
openenv validate --url https://<your-space>.hf.space
```

## Reward function design

| Signal | Value | When |
| ---: | ---: | --- |
| Correct classification | +0.20 | First correct `classify` action |
| Correct action commit | +0.30 | First correct `resolve` or `escalate` |
| Good response | +0.30 | First correct `respond` text |
| Correct resolution | +0.20 | Resolution text matches approved wording |
| Wrong action | -0.20 | Incorrect classification, response, or commit |
| Repeated action | -0.30 | Same fingerprint as previous step |
| Ignored history | -0.20 | HARD only: first `respond` missing history ack |
| Efficiency penalty | -0.05 | Each step beyond optimal count |

Raw rewards are normalized to **[0, 1]** at episode end using min-max bounds `[-3.0, 1.2]`.

## Baseline scores

> Run `python inference.py` locally to fill these. Scores depend on the LLM used.

| Task | Score (0-1) | Notes |
| --- | ---: | --- |
| support_classify | 0.76 | Single `classify` action; most models score high |
| support_routing | 0.05 | Needs classify + correct resolve/escalate |
| support_resolution | 0.06 | Full multi-step; exact strings + history ack |

## License

MIT
