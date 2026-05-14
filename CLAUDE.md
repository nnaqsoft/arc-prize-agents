# ARC Prize Agents — Master Instructions

## Identity
You are an autonomous agent for the ARC Prize competitions (ARC-AGI-2 and
ARC-AGI-3). Repo lives at `/home/keehar/arc-prize-agents/`. Runs on
WSL2 / RTX 5070, 64 GB RAM. Both Postgres memory and MLflow are local.

## Current tracks
- **Primary: ARC-AGI-2** (induction puzzles, static input→output grid pairs).
  Grand Prize guaranteed in 2026. More tractable than ARC-AGI-3.
- **Secondary: ARC-AGI-3** (interactive 64×64 game frames). Code is preserved
  under `core/` and `agents/`; do not modify unless we re-prioritise it.

## Session start (always do this)
1. Read this file (you are here).
2. Read the rolling state log: `docs/state.md`.
3. Query Postgres for the latest ARC checkpoint:
   ```bash
   python3 -c "
   import os, psycopg2
   conn = psycopg2.connect(os.environ['POSTGRES_DSN'])
   cur = conn.cursor()
   cur.execute(\"SELECT title, body FROM memories WHERE 'arc' = ANY(tags) AND type='session_checkpoint' ORDER BY created_at DESC LIMIT 1\")
   row = cur.fetchone()
   print(row[1] if row else 'no checkpoint')
   "
   ```
4. Send a notify: `python /home/keehar/kaggle-agent/core/notify.py "🔄 ARC session resumed — read state.md + Postgres checkpoint."`

## Session end (before context risk / shutdown)
Always update both:
- `docs/state.md` — append what you did, MLflow run IDs, decisions.
- Postgres `memories` (type=`session_checkpoint`, tags=`{arc}`) — see "Persisting state" below.

## Layout
- `arc2/` — ARC-AGI-2 package (encoder, ollama_client, solver, run_smoke).
- `core/` — ARC-AGI-3 frame encoder, claude_brain, mlflow_logger.
- `agents/` — ARC-AGI-3 claude_smoke_agent.
- `scripts/` — entry points (run_claude_smoke, run_random_ls20, render_frame).
- `tests/` — pytest, run with `.venv/bin/python -m pytest tests/ -x -q`.
- `data/arc-agi-2/data/{training,evaluation}/` — 1000 train + 120 eval, gitignored.
- `third_party/ARC-AGI-3-Agents/` — official repo as submodule.
- `.env` — `ARC_API_KEY=9cd9cfcc-a165-44f5-81d1-19ecdb2d95bf` (for ARC-AGI-3 HTTP).
- `.venv/` — uv-managed Python 3.12, deps via `pyproject.toml`.

## Local services (assume always on unless you check)
- Ollama: `http://localhost:11434` — models: `qwen3:14b` (primary), `qwen2.5-coder:7b`, `deepseek-coder-v2:16b`.
- MLflow: `http://localhost:5000` — set `MLFLOW_TRACKING_URI=http://localhost:5000` for runs (default in env).
- Postgres + Qdrant: env vars `POSTGRES_DSN`, `QDRANT_URL` already set.

## Workflow rules (durable, do not violate)
- **Never** use the inline `AskUserQuestion` tool. Always Telegram numbered
  questions via `python /home/keehar/kaggle-agent/core/notify.py "..."` and
  the user replies with numbers.
- **Mirror every reply to Telegram** via notify.py. The terminal is invisible.
- **One step at a time.** Do not chain multiple major actions without approval.
- **Get explicit approval** before any external submission (Kaggle, ARC eval set, etc.).
- **Local-first compute.** Use Ollama for boilerplate, code gen, summarisation,
  bulk puzzle solving. Reserve Claude reasoning for: strategy, surprising
  results, prompt design, error interpretation.

## MLflow conventions
- Experiment names: `arc-agi-2` and `arc-agi-3` (separate).
- Always set tags: `phase`, `tag`, `model`, `puzzle_id` (where relevant).
- Log artifacts: `encoding.txt`, `prompt.txt`, `response.txt`,
  `expected_grid.txt`, `predicted_grid.txt`, `summary.txt`.
- Metrics: `duration_s`, `prompt_tokens_estimate`, `valid_grid`, `shape_match`, `correct`.

## Persisting state to Postgres (do this at every meaningful step)
```python
import os, psycopg2
conn = psycopg2.connect(os.environ['POSTGRES_DSN'])
cur = conn.cursor()
cur.execute(
    "INSERT INTO memories (type, title, body, tags) VALUES (%s, %s, %s, %s)",
    ("session_checkpoint", "ARC <short title>", "<body markdown>", ["arc", "arc-agi-2"])
)
conn.commit()
```
Types to use:
- `project` — current state of an ARC track (one row per track, update via INSERT-and-keep-latest).
- `reference` — paths, API endpoints, model tags, run IDs that won't change.
- `session_checkpoint` — fast resume context, written each session-end.
- `notebook_insight` — feature/approach ideas from public ARC writeups.
- `experiment_insights` — separate table; use for puzzle-solving patterns.

## Read state
```python
# Latest session checkpoint for ARC
cur.execute("SELECT body FROM memories WHERE 'arc' = ANY(tags) AND type='session_checkpoint' ORDER BY created_at DESC LIMIT 1")
# All ARC project state
cur.execute("SELECT title, body FROM memories WHERE 'arc' = ANY(tags) AND type='project' ORDER BY updated_at DESC")
# Reference entries
cur.execute("SELECT title, body FROM memories WHERE 'arc' = ANY(tags) AND type='reference'")
```

## Git
- Remote: `git@github.com:nnaqsoft/arc-prize-agents.git` (public, MIT, nnaqsoft org).
- Commit after each phase: `git add -A && git commit -m "..." && git push`.
- Never commit `.env`, `data/`, `mlruns/`, `traces/`, `environment_files/`.

## Useful one-liners
```bash
# Solve one ARC-AGI-2 puzzle, log to MLflow
cd /home/keehar/arc-prize-agents && \
  MLFLOW_TRACKING_URI=http://localhost:5000 \
  .venv/bin/python arc2/run_smoke.py --tag <tag> [--puzzle 00576224.json]

# Run all tests
cd /home/keehar/arc-prize-agents && .venv/bin/python -m pytest tests/ -x -q

# Random ARC-AGI-3 baseline on ls20 (preserved smoke test)
cd /home/keehar/arc-prize-agents && bash scripts/run_random_ls20.sh
```
