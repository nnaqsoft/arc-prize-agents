# ARC Prize Agents — Rolling State Log

This is the source of truth for "what's the current status, what was just
done, what's next". Append new entries at the top. Each entry: ISO date +
short title + 3–6 bullets. Do NOT rewrite history — only append.

---

## 2026-05-13 — ARC-AGI-2 smoke solver shipped

- Built `arc2/` package: encoder (palette legend mandatory), ollama_client
  (POST /api/chat + format=json), solver (strict-JSON parse + grid validate
  + ground-truth diff), run_smoke (MLflow logging).
- Tests: 11/11 passing (3 ARC-AGI-3 + 8 ARC-AGI-2).
- First smoke run: puzzle `00576224.json` solved correctly by qwen3:14b
  in 45.97s with a ~109-token prompt.
- MLflow run id: `ac850f2b1afa467daf05195af3a11938` (experiment 38).
- Committed: `bebce5b` on `main`, pushed to `nnaqsoft/arc-prize-agents`.
- **Next options (waiting on user):**
  1) Bulk-run solver across all 1000 training puzzles for a qwen3:14b baseline.
  2) Bulk-run on the 120 evaluation puzzles only (cleaner signal).
  3) Iterate on prompt/encoding (CoT, retry, multi-sample voting) before scaling.
  4) Pivot back to ARC-AGI-3.

## 2026-05-13 — Repo rename + dataset clone

- Renamed GitHub repo `nnaqsoft/arc-agi-3-agent` → `nnaqsoft/arc-prize-agents`.
- Moved local checkout `~/arc-agi-3-agent` → `~/arc-prize-agents`.
- Cloned ARC-AGI-2 dataset into `data/arc-agi-2/` (1000 train + 120 eval).
- ARC-AGI-3 code preserved untouched under `core/`, `agents/`, `scripts/`.

## 2026-05-13 — ARC-AGI-3 random + claude smoke tests landed

- Random agent baseline run on `ls20` (5 episodes, 0/7 levels — expected,
  MAX_ACTIONS=80 cap).
- Claude smoke agent: subprocess wrapper around `claude --print
  --output-format json --model sonnet --no-session-persistence
  --setting-sources user --disable-slash-commands`. Working with the
  subscription, no API key needed.
- Frame encoder: flood-fill connected components + palette names.
- MLflow logger: local `mlruns/` by default, override with env.

---

(Older entries: see git log.)
