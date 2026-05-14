#!/usr/bin/env python3
"""Single-puzzle smoke run for ARC-AGI-2 + MLflow log.

Default: solve the first training puzzle alphabetically with qwen3:14b.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mlflow  # noqa: E402

from arc2.encoder import ArcPuzzle, render_grid_ascii  # noqa: E402
from arc2.solver import render_solve_summary, solve_puzzle  # noqa: E402
from core.mlflow_logger import start_run  # noqa: E402

logger = logging.getLogger("arc2_smoke")

DATA_DIR_DEFAULT = REPO_ROOT / "data" / "arc-agi-2" / "data"


def pick_puzzle(training_dir: Path) -> Path:
    files = sorted(p for p in training_dir.iterdir() if p.suffix == ".json")
    if not files:
        raise SystemExit(f"No puzzles found in {training_dir}")
    return files[0]


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default=str(DATA_DIR_DEFAULT),
        help="Path to the ARC-AGI-2 data dir containing training/ and evaluation/",
    )
    p.add_argument("--split", default="training", choices=["training", "evaluation"])
    p.add_argument("--puzzle", default=None, help="Specific puzzle filename (overrides --split-first)")
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--tag", default="smoke")
    args = p.parse_args()

    data_dir = Path(args.data_dir) / args.split
    if args.puzzle:
        puzzle_path = data_dir / args.puzzle
    else:
        puzzle_path = pick_puzzle(data_dir)
    if not puzzle_path.exists():
        raise SystemExit(f"Puzzle not found: {puzzle_path}")

    puzzle_id = puzzle_path.stem
    data = json.loads(puzzle_path.read_text())
    puzzle = ArcPuzzle.from_json(puzzle_id, data)
    logger.info(
        "Loaded puzzle %s (train=%d, test=%d)",
        puzzle_id,
        len(puzzle.train_pairs),
        len(puzzle.test_inputs),
    )

    params = {
        "agent": "arc2_smoke",
        "model": args.model,
        "encoder": "arc2_v1",
        "split": args.split,
        "puzzle_id": puzzle_id,
        "test_index": 0,
    }

    with start_run(
        run_name=f"arc2-smoke-{puzzle_id}",
        tags={"phase": "arc2-smoke", "tag": args.tag},
        params=params,
    ) as run:
        result = solve_puzzle(puzzle, test_index=0, model=args.model)

        mlflow.log_metric("duration_s", result.duration_s)
        mlflow.log_metric("prompt_tokens_estimate", result.prompt_tokens_estimate)
        mlflow.log_metric("valid_grid", int(result.valid_grid))
        mlflow.log_metric("shape_match", int(result.shape_match))
        mlflow.log_metric("correct", int(result.correct))

        mlflow.log_text(result.encoding_text, "encoding.txt")
        mlflow.log_text(result.user_prompt, "prompt.txt")
        mlflow.log_text(result.raw_response_text, "response.txt")
        if result.expected_output is not None:
            mlflow.log_text(render_grid_ascii(result.expected_output), "expected_grid.txt")
        if result.parsed_output is not None:
            mlflow.log_text(render_grid_ascii(result.parsed_output), "predicted_grid.txt")
        mlflow.log_text(render_solve_summary(result), "summary.txt")

        logger.info("-" * 60)
        logger.info("\n%s", render_solve_summary(result))
        logger.info("-" * 60)
        logger.info("MLflow run: %s", run.info.run_id)

    return 0 if result.correct else 1


if __name__ == "__main__":
    sys.exit(main())
