#!/usr/bin/env python3
"""Entry point for the Claude smoke-test episode.

Usage:
  scripts/run_claude_smoke.py [--game ls20-9607627b] [--tag smoke]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import mlflow  # noqa: E402
from arc_agi import Arcade  # noqa: E402

from agents.claude_smoke_agent import run_smoke_episode  # noqa: E402
from core.mlflow_logger import start_run  # noqa: E402

logger = logging.getLogger("claude_smoke")


def pick_game_id(arcade: Arcade, slug_or_id: str) -> str:
    envs = arcade.get_environments()
    ids = [e.game_id for e in envs]
    exact = [g for g in ids if g == slug_or_id]
    if exact:
        return exact[0]
    prefix = [g for g in ids if g.startswith(slug_or_id)]
    if prefix:
        return prefix[0]
    raise SystemExit(f"No game matching {slug_or_id!r}. Available: {ids[:5]}...")


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ls20", help="Game id or prefix (default: ls20)")
    p.add_argument("--tag", default="smoke", help="MLflow tag suffix")
    args = p.parse_args()

    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key:
        raise SystemExit("ARC_API_KEY not set — check .env")

    recordings_dir = REPO_ROOT / "traces" / "claude_smoke"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    arcade = Arcade(arc_api_key=api_key, recordings_dir=str(recordings_dir))
    game_id = pick_game_id(arcade, args.game)
    logger.info("Resolved game: %s", game_id)

    card_id = arcade.open_scorecard(tags=[f"claude-smoke-{args.tag}"])
    logger.info("Opened scorecard: %s", card_id)

    env = arcade.make(
        game_id,
        scorecard_id=card_id,
        save_recording=True,
        include_frame_data=True,
    )
    if env is None:
        raise SystemExit(f"Arcade.make returned None for {game_id}")

    params = {
        "agent": "claude_smoke",
        "encoder": "symbolic_v1",
        "model": "sonnet",
        "game_id": game_id,
        "card_id": card_id,
    }

    with start_run(run_name=f"smoke-{args.tag}", tags={"phase": "step4-smoke"}, params=params) as run:
        try:
            result = run_smoke_episode(env)
        finally:
            scorecard = arcade.close_scorecard(card_id)

        mlflow.log_metric("claude_calls", 1)
        mlflow.log_metric("claude_duration_s", result.claude_duration_s)
        mlflow.log_metric("actions_taken", result.actions_taken)
        mlflow.log_metric("levels_completed", result.levels_completed)
        mlflow.log_metric("fallback_used", int(result.fallback_used))
        mlflow.log_metric("action_valid", int(result.chose_action_valid))
        mlflow.set_tag("chose_action_id", str(result.chose_action_id))
        mlflow.set_tag("final_state", result.final_state)

        mlflow.log_text(result.encoding_text, "frame_encoding.txt")
        mlflow.log_text(result.claude_prompt, "claude_prompt.txt")
        mlflow.log_text(result.claude_response_text, "claude_response.txt")
        if scorecard is not None:
            mlflow.log_text(scorecard.model_dump_json(indent=2), "scorecard.json")

        logger.info("-" * 60)
        logger.info("Smoke run summary:")
        logger.info(" chose action_id   : %s", result.chose_action_id)
        logger.info(" action was valid  : %s", result.chose_action_valid)
        logger.info(" fallback used     : %s", result.fallback_used)
        logger.info(" final state       : %s", result.final_state)
        logger.info(" levels completed  : %d", result.levels_completed)
        logger.info(" actions taken     : %d", result.actions_taken)
        logger.info(" claude duration_s : %.2f", result.claude_duration_s)
        logger.info(" MLflow run        : %s", run.info.run_id)
        logger.info("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
