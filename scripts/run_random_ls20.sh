#!/usr/bin/env bash
# Run the official random baseline agent against the ls20 game.
# Usage: ./scripts/run_random_ls20.sh [num_episodes]
#
# Each episode is a single run of the random agent against ls20.
# The random agent itself plays up to MAX_ACTIONS=80 actions per run,
# stopping early if it reaches GameState.WIN.

set -euo pipefail

NUM_EPISODES="${1:-5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FW_DIR="$REPO_ROOT/third_party/ARC-AGI-3-Agents"
TRACE_DIR="$REPO_ROOT/traces/random_ls20_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$TRACE_DIR"
echo "Repo:    $REPO_ROOT"
echo "Trace:   $TRACE_DIR"
echo "Episodes: $NUM_EPISODES"

cd "$FW_DIR"
for i in $(seq 1 "$NUM_EPISODES"); do
  echo ""
  echo "=== Episode $i / $NUM_EPISODES ==="
  .venv/bin/python main.py --agent=random --game=ls20 --tags="random-baseline,ep$i" \
    2>&1 | tee "$TRACE_DIR/episode_${i}.log"
done

echo ""
echo "All episodes complete. Logs in: $TRACE_DIR"
