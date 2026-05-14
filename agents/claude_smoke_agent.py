"""Minimal Claude-powered smoke-test agent.

Spec (per step-4 design):
- Run exactly one episode against one game.
- After the initial RESET, make exactly ONE call to the Claude CLI brain
  with a symbolic encoding of the current frame and the per-frame
  available_actions list.
- Parse Claude's chosen action_id. Validate it against available_actions.
  If invalid (or unparseable), fall back to the first available action and
  record the fallback so MLflow shows the smoke wiring still survived.
- Submit that single action, then stop.

No memory, no multi-step planning. The point is to prove the end-to-end
wiring (Arcade -> symbolic encoder -> Claude -> action -> Arcade) works.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from arcengine import FrameDataRaw, GameAction, GameState

from core.claude_brain import call_claude
from core.frame_encoder import SymbolicEncoding, encode_frame

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a tactical agent playing the ARC-AGI-3 game ls20.
You receive a compact symbolic description of one frame.

ALWAYS obey these rules:
1. You MUST choose exactly one action_id from the `available_actions` list provided.
2. NEVER choose an action_id that is not in `available_actions`.
3. Respond with ONLY a JSON object of the form
     {"action_id": <int>, "reasoning": "<one short sentence>"}
   No prose before or after.

Action conventions (template-only — may vary per game):
  1=Up   2=Down   3=Left   4=Right
  5=Interact   6=Click(x,y in 0..63)   7=Undo   0=Reset

For ls20 you should never need ACTION6 (no x,y), so prefer the basic
movement / interact actions when available.
"""


@dataclass
class SmokeRunResult:
    chose_action_id: int | None
    chose_action_valid: bool
    fallback_used: bool
    encoding_text: str
    claude_prompt: str
    claude_response_text: str
    claude_duration_s: float
    levels_completed: int
    final_state: str
    actions_taken: int


def _build_user_prompt(enc: SymbolicEncoding) -> str:
    return (
        "Current frame:\n"
        f"```\n{enc.to_prompt_text(max_objects=20)}\n```\n\n"
        "Pick exactly one action_id from `available_actions` "
        f"({enc.available_actions}). Reply as JSON only."
    )


def _parse_action_id(text: str) -> int | None:
    """Extract an integer action_id from Claude's reply."""
    # Try strict JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("action_id"), int):
            return int(obj["action_id"])
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object inside arbitrary text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("action_id"), int):
                return int(obj["action_id"])
        except json.JSONDecodeError:
            pass
    # Fall back to a bare integer in the response.
    for token in text.replace(",", " ").split():
        token = token.strip(".:;()[]{}'\"")
        if token.lstrip("-").isdigit():
            return int(token)
    return None


def run_smoke_episode(env: Any) -> SmokeRunResult:
    """Run a single smoke episode against an already-built EnvironmentWrapper.

    `env` must expose `.reset()` and `.step(action, data=..., reasoning=...)`
    matching arc_agi.wrapper.EnvironmentWrapper. The wrapper's constructor
    already issues the first RESET, so we go straight to one Claude call.
    """
    actions_taken = 0

    # The framework auto-resets on env construction. observation_space holds
    # the first FrameData. If for any reason we got NOT_PLAYED, issue RESET.
    frame: FrameDataRaw = env.observation_space
    if frame is None or frame.state is GameState.NOT_PLAYED:
        frame = env.reset()
        actions_taken += 1

    if frame is None:
        raise RuntimeError("Environment returned no frame after reset")

    enc = encode_frame(frame)
    user_prompt = _build_user_prompt(enc)

    logger.info("Calling Claude (one shot) — available_actions=%s", enc.available_actions)
    result = call_claude(
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        model="sonnet",
        timeout_s=120.0,
    )

    chose = _parse_action_id(result.text)
    fallback = False
    if chose is None or chose not in enc.available_actions:
        # Validate / fall back.
        logger.warning("Claude returned invalid action %r; falling back", chose)
        fallback = True
        chose = enc.available_actions[0] if enc.available_actions else 0

    valid = chose in enc.available_actions

    # Build the GameAction object and submit it.
    action = GameAction.from_id(chose) if hasattr(GameAction, "from_id") else _action_from_id(chose)
    data: dict[str, Any] = {}
    if action.is_complex():  # ACTION6 — needs x, y
        data["x"] = 32
        data["y"] = 32
    reasoning = {
        "source": "claude_smoke_agent",
        "fallback_used": fallback,
        "raw_text": result.text[:1000],
    }
    next_frame = env.step(action, data=data, reasoning=reasoning)
    actions_taken += 1

    final_state = (
        next_frame.state.name
        if next_frame is not None and hasattr(next_frame.state, "name")
        else "UNKNOWN"
    )
    levels = next_frame.levels_completed if next_frame is not None else 0

    return SmokeRunResult(
        chose_action_id=chose,
        chose_action_valid=valid,
        fallback_used=fallback,
        encoding_text=enc.to_prompt_text(max_objects=20),
        claude_prompt=user_prompt,
        claude_response_text=result.text,
        claude_duration_s=result.duration_s,
        levels_completed=int(levels),
        final_state=final_state,
        actions_taken=actions_taken,
    )


def _action_from_id(action_id: int) -> GameAction:
    """Map an int to a GameAction. arcengine exposes this as a helper on some
    versions; fall back to a manual lookup."""
    for a in GameAction:
        if a.value[0] == action_id:
            return a
    raise ValueError(f"Unknown action_id={action_id}")
