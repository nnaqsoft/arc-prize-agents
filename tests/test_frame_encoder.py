"""Sanity tests for the symbolic frame encoder."""
from __future__ import annotations

from core.frame_encoder import encode_frame


def _make_grid_with_object() -> list[list[int]]:
    """A 64x64 grid with a 5x5 yellow (11) square on a default-background (5) field."""
    g = [[5 for _ in range(64)] for _ in range(64)]
    for y in range(20, 25):
        for x in range(30, 35):
            g[y][x] = 11
    return g


def test_encoder_includes_available_actions() -> None:
    grid = _make_grid_with_object()
    enc = encode_frame(
        {
            "game_id": "test-game",
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 7,
            "available_actions": [1, 2, 3, 4],
            "frame": [grid],
        }
    )
    text = enc.to_prompt_text()
    assert "available_actions=[1, 2, 3, 4]" in text
    assert "you MUST pick exactly one action_id" in text


def test_encoder_finds_object() -> None:
    grid = _make_grid_with_object()
    enc = encode_frame(
        {
            "game_id": "test-game",
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 7,
            "available_actions": [1],
            "frame": [grid],
        }
    )
    assert len(enc.objects) == 1
    obj = enc.objects[0]
    assert obj.color == 11
    assert obj.area == 25
    assert obj.bbox == (20, 30, 24, 34)


def test_encoder_empty_available_actions_still_renders() -> None:
    grid = [[5 for _ in range(64)] for _ in range(64)]
    enc = encode_frame(
        {
            "game_id": "test-game",
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 1,
            "available_actions": [],
            "frame": [grid],
        }
    )
    text = enc.to_prompt_text()
    assert "available_actions=[]" in text
