"""Token-efficient encoder for ARC-AGI-2 puzzles.

Layout (per user spec — no prose, palette legend always present):

    dims: train inputs H×W -> outputs H×W  (one line, summary across pairs)
    palette: 0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=magenta 7=orange 8=azure 9=maroon

    train:
    [1]
    in:
    <row of space-separated digits>
    ...
    out:
    <row of space-separated digits>
    ...

    [2]
    ...

    test:
    in:
    <row of space-separated digits>
    ...

That is the entire prompt body. No headers like "Here is the puzzle:" or
"Please respond with...". Those go in the system prompt of the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Standard ARC-AGI palette (Chollet's original notebook).
ARC_PALETTE: dict[int, str] = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "maroon",
}


@dataclass
class ArcPuzzle:
    puzzle_id: str
    train_pairs: list[tuple[list[list[int]], list[list[int]]]]
    test_inputs: list[list[list[int]]]
    test_outputs: list[list[list[int]] | None]  # ground-truth, may be None for unseen

    @classmethod
    def from_json(cls, puzzle_id: str, data: dict) -> "ArcPuzzle":
        train_pairs = [
            (ex["input"], ex["output"]) for ex in data.get("train", [])
        ]
        test_inputs = [ex["input"] for ex in data.get("test", [])]
        test_outputs = [ex.get("output") for ex in data.get("test", [])]
        return cls(
            puzzle_id=puzzle_id,
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
        )


def _grid_lines(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def _dims_summary(pairs: Iterable[tuple[list[list[int]], list[list[int]]]]) -> str:
    parts: list[str] = []
    for i, (a, b) in enumerate(pairs, start=1):
        ih, iw = len(a), (len(a[0]) if a else 0)
        oh, ow = len(b), (len(b[0]) if b else 0)
        parts.append(f"{ih}x{iw}->{oh}x{ow}")
    return " ".join(parts)


def _palette_line() -> str:
    return "palette: " + " ".join(f"{k}={v}" for k, v in ARC_PALETTE.items())


def encode_puzzle(
    puzzle: ArcPuzzle,
    *,
    test_index: int = 0,
    include_train_dims_summary: bool = True,
) -> str:
    """Encode a puzzle for one specific test input (default: index 0).

    Always emits palette legend. No verbose prose.
    """
    lines: list[str] = []
    if include_train_dims_summary:
        lines.append("dims: " + _dims_summary(puzzle.train_pairs))
    lines.append(_palette_line())
    lines.append("")
    lines.append("train:")
    for i, (inp, out) in enumerate(puzzle.train_pairs, start=1):
        lines.append(f"[{i}]")
        lines.append("in:")
        lines.append(_grid_lines(inp))
        lines.append("out:")
        lines.append(_grid_lines(out))
        lines.append("")
    lines.append("test:")
    lines.append("in:")
    lines.append(_grid_lines(puzzle.test_inputs[test_index]))
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough estimate — Qwen tokenizer averages ~3.5 chars/token on grid digits."""
    return max(1, len(text) // 3)


def validate_grid(g: object) -> tuple[bool, str]:
    """Check that g is a non-empty list[list[int]] with values in 0..9."""
    if not isinstance(g, list) or not g:
        return False, "grid is not a non-empty list"
    if not all(isinstance(row, list) and row for row in g):
        return False, "grid contains an empty or non-list row"
    w = len(g[0])
    if not all(len(row) == w for row in g):
        return False, "grid is not rectangular"
    for row in g:
        for v in row:
            if not isinstance(v, int) or not (0 <= v <= 9):
                return False, f"cell {v!r} not an int in 0..9"
    return True, ""


def render_grid_ascii(g: list[list[int]]) -> str:
    return _grid_lines(g)
