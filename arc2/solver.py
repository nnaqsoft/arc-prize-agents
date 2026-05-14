"""Single-puzzle ARC-AGI-2 solver using a local Ollama model.

Spec:
- Take one ArcPuzzle, encode it via arc2.encoder.encode_puzzle.
- One Ollama chat call with format="json" (strict JSON output).
- Parse the JSON; expect {"output": [[...],...], "reasoning": "..."}.
- Validate the output grid (0..9 ints, rectangular, non-empty).
- Compare to ground truth (if known): exact element-wise equality.
- Return a SolveResult; the caller (run_smoke) is responsible for MLflow.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .encoder import (
    ArcPuzzle,
    encode_puzzle,
    estimate_tokens,
    render_grid_ascii,
    validate_grid,
)
from .ollama_client import OllamaResult, chat

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You solve ARC-AGI-2 puzzles. Each puzzle gives you 2 or more "
    "(input grid, output grid) training pairs and ONE test input grid. "
    "Infer the transformation rule from the training pairs, then apply it "
    "to the test input. Reply with strict JSON ONLY, no markdown, no prose, "
    "matching this schema: "
    '{"output": [[int, ...], ...], "reasoning": "<one short sentence>"}. '
    "Output cells are integers 0..9. The output grid may be a different "
    "shape than the input — match the pattern shown in the training pairs."
)


@dataclass
class SolveResult:
    puzzle_id: str
    test_index: int
    model: str
    encoding_text: str
    prompt_tokens_estimate: int
    user_prompt: str
    raw_response_text: str
    parsed_output: list[list[int]] | None
    parsed_reasoning: str
    parse_error: str
    valid_grid: bool
    grid_validation_error: str
    expected_output: list[list[int]] | None
    correct: bool
    shape_match: bool
    duration_s: float


def _try_parse(text: str) -> tuple[dict | None, str]:
    """Parse the model output as JSON, tolerating leading/trailing junk."""
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    # Try to extract the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), ""
        except json.JSONDecodeError as e:
            return None, f"JSONDecodeError after slice: {e}"
    return None, "no JSON object found"


def solve_puzzle(
    puzzle: ArcPuzzle,
    *,
    test_index: int = 0,
    model: str = "qwen3:14b",
    num_ctx: int = 8192,
) -> SolveResult:
    encoding = encode_puzzle(puzzle, test_index=test_index)
    user_prompt = encoding
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        "Solving %s test[%d] with %s — encoding ~%d tokens",
        puzzle.puzzle_id,
        test_index,
        model,
        estimate_tokens(encoding),
    )
    res: OllamaResult = chat(
        messages,
        model=model,
        json_format=True,
        temperature=0.0,
        num_ctx=num_ctx,
    )

    parsed, parse_err = _try_parse(res.text)
    parsed_output: list[list[int]] | None = None
    parsed_reasoning = ""
    valid = False
    valid_err = ""
    if parsed is not None and isinstance(parsed, dict):
        candidate = parsed.get("output")
        parsed_reasoning = str(parsed.get("reasoning", ""))[:500]
        ok, err = validate_grid(candidate)
        if ok:
            parsed_output = candidate  # type: ignore[assignment]
            valid = True
        else:
            valid_err = err
    else:
        valid_err = "parse failed"

    expected = puzzle.test_outputs[test_index] if test_index < len(puzzle.test_outputs) else None
    correct = False
    shape_match = False
    if parsed_output is not None and expected is not None:
        shape_match = (
            len(parsed_output) == len(expected)
            and all(len(r) == len(expected[0]) for r in parsed_output)
        )
        correct = parsed_output == expected

    return SolveResult(
        puzzle_id=puzzle.puzzle_id,
        test_index=test_index,
        model=model,
        encoding_text=encoding,
        prompt_tokens_estimate=estimate_tokens(encoding),
        user_prompt=user_prompt,
        raw_response_text=res.text,
        parsed_output=parsed_output,
        parsed_reasoning=parsed_reasoning,
        parse_error=parse_err,
        valid_grid=valid,
        grid_validation_error=valid_err,
        expected_output=expected,
        correct=correct,
        shape_match=shape_match,
        duration_s=res.duration_s,
    )


def render_solve_summary(r: SolveResult) -> str:
    parts: list[str] = []
    parts.append(f"puzzle: {r.puzzle_id}  test[{r.test_index}]  model: {r.model}")
    parts.append(f"duration_s: {r.duration_s:.2f}   prompt_tokens_est: {r.prompt_tokens_estimate}")
    parts.append(f"valid_grid: {r.valid_grid}   shape_match: {r.shape_match}   correct: {r.correct}")
    if r.expected_output is not None:
        parts.append("expected:")
        parts.append(render_grid_ascii(r.expected_output))
    if r.parsed_output is not None:
        parts.append("predicted:")
        parts.append(render_grid_ascii(r.parsed_output))
    if not r.valid_grid:
        parts.append(f"validation_error: {r.grid_validation_error}")
    if r.parse_error:
        parts.append(f"parse_error: {r.parse_error}")
    return "\n".join(parts)
