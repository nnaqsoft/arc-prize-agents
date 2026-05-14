"""Sanity tests for the ARC-AGI-2 encoder."""
from __future__ import annotations

from arc2.encoder import ARC_PALETTE, ArcPuzzle, encode_puzzle, validate_grid


def _small_puzzle() -> ArcPuzzle:
    return ArcPuzzle(
        puzzle_id="test",
        train_pairs=[
            ([[1, 2], [3, 4]], [[4, 3], [2, 1]]),
            ([[5, 0], [0, 5]], [[5, 0], [0, 5]]),
        ],
        test_inputs=[[[6, 7], [8, 9]]],
        test_outputs=[[[9, 8], [7, 6]]],
    )


def test_encoding_contains_palette_legend() -> None:
    txt = encode_puzzle(_small_puzzle())
    assert "palette:" in txt
    for k, name in ARC_PALETTE.items():
        assert f"{k}={name}" in txt


def test_encoding_section_order() -> None:
    txt = encode_puzzle(_small_puzzle())
    dims_idx = txt.find("dims:")
    pal_idx = txt.find("palette:")
    train_idx = txt.find("train:")
    test_idx = txt.find("test:")
    assert 0 <= dims_idx < pal_idx < train_idx < test_idx


def test_encoding_includes_dims_for_each_pair() -> None:
    txt = encode_puzzle(_small_puzzle())
    assert "2x2->2x2" in txt


def test_encoding_includes_test_input_only_no_output() -> None:
    txt = encode_puzzle(_small_puzzle())
    # The test output (target) must NOT appear in the encoding.
    assert "9 8" not in txt
    assert "7 6" not in txt
    # The test input MUST appear.
    assert "6 7" in txt
    assert "8 9" in txt


def test_validate_grid_accepts_valid() -> None:
    ok, err = validate_grid([[0, 1, 2], [3, 4, 5]])
    assert ok and err == ""


def test_validate_grid_rejects_bad_value() -> None:
    ok, err = validate_grid([[0, 1, 10]])
    assert not ok and "0..9" in err


def test_validate_grid_rejects_non_rectangular() -> None:
    ok, err = validate_grid([[0, 1], [2]])
    assert not ok and "rectangular" in err


def test_arcpuzzle_from_json() -> None:
    p = ArcPuzzle.from_json(
        "p1",
        {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        },
    )
    assert p.puzzle_id == "p1"
    assert len(p.train_pairs) == 1
    assert p.train_pairs[0] == ([[1]], [[2]])
    assert p.test_inputs == [[[3]]]
    assert p.test_outputs == [[[4]]]
