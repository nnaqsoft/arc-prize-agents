#!/usr/bin/env python3
"""Render one frame from a recording as both ASCII and PNG.

Usage:
  scripts/render_frame.py <recording.jsonl> [--index N] [--png OUT.png]

A recording.jsonl is one JSON object per line. Each line that has a
top-level 'frame' field is a FrameData snapshot — 'frame' is a list of
grids (usually length 1), each grid is a 2D list of int color indices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 16-color palette used by ARC-AGI (each cell value 0..15 maps to a color).
# Approximated for terminal display; the exact RGB values used by the
# official renderer live in arc_agi/render.py.
ARC_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0 black
    (30, 147, 255),   # 1 blue
    (249, 60, 49),    # 2 red
    (79, 204, 48),    # 3 green
    (255, 220, 0),    # 4 yellow
    (153, 153, 153),  # 5 grey
    (229, 58, 163),   # 6 magenta
    (255, 133, 27),   # 7 orange
    (135, 216, 241),  # 8 cyan
    (146, 18, 49),    # 9 maroon
    (255, 255, 255),  # 10 white
    (96, 32, 168),    # 11 purple
    (96, 96, 96),     # 12 dark grey
    (32, 32, 32),     # 13 near-black
    (179, 179, 179),  # 14 light grey
    (224, 224, 224),  # 15 very light grey
]

# Single-char glyph per palette index for an ASCII rendering.
GLYPHS = ".#@%+*=-oXOQ&$wM"


def find_frame(path: Path, index: int) -> dict:
    """Return the index-th line that has a 'frame' field."""
    found = 0
    with path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Recording wraps each entry as {"timestamp": ..., "data": {...}}
            inner = obj.get("data", obj) if isinstance(obj.get("data"), dict) else obj
            if isinstance(inner.get("frame"), list):
                if found == index:
                    return inner
                found += 1
    raise IndexError(f"Recording has fewer than {index + 1} frames")


def render_ascii(grid: list[list[int]]) -> str:
    out_lines = []
    for row in grid:
        out_lines.append("".join(GLYPHS[v % len(GLYPHS)] for v in row))
    return "\n".join(out_lines)


def render_ascii_compact(grid: list[list[int]]) -> str:
    """2x1 compact rendering: each output char represents two input cells side-by-side
    via a unicode half-block — but here we just use the same glyph twice so terminals
    without unicode still work. Skipping; full 64x64 is fine in a wide terminal."""
    return render_ascii(grid)


def render_png(grid: list[list[int]], out_path: Path, scale: int = 8) -> None:
    """Save the grid as a PNG using the ARC palette, upscaled `scale`x."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed — skipping PNG output.", file=sys.stderr)
        return
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ARC_PALETTE[grid[y][x] % len(ARC_PALETTE)]
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    img.save(out_path)
    print(f"PNG written to {out_path} ({w}x{h} scaled {scale}x)", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("recording")
    p.add_argument("--index", type=int, default=0, help="Which frame to render (0-based)")
    p.add_argument("--png", default=None, help="Optional PNG output path")
    p.add_argument("--no-ascii", action="store_true")
    args = p.parse_args()

    path = Path(args.recording)
    obj = find_frame(path, args.index)
    frame_list = obj["frame"]
    if not frame_list:
        print("frame is empty", file=sys.stderr)
        return 1
    grid = frame_list[0]  # most frames have a single grid

    print(f"# Recording: {path.name}")
    print(f"# Frame index: {args.index}")
    print(f"# Game state: {obj.get('state')}")
    print(f"# Levels completed: {obj.get('levels_completed')}")
    print(f"# Available actions: {obj.get('available_actions')}")
    print(f"# Grid shape: {len(grid)} x {len(grid[0])}")
    palette_vals = sorted({v for row in grid for v in row})
    print(f"# Distinct color indices in this frame: {palette_vals}")
    print()
    if not args.no_ascii:
        print(render_ascii(grid))
    if args.png:
        render_png(grid, Path(args.png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
