"""Symbolic frame encoder for ARC-AGI-3.

Goal: turn a 64x64 int grid into a compact textual description that an LLM
can reason over cheaply. The encoder is intentionally simple — it groups
contiguous same-colored cells into "objects" via flood fill, captures
bounding boxes / sizes / centroids, and ALWAYS includes the per-frame
`available_actions` list. Downstream agents are required by contract to
pick actions only from that list.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

# Palette names mirror arc_agi/rendering.py COLOR_MAP.
PALETTE_NAMES: dict[int, str] = {
    0: "White",
    1: "Off-white",
    2: "Neutral-light",
    3: "Neutral",
    4: "Off-black",
    5: "Black (default bg)",
    6: "Magenta",
    7: "Magenta-light",
    8: "Red",
    9: "Blue",
    10: "Blue-light",
    11: "Yellow",
    12: "Orange",
    13: "Maroon",
    14: "Green",
    15: "Purple",
}

DEFAULT_BACKGROUND = 5  # arcengine/camera.py


@dataclass
class SymbolicObject:
    color: int
    color_name: str
    area: int
    bbox: tuple[int, int, int, int]   # (top, left, bottom, right) inclusive
    centroid: tuple[float, float]     # (y, x)

    def to_line(self) -> str:
        t, l, b, r = self.bbox
        cy, cx = self.centroid
        return (
            f"  color={self.color:2d} ({self.color_name:<20}) "
            f"bbox=(t={t:2d},l={l:2d},b={b:2d},r={r:2d}) "
            f"area={self.area:4d} centroid=(y={cy:5.1f},x={cx:5.1f})"
        )


@dataclass
class SymbolicEncoding:
    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    frame_shape: tuple[int, int]
    distinct_colors: list[int]
    background_color: int
    objects: list[SymbolicObject] = field(default_factory=list)
    truncated_objects: int = 0  # number dropped if we cap the list

    def to_prompt_text(self, max_objects: int = 30) -> str:
        """Render as a compact text block for an LLM prompt.

        Always renders the available_actions line near the top and explicitly
        instructs the model that no other action is legal.
        """
        h, w = self.frame_shape
        lines: list[str] = []
        lines.append(f"ARC-AGI-3 frame  game_id={self.game_id}")
        lines.append(f"state={self.state}  levels={self.levels_completed}/{self.win_levels}")
        lines.append(
            "available_actions=" + repr(self.available_actions)
            + "   (you MUST pick exactly one action_id from this list)"
        )
        lines.append(f"frame_shape={h}x{w}  background_color={self.background_color}")
        lines.append(f"distinct_colors={self.distinct_colors}")
        lines.append("objects (largest first):")
        objs = self.objects[:max_objects]
        if not objs:
            lines.append("  (no non-background objects)")
        for obj in objs:
            lines.append(obj.to_line())
        if self.truncated_objects > 0:
            lines.append(f"  ... +{self.truncated_objects} smaller objects omitted")
        return "\n".join(lines)


def _flood_fill_components(
    grid: list[list[int]],
    background: int,
) -> list[list[tuple[int, int]]]:
    """Return a list of components; each component is a list of (y, x) coords
    of same-colored, 4-connected cells. Background-colored cells are skipped.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    visited = [[False] * w for _ in range(h)]
    components: list[list[tuple[int, int]]] = []

    for y0 in range(h):
        for x0 in range(w):
            if visited[y0][x0] or grid[y0][x0] == background:
                continue
            color = grid[y0][x0]
            queue = deque([(y0, x0)])
            visited[y0][x0] = True
            comp: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and not visited[ny][nx]
                        and grid[ny][nx] == color
                    ):
                        visited[ny][nx] = True
                        queue.append((ny, nx))
            components.append(comp)
    return components


def encode_frame(frame_data: Any, max_objects: int = 30) -> SymbolicEncoding:
    """Encode a FrameData (or dict shaped like one) into a SymbolicEncoding.

    The encoder always populates `available_actions`. If the field is empty
    we still emit it explicitly — the agent contract is "never act outside
    this list", and an empty list means the agent can only call RESET.
    """
    # Accept FrameData (Pydantic), FrameDataRaw (frame as ndarray list), and plain dicts.
    if hasattr(frame_data, "model_dump"):
        data = frame_data.model_dump()
        # FrameDataRaw stores the frame on a PrivateAttr; pull it explicitly.
        raw_frame_attr = getattr(frame_data, "frame", None)
        if raw_frame_attr is not None and not data.get("frame"):
            data["frame"] = [
                arr.tolist() if hasattr(arr, "tolist") else arr for arr in raw_frame_attr
            ]
    elif isinstance(frame_data, dict):
        data = dict(frame_data)
    else:
        raise TypeError(f"Cannot encode frame of type {type(frame_data)!r}")

    raw_frame = data.get("frame", [])
    if raw_frame and hasattr(raw_frame[0], "tolist"):
        raw_frame = [arr.tolist() for arr in raw_frame]
    grid = raw_frame[0] if raw_frame else [[DEFAULT_BACKGROUND]]
    h = len(grid)
    w = len(grid[0]) if h else 0

    state = data.get("state")
    state_name = state.name if hasattr(state, "name") else str(state)

    distinct = sorted({v for row in grid for v in row})
    bg = DEFAULT_BACKGROUND if DEFAULT_BACKGROUND in distinct else (distinct[0] if distinct else DEFAULT_BACKGROUND)
    # Heuristic refinement: if a non-default color dominates >50% of cells,
    # treat THAT as background. This keeps the "objects" list focused on
    # foreground features instead of swamping it with floor tiles.
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    total = h * w
    if total:
        dominant_color, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        if dominant_count / total > 0.5:
            bg = dominant_color

    components = _flood_fill_components(grid, background=bg)
    objs: list[SymbolicObject] = []
    for comp in components:
        ys = [y for y, _ in comp]
        xs = [x for _, x in comp]
        color = grid[comp[0][0]][comp[0][1]]
        area = len(comp)
        bbox = (min(ys), min(xs), max(ys), max(xs))
        centroid = (sum(ys) / area, sum(xs) / area)
        objs.append(
            SymbolicObject(
                color=color,
                color_name=PALETTE_NAMES.get(color, f"unknown_{color}"),
                area=area,
                bbox=bbox,
                centroid=centroid,
            )
        )
    objs.sort(key=lambda o: o.area, reverse=True)

    truncated = max(0, len(objs) - max_objects)

    return SymbolicEncoding(
        game_id=data.get("game_id", ""),
        state=state_name,
        levels_completed=int(data.get("levels_completed", 0)),
        win_levels=int(data.get("win_levels", 0)),
        available_actions=list(data.get("available_actions", []) or []),
        frame_shape=(h, w),
        distinct_colors=distinct,
        background_color=bg,
        objects=objs,
        truncated_objects=truncated,
    )
