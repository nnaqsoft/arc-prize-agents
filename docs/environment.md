# ARC-AGI-3 Environment — Technical Reference

> This document is the agent's source-of-truth for what the environment looks
> like and how scoring works. All claims are cited to either the installed
> `arcengine` / `arc_agi` source or to the official ARC Prize blog.

## 1. Frame format

`FrameData` is a Pydantic model (`arcengine/enums.py`):

```python
class FrameData(BaseModel):
    game_id: str = ""
    frame: list[list[list[int]]] = []        # list of 2D grids, each 64x64 ints 0..15
    state: GameState = GameState.NOT_PLAYED  # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int = Field(0, ge=0, le=254)
    win_levels: int = Field(0, ge=0, le=254)
    action_input: ActionInput                # echo of last action
    guid: Optional[str] = None               # session GUID
    full_reset: bool = False
    available_actions: list[int] = []        # subset of {0..7}
```

- Grid is **64×64**, dtype int8 internally (`MAX_DIMENSION=64` in `arcengine/camera.py`).
- `frame` is a *list* of grids — single-grid for most games (ls20 confirmed), but the
  schema permits multi-view games.
- **16-color palette** (`arc_agi/rendering.py` `COLOR_MAP`):

| Value | Hex | Name |
|------:|:----|:-----|
| 0 | `#FFFFFFFF` | White |
| 1 | `#CCCCCCFF` | Off-white |
| 2 | `#999999FF` | Neutral light |
| 3 | `#666666FF` | Neutral |
| 4 | `#333333FF` | Off-black |
| 5 | `#000000FF` | Black (default background) |
| 6 | `#E53AA3FF` | Magenta |
| 7 | `#FF7BCCFF` | Magenta light |
| 8 | `#F93C31FF` | Red |
| 9 | `#1E93FFFF` | Blue |
| 10 | `#88D8F1FF` | Blue light |
| 11 | `#FFDC00FF` | Yellow |
| 12 | `#FF851BFF` | Orange |
| 13 | `#921231FF` | Maroon |
| 14 | `#4FCC30FF` | Green |
| 15 | `#A356D6FF` | Purple |

- The palette is game-agnostic at the engine layer. Per-game tile meanings (e.g.
  in ls20: walls=10, floor=8) are only surfaced in template prompts.

## 2. Actions

```python
class GameAction(Enum):
    RESET   = (0, SimpleAction)
    ACTION1 = (1, SimpleAction)
    ACTION2 = (2, SimpleAction)
    ACTION3 = (3, SimpleAction)
    ACTION4 = (4, SimpleAction)
    ACTION5 = (5, SimpleAction)
    ACTION6 = (6, ComplexAction)   # ONLY complex: needs x,y in [0,63]
    ACTION7 = (7, SimpleAction)
```

- `available_actions` per frame restricts the legal subset; the agent **must**
  consult it every turn.
- Conventional template meanings (from `agents/templates/multimodal.py` —
  templates only, not engine guarantees): 1=Up, 2=Down, 3=Left, 4=Right,
  5=Interact, 6=Click(x,y), 7=Undo.
- A per-step `reasoning` blob can be attached. Hard limit:
  `MAX_REASONING_BYTES = 16 * 1024` (16 KB) — `arcengine/enums.py`.
- RESET semantics:
  - First RESET starts a new "play" with a fresh `guid`.
  - Mid-game RESETs count as `resets` (scorecard tracks).
  - After WIN/GAME_OVER, RESET starts a fresh playthrough; `full_reset=True`
    if the engine fully reinitialized.

## 3. Scoring

Scoring code: `arc_agi/scorecard.py`.

Per level:
```python
score = min((baseline_actions / actions_taken) * 100, 100.0)
```
- Failing to complete a level → 0 for that level.
- **Score = baseline-action efficiency, capped at 100.**

Per environment (one game):
```python
env_score = mean(level_scores)
```

Leaderboard metric:
- Mean of `env_score` across all 25 games in a card (per ARC Prize preview blog).
- Expressed as a percentage.

`EnvironmentScore` fields: `id, guid, score, levels_completed, actions, resets,
state, completed, level_scores, level_actions, level_baseline_actions,
number_of_levels, number_of_environments, message`.

Win definition: `levels_completed == number_of_levels` AND `state == WIN`.

## 4. Session flow

Online flow (used for leaderboard submissions):

1. `Arcade()` — auto-fetches anonymous key if not provided.
2. `arcade.open_scorecard(...)` → POST `/api/scorecard/open` → `scorecard_id`.
3. `arcade.list_games()` → GET `/api/games` → 25 game defs.
4. Per game: `env = arcade.make(game_id, scorecard_id=...)` — RESETs implicitly,
   returns first `FrameData` with fresh `guid`.
5. Loop: `env.step(action, data={x,y}, reasoning={...})` → POST
   `/api/cmd/ACTION{N}` → next `FrameData`. Until WIN / GAME_OVER /
   local `MAX_ACTIONS` cap (default 80, set in `agents/agent.py`).
6. `arcade.close_scorecard(scorecard_id)` — finalizes.

- HTTP timeout: 10 s per step (hard-coded in `remote_wrapper.py`).
- Rate limits: **not documented** anywhere in the toolkit or docs site.
- Modes: `NORMAL` (local engine + remote scorecard), `ONLINE` (everything remote),
  `OFFLINE` (purely local, no leaderboard).

## 5. Recording format

JSONL, one event per line, written under
`{recordings_dir}/{scorecard_id}/{game_id}-{guid}.jsonl`:

```json
{
  "timestamp": "...",
  "data": {
    "game_id": "ls20-9607627b",
    "frame": [[[ /* 64x64 ints */ ]]],
    "state": "NOT_FINISHED",
    "levels_completed": 0, "win_levels": 7,
    "action_input": {"id": 0, "data": {}, "reasoning": null},
    "guid": "...", "full_reset": false,
    "available_actions": [1,2,3,4]
  }
}
```

## 6. Context-management implications

- A raw 64×64 grid dumped as JSON is roughly **3 000–4 500 tokens**.
- A 25-game × 80-action card with naive observations would burn **6–9 M
  tokens** before any reasoning — economically unworkable.

### Preview-competition results (signal)

From [arcprize.org/blog/arc-agi-3-preview](https://arcprize.org/blog/arc-agi-3-preview):

| Team | Score | Approach |
|------|------:|----------|
| StochasticGoose | 12.58% | Pure CNN policy. No LLM. |
| Fluxonian       | 8.04%  | LLM generates a DSL program; LLM rarely called per-step. |
| Blind Squirrel  | 6.71%  | State-graph search + ResNet18 vision. |
| (Anthropic, Jul 2025) frontier LLMs naive | 0.51% | Whole-frame LLM-in-the-loop. |
| Humans | ~100% | — |

**Takeaway: every competitive entry minimizes per-step LLM calls and uses
external memory or symbolic abstractions.**

### Architecture implications for our agent

1. Never ship raw grids as text. Compress to symbolic (object lists, palette
   legend + run-length / patch description) or use multimodal image input.
2. External memory: persist `frame_hash → outcome` outside the LLM prompt.
3. Hierarchical control: planner LLM picks a subgoal every N steps; tactical
   loop (deterministic / small policy) executes within.
4. Cache per-game action semantics in a system prompt once per game.
5. Truncate frame history to last K (5–10) frames; summarize older trajectories.
6. Use the 16 KB `reasoning` cap as a forcing function for compactness.

## Sources

All file paths below are under
`/home/keehar/arc-agi-3-agent/third_party/ARC-AGI-3-Agents/`:

- `.venv/lib/python3.12/site-packages/arcengine/enums.py` — `FrameData`, `GameAction`, `GameState`, `MAX_REASONING_BYTES`
- `.venv/lib/python3.12/site-packages/arcengine/camera.py` — `MAX_DIMENSION=64`, default background color
- `.venv/lib/python3.12/site-packages/arc_agi/scorecard.py` — scoring formula
- `.venv/lib/python3.12/site-packages/arc_agi/base.py` — `Arcade`, `OperationMode`, session flow
- `.venv/lib/python3.12/site-packages/arc_agi/wrapper.py` — recording format
- `.venv/lib/python3.12/site-packages/arc_agi/remote_wrapper.py` — REST endpoints
- `.venv/lib/python3.12/site-packages/arc_agi/rendering.py` — palette
- `agents/agent.py` — `MAX_ACTIONS=80`
- `agents/templates/{random_agent.py, multimodal.py, llm_agents.py, reasoning_agent.py, langgraph_thinking/agent.py}` — reference compression strategies
- External: <https://arcprize.org/blog/arc-agi-3-preview>, <https://arcprize.org/arc-agi/3>
