"""Wrapper around the `claude` CLI used as the agent's LLM brain.

We invoke `claude -p ... --bare --output-format json --model sonnet` as a
subprocess so the agent uses the user's Claude subscription rather than
the Anthropic API directly. `--bare` skips hooks, auto-memory, CLAUDE.md
discovery and other Claude Code conveniences that would leak unrelated
context into our agent.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "sonnet"


@dataclass
class ClaudeCallResult:
    raw_stdout: str
    parsed: dict[str, Any] | None
    text: str
    exit_code: int
    duration_s: float


class ClaudeBrainError(RuntimeError):
    pass


def call_claude(
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    json_schema: dict[str, Any] | None = None,
    cwd: str | Path | None = None,
    timeout_s: float = 120.0,
) -> ClaudeCallResult:
    """Invoke claude CLI in --print mode and return both the parsed envelope
    and the inner text.

    Args:
        user_prompt: the prompt to send.
        system_prompt: appended to the default system prompt via --append-system-prompt.
        model: 'sonnet', 'opus', 'haiku', or a full model id.
        json_schema: if given, passed to --json-schema for structured output.
        cwd: working directory for the subprocess (a clean tmpdir is fine).
        timeout_s: hard subprocess timeout.

    Raises:
        ClaudeBrainError: on non-zero exit or unparseable output.
    """
    import time

    args: list[str] = [
        "claude",
        "--print",
        "--output-format", "json",
        "--model", model,
        "--no-session-persistence",
        # Skip CLAUDE.md auto-discovery, hooks, MCP, and other Claude Code
        # features so the agent gets a clean context, but DO NOT pass --bare
        # — that disables OAuth/keychain, which is how the subscription
        # login is read.
        "--setting-sources", "user",
        "--disable-slash-commands",
    ]
    if system_prompt:
        args.extend(["--append-system-prompt", system_prompt])
    if json_schema is not None:
        args.extend(["--json-schema", json.dumps(json_schema)])
    args.append(user_prompt)

    t0 = time.time()
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
        timeout=timeout_s,
    )
    duration = time.time() - t0

    if proc.returncode != 0:
        raise ClaudeBrainError(
            f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )

    raw = proc.stdout
    parsed: dict[str, Any] | None
    text: str
    try:
        parsed = json.loads(raw)
        # --output-format json wraps the response as
        # {"type": "result", "subtype": "success", "result": "...assistant text..."}
        # or similar; the agent text lives under 'result'.
        text = parsed.get("result") or parsed.get("text") or ""
        if not text:
            # Some versions nest differently; fall back to the whole envelope.
            text = json.dumps(parsed)
    except json.JSONDecodeError:
        parsed = None
        text = raw

    return ClaudeCallResult(
        raw_stdout=raw,
        parsed=parsed,
        text=text.strip(),
        exit_code=proc.returncode,
        duration_s=duration,
    )
