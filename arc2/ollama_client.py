"""Tiny Ollama HTTP client.

We POST to /api/chat with optional format="json" for strict-JSON output.
Default host is OLLAMA_HOST env var or http://localhost:11434.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT_S = 600


@dataclass
class OllamaResult:
    text: str
    model: str
    duration_s: float
    raw: dict[str, Any]


class OllamaError(RuntimeError):
    pass


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = "qwen3:14b",
    host: str = DEFAULT_HOST,
    json_format: bool = False,
    temperature: float = 0.0,
    num_ctx: int = 8192,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> OllamaResult:
    """One non-streaming chat call.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        model: Ollama model tag (e.g. "qwen3:14b").
        json_format: if True, send `format="json"` to force strict-JSON output.
        temperature: 0 = greedy (default — we want deterministic reasoning).
        num_ctx: context window (Ollama default is 2048, often too small).
    """
    url = host.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    if json_format:
        payload["format"] = "json"

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    duration = time.time() - t0
    if resp.status_code != 200:
        raise OllamaError(f"Ollama {url} returned {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    msg = body.get("message") or {}
    text = msg.get("content", "")
    return OllamaResult(text=text, model=model, duration_s=duration, raw=body)
