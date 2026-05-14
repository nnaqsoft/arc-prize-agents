"""Minimal MLflow wrapper for ARC-AGI-3 runs.

Tracking URI defaults to a local mlruns/ dir at the repo root; can be
overridden via MLFLOW_TRACKING_URI in the env.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"file:{REPO_ROOT / 'mlruns'}"
DEFAULT_EXPERIMENT = "arc-agi-3"


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> None:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


@contextmanager
def start_run(
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run, log params and tags up front, and yield it."""
    configure_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        if params:
            mlflow.log_params({k: _safe_param(v) for k, v in params.items()})
        yield run


def _safe_param(v: Any) -> str:
    # MLflow stringifies anyway, but we cap length so a giant prompt doesn't
    # blow up the params store.
    s = str(v)
    return s if len(s) <= 500 else (s[:497] + "...")
