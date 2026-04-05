from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.config import AppConfig
from src.logging.session import STATE_SUFFIX, TRACE_SUFFIX
from src.models import SessionState


ArtifactType = Literal[
    "log",
    "state",
    "trace",
    "source",
    "context",
    "dashboard_spec",
    "figures",
    "transformed_dataset",
]

ARTIFACT_CONTENT_TYPES: dict[ArtifactType, str] = {
    "log": "text/markdown; charset=utf-8",
    "state": "application/json",
    "trace": "application/json",
    "source": "application/octet-stream",
    "context": "text/plain; charset=utf-8",
    "dashboard_spec": "application/json",
    "figures": "application/json",
    "transformed_dataset": "application/octet-stream",
}


def log_path(config: AppConfig, session_id: str) -> Path:
    return config.logs_dir / f"{session_id}.log"


def state_path(config: AppConfig, session_id: str) -> Path:
    return config.logs_dir / f"{session_id}{STATE_SUFFIX}"


def trace_path(config: AppConfig, session_id: str) -> Path:
    return config.logs_dir / f"{session_id}{TRACE_SUFFIX}"


def dashboard_spec_path(config: AppConfig, session_id: str) -> Path:
    return config.outputs_dir / f"dashboard_spec_{session_id}.json"


def figures_path(config: AppConfig, session_id: str) -> Path:
    return config.outputs_dir / f"figures_{session_id}.json"


def context_path(config: AppConfig, session_id: str) -> Path:
    return config.outputs_dir / f"context_{session_id}.txt"


def transformed_dataset_path(config: AppConfig, session_id: str) -> Path:
    return config.outputs_dir / f"transformed_{session_id}.parquet"


def source_path(config: AppConfig, session_id: str, suffix: str) -> Path:
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return config.outputs_dir / f"source_{session_id}{normalized_suffix}"


def discover_source_path(config: AppConfig, session_id: str, state: SessionState | None = None) -> Path | None:
    if state and state.data_path:
        candidate = Path(state.data_path)
        if candidate.exists():
            return candidate
    matches = sorted(config.outputs_dir.glob(f"source_{session_id}.*"))
    return matches[0] if matches else None


def resolve_artifact_path(
    config: AppConfig,
    session_id: str,
    artifact_type: ArtifactType,
    state: SessionState | None = None,
) -> Path | None:
    if artifact_type == "log":
        path = log_path(config, session_id)
        return path if path.exists() else None
    if artifact_type == "state":
        path = state_path(config, session_id)
        return path if path.exists() else None
    if artifact_type == "trace":
        path = trace_path(config, session_id)
        return path if path.exists() else None
    if artifact_type == "source":
        return discover_source_path(config, session_id, state=state)
    if artifact_type == "context":
        if state and state.description_path:
            candidate = Path(state.description_path)
            if candidate.exists():
                return candidate
        path = context_path(config, session_id)
        return path if path.exists() else None
    if artifact_type == "dashboard_spec":
        path = dashboard_spec_path(config, session_id)
        return path if path.exists() else None
    if artifact_type == "figures":
        path = figures_path(config, session_id)
        return path if path.exists() else None
    if artifact_type == "transformed_dataset":
        if state and state.transformed_dataset:
            candidate = Path(state.transformed_dataset)
            if candidate.exists():
                return candidate
        path = transformed_dataset_path(config, session_id)
        return path if path.exists() else None
    return None
