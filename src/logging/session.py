from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.models import ExecutionTrace, SessionState


SPEC_START = "BEGIN_DASHBOARD_SPEC"
SPEC_END = "END_DASHBOARD_SPEC"
STATE_SUFFIX = ".state.json"
TRACE_SUFFIX = ".trace.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def make_session_id(now: datetime | None = None) -> str:
    timestamp = now or datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


@dataclass
class SessionLogger:
    path: Path

    def write_line(self, line: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    def section(self, title: str) -> None:
        self.write_line(f"## {title}")

    def log_kv(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            self.write_line(f"- {key}: {value}")

    def log_block(self, title: str, content: str) -> None:
        self.write_line(f"### {title}")
        self.write_line("```")
        self.write_line(content)
        self.write_line("```")

    def log_json(self, title: str, payload: dict[str, Any]) -> None:
        self.write_line(f"### {title}")
        self.write_line("```json")
        self.write_line(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        self.write_line("```")

    def log_tool_call(self, index: int, tool_name: str, reasoning: str, params: dict[str, Any]) -> None:
        self.write_line(f"### Tool Call {index}: {tool_name}")
        if reasoning:
            self.write_line(f"- reasoning: {reasoning}")
        self.write_line("```json")
        self.write_line(json.dumps(params, indent=2, sort_keys=True, default=_json_default))
        self.write_line("```")

    def log_dashboard_spec(self, payload: dict[str, Any]) -> None:
        self.write_line(SPEC_START)
        self.write_line(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        self.write_line(SPEC_END)

    def artifact_path(self, suffix: str) -> Path:
        stem = self.path.stem
        return self.path.with_name(f"{stem}{suffix}")

    def write_artifact(self, suffix: str, payload: dict[str, Any]) -> Path:
        artifact_path = self.artifact_path(suffix)
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return artifact_path

    def log_session_state(self, state: SessionState) -> Path:
        artifact_path = self.write_artifact(STATE_SUFFIX, state.model_dump(mode="python"))
        self.log_kv({"session_state_artifact": str(artifact_path)})
        return artifact_path

    def log_execution_trace(self, trace: ExecutionTrace) -> Path:
        artifact_path = self.write_artifact(TRACE_SUFFIX, trace.model_dump(mode="python"))
        self.log_kv({"execution_trace_artifact": str(artifact_path)})
        return artifact_path


def init_session_logger(logs_dir: Path) -> SessionLogger:
    session_id = make_session_id()
    log_path = logs_dir / f"session_{session_id}.log"
    logger.info("Session log path: {}", log_path)
    return SessionLogger(path=log_path)


def load_dashboard_spec(session_log: Path) -> dict[str, Any]:
    state_path = session_log.with_name(f"{session_log.stem}{STATE_SUFFIX}")
    if state_path.exists():
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        active_spec = state_payload.get("active_spec")
        if isinstance(active_spec, dict):
            return active_spec
    if not session_log.exists():
        raise FileNotFoundError(f"Session log not found: {session_log}")
    content = session_log.read_text(encoding="utf-8")
    start = content.find(SPEC_START)
    end = content.find(SPEC_END)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Dashboard spec not found in session log.")
    raw_json = content[start + len(SPEC_START):end].strip()
    return json.loads(raw_json)


def load_session_metadata(session_log: Path) -> dict[str, str]:
    if not session_log.exists():
        raise FileNotFoundError(f"Session log not found: {session_log}")
    metadata: dict[str, str] = {}
    for line in session_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- "):
            key_value = line[2:].split(":", maxsplit=1)
            if len(key_value) == 2:
                key, value = key_value[0].strip(), key_value[1].strip()
                metadata[key] = value
    return metadata


def load_session_state(session_log: Path) -> SessionState:
    state_path = session_log.with_name(f"{session_log.stem}{STATE_SUFFIX}")
    if not state_path.exists():
        raise FileNotFoundError(f"Session state artifact not found: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return SessionState.model_validate(payload)


def load_execution_trace(session_log: Path) -> ExecutionTrace:
    trace_path = session_log.with_name(f"{session_log.stem}{TRACE_SUFFIX}")
    if not trace_path.exists():
        raise FileNotFoundError(f"Execution trace artifact not found: {trace_path}")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    return ExecutionTrace.model_validate(payload)
