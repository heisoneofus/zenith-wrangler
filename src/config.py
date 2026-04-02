from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-5.2"
    analysis_temperature: float = 0.3
    tool_temperature: float = 0.1
    api_key: str | None = None


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    logs_dir: Path
    outputs_dir: Path
    llm: LLMConfig = LLMConfig()
    sample_rows: int = 5000

    @staticmethod
    def default(root_dir: Path) -> "AppConfig":
        return AppConfig(
            root_dir=root_dir,
            logs_dir=root_dir / "logs",
            outputs_dir=root_dir / "outputs",
        )
