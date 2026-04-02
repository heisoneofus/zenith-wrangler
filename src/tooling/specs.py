from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models import DashboardSpec, VisualSpec


ToolCategory = Literal["loader", "cleaning", "transform", "visualization", "dashboard"]
OutputKind = Literal["dataframe", "figure", "dashboard"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


class ReadCsvParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sample_rows: int | None = Field(default=None, ge=1)


class ReadExcelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sample_rows: int | None = Field(default=None, ge=1)


class ReadParquetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sample_rows: int | None = Field(default=None, ge=1)


class DropDuplicatesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


FillStrategy = Literal["auto", "mean", "median", "mode", "forward", "backward", "constant"]


class FillMissingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: FillStrategy = "auto"
    fill_value: dict[str, Any] | None = None


OutlierMethod = Literal["iqr", "zscore"]


class RemoveOutliersParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str] | None = None
    factor: float = Field(default=1.5, gt=0)
    method: OutlierMethod = "iqr"


PivotAgg = Literal["mean", "sum", "median", "count", "min", "max"]


class PivotDataParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: str
    columns: str
    values: str
    aggfunc: PivotAgg = "mean"


class AggregateByParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_by: list[str] = Field(min_length=1)
    metrics: list[str] | dict[str, str | list[str]]
    agg: str | dict[str, str] = "sum"


class FlattenNestedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_depth: int = Field(default=1, ge=0)


class CreateFigureParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: VisualSpec


class BuildDashboardParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    design: DashboardSpec


@dataclass
class ToolSpec:
    name: str
    description: str
    category: ToolCategory
    input_model: type[BaseModel]
    output_kind: OutputKind
    requires_context: tuple[str, ...] = ()
    produces_context: tuple[str, ...] = ()
    usage_guidance: str = ""
    public_safe: bool = True
    deterministic: bool = True
    version: str = "1.0.0"
    examples: list[dict[str, Any]] = field(default_factory=list)
    execute: Callable[..., Any] = field(default=lambda *_args, **_kwargs: None)

    def validate_params(self, raw: dict[str, Any]) -> BaseModel:
        return self.input_model.model_validate(raw)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires_context": list(self.requires_context),
            "produces_context": list(self.produces_context),
            "usage_guidance": self.usage_guidance,
            "public_safe": self.public_safe,
            "deterministic": self.deterministic,
            "version": self.version,
            "output_kind": self.output_kind,
            "input_schema": self.input_model.model_json_schema(),
            "examples": _json_safe(self.examples),
        }

    def to_json_schema(self) -> dict[str, Any]:
        return self.to_prompt_dict()
