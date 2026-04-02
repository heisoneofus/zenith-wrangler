from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricsAnalysis(BaseModel):
    primary_metrics: list[str] = Field(default_factory=list)
    secondary_metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class DataQualityIssue(BaseModel):
    type: Literal["missing", "duplicates", "outliers", "types", "other"]
    columns: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "low"
    action: str = ""


class DataQualityAssessment(BaseModel):
    issues: list[DataQualityIssue] = Field(default_factory=list)
    suggested_operations: list[str] = Field(default_factory=list)
    notes: str = ""


class VisualSpec(BaseModel):
    title: str
    chart_type: Literal["line", "bar", "scatter", "histogram", "box", "heatmap", "area", "pie"]
    x: str | None = None
    y: str | None = None
    color: str | None = None
    aggregation: Literal["sum", "mean", "median", "count"] | None = None
    description: str | None = None


class DashboardSpec(BaseModel):
    title: str = "Zenith Wrangler Dashboard"
    layout: Literal["grid", "tabs", "sections"] = "grid"
    visuals: list[VisualSpec] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    notes: str = ""


class LLMAnalysisResponse(BaseModel):
    """LLM response model without data_schema (which we populate separately)."""
    metrics: MetricsAnalysis
    quality: DataQualityAssessment
    design: DashboardSpec
    sampled_rows: int


class AnalysisReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    metrics: MetricsAnalysis
    quality: DataQualityAssessment
    design: DashboardSpec
    sampled_rows: int
    data_schema: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_schema_key(cls, value: object) -> object:
        if isinstance(value, dict) and "data_schema" not in value and "schema" in value:
            normalized = dict(value)
            normalized["data_schema"] = normalized.pop("schema")
            return normalized
        return value
