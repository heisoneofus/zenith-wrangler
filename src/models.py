from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    id: str = Field(default_factory=lambda: _make_id("visual"))
    title: str
    chart_type: Literal["line", "bar", "scatter", "histogram", "box", "heatmap", "area", "pie"]
    x: str | None = None
    y: str | None = None
    color: str | None = None
    aggregation: Literal["sum", "mean", "median", "count"] | None = None
    description: str | None = None
    source_dataframe_ref: str | None = None
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    pinned: bool = False
    status: Literal["proposed", "approved", "rendered", "revised"] = "proposed"


class DashboardSpec(BaseModel):
    id: str = Field(default_factory=lambda: _make_id("dashboard"))
    title: str = "Zenith Wrangler Dashboard"
    layout: Literal["grid", "tabs", "sections"] = "grid"
    theme: Literal["light", "dark"] = "light"
    visuals: list[VisualSpec] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    notes: str = ""
    plan_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    data_quality_summary: list[str] = Field(default_factory=list)
    transform_history: list[str] = Field(default_factory=list)
    approval_status: Literal["draft", "approved", "reviewed"] = "draft"


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


class DashboardPatchOperation(BaseModel):
    op: Literal[
        "add_visual",
        "remove_visual",
        "replace_visual",
        "change_layout",
        "add_filter",
        "remove_filter",
        "remap_field",
        "change_aggregation",
        "change_theme",
    ]
    target_visual_id: str | None = None
    target_visual_title: str | None = None
    target_field: Literal["x", "y", "color"] | None = None
    value: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class DashboardPatch(BaseModel):
    prompt: str = ""
    operations: list[DashboardPatchOperation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolEvent(BaseModel):
    event_type: Literal["tool_call", "tool_result", "warning", "repair", "approval", "review"]
    tool_name: str = ""
    status: Literal["planned", "completed", "failed", "skipped", "repaired", "approved"] = "planned"
    reasoning: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    dataframe_ref: str | None = None
    started_at: str = Field(default_factory=_utc_now)
    ended_at: str | None = None


class ExecutionTrace(BaseModel):
    session_id: str = ""
    workflow: list[str] = Field(default_factory=lambda: ["profile", "propose_plan", "approve_edit", "execute_review"])
    current_stage: Literal["profile", "propose_plan", "approve_edit", "execute_review"] = "profile"
    status: Literal["planned", "approved", "executed", "repaired", "failed"] = "planned"
    events: list[ToolEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    approvals: list[str] = Field(default_factory=list)
    repair_notes: list[str] = Field(default_factory=list)


class PlanProposal(BaseModel):
    session_id: str = ""
    mode: Literal["heuristic", "llm"] = "heuristic"
    user_goal: str = ""
    summary: str = ""
    profile_notes: list[str] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    transform_plan: list[str] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
    approval_required: bool = True
    approved: bool = False
    design: DashboardSpec = Field(default_factory=DashboardSpec)


class SessionState(BaseModel):
    session_id: str
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    status: Literal["draft", "planned", "approved", "executed", "reviewed", "failed"] = "draft"
    data_path: str = ""
    description_path: str = ""
    transformed_dataset: str = ""
    output_path: str = ""
    user_goal: str = ""
    analysis: AnalysisReport | None = None
    plan: PlanProposal | None = None
    trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
    active_spec: DashboardSpec = Field(default_factory=DashboardSpec)
    spec_versions: list[DashboardSpec] = Field(default_factory=list)
    pending_patch: DashboardPatch | None = None
    decisions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
