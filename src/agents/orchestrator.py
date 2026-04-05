from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.config import AppConfig
from src.dashboard.builder import DashboardResult, build_dashboard
from src.models import AnalysisReport, DashboardSpec, ExecutionTrace, ToolEvent, VisualSpec
from src.tooling import (
    AggregateByParams,
    BuildDashboardParams,
    CreateFigureParams,
    DropDuplicatesParams,
    FillMissingParams,
    FlattenNestedParams,
    PivotDataParams,
    ReadCsvParams,
    ReadExcelParams,
    ReadParquetParams,
    RemoveOutliersParams,
    ToolSpec,
)
from src.tools import cleaning, loaders, transforms, visualization


_IDENTIFIER_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|identifier|transaction_id|person_id|user_id)", re.IGNORECASE)


def _looks_like_identifier_name(column: str) -> bool:
    return bool(_IDENTIFIER_NAME_PATTERN.search(column))


def _is_numeric_dtype_name(dtype_name: str) -> bool:
    lowered = dtype_name.lower()
    return any(token in lowered for token in ("int", "float", "double", "decimal", "number"))


def _sanitize_dataframe_ref(reference: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", reference.strip()).strip("_").lower()
    return normalized or fallback


def _resolve_context_dataframe(
    context: dict[str, Any],
    dataframe_ref: str | None = None,
    *,
    default_to_baseline: bool = False,
) -> pd.DataFrame:
    raw_dataframes = context.get("dataframes")
    if isinstance(raw_dataframes, dict):
        dataframes = {
            key: value
            for key, value in raw_dataframes.items()
            if isinstance(key, str) and isinstance(value, pd.DataFrame)
        }
    else:
        dataframes: dict[str, pd.DataFrame] = {}
    if dataframe_ref:
        normalized_ref = _sanitize_dataframe_ref(dataframe_ref, dataframe_ref)
        if normalized_ref not in dataframes:
            raise ValueError(f"Dataframe reference '{normalized_ref}' is missing from context.")
        return dataframes[normalized_ref]

    if default_to_baseline:
        baseline_ref = context.get("baseline_dataframe_ref")
        if baseline_ref and baseline_ref in dataframes:
            return dataframes[baseline_ref]

    active_ref = context.get("active_dataframe_ref")
    if active_ref and active_ref in dataframes:
        return dataframes[active_ref]

    fallback_dataframe = context["dataframe"] if "dataframe" in context else None
    if isinstance(fallback_dataframe, pd.DataFrame):
        return fallback_dataframe
    raise ValueError("No dataframe available in context.")


def _store_dataframe(
    context: dict[str, Any],
    reference: str,
    dataframe: pd.DataFrame,
    *,
    set_active: bool,
    set_baseline: bool,
) -> None:
    dataframes = context.setdefault("dataframes", {})
    dataframes[reference] = dataframe
    if set_baseline:
        context["baseline_dataframe_ref"] = reference
    if set_active:
        context["active_dataframe_ref"] = reference
        context["dataframe"] = dataframe


def _next_derived_dataframe_ref(context: dict[str, Any], tool_name: str) -> str:
    counter = context.get("derived_dataframe_counter", 0) + 1
    context["derived_dataframe_counter"] = counter
    return _sanitize_dataframe_ref(f"{tool_name}_{counter}", f"derived_{counter}")


def _safe_create_figure(context: dict[str, Any], spec: VisualSpec | dict[str, Any], dataframe_ref: str | None = None):
    dataframe = _resolve_context_dataframe(context, dataframe_ref=dataframe_ref, default_to_baseline=True)
    visual_spec = spec if isinstance(spec, VisualSpec) else VisualSpec.model_validate(spec)
    try:
        return visualization.create_figure(dataframe, visual_spec)
    except Exception as exc:
        return visualization.error_figure(visual_spec.title, f"Unable to render chart: {exc}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_tool_specs(self) -> list[ToolSpec]:
        return [self._tools[name] for name in self.list_tools()]

    def export_tool_catalog(self) -> list[dict[str, Any]]:
        return [tool.to_json_schema() for tool in self.list_tool_specs()]

    def get_tool_prompt_context(self) -> dict[str, Any]:
        return {"tools": [tool.to_prompt_dict() for tool in self.list_tool_specs()]}


@dataclass
class ToolCall:
    tool_name: str
    params: dict[str, Any]
    reasoning: str = ""


@dataclass
class ExecutionResult:
    dataframe: pd.DataFrame | None
    baseline_dataframe: pd.DataFrame | None
    figures: list[Any]
    dashboard: DashboardResult | None
    output_path: Path | None
    transformations_applied: list[str]
    warnings: list[str]


class Orchestrator:
    def __init__(self, config: AppConfig, registry: ToolRegistry):
        self.config = config
        self.registry = registry

    def _identifier_columns_from_analysis(self, analysis: AnalysisReport) -> set[str]:
        return {column for column in analysis.data_schema if _looks_like_identifier_name(column)}

    def _metric_candidates(self, analysis: AnalysisReport, identifier_columns: set[str]) -> list[str]:
        preferred = analysis.metrics.primary_metrics + analysis.metrics.secondary_metrics
        schema_numeric = [
            column
            for column, dtype_name in analysis.data_schema.items()
            if _is_numeric_dtype_name(dtype_name) and column not in identifier_columns
        ]
        ordered = preferred + schema_numeric
        deduped: list[str] = []
        for column in ordered:
            if column in deduped or column in identifier_columns:
                continue
            if column in analysis.data_schema:
                deduped.append(column)
        return deduped

    def _grouping_candidates(self, analysis: AnalysisReport, identifier_columns: set[str]) -> list[str]:
        preferred = analysis.metrics.time_fields + analysis.metrics.dimensions
        schema_categorical = [
            column
            for column, dtype_name in analysis.data_schema.items()
            if not _is_numeric_dtype_name(dtype_name) and column not in identifier_columns
        ]
        ordered = preferred + schema_categorical
        deduped: list[str] = []
        for column in ordered:
            if column in deduped or column in identifier_columns:
                continue
            if column in analysis.data_schema:
                deduped.append(column)
        return deduped

    def _build_aggregate_tool_call(self, operation: str, analysis: AnalysisReport | None) -> ToolCall | None:
        if analysis is None:
            return None
        identifier_columns = self._identifier_columns_from_analysis(analysis)
        grouping_candidates = self._grouping_candidates(analysis, identifier_columns)
        metric_candidates = self._metric_candidates(analysis, identifier_columns)
        if not grouping_candidates or not metric_candidates:
            return None
        return ToolCall(
            tool_name="aggregate_by",
            params={
                "group_by": [grouping_candidates[0]],
                "metrics": [metric_candidates[0]],
                "agg": "sum",
                "dataframe_ref": "baseline",
                "output_dataframe_ref": _sanitize_dataframe_ref(
                    f"aggregate_{metric_candidates[0]}_by_{grouping_candidates[0]}",
                    "aggregate_result",
                ),
            },
            reasoning=f"Apply transform recommendation: {operation}",
        )

    def _build_pivot_tool_call(self, operation: str, analysis: AnalysisReport | None) -> ToolCall | None:
        if analysis is None:
            return None
        identifier_columns = self._identifier_columns_from_analysis(analysis)
        metric_candidates = self._metric_candidates(analysis, identifier_columns)
        if not metric_candidates:
            return None

        heatmap_spec = next((spec for spec in analysis.design.visuals if spec.chart_type == "heatmap"), None)
        if heatmap_spec and heatmap_spec.x and heatmap_spec.y:
            values = heatmap_spec.color if heatmap_spec.color in analysis.data_schema else metric_candidates[0]
            if values in analysis.data_schema:
                return ToolCall(
                    tool_name="pivot_data",
                    params={
                        "index": heatmap_spec.y,
                        "columns": heatmap_spec.x,
                        "values": values,
                        "aggfunc": "mean",
                        "dataframe_ref": "baseline",
                        "output_dataframe_ref": _sanitize_dataframe_ref(
                            f"pivot_{values}_by_{heatmap_spec.y}_{heatmap_spec.x}",
                            "pivot_result",
                        ),
                    },
                    reasoning=f"Apply transform recommendation: {operation}",
                )

        grouping_candidates = self._grouping_candidates(analysis, identifier_columns)
        if len(grouping_candidates) < 2:
            return None
        return ToolCall(
            tool_name="pivot_data",
            params={
                "index": grouping_candidates[0],
                "columns": grouping_candidates[1],
                "values": metric_candidates[0],
                "aggfunc": "mean",
                "dataframe_ref": "baseline",
                "output_dataframe_ref": _sanitize_dataframe_ref(
                    f"pivot_{metric_candidates[0]}_by_{grouping_candidates[0]}_{grouping_candidates[1]}",
                    "pivot_result",
                ),
            },
            reasoning=f"Apply transform recommendation: {operation}",
        )

    def _sanitize_visual_spec(self, spec: VisualSpec, analysis: AnalysisReport) -> VisualSpec:
        identifier_columns = self._identifier_columns_from_analysis(analysis)
        if not identifier_columns:
            return spec

        updates: dict[str, str | None] = {}
        metric_candidates = self._metric_candidates(analysis, identifier_columns)
        grouping_candidates = self._grouping_candidates(analysis, identifier_columns)

        if spec.y in identifier_columns and metric_candidates:
            replacement = next((column for column in metric_candidates if column != spec.x), metric_candidates[0])
            updates["y"] = replacement

        if spec.x in identifier_columns and grouping_candidates:
            replacement = next((column for column in grouping_candidates if column != spec.y), grouping_candidates[0])
            updates["x"] = replacement

        if spec.color in identifier_columns:
            updates["color"] = None

        if not updates:
            return spec
        return spec.model_copy(update=updates)

    def _parse_operation_to_tool(self, operation: str, analysis: AnalysisReport | None = None) -> ToolCall | None:
        """Parse operation description strings into ToolCall objects."""
        op_lower = operation.lower()

        # Drop columns (e.g., "Drop 'Unnamed: 70'")
        if op_lower.startswith("drop"):
            # These are column drops, not duplicate drops - skip for now
            # as there's no registered tool for dropping specific columns
            return None

        # Remove duplicates
        if "duplicate" in op_lower:
            return ToolCall(
                tool_name="drop_duplicates",
                params={},
                reasoning=f"Apply data quality fix: {operation}"
            )

        # Fill missing values
        if "missing" in op_lower or "null" in op_lower or "nan" in op_lower or "fill" in op_lower:
            return ToolCall(
                tool_name="fill_missing",
                params={},
                reasoning=f"Apply data quality fix: {operation}"
            )

        # Remove outliers
        if "outlier" in op_lower:
            return ToolCall(
                tool_name="remove_outliers",
                params={},
                reasoning=f"Apply data quality fix: {operation}"
            )

        # Flatten nested / JSON-like values
        flatten_terms = ("flatten", "nested", "json", "list-like", "dict-like", "array-like")
        if any(term in op_lower for term in flatten_terms):
            return ToolCall(
                tool_name="flatten_nested",
                params={"max_depth": 1},
                reasoning=f"Apply data quality fix: {operation}"
            )

        aggregate_terms = ("aggregate", "group", "rollup", "summary", "summarize", "kpi")
        if any(term in op_lower for term in aggregate_terms):
            return self._build_aggregate_tool_call(operation, analysis)

        pivot_terms = ("pivot", "matrix", "cross-tab", "crosstab", "heatmap")
        if any(term in op_lower for term in pivot_terms):
            return self._build_pivot_tool_call(operation, analysis)

        # If no match, skip this operation
        return None

    def plan_execution(
        self,
        analysis: AnalysisReport,
        data_path: Path,
        design_override: DashboardSpec | None = None,
    ) -> list[ToolCall]:
        plan: list[ToolCall] = []
        loader = loaders.detect_loader(data_path)
        plan.append(
            ToolCall(
                tool_name=loader,
                params={"path": data_path, "sample_rows": None},
                reasoning="Load full dataset for execution.",
            )
        )

        # Parse suggested operations into actual tool calls
        for op in analysis.quality.suggested_operations:
            tool_call = self._parse_operation_to_tool(op, analysis=analysis)
            if tool_call:
                plan.append(tool_call)

        design_source = design_override or analysis.design
        safe_visuals = [self._sanitize_visual_spec(spec, analysis) for spec in design_source.visuals]
        safe_design = design_source.model_copy(update={"visuals": safe_visuals})

        for safe_spec in safe_visuals:
            plan.append(
                ToolCall(
                    tool_name="create_figure",
                    params={"spec": safe_spec.model_dump()},
                    reasoning=f"Generate {safe_spec.chart_type} for {safe_spec.title}.",
                )
            )
        plan.append(
            ToolCall(
                tool_name="build_dashboard",
                params={"design": safe_design.model_dump()},
                reasoning="Assemble Dash layout and callbacks.",
            )
        )
        return plan

    def execute_plan(
        self,
        plan: list[ToolCall],
        output_format: str,
        output_path: Path,
        port: int,
        logger_ctx,
        trace: ExecutionTrace | None = None,
        defer_export: bool = False,
    ) -> ExecutionResult:
        context: dict[str, Any] = {
            "figures": [],
            "transformations": [],
            "dataframes": {},
            "derived_dataframe_counter": 0,
            "derived_dataframe_refs": [],
            "guardrail_warnings": [],
        }
        output: Path | None = None
        dashboard: DashboardResult | None = None
        for index, call in enumerate(plan, start=1):
            tool = self.registry.get(call.tool_name)
            validated_params = tool.validate_params(call.params).model_dump(mode="python", exclude_none=True)
            logger_ctx.log_tool_call(index, call.tool_name, call.reasoning, validated_params)
            if trace is not None:
                trace.events.append(
                    ToolEvent(
                        event_type="tool_call",
                        tool_name=call.tool_name,
                        status="planned",
                        reasoning=call.reasoning,
                        params=_json_safe_payload(validated_params),
                        started_at=_utc_now(),
                    )
                )
            previous_dataframe = context.get("dataframe")
            try:
                result = tool.execute(context, **validated_params)
            except Exception as exc:
                message = f"Pipeline step failed at '{call.tool_name}'."
                if hasattr(logger_ctx, "log_block"):
                    logger_ctx.log_block("Execution Error", f"{message}\n{exc}")
                if trace is not None:
                    trace.events.append(
                        ToolEvent(
                            event_type="tool_result",
                            tool_name=call.tool_name,
                            status="failed",
                            reasoning=call.reasoning,
                            params=_json_safe_payload(validated_params),
                            message=f"{message} Reason: {exc}",
                            started_at=_utc_now(),
                            ended_at=_utc_now(),
                        )
                    )
                    trace.status = "failed"
                    trace.warnings.append(str(exc))
                if output_format in {"server", "dash"}:
                    if defer_export:
                        return ExecutionResult(
                            dataframe=context.get("dataframe"),
                            baseline_dataframe=self._baseline_dataframe(context),
                            figures=context.get("figures", []),
                            dashboard=dashboard,
                            output_path=None,
                            transformations_applied=context.get("transformations", []),
                            warnings=context.get("guardrail_warnings", []),
                        )
                    error_app = visualization.build_error_app(
                        title="Dashboard Generation Failed",
                        message="The dashboard could not be generated. Review the error details below.",
                        details=f"{message}\nReason: {exc}",
                    )
                    output = visualization.export_dashboard(
                        output_format=output_format,
                        output_path=output_path,
                        title="Dashboard Generation Failed",
                        figures=[],
                        app=error_app,
                        port=port,
                    )
                    return ExecutionResult(
                        dataframe=context.get("dataframe"),
                        baseline_dataframe=self._baseline_dataframe(context),
                        figures=context.get("figures", []),
                        dashboard=dashboard,
                        output_path=output,
                        transformations_applied=context.get("transformations", []),
                        warnings=context.get("guardrail_warnings", []),
                    )
                raise
            skip_storage = False
            guardrail_warning = self._guard_destructive_transform(
                call.tool_name,
                previous_dataframe if isinstance(previous_dataframe, pd.DataFrame) else None,
                result if isinstance(result, pd.DataFrame) else None,
            )
            if guardrail_warning:
                skip_storage = True
                context["guardrail_warnings"].append(guardrail_warning)
                if trace is not None:
                    trace.events.append(
                        ToolEvent(
                            event_type="warning",
                            tool_name=call.tool_name,
                            status="skipped",
                            reasoning=call.reasoning,
                            params=_json_safe_payload(validated_params),
                            message=guardrail_warning,
                            started_at=_utc_now(),
                            ended_at=_utc_now(),
                        )
                    )
            for produced in tool.produces_context:
                if produced == "dataframe":
                    if skip_storage:
                        continue
                    if tool.category == "loader":
                        _store_dataframe(context, "raw", result, set_active=True, set_baseline=False)
                        _store_dataframe(context, "baseline", result, set_active=True, set_baseline=True)
                    elif tool.category == "cleaning":
                        _store_dataframe(context, "baseline", result, set_active=True, set_baseline=True)
                        context["transformations"].append(call.tool_name)
                    elif tool.category == "transform":
                        output_reference = validated_params.get("output_dataframe_ref")
                        reference = (
                            _sanitize_dataframe_ref(output_reference, call.tool_name)
                            if output_reference
                            else _next_derived_dataframe_ref(context, call.tool_name)
                        )
                        _store_dataframe(context, reference, result, set_active=True, set_baseline=False)
                        context["derived_dataframe_refs"].append(reference)
                        context["last_derived_dataframe_ref"] = reference
                        context["transformations"].append(call.tool_name)
                    else:
                        _store_dataframe(context, "baseline", result, set_active=True, set_baseline=True)
                elif produced == "figure":
                    context["figures"].append(result)
                elif produced == "dashboard":
                    dashboard = result
                    context["dashboard"] = result
            if trace is not None and not skip_storage:
                trace.events.append(
                    ToolEvent(
                        event_type="tool_result",
                        tool_name=call.tool_name,
                        status="completed",
                        reasoning=call.reasoning,
                        params=_json_safe_payload(validated_params),
                        outputs=self._summarize_result(call.tool_name, result),
                        started_at=_utc_now(),
                        ended_at=_utc_now(),
                    )
                )

        figures = context.get("figures", [])
        dashboard = dashboard or context.get("dashboard")
        if defer_export:
            return ExecutionResult(
                dataframe=context.get("dataframe"),
                baseline_dataframe=self._baseline_dataframe(context),
                figures=figures,
                dashboard=dashboard,
                output_path=None,
                transformations_applied=context.get("transformations", []),
                warnings=context.get("guardrail_warnings", []),
            )
        output = visualization.export_dashboard(
            output_format=output_format,
            output_path=output_path,
            title=dashboard.spec.title if dashboard else "Zenith Wrangler Dashboard",
            figures=figures,
            app=dashboard.app if dashboard else None,
            port=port,
        )
        return ExecutionResult(
            dataframe=context.get("dataframe"),
            baseline_dataframe=self._baseline_dataframe(context),
            figures=figures,
            dashboard=dashboard,
            output_path=output,
            transformations_applied=context.get("transformations", []),
            warnings=context.get("guardrail_warnings", []),
        )

    def _summarize_result(self, tool_name: str, result: Any) -> dict[str, Any]:
        if isinstance(result, pd.DataFrame):
            return {
                "rows": len(result),
                "columns": list(result.columns),
            }
        title = getattr(getattr(result, "layout", None), "title", None)
        if title is not None and hasattr(title, "text"):
            return {"figure_title": title.text or tool_name}
        dashboard_spec = getattr(result, "spec", None)
        if dashboard_spec is not None:
            return {
                "dashboard_title": dashboard_spec.title,
                "visual_count": len(dashboard_spec.visuals),
            }
        return {"result_type": type(result).__name__}

    def _guard_destructive_transform(
        self,
        tool_name: str,
        previous_dataframe: pd.DataFrame | None,
        result_dataframe: pd.DataFrame | None,
    ) -> str | None:
        if tool_name not in {"remove_outliers", "drop_duplicates"}:
            return None
        if previous_dataframe is None or result_dataframe is None or previous_dataframe.empty:
            return None
        removed_rows = len(previous_dataframe) - len(result_dataframe)
        if removed_rows <= 0:
            return None
        removed_ratio = removed_rows / max(len(previous_dataframe), 1)
        if removed_ratio < 0.35:
            return None
        return (
            f"Skipped `{tool_name}` because it removed {removed_rows} rows "
            f"({removed_ratio:.0%} of the current dataset), exceeding the analyst safety threshold."
        )

    def _baseline_dataframe(self, context: dict[str, Any]) -> pd.DataFrame | None:
        baseline_ref = context.get("baseline_dataframe_ref")
        dataframes = context.get("dataframes", {})
        if isinstance(baseline_ref, str) and isinstance(dataframes, dict):
            baseline = dataframes.get(baseline_ref)
            if isinstance(baseline, pd.DataFrame):
                return baseline
        dataframe = context.get("dataframe")
        return dataframe if isinstance(dataframe, pd.DataFrame) else None


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="read_csv",
            description="Load CSV data.",
            category="loader",
            input_model=ReadCsvParams,
            output_kind="dataframe",
            produces_context=("dataframe",),
            usage_guidance="Use for comma-separated flat files.",
            public_safe=False,
            examples=[{"path": "data/sales.csv"}, {"path": "data/sales.csv", "sample_rows": 500}],
            execute=lambda ctx, path, sample_rows=None: loaders.read_csv(path, sample_rows=sample_rows),
        )
    )
    registry.register(
        ToolSpec(
            name="read_excel",
            description="Load Excel data.",
            category="loader",
            input_model=ReadExcelParams,
            output_kind="dataframe",
            produces_context=("dataframe",),
            usage_guidance="Use for .xlsx/.xls spreadsheets.",
            public_safe=False,
            examples=[{"path": "data/sales.xlsx"}, {"path": "data/sales.xlsx", "sample_rows": 250}],
            execute=lambda ctx, path, sample_rows=None: loaders.read_excel(path, sample_rows=sample_rows),
        )
    )
    registry.register(
        ToolSpec(
            name="read_parquet",
            description="Load Parquet data.",
            category="loader",
            input_model=ReadParquetParams,
            output_kind="dataframe",
            produces_context=("dataframe",),
            usage_guidance="Use for columnar parquet datasets.",
            public_safe=False,
            examples=[{"path": "data/sales.parquet"}, {"path": "data/sales.parquet", "sample_rows": 1000}],
            execute=lambda ctx, path, sample_rows=None: loaders.read_parquet(path, sample_rows=sample_rows),
        )
    )
    registry.register(
        ToolSpec(
            name="drop_duplicates",
            description="Remove duplicate rows.",
            category="cleaning",
            input_model=DropDuplicatesParams,
            output_kind="dataframe",
            requires_context=("dataframe",),
            produces_context=("dataframe",),
            usage_guidance="Use when duplicate records affect metrics.",
            examples=[{}],
            execute=lambda ctx: cleaning.drop_duplicates(_resolve_context_dataframe(ctx)),
        )
    )
    registry.register(
        ToolSpec(
            name="fill_missing",
            description="Fill missing values.",
            category="cleaning",
            input_model=FillMissingParams,
            output_kind="dataframe",
            requires_context=("dataframe",),
            produces_context=("dataframe",),
            usage_guidance="Prefer `auto` unless domain-specific constants are required.",
            examples=[{"strategy": "auto"}, {"strategy": "constant", "fill_value": {"region": "Unknown"}}],
            execute=lambda ctx, strategy="auto", fill_value=None: cleaning.fill_missing(
                _resolve_context_dataframe(ctx), strategy=strategy, fill_value=fill_value
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="remove_outliers",
            description="Remove outliers using IQR or Z-score method.",
            category="cleaning",
            input_model=RemoveOutliersParams,
            output_kind="dataframe",
            requires_context=("dataframe",),
            produces_context=("dataframe",),
            usage_guidance="Use on numeric fields after deciding acceptable sensitivity.",
            examples=[{"method": "iqr", "factor": 1.5}, {"columns": ["sales"], "method": "zscore", "factor": 3.0}],
            execute=lambda ctx, columns=None, factor=1.5, method="iqr": cleaning.remove_outliers(
                _resolve_context_dataframe(ctx), columns=columns, factor=factor, method=method
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="pivot_data",
            description="Pivot data for heatmaps.",
            category="transform",
            input_model=PivotDataParams,
            output_kind="dataframe",
            requires_context=("dataframe",),
            produces_context=("dataframe",),
            usage_guidance="Use for matrix-like analysis where dimensions form rows/columns.",
            examples=[
                {"index": "region", "columns": "month", "values": "sales", "aggfunc": "mean"},
                {
                    "index": "region",
                    "columns": "month",
                    "values": "sales",
                    "aggfunc": "mean",
                    "dataframe_ref": "baseline",
                    "output_dataframe_ref": "sales_matrix",
                },
            ],
            execute=lambda ctx, index, columns, values, aggfunc="mean", dataframe_ref=None, output_dataframe_ref=None: transforms.pivot_data(
                _resolve_context_dataframe(ctx, dataframe_ref=dataframe_ref),
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="aggregate_by",
            description="Aggregate data by dimensions with flexible aggregation functions.",
            category="transform",
            input_model=AggregateByParams,
            output_kind="dataframe",
            requires_context=("dataframe",),
            produces_context=("dataframe",),
            usage_guidance="Use before plotting KPIs or grouped summaries.",
            examples=[
                {"group_by": ["region"], "metrics": ["sales"], "agg": "sum"},
                {"group_by": ["region"], "metrics": {"sales": "sum", "margin": "mean"}, "agg": "sum"},
                {
                    "group_by": ["region"],
                    "metrics": ["sales"],
                    "agg": "sum",
                    "dataframe_ref": "baseline",
                    "output_dataframe_ref": "sales_by_region",
                },
            ],
            execute=lambda ctx, group_by, metrics, agg="sum", dataframe_ref=None, output_dataframe_ref=None: transforms.aggregate_by(
                _resolve_context_dataframe(ctx, dataframe_ref=dataframe_ref),
                group_by=group_by,
                metrics=metrics,
                agg=agg,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="flatten_nested",
            description="Flatten nested JSON/dict structures including lists of dicts.",
            category="transform",
            input_model=FlattenNestedParams,
            output_kind="dataframe",
            requires_context=("dataframe",),
            produces_context=("dataframe",),
            usage_guidance="Use when columns store dictionaries or list-of-dict payloads.",
            examples=[
                {"max_depth": 1},
                {"max_depth": 2},
                {"max_depth": 2, "dataframe_ref": "baseline", "output_dataframe_ref": "flattened_events"},
            ],
            execute=lambda ctx, max_depth=1, dataframe_ref=None, output_dataframe_ref=None: transforms.flatten_nested(
                _resolve_context_dataframe(ctx, dataframe_ref=dataframe_ref),
                max_depth=max_depth,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="create_figure",
            description="Create Plotly figure.",
            category="visualization",
            input_model=CreateFigureParams,
            output_kind="figure",
            requires_context=("dataframe",),
            produces_context=("figure",),
            usage_guidance=(
                "Use after cleaning/transforms with a valid chart specification. "
                "Defaults to the baseline cleaned dataframe unless `dataframe_ref` is provided."
            ),
            examples=[
                {"spec": {"title": "Sales by Region", "chart_type": "bar", "x": "region", "y": "sales"}},
                {
                    "dataframe_ref": "sales_by_region",
                    "spec": {"title": "Sales by Region", "chart_type": "bar", "x": "region", "y": "sales"},
                },
            ],
            execute=lambda ctx, spec, dataframe_ref=None: _safe_create_figure(
                ctx,
                spec,
                dataframe_ref=dataframe_ref,
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="build_dashboard",
            description="Build Dash app layout and callbacks.",
            category="dashboard",
            input_model=BuildDashboardParams,
            output_kind="dashboard",
            requires_context=("dataframe",),
            produces_context=("dashboard",),
            usage_guidance=(
                "Use as the final assembly step after figure generation planning. "
                "Defaults to the baseline cleaned dataframe unless `dataframe_ref` is provided."
            ),
            examples=[
                {"design": {"title": "Sales Dashboard", "layout": "grid", "visuals": []}},
                {
                    "dataframe_ref": "sales_by_region",
                    "design": {"title": "Regional Dashboard", "layout": "grid", "visuals": []},
                },
            ],
            execute=lambda ctx, design, dataframe_ref=None: build_dashboard(
                _resolve_context_dataframe(ctx, dataframe_ref=dataframe_ref, default_to_baseline=True),
                design if isinstance(design, DashboardSpec) else DashboardSpec.model_validate(design),
            ),
        )
    )
    return registry


def export_tool_catalog() -> list[dict[str, Any]]:
    return build_registry().export_tool_catalog()


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            safe[key] = str(value)
        elif isinstance(value, dict):
            safe[key] = _json_safe_payload(value)
        elif isinstance(value, list):
            safe[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            safe[key] = value
    return safe


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
