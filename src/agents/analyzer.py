from __future__ import annotations

import json
import re
import pandas as pd
from loguru import logger

from src.config import AppConfig
from src.models import (
    AnalysisReport,
    DashboardSpec,
    DataQualityAssessment,
    DataQualityIssue,
    LLMAnalysisResponse,
    MetricsAnalysis,
    VisualSpec,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


_IDENTIFIER_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|identifier|transaction_id|person_id|user_id)", re.IGNORECASE)


def _looks_like_identifier_name(column: str) -> bool:
    return bool(_IDENTIFIER_NAME_PATTERN.search(column))


def _is_identifier_series(series: pd.Series, column: str) -> bool:
    if _looks_like_identifier_name(column):
        return True
    non_null = series.dropna()
    if len(non_null) < 20:
        return False
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    is_int_like = pd.api.types.is_integer_dtype(non_null) or pd.api.types.is_object_dtype(non_null)
    return is_int_like and unique_ratio >= 0.95


def _detect_identifier_fields(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if _is_identifier_series(df[column], column)]


def _detect_time_fields(df: pd.DataFrame) -> list[str]:
    time_fields: list[str] = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            time_fields.append(column)
            continue
        if "date" in column.lower() or "time" in column.lower():
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() > 0.7:
                time_fields.append(column)
    return time_fields


def _heuristic_analysis(df: pd.DataFrame, description: str | None) -> AnalysisReport:
    all_numeric_cols = df.select_dtypes(include="number").columns.tolist()
    time_fields = _detect_time_fields(df)
    identifier_fields = _detect_identifier_fields(df)
    numeric_cols = [col for col in all_numeric_cols if col not in identifier_fields]
    categorical_cols = [col for col in df.columns if col not in all_numeric_cols and col not in identifier_fields]

    variances = (
        df[numeric_cols].var(numeric_only=True).sort_values(ascending=False) if numeric_cols else pd.Series()
    )
    primary_metrics = variances.index[:2].tolist() if not variances.empty else numeric_cols[:2]
    secondary_metrics = [col for col in numeric_cols if col not in primary_metrics]

    metrics = MetricsAnalysis(
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        dimensions=categorical_cols,
        time_fields=time_fields,
        notes=(
            f"Description context used: {bool(description)}"
            + (f". Identifier-like fields excluded from metric selection: {', '.join(identifier_fields)}" if identifier_fields else "")
        ),
    )

    issues: list[DataQualityIssue] = []
    suggested_ops: list[str] = []

    missing_cols = [col for col in df.columns if df[col].isna().any()]
    if missing_cols:
        issues.append(DataQualityIssue(type="missing", columns=missing_cols, severity="medium", action="fill_missing"))
        suggested_ops.append("fill_missing")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        issues.append(
            DataQualityIssue(type="duplicates", columns=[], severity="low", action=f"drop_duplicates ({duplicate_count})")
        )
        suggested_ops.append("drop_duplicates")

    outlier_cols: list[str] = []
    for column in numeric_cols:
        series = df[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).mean()
        if outliers > 0.02:
            outlier_cols.append(column)
    if outlier_cols:
        issues.append(DataQualityIssue(type="outliers", columns=outlier_cols, severity="medium", action="remove_outliers"))
        suggested_ops.append("remove_outliers")

    if categorical_cols and primary_metrics:
        suggested_ops.append("aggregate grouped metrics for chart-ready summaries")
    if time_fields and categorical_cols and primary_metrics:
        suggested_ops.append("pivot data for heatmap-style matrix analysis")

    quality = DataQualityAssessment(issues=issues, suggested_operations=sorted(set(suggested_ops)))

    visuals: list[VisualSpec] = []
    if time_fields and primary_metrics:
        visuals.append(
            VisualSpec(
                title=f"{primary_metrics[0]} over time",
                chart_type="line",
                x=time_fields[0],
                y=primary_metrics[0],
            )
        )
    if categorical_cols and primary_metrics:
        category = categorical_cols[0]
        cardinality = int(df[category].nunique(dropna=True))
        prefers_pie = 0 < cardinality <= 8
        visuals.append(
            VisualSpec(
                title=f"{primary_metrics[0]} by {category}",
                chart_type="pie" if prefers_pie else "bar",
                x=category,
                y=primary_metrics[0],
                aggregation="sum",
            )
        )
    if time_fields and categorical_cols and primary_metrics:
        visuals.append(
            VisualSpec(
                title=f"{primary_metrics[0]} heatmap by {categorical_cols[0]} and {time_fields[0]}",
                chart_type="heatmap",
                x=time_fields[0],
                y=categorical_cols[0],
                color=primary_metrics[0],
            )
        )
    if len(primary_metrics) >= 2:
        visuals.append(
            VisualSpec(
                title=f"{primary_metrics[0]} vs {primary_metrics[1]}",
                chart_type="scatter",
                x=primary_metrics[0],
                y=primary_metrics[1],
            )
        )
    if not visuals and numeric_cols:
        visuals.append(
            VisualSpec(
                title=f"Distribution of {numeric_cols[0]}",
                chart_type="histogram",
                x=numeric_cols[0],
            )
        )

    filters = [column for column in categorical_cols if df[column].nunique(dropna=True) <= 50][:2]
    design = DashboardSpec(
        title="Zenith Wrangler Dashboard",
        layout="grid",
        visuals=visuals,
        filters=filters,
        notes="Heuristic design (LLM unavailable).",
    )

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return AnalysisReport(
        metrics=metrics,
        quality=quality,
        design=design,
        sampled_rows=len(df),
        data_schema=schema,
    )


class Analyzer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        if OpenAI is not None:
            try:
                self.client = OpenAI(api_key=self.config.llm.api_key)
            except Exception as exc:  # pragma: no cover - network/env dependent
                logger.warning("OpenAI client unavailable: {}", exc)

    def run_analysis(self, df: pd.DataFrame, description: str | None, logger_ctx) -> AnalysisReport:
        if self.client is None:
            return _heuristic_analysis(df, description)

        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        prompt = {
            "task": "Analyze dataset and propose dashboard plan.",
            "schema": schema,
            "sample_rows": df.head(30).to_dict(orient="records"),
            "description": description or "",
            "instructions": [
                "Identify primary and secondary metrics.",
                "Assess data quality issues and cleaning operations.",
                "Propose dashboard layout, visuals, and filters.",
                "Explicitly inspect columns for nested-like content: dict-like, list-like, JSON-like, and CSV stringified nested values (e.g., \"['Drama']\", \"['Crime', 'Drama']\", '{\"a\": 1}').",
                "If nested/stringified nested columns are present and flattening would improve downstream visuals or filters, recommend `flatten_nested` and name the source column(s).",
                "Identify identifier-like fields (e.g., user_id, person_id, transaction_id, UUID/GUID) and avoid treating them as continuous metrics for charts.",
                "Use identifier-like fields as keys only; do not prioritize them as KPI y-axes unless the user explicitly requests ID-level analysis.",
                "When grouped rollups would improve KPI readability, recommend `aggregate_by` and mention grouping columns, metric columns, and aggregation intent.",
                "When matrix/cross-tab structure would improve heatmap readiness, recommend `pivot_data` and mention index, columns, values, and aggfunc intent.",
                "Aim for a varied chart mix when appropriate (line, bar, scatter, histogram, box, area, heatmap, pie) instead of defaulting to only timeline/bar views.",
                "Use only column names that exist in the provided schema.",
                "For visual mappings (`x`, `y`, `color`, and optional encodings such as `shape`, `size`, `symbol`, `facet_row`, `facet_col`), select only schema-present columns.",
                "Required plotting fields must exist; optional encodings should be omitted when unavailable.",
                "If a desired grouping dimension is missing, either omit that encoding or propose an explicit transform that creates the field before visualization.",
                "Do not reference downstream flattened fields unless the same response also recommends the transform that creates them (for example, `flatten_nested` with source columns).",
                "Do not invent derived field names unless the response also includes the transform step that creates them.",
                "Return concise decision notes instead of chain-of-thought.",
            ],
        }
        logger_ctx.log_json("Metrics/Data Quality Prompt", prompt)

        try:
            response = self.client.responses.parse(
                model=self.config.llm.model,
                temperature=self.config.llm.analysis_temperature,
                input=json.dumps(prompt, default=str),
                text_format=LLMAnalysisResponse,
            )
            parsed = response.output_parsed
            payload = response.output_text
            if payload:
                logger_ctx.log_block("Metrics/Data Quality Response", payload)

            # Convert LLM response to full AnalysisReport by adding data_schema
            if isinstance(parsed, LLMAnalysisResponse):
                llm_response = parsed
            elif parsed is not None:
                llm_response = LLMAnalysisResponse.model_validate(parsed)
            else:
                llm_response = LLMAnalysisResponse.model_validate_json(payload)

            return AnalysisReport(
                metrics=llm_response.metrics,
                quality=llm_response.quality,
                design=llm_response.design,
                sampled_rows=llm_response.sampled_rows,
                data_schema=schema,
            )
        except Exception as exc:  # pragma: no cover - network/env dependent
            logger.warning("LLM analysis failed, falling back to heuristic: {}", exc)
            return _heuristic_analysis(df, description)
