from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from textwrap import dedent
from typing import Any

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.models import DashboardSpec, SessionState, VisualSpec
from src.tools.visualization import create_figure, error_figure

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

DEFAULT_NOTES = (
    "Adaptive analytics workspace tuned for high-clarity exploration, responsive charting, "
    "and rapid signal discovery."
)

DATE_HINT_TOKENS = ("date", "time", "timestamp", "_at", "_on")
DATE_VALUE_PATTERN = re.compile(
    r"(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|"
    r"[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}"
    r")"
)


@dataclass
class DashboardResult:
    app: Dash
    spec: DashboardSpec


def _store_payload(session_state: SessionState | None, design: DashboardSpec) -> dict[str, Any]:
    if session_state is None:
        return {
            "plan": None,
            "trace": None,
            "status": design.approval_status,
            "spec_versions": [design.model_dump(mode="python")],
        }
    return session_state.model_dump(mode="python")


def _looks_date_like_name(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in DATE_HINT_TOKENS)


def _looks_date_like_values(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip()
    if sample.empty:
        return False

    sample = sample[sample.ne("")].head(25)
    if sample.empty:
        return False

    return sample.str.contains(DATE_VALUE_PATTERN, regex=True).mean() >= 0.8


def _coerce_datetime_filter_series(column: str, series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return None

    if not (_looks_date_like_name(column) or _looks_date_like_values(series)):
        return None

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    threshold = 0.7 if _looks_date_like_name(column) else 0.95
    if parsed.notna().mean() >= threshold:
        return parsed
    return None


def _is_date_filter(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return _coerce_datetime_filter_series(column, df[column]) is not None


def _serialize_date_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return timestamp.date().isoformat()


def _render_visual(df: pd.DataFrame, spec: VisualSpec, theme: str = "light") -> Any:
    try:
        return create_figure(df, spec, theme=theme)
    except Exception as exc:
        return error_figure(spec.title, f"Unable to render chart: {exc}", theme=theme)


def _display_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _theme_badge(theme: str) -> str:
    return "Nebula Dark Mode" if theme == "dark" else "Photon Light Mode"


def _theme_status(theme: str) -> str:
    return "Dark spectrum engaged" if theme == "dark" else "Light spectrum engaged"


def _root_class(theme: str) -> str:
    return f"dashboard-shell theme-{theme}"


def _visual_caption(spec: VisualSpec) -> str:
    if spec.description:
        return spec.description
    axes: list[str] = []
    if spec.x:
        axes.append(f"x: {_display_name(spec.x)}")
    if spec.y:
        axes.append(f"y: {_display_name(spec.y)}")
    if spec.color:
        axes.append(f"color: {_display_name(spec.color)}")
    return " | ".join(axes) if axes else "Generated from the current dataset context."


def _bullet_list(items: list[str], empty_message: str) -> Any:
    values = items or [empty_message]
    return html.Ul([html.Li(item) for item in values], className="sidebar-list")


def _workspace_status(design: DashboardSpec, session_state: SessionState | None) -> str:
    if session_state is None:
        return "Plan status: local dashboard view"
    return f"Plan status: {session_state.status}"


def _version_options(spec_versions: list[DashboardSpec]) -> list[dict[str, Any]]:
    return [
        {"label": f"Version {index + 1}", "value": index}
        for index, _spec in enumerate(spec_versions)
    ]


def _resolve_spec_versions(session_state: SessionState | None, design: DashboardSpec) -> list[DashboardSpec]:
    if session_state is None or not session_state.spec_versions:
        return [design]
    return session_state.spec_versions


def _serialize_spec_versions(spec_versions: list[DashboardSpec]) -> list[dict[str, Any]]:
    return [spec.model_dump(mode="python") for spec in spec_versions]


def _deserialize_spec_versions(payload: list[dict[str, Any]] | None, fallback: list[DashboardSpec]) -> list[DashboardSpec]:
    if not payload:
        return fallback
    return [DashboardSpec.model_validate(item) for item in payload]


def _all_filter_columns(spec_versions: list[DashboardSpec]) -> list[str]:
    filters: list[str] = []
    for spec in spec_versions:
        for filter_name in spec.filters:
            if filter_name not in filters:
                filters.append(filter_name)
    return filters


def _version_summary(spec_versions: list[DashboardSpec], version_index: int, compare_mode: bool) -> list[str]:
    active = spec_versions[version_index]
    summary = [
        f"Layout: {active.layout}",
        f"Theme: {active.theme}",
        f"Visuals: {len(active.visuals)}",
        f"Filters: {len(active.filters)}",
    ]
    if compare_mode and version_index > 0:
        previous = spec_versions[version_index - 1]
        summary.append(
            f"Compared with Version {version_index}: visuals {len(previous.visuals)} -> {len(active.visuals)}, "
            f"filters {len(previous.filters)} -> {len(active.filters)}"
        )
    return summary


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.select_dtypes(include="number").columns]


def _dimension_columns(df: pd.DataFrame) -> list[str]:
    dimensions: list[str] = []
    numeric_columns = set(_numeric_columns(df))
    for column in df.columns:
        series = df[column]
        if column not in numeric_columns:
            dimensions.append(column)
            continue
        if series.nunique(dropna=True) <= 16:
            dimensions.append(column)
    return dimensions


def _match_prompt_columns(df: pd.DataFrame, prompt: str) -> list[str]:
    if not prompt.strip():
        return []
    normalized_prompt = _normalize_phrase(prompt)
    matches: list[str] = []
    for column in df.columns:
        label = _normalize_phrase(column)
        if label and (label in normalized_prompt or label.replace(" ", "") in normalized_prompt.replace(" ", "")):
            matches.append(column)
    return matches


def _keyword_chart_type(prompt: str) -> str | None:
    lowered = prompt.lower()
    for needle, chart_type in [
        ("heatmap", "heatmap"),
        ("heat map", "heatmap"),
        ("scatter", "scatter"),
        ("box", "box"),
        ("histogram", "histogram"),
        ("line", "line"),
        ("area", "area"),
        ("pie", "pie"),
        ("bar", "bar"),
    ]:
        if needle in lowered:
            return chart_type
    return None


def _prompt_time_grain(prompt: str) -> str | None:
    lowered = prompt.lower()
    if any(token in lowered for token in [" by day", " daily", " per day", "group by date", "group by day"]):
        return "day"
    if any(token in lowered for token in [" by week", " weekly", " per week", "group by week"]):
        return "week"
    if any(token in lowered for token in [" by month", " monthly", " per month", "group by month"]):
        return "month"
    return None


def _next_chart_type(current: str) -> str:
    cycle = ["bar", "line", "scatter", "box", "histogram", "area", "pie", "heatmap"]
    if current not in cycle:
        return cycle[0]
    return cycle[(cycle.index(current) + 1) % len(cycle)]


def _first_distinct(items: list[str], exclude: set[str] | None = None) -> str | None:
    blocked = exclude or set()
    for item in items:
        if item not in blocked:
            return item
    return None


def _regenerate_visual_spec(df: pd.DataFrame, spec: VisualSpec, prompt: str) -> VisualSpec:
    updated = spec.model_copy(deep=True)
    matched_columns = _match_prompt_columns(df, prompt)
    numeric_columns = _numeric_columns(df)
    dimension_columns = _dimension_columns(df)
    date_columns = [column for column in df.columns if _is_date_filter(df, column)]
    chart_type = _keyword_chart_type(prompt) or _next_chart_type(updated.chart_type)
    requested_time_grain = _prompt_time_grain(prompt)
    mentioned_numeric = [column for column in matched_columns if column in numeric_columns]
    mentioned_dimensions = [column for column in matched_columns if column in dimension_columns]

    default_metric = _first_distinct(mentioned_numeric) or updated.y or _first_distinct(numeric_columns) or updated.x
    default_dimension = (
        _first_distinct(mentioned_dimensions)
        or updated.x
        or _first_distinct(date_columns)
        or _first_distinct(dimension_columns)
        or _first_distinct(df.columns.tolist())
    )
    secondary_dimension = _first_distinct(
        [column for column in mentioned_dimensions + dimension_columns if column != default_dimension],
        exclude={default_dimension} if default_dimension else None,
    )
    secondary_metric = _first_distinct(
        [column for column in mentioned_numeric + numeric_columns if column != default_metric],
        exclude={default_metric} if default_metric else None,
    )

    updated.chart_type = chart_type  # type: ignore[assignment]
    updated.color = None
    updated.time_grain = requested_time_grain

    if chart_type in {"bar", "line", "area"}:
        updated.x = _first_distinct(date_columns) if requested_time_grain and date_columns else default_dimension
        updated.y = default_metric if default_metric in numeric_columns else None
        updated.aggregation = "mean" if updated.y else "count"
    elif chart_type == "scatter":
        updated.x = default_metric if default_metric in numeric_columns else _first_distinct(numeric_columns)
        updated.y = secondary_metric or updated.y or updated.x
        updated.aggregation = None
        updated.color = secondary_dimension
    elif chart_type == "box":
        updated.x = default_dimension
        updated.y = default_metric if default_metric in numeric_columns else _first_distinct(numeric_columns)
        updated.aggregation = None
        updated.color = secondary_dimension
    elif chart_type == "histogram":
        updated.x = default_metric if default_metric in numeric_columns else _first_distinct(numeric_columns)
        updated.y = None
        updated.aggregation = None
    elif chart_type == "pie":
        updated.x = default_dimension
        updated.y = default_metric if default_metric in numeric_columns else None
        updated.aggregation = "sum" if updated.y else "count"
    elif chart_type == "heatmap":
        updated.x = default_dimension
        updated.y = secondary_dimension or _first_distinct(dimension_columns, exclude={default_dimension} if default_dimension else None)
        updated.aggregation = "count"
    else:
        updated.x = _first_distinct(date_columns) if requested_time_grain and date_columns else default_dimension
        updated.y = default_metric if default_metric in numeric_columns else updated.y
        updated.aggregation = updated.aggregation or "mean"

    focus_text = prompt.strip() or "a different analytical angle"
    updated.title = f"{spec.title} Reframed"
    updated.description = f"Regenerated to focus on {focus_text}."
    updated.rationale = f"Navigator regenerated this chart to focus on {focus_text} while preserving valid dataset fields."
    updated.warnings = [warning for warning in updated.warnings if "Regenerated" not in warning]
    updated.warnings.append(f"Regenerated from prompt: {focus_text}.")
    updated.confidence = min((updated.confidence or 0.62) + 0.04, 0.95)
    updated.status = "revised"
    return updated


def _llm_regenerate_visual_spec(
    df: pd.DataFrame,
    spec: VisualSpec,
    prompt: str,
    llm_api_key: str | None,
) -> VisualSpec | None:
    api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
    if not prompt.strip() or not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)
    schema = {column: str(dtype) for column, dtype in df.dtypes.items()}
    request = {
        "task": "Regenerate one dashboard visual from a user prompt.",
        "instructions": [
            "Return a valid VisualSpec using only columns present in the schema.",
            "Prefer a time field on x when the user asks for timeline/date/day/week/month grouping.",
            "Use `time_grain` when the user asks for day, week, or month grouping.",
            "Set aggregation when grouped rollups are needed.",
            "Do not invent fields that are not in the schema.",
            "Keep the title concise and user-facing.",
        ],
        "schema": schema,
        "current_spec": spec.model_dump(mode="python"),
        "prompt": prompt,
        "sample_rows": df.head(20).to_dict(orient="records"),
    }
    response = client.responses.parse(
        model="gpt-5.2",
        temperature=0.2,
        input=json.dumps(request, default=str),
        text_format=VisualSpec,
    )
    parsed = response.output_parsed
    if isinstance(parsed, VisualSpec):
        regenerated = parsed
    elif parsed is not None:
        regenerated = VisualSpec.model_validate(parsed)
    else:
        regenerated = VisualSpec.model_validate_json(response.output_text)

    return regenerated.model_copy(
        update={
            "id": spec.id,
            "pinned": spec.pinned,
            "source_dataframe_ref": regenerated.source_dataframe_ref or spec.source_dataframe_ref or "baseline",
            "description": regenerated.description or f"Regenerated to focus on {prompt.strip()}.",
            "rationale": regenerated.rationale or f"Navigator regenerated this chart around '{prompt.strip()}'.",
            "warnings": list(regenerated.warnings) + [f"Regenerated from prompt: {prompt.strip()}."],
            "confidence": regenerated.confidence if regenerated.confidence is not None else min((spec.confidence or 0.62) + 0.08, 0.96),
            "status": "revised",
        }
    )


def _quality_rows(design: DashboardSpec, session_state: SessionState | None) -> list[dict[str, str]]:
    if session_state is not None and session_state.analysis is not None:
        rows: list[dict[str, str]] = []
        for issue in session_state.analysis.quality.issues:
            columns = ", ".join(issue.columns) if issue.columns else "dataset-wide"
            summary = issue.action or f"{issue.type.replace('_', ' ').title()} on {columns}"
            rows.append({"severity": issue.severity, "summary": summary})
        if rows:
            return rows

    rows = []
    for item in design.data_quality_summary:
        match = re.search(r"\((high|medium|low)\)", item, flags=re.IGNORECASE)
        severity = (match.group(1).lower() if match else "low")
        rows.append({"severity": severity, "summary": item})
    return rows


def _quality_score(design: DashboardSpec, session_state: SessionState | None) -> int:
    penalties = {"high": 24, "medium": 12, "low": 5}
    score = 100
    for row in _quality_rows(design, session_state):
        score -= penalties.get(row["severity"], 5)
    return max(0, min(100, score))


def _quality_badge_label(severity: str) -> str:
    return severity.strip().title()


def _build_plan_panel(design: DashboardSpec, session_state: SessionState | None, spec_versions: list[DashboardSpec]) -> html.Section:
    plan = session_state.plan if session_state is not None else None
    return html.Section(
        id="plan-panel",
        className="panel-card sidebar-card",
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.Div(
                        [
                            html.H2("Navigator Plan"),
                            html.P("Inspect the current proposal, workflow stage, and analyst-facing guardrails."),
                        ]
                    ),
                    html.Span(_workspace_status(design, session_state), id="workspace-status", className="section-meta"),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Workflow", className="control-label"),
                    _bullet_list(
                        list(session_state.trace.workflow) if session_state is not None else ["profile", "propose_plan", "approve_edit", "execute_review"],
                        "Workflow not available.",
                    ),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Proposal Summary", className="control-label"),
                    html.P(plan.summary if plan is not None else design.plan_summary or DEFAULT_NOTES, className="sidebar-copy"),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Profile Notes", className="control-label"),
                    _bullet_list(plan.profile_notes if plan is not None else design.assumptions, "No profile notes recorded."),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Version History", className="control-label"),
                    dcc.Dropdown(
                        id="version-select",
                        options=_version_options(spec_versions),
                        value=len(spec_versions) - 1,
                        clearable=False,
                        className="theme-dropdown",
                    ),
                ],
            ),
        ],
    )


def _build_workspace_controls(spec_versions: list[DashboardSpec]) -> html.Div:
    return html.Div(
        id="workspace-controls",
        className="action-bar action-bar--review",
        children=[
            html.Button("Approve Plan", id="approve-plan", n_clicks=0, className="action-button"),
            html.Button("Export Plan", id="export-plan", n_clicks=0, className="action-button action-button--subtle action-button--small"),
            html.Button(
                "Export Provenance",
                id="export-provenance",
                n_clicks=0,
                className="action-button action-button--subtle action-button--small",
            ),
            html.Button("Export Trace", id="export-trace", n_clicks=0, className="action-button action-button--subtle"),
            html.Span(
                f"{len(spec_versions)} tracked version(s)",
                id="action-status",
                className="section-meta",
            ),
            dcc.Download(id="plan-download"),
            dcc.Download(id="provenance-download"),
            dcc.Download(id="trace-download"),
        ],
    )


def _build_provenance_panel(design: DashboardSpec, session_state: SessionState | None, spec_versions: list[DashboardSpec]) -> html.Section:
    trace = session_state.trace if session_state is not None else None
    quality_rows = _quality_rows(design, session_state)
    quality_score = _quality_score(design, session_state)
    return html.Section(
        id="provenance-panel",
        className="panel-card sidebar-card provenance-card",
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.Div(
                        [
                            html.H2("Provenance Drawer"),
                            html.P("Why each chart exists, what transforms ran, and how revisions have changed the spec."),
                        ]
                    ),
                    html.Span("Inspectable run", className="section-meta"),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Data Quality", className="control-label"),
                    html.Div(
                        className="quality-score-card",
                        children=[
                            html.Span("Overall Quality Score", className="quality-score-label"),
                            html.Div(
                                [
                                    html.Span(str(quality_score), className="quality-score-value"),
                                    html.Span("/100", className="quality-score-scale"),
                                ],
                                className="quality-score-stack",
                            ),
                        ],
                    ),
                    html.Div(
                        className="quality-list",
                        children=[
                            html.Div(
                                className=f"quality-item quality-item--{row['severity']}",
                                children=[
                                    html.Span(_quality_badge_label(row["severity"]), className="quality-badge"),
                                    html.P(row["summary"], className="quality-copy"),
                                ],
                            )
                            for row in quality_rows
                        ]
                        or [
                            html.Div(
                                className="quality-item quality-item--neutral",
                                children=[
                                    html.Span("Clear", className="quality-badge"),
                                    html.P("No material quality issues were recorded for this run.", className="quality-copy"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Transform History", className="control-label"),
                    _bullet_list(design.transform_history, "No transforms recorded."),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Version Diff", className="control-label"),
                    html.Div(id="version-summary", className="sidebar-copy"),
                ],
            ),
            html.Div(
                className="sidebar-section",
                children=[
                    html.Span("Review Notes", className="control-label"),
                    _bullet_list(trace.repair_notes if trace is not None else [], "No critic repairs were needed."),
                ],
            ),
        ],
    )


def _plan_export_markdown(payload: dict[str, Any], spec_versions: list[DashboardSpec], version_index: int) -> str:
    plan = payload.get("plan") or {}
    active_spec = spec_versions[version_index]
    lines = [
        "# Navigator Plan",
        "",
        f"- Session: {payload.get('session_id', 'local-session')}",
        f"- Status: {payload.get('status', active_spec.approval_status)}",
        f"- Active Version: {version_index + 1}",
        "",
        "## Summary",
        plan.get("summary") or active_spec.plan_summary or DEFAULT_NOTES,
        "",
        "## Workflow",
    ]
    for step in ((payload.get("trace") or {}).get("workflow") or []):
        lines.append(f"- {step}")
    lines.extend(["", "## Profile Notes"])
    for note in plan.get("profile_notes") or active_spec.assumptions or ["No profile notes recorded."]:
        lines.append(f"- {note}")
    lines.extend(["", "## Visual Queue"])
    for visual in active_spec.visuals:
        lines.append(f"- {visual.title} [{visual.chart_type}]")
    return "\n".join(lines)


def _provenance_export_markdown(
    payload: dict[str, Any],
    spec_versions: list[DashboardSpec],
    version_index: int,
    compare_mode: bool,
) -> str:
    active_spec = spec_versions[version_index]
    trace = payload.get("trace") or {}
    lines = [
        "# Provenance Drawer",
        "",
        f"- Quality Score: {_quality_score(active_spec, None)}/100",
        f"- Active Version: {version_index + 1}",
        "",
        "## Data Quality",
    ]
    for row in _quality_rows(active_spec, None) or [{"severity": "neutral", "summary": "No material quality issues recorded."}]:
        lines.append(f"- {row['severity'].title()}: {row['summary']}")
    lines.extend(["", "## Transform History"])
    for item in active_spec.transform_history or ["No transforms recorded."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Version Diff"])
    for item in _version_summary(spec_versions, version_index, compare_mode):
        lines.append(f"- {item}")
    lines.extend(["", "## Review Notes"])
    for item in trace.get("repair_notes") or ["No critic repairs were needed."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Visual Rationales"])
    for visual in active_spec.visuals:
        lines.append(f"- {visual.title}: {visual.rationale or 'No rationale recorded.'}")
    return "\n".join(lines)


def _workspace_grid_class(approved: bool) -> str:
    return "workspace-grid workspace-grid--approved" if approved else "workspace-grid"


def _workspace_main_class(approved: bool) -> str:
    return "workspace-main workspace-main--approved" if approved else "workspace-main"


def _sidebar_panel_class(base: str, approved: bool) -> str:
    return f"{base} sidebar-card sidebar-card--hidden" if approved else f"{base} sidebar-card"


def _filter_panel_class(approved: bool) -> str:
    return "panel-card filter-panel filter-panel--compact" if approved else "panel-card filter-panel"


def _controls_class(approved: bool) -> str:
    return "action-bar action-bar--approved" if approved else "action-bar action-bar--review"


def _tab_bar_class(approved: bool, visual_count: int) -> str:
    if approved or visual_count <= 1:
        return "visual-tabs visual-tabs--hidden"
    return "visual-tabs"


def _tab_button_class(selected: bool, hidden: bool = False) -> str:
    if hidden:
        return "visual-tab-button visual-tab-button--hidden"
    if selected:
        return "visual-tab-button visual-tab-button--selected"
    return "visual-tab-button"


def _dashboard_styles() -> str:
    return dedent(
        """
        :root {
            color-scheme: dark;
        }
        * {
            box-sizing: border-box;
        }
        html, body {
            margin: 0;
            padding: 0;
            min-height: 100%;
            background: #050816;
        }
        body {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        }
        .dashboard-shell {
            --bg-primary: #f4f7ff;
            --bg-secondary: #e9f4ff;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: #ffffff;
            --surface-soft: #edf3ff;
            --border: rgba(71, 103, 190, 0.18);
            --border-strong: rgba(71, 103, 190, 0.32);
            --text-primary: #10203d;
            --text-secondary: #5f6f8e;
            --accent-primary: #246bff;
            --accent-secondary: #11c7b3;
            --accent-tertiary: #6d5efc;
            --chip-bg: rgba(36, 107, 255, 0.08);
            --shadow: 0 24px 60px rgba(72, 94, 161, 0.18);
            --grid-border: rgba(71, 103, 190, 0.12);
            min-height: 100vh;
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top left, rgba(36, 107, 255, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(17, 199, 179, 0.18), transparent 24%),
                linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            transition: background 0.3s ease, color 0.3s ease;
        }
        .dashboard-shell.theme-dark {
            --bg-primary: #050816;
            --bg-secondary: #0b1431;
            --surface: rgba(10, 18, 42, 0.78);
            --surface-strong: #0f1a38;
            --surface-soft: #132247;
            --border: rgba(101, 140, 255, 0.2);
            --border-strong: rgba(80, 227, 194, 0.34);
            --text-primary: #eef4ff;
            --text-secondary: #9dafd4;
            --accent-primary: #5ea1ff;
            --accent-secondary: #50e3c2;
            --accent-tertiary: #a46bff;
            --chip-bg: rgba(94, 161, 255, 0.12);
            --shadow: 0 26px 80px rgba(0, 0, 0, 0.42);
            --grid-border: rgba(101, 140, 255, 0.16);
        }
        .dashboard-shell::before,
        .dashboard-shell::after {
            content: "";
            position: absolute;
            inset: auto;
            width: 32rem;
            height: 32rem;
            border-radius: 50%;
            filter: blur(40px);
            pointer-events: none;
            opacity: 0.46;
        }
        .dashboard-shell::before {
            top: -8rem;
            right: -10rem;
            background: radial-gradient(circle, var(--accent-primary) 0%, transparent 62%);
        }
        .dashboard-shell::after {
            bottom: -12rem;
            left: -8rem;
            background: radial-gradient(circle, var(--accent-secondary) 0%, transparent 56%);
        }
        .dashboard-frame {
            position: relative;
            z-index: 1;
            max-width: 1540px;
            margin: 0 auto;
            padding: 32px 24px 48px;
        }
        .hero-panel,
        .panel-card,
        .graph-card,
        .empty-state {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 28px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }
        .hero-panel {
            padding: 20px 22px;
            margin-bottom: 18px;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.6fr);
            gap: 16px;
            align-items: start;
        }
        .eyebrow-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-strong);
            color: var(--accent-secondary);
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 12px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .dashboard-title {
            margin: 10px 0 8px;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 0.98;
            letter-spacing: -0.04em;
        }
        .dashboard-copy {
            margin: 0;
            max-width: 52rem;
            color: var(--text-secondary);
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .signal-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 16px;
        }
        .signal-chip {
            min-width: 104px;
            padding: 10px 12px;
            border-radius: 20px;
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            border: 1px solid var(--border);
        }
        .signal-chip-label {
            display: block;
            color: var(--text-secondary);
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .signal-chip-value {
            display: block;
            margin-top: 4px;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        .theme-panel {
            padding: 16px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent);
            border: 1px solid var(--border);
        }
        .theme-panel h2 {
            margin: 0;
            font-size: 1rem;
        }
        .theme-panel p {
            margin: 6px 0 0;
            color: var(--text-secondary);
            line-height: 1.5;
            font-size: 0.92rem;
        }
        .theme-toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-top: 18px;
        }
        .theme-status {
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 12px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--accent-primary);
        }
        .theme-toggle.form-switch {
            margin: 0;
            padding-left: 0;
        }
        .theme-toggle .form-check-input {
            width: 3.4rem;
            height: 1.9rem;
            margin-left: 0;
            background-color: rgba(255, 255, 255, 0.12);
            border-color: var(--border-strong);
            box-shadow: none;
            cursor: pointer;
        }
        .theme-toggle .form-check-input:checked {
            background-color: var(--accent-primary);
            border-color: var(--accent-primary);
        }
        .panel-card {
            padding: 20px;
            margin-bottom: 18px;
        }
        .section-heading {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 18px;
        }
        .section-heading h2 {
            margin: 0;
            font-size: 1.1rem;
        }
        .section-heading p {
            margin: 8px 0 0;
            color: var(--text-secondary);
        }
        .section-meta {
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border);
            color: var(--accent-primary);
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 12px;
            white-space: nowrap;
        }
        .workspace-grid {
            display: grid;
            grid-template-columns: minmax(260px, 0.9fr) minmax(0, 1.6fr) minmax(260px, 0.95fr);
            gap: 22px;
            align-items: start;
        }
        .workspace-grid.workspace-grid--approved {
            grid-template-columns: minmax(0, 1fr);
        }
        .workspace-main {
            min-width: 0;
        }
        .workspace-main.workspace-main--approved {
            grid-column: 1 / -1;
        }
        .sidebar-card {
            position: sticky;
            top: 20px;
        }
        .sidebar-card.sidebar-card--hidden {
            display: none;
        }
        .sidebar-section + .sidebar-section {
            margin-top: 18px;
        }
        .sidebar-list {
            margin: 0;
            padding-left: 18px;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .sidebar-copy {
            margin: 0;
            color: var(--text-secondary);
            line-height: 1.7;
        }
        .action-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            margin-bottom: 14px;
        }
        .action-bar.action-bar--approved #approve-plan {
            display: none;
        }
        .action-button {
            border: 1px solid var(--border-strong);
            border-radius: 999px;
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            color: var(--text-primary);
            padding: 10px 16px;
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
        }
        .action-button--subtle {
            border-color: var(--border);
            color: var(--text-secondary);
        }
        .action-button--small {
            padding: 8px 12px;
            font-size: 11px;
        }
        .filter-field {
            height: 100%;
            padding: 14px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), transparent);
            border: 1px solid var(--grid-border);
        }
        .filter-panel .section-heading p {
            font-size: 0.92rem;
        }
        .filter-panel.filter-panel--compact {
            padding: 14px 16px;
        }
        .filter-panel.filter-panel--compact .section-heading {
            margin-bottom: 10px;
        }
        .filter-panel.filter-panel--compact .section-heading h2 {
            font-size: 0.96rem;
        }
        .filter-panel.filter-panel--compact .section-heading p {
            display: none;
        }
        .filter-panel.filter-panel--compact .filter-field {
            padding: 10px 12px;
            border-radius: 18px;
        }
        .filter-panel.filter-panel--compact .control-label {
            margin-bottom: 6px;
            font-size: 10px;
        }
        .control-label {
            display: block;
            margin-bottom: 12px;
            color: var(--text-secondary);
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 12px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .theme-dropdown .Select-control,
        .theme-dropdown .Select-menu-outer,
        .DateRangePickerInput,
        .DateInput,
        .DateInput_input {
            background: var(--surface-soft) !important;
            border-color: var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 16px !important;
            box-shadow: none !important;
        }
        .theme-dropdown .Select-control,
        .DateRangePickerInput {
            min-height: 46px !important;
        }
        .theme-dropdown .Select-placeholder,
        .theme-dropdown .Select-value-label,
        .theme-dropdown .Select-input input,
        .theme-dropdown .Select-arrow-zone,
        .DateInput_input {
            color: var(--text-primary) !important;
        }
        .theme-dropdown .Select-menu-outer {
            margin-top: 8px;
            border: 1px solid var(--border) !important;
        }
        .dashboard-shell.theme-dark .theme-dropdown .Select-control,
        .dashboard-shell.theme-dark .theme-dropdown .Select-menu-outer,
        .dashboard-shell.theme-dark .DateRangePickerInput,
        .dashboard-shell.theme-dark .DateInput,
        .dashboard-shell.theme-dark .DateInput_input {
            background: #122146 !important;
        }
        .theme-dropdown .Select-option {
            background: var(--surface-soft) !important;
            color: var(--text-primary) !important;
        }
        .theme-dropdown .Select-option.is-focused {
            background: var(--chip-bg) !important;
        }
        .dashboard-shell.theme-dark .theme-dropdown .Select-placeholder,
        .dashboard-shell.theme-dark .theme-dropdown .Select-value-label,
        .dashboard-shell.theme-dark .theme-dropdown .Select-input input,
        .dashboard-shell.theme-dark .theme-dropdown .Select-arrow-zone,
        .dashboard-shell.theme-dark .DateInput_input {
            color: #e7f0ff !important;
            opacity: 1 !important;
        }
        .DateRangePickerInput_arrow_svg,
        .DateInput_input__focused,
        .CalendarMonth_caption,
        .DayPicker_weekHeader_li,
        .DayPickerNavigation_button__default {
            color: var(--text-primary) !important;
            fill: var(--text-primary) !important;
        }
        .CalendarMonth,
        .DayPicker,
        .DayPicker_transitionContainer,
        .CalendarDay__default,
        .CalendarMonthGrid {
            background: var(--surface-strong) !important;
            color: var(--text-primary) !important;
        }
        .CalendarDay__selected,
        .CalendarDay__selected_span {
            background: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            color: #ffffff !important;
        }
        .dashboard-section {
            margin-bottom: 12px;
        }
        .visual-tabs {
            margin-bottom: 16px;
        }
        .visual-tabs.visual-tabs--hidden {
            display: none;
        }
        .visual-tab-strip {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 4px;
            scrollbar-width: thin;
        }
        .visual-tab-button {
            flex: 0 0 auto;
            max-width: 220px;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-primary);
            padding: 10px 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 0.92rem;
            text-align: left;
            cursor: pointer;
        }
        .visual-tab-button.visual-tab-button--selected {
            border-color: var(--border-strong);
            background: var(--chip-bg);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
        }
        .visual-tab-button.visual-tab-button--hidden {
            display: none;
        }
        .graph-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 22px;
        }
        .graph-slot.graph-slot--hidden {
            display: none;
        }
        .graph-slot.graph-slot--focus {
            grid-column: 1 / -1;
        }
        .graph-card {
            padding: 18px 18px 10px;
            min-height: 100%;
        }
        .graph-card.graph-card--hidden {
            display: none;
        }
        .graph-meta {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 8px;
            padding: 0 6px;
        }
        .graph-title {
            margin: 0 6px 12px;
            font-size: clamp(1.3rem, 2.4vw, 1.9rem);
            line-height: 1.15;
            letter-spacing: -0.03em;
        }
        .graph-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 8px 6px 12px;
        }
        .graph-action-button {
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--surface-soft);
            color: var(--text-primary);
            padding: 8px 12px;
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
        }
        .graph-prompt-input {
            flex: 1 1 260px;
            min-height: 40px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--surface-soft);
            color: var(--text-primary);
            padding: 10px 14px;
            outline: none;
        }
        .graph-prompt-input::placeholder {
            color: var(--text-secondary);
            opacity: 0.95;
        }
        .graph-rationale {
            margin: 0 6px 12px;
            color: var(--text-secondary);
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .workspace-grid.workspace-grid--approved .graph-slot {
            display: block;
        }
        .workspace-grid.workspace-grid--approved .graph-slot.graph-slot--full {
            grid-column: auto;
        }
        .workspace-grid.workspace-grid--approved .graph-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .workspace-grid.workspace-grid--approved .graph-card {
            min-height: 100%;
        }
        .graph-copy {
            margin: 0;
            color: var(--text-secondary);
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .graph-kind {
            align-self: flex-start;
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border);
            color: var(--accent-secondary);
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .quality-score-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
            padding: 14px 16px;
            border-radius: 18px;
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            border: 1px solid var(--border);
        }
        .quality-score-label {
            color: var(--text-secondary);
            font-size: 0.92rem;
        }
        .quality-score-stack {
            display: flex;
            align-items: baseline;
            gap: 4px;
        }
        .quality-score-value {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        }
        .quality-score-scale {
            color: var(--text-secondary);
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 0.9rem;
        }
        .quality-list {
            display: grid;
            gap: 10px;
        }
        .quality-item {
            padding: 12px 14px;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.03);
        }
        .quality-item--high {
            border-color: rgba(255, 108, 129, 0.5);
            background: rgba(255, 108, 129, 0.08);
        }
        .quality-item--medium {
            border-color: rgba(255, 191, 92, 0.45);
            background: rgba(255, 191, 92, 0.08);
        }
        .quality-item--low,
        .quality-item--neutral {
            border-color: rgba(80, 227, 194, 0.28);
            background: rgba(80, 227, 194, 0.06);
        }
        .quality-badge {
            display: inline-flex;
            margin-bottom: 8px;
            padding: 4px 9px;
            border-radius: 999px;
            font-family: 'IBM Plex Mono', 'Consolas', monospace;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            background: rgba(255, 255, 255, 0.1);
        }
        .quality-copy {
            margin: 0;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .dashboard-graph .js-plotly-plot .plotly,
        .dashboard-graph .js-plotly-plot,
        .dashboard-graph {
            border-radius: 24px;
            overflow: hidden;
        }
        .dashboard-shell .js-plotly-plot .plotly .modebar {
            background: rgba(0, 0, 0, 0.18);
            border-radius: 12px;
            padding: 4px;
        }
        .dashboard-shell .js-plotly-plot .plotly .modebar-btn path {
            fill: var(--text-secondary);
        }
        .empty-state {
            padding: 34px;
        }
        .empty-state h2 {
            margin: 14px 0 10px;
            font-size: 1.4rem;
        }
        .empty-state p {
            margin: 0;
            color: var(--text-secondary);
            line-height: 1.7;
            max-width: 52rem;
        }
        @media (max-width: 980px) {
            .dashboard-frame {
                padding: 22px 16px 36px;
            }
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .workspace-grid {
                grid-template-columns: 1fr;
            }
            .sidebar-card {
                position: static;
            }
            .graph-grid {
                grid-template-columns: 1fr;
            }
            .workspace-grid.workspace-grid--approved .graph-grid {
                grid-template-columns: 1fr;
            }
            .hero-panel,
            .panel-card,
            .graph-card,
            .empty-state {
                border-radius: 24px;
            }
            .dashboard-title {
                font-size: clamp(2.2rem, 11vw, 3.6rem);
            }
            .graph-meta,
            .section-heading,
            .theme-toggle-row {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        """
    )


def _dashboard_index_string() -> str:
    return (
        "<!DOCTYPE html>"
        "<html>"
        "<head>"
        "{%metas%}"
        "<title>{%title%}</title>"
        "{%favicon%}"
        "{%css%}"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap' rel='stylesheet'>"
        f"<style>{_dashboard_styles()}</style>"
        "</head>"
        "<body>"
        "{%app_entry%}"
        "<footer>"
        "{%config%}"
        "{%scripts%}"
        "{%renderer%}"
        "</footer>"
        "</body>"
        "</html>"
    )


def _build_filter_components(df: pd.DataFrame, filters: list[str]) -> list[Any]:
    components: list[Any] = []
    for column in filters:
        if column not in df.columns:
            continue
        parsed_dates = _coerce_datetime_filter_series(column, df[column])
        if parsed_dates is not None:
            series = parsed_dates.dropna()
            if series.empty:
                continue
            min_date = series.min()
            max_date = series.max()
            control = dcc.DatePickerRange(
                id=f"filter-{column}",
                start_date=_serialize_date_value(min_date),
                end_date=_serialize_date_value(max_date),
                min_date_allowed=_serialize_date_value(min_date),
                max_date_allowed=_serialize_date_value(max_date),
                display_format="MMM D, YYYY",
                className="theme-date-range",
                number_of_months_shown=1,
            )
        else:
            values = df[column].dropna().unique().tolist()
            options = [{"label": "All", "value": "__all__"}] + [
                {"label": str(value), "value": value} for value in values[:200]
            ]
            control = dcc.Dropdown(
                id=f"filter-{column}",
                options=options,
                value="__all__",
                clearable=False,
                searchable=True,
                className="theme-dropdown",
            )

        components.append(
            dbc.Col(
                html.Div(
                    [
                        html.Label(_display_name(column), className="control-label"),
                        control,
                    ],
                    className="filter-field",
                ),
                xs=12,
                md=6,
                xl=4,
            )
        )
    return components


def _apply_filters(df: pd.DataFrame, filters: list[str], values: dict[str, Any]) -> pd.DataFrame:
    result = df
    for column in filters:
        if column not in df.columns:
            continue
        value = values.get(column)
        if value is None:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            start, end = value
            if start and end:
                series = _coerce_datetime_filter_series(column, result[column])
                if series is None:
                    continue
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if len(str(end)) <= 10:
                    end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                result = result[series.between(start_ts, end_ts, inclusive="both")]
        elif value != "__all__":
            result = result[result[column] == value]
    return result


def _build_hero(design: DashboardSpec, df: pd.DataFrame, filter_count: int) -> html.Section:
    return html.Section(
        className="hero-panel",
        children=[
            html.Div(
                className="hero-grid",
                children=[
                    html.Div(
                        [
                            html.Span(_theme_badge("light"), id="theme-badge", className="eyebrow-pill"),
                            html.H1(design.title, className="dashboard-title"),
                            html.P(design.notes or DEFAULT_NOTES, className="dashboard-copy"),
                            html.Div(
                                className="signal-grid",
                                children=[
                                    html.Div(
                                        [
                                            html.Span("Rows", className="signal-chip-label"),
                                            html.Span(f"{len(df):,}", className="signal-chip-value"),
                                        ],
                                        className="signal-chip",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Visuals", className="signal-chip-label"),
                                            html.Span(str(len(design.visuals)), className="signal-chip-value"),
                                        ],
                                        className="signal-chip",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Filters", className="signal-chip-label"),
                                            html.Span(str(filter_count), className="signal-chip-value"),
                                        ],
                                        className="signal-chip",
                                    ),
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        className="theme-panel",
                        children=[
                            html.H2("Adaptive Theme Engine"),
                            html.P(
                                "Flip the interface between crafted light and dark palettes. "
                                "The surrounding UI and each Plotly visualization update together."
                            ),
                            html.Div(
                                className="theme-toggle-row",
                                children=[
                                    html.Span(_theme_status("light"), id="theme-status", className="theme-status"),
                                    dbc.Switch(id="theme-toggle", value=False, className="theme-toggle"),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


def _build_filter_panel(filter_components: list[Any], filter_count: int, approved: bool = False) -> html.Section | None:
    if not filter_components:
        return None
    return html.Section(
        id="filter-panel",
        className=_filter_panel_class(approved),
        children=[
            html.Div(
                className="section-heading",
                children=[
                    html.Div(
                        [
                            html.H2("Signal Filters"),
                            html.P("Constrain the active dataset slice and recompute every visual in real time."),
                        ]
                    ),
                    html.Span(f"{filter_count} active controls", className="section-meta"),
                ],
            ),
            dbc.Row(filter_components, className="g-4"),
        ],
    )


def _graph_card_class(spec: VisualSpec, pinned_ids: list[str] | None = None) -> str:
    if spec.description == "__hidden_slot__":
        return "graph-card graph-card--hidden"
    return "graph-card"


def _placeholder_visual(index: int) -> VisualSpec:
    return VisualSpec(
        title=f"Hidden Slot {index + 1}",
        chart_type="bar",
        description="__hidden_slot__",
        rationale="",
    )


def _visual_slots(spec: DashboardSpec, slot_count: int) -> list[VisualSpec]:
    visuals = list(spec.visuals)
    while len(visuals) < slot_count:
        visuals.append(_placeholder_visual(len(visuals)))
    return visuals


def _build_graph_card(
    df: pd.DataFrame,
    spec: VisualSpec,
    theme: str,
    index: int,
    pinned_ids: list[str] | None = None,
) -> html.Article:
    return html.Article(
        id=f"graph-card-{index}",
        className=_graph_card_class(spec, pinned_ids),
        children=[
            html.H3(spec.title, id=f"graph-title-{index}", className="graph-title"),
            html.Div(
                className="graph-meta",
                children=[
                    html.P(_visual_caption(spec), id=f"graph-caption-{index}", className="graph-copy"),
                    html.Span(spec.chart_type, id=f"graph-kind-{index}", className="graph-kind"),
                ],
            ),
            html.Div(
                className="graph-actions",
                children=[
                    html.Button("Regenerate", id=f"regenerate-{index}", n_clicks=0, className="graph-action-button"),
                    dcc.Input(
                        id=f"regenerate-prompt-{index}",
                        type="text",
                        value="",
                        debounce=True,
                        placeholder="Tell Navigator what this regeneration should focus on...",
                        className="graph-prompt-input",
                    ),
                ],
            ),
            html.P(
                spec.rationale or "Navigator rationale will appear here once the plan is available.",
                id=f"graph-rationale-{index}",
                className="graph-rationale",
            ),
            dcc.Graph(
                id=f"graph-{index}",
                figure=_render_visual(df, spec, theme),
                className="dashboard-graph",
                config=GRAPH_CONFIG,
            ),
        ],
    )


def _build_visual_section(
    df: pd.DataFrame,
    design: DashboardSpec,
    theme: str,
    approved: bool,
    focus_index: int,
    actual_visual_count: int | None = None,
    pinned_ids: list[str] | None = None,
) -> html.Section:
    visual_count = actual_visual_count if actual_visual_count is not None else len(design.visuals)
    if visual_count == 0:
        return html.Section(
            className="empty-state",
            children=[
                html.Span("Visualization Queue Empty", className="eyebrow-pill"),
                html.H2("No visualizations were produced for this dataset."),
                html.P(
                    "Try regenerating the dashboard with a more specific prompt, a different layout, "
                    "or a refined set of metrics and dimensions."
                ),
            ],
        )

    safe_focus = min(max(focus_index, 0), visual_count - 1)
    focus_spec = design.visuals[safe_focus]
    graph_slots = []
    for index, spec in enumerate(design.visuals):
        slot_class = "graph-slot graph-slot--full" if approved else "graph-slot graph-slot--focus"
        if spec.description == "__hidden_slot__":
            slot_class = "graph-slot graph-slot--hidden"
        elif not approved and index != safe_focus:
            slot_class = "graph-slot graph-slot--hidden"
        graph_slots.append(
            html.Div(
                _build_graph_card(df, spec, theme, index, pinned_ids=pinned_ids),
                id=f"graph-slot-{index}",
                className=slot_class,
            )
        )

    return html.Section(
        className="dashboard-section",
        children=[
            html.Div(
                id="visual-tabs",
                className=_tab_bar_class(approved, visual_count),
                children=[
                    html.Div(
                        className="visual-tab-strip",
                        children=[
                            html.Button(
                                f"{index + 1}. {spec.title}",
                                id=f"tab-{index}",
                                n_clicks=0,
                                className=_tab_button_class(index == safe_focus, spec.description == "__hidden_slot__"),
                            )
                            for index, spec in enumerate(design.visuals)
                        ],
                    ),
                ],
            ),
            html.Div(graph_slots, className="graph-grid"),
        ],
    )


def _filter_input_definitions(df: pd.DataFrame, filters: list[str]) -> list[tuple[str, bool]]:
    definitions: list[tuple[str, bool]] = []
    for column in filters:
        if column not in df.columns:
            continue
        definitions.append((column, _is_date_filter(df, column)))
    return definitions


def _collect_filter_values(columns: list[tuple[str, bool]], args: tuple[Any, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    arg_index = 0
    for column, is_date_filter in columns:
        if is_date_filter:
            values[column] = (args[arg_index], args[arg_index + 1])
            arg_index += 2
        else:
            values[column] = args[arg_index]
            arg_index += 1
    return values


def build_dashboard(
    df: pd.DataFrame,
    design: DashboardSpec,
    session_state: SessionState | None = None,
    llm_api_key: str | None = None,
) -> DashboardResult:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = design.title
    app.index_string = _dashboard_index_string()

    initial_theme = design.theme or "light"
    spec_versions = _resolve_spec_versions(session_state, design)
    all_filters = _all_filter_columns(spec_versions)
    filter_components = _build_filter_components(df, all_filters)
    filter_columns = _filter_input_definitions(df, all_filters)
    initial_version = len(spec_versions) - 1
    active_design = spec_versions[initial_version]
    max_visuals = max((len(spec.visuals) for spec in spec_versions), default=0)
    display_design = active_design.model_copy(update={"visuals": _visual_slots(active_design, max_visuals)})
    initial_approval = False
    initial_tab = 0

    app.layout = html.Div(
        id="dashboard-root",
        className=_root_class(initial_theme),
        children=[
            dcc.Store(id="session-state-store", data=_store_payload(session_state, design)),
            dcc.Store(id="spec-versions-store", data=_serialize_spec_versions(spec_versions)),
            dcc.Store(id="approval-store", data=initial_approval),
            dcc.Store(id="version-store", data=initial_version),
            dcc.Store(id="tab-store", data=initial_tab),
            html.Div(
                className="dashboard-frame",
                children=[
                    _build_hero(active_design, df, len(filter_components)),
                    *([_build_filter_panel(filter_components, len(filter_components), approved=initial_approval)] if filter_components else []),
                    html.Div(
                        id="workspace-grid",
                        className=_workspace_grid_class(initial_approval),
                        children=[
                            _build_plan_panel(active_design, session_state, spec_versions),
                            html.Div(
                                id="workspace-main",
                                className=_workspace_main_class(initial_approval),
                                children=[
                                    html.Section(
                                        id="workspace-panel",
                                        className="panel-card",
                                        children=[
                                            _build_workspace_controls(spec_versions),
                                            _build_visual_section(
                                                df,
                                                display_design,
                                                initial_theme,
                                                approved=initial_approval,
                                                focus_index=initial_tab,
                                                actual_visual_count=len(active_design.visuals),
                                            ),
                                        ],
                                    )
                                ],
                            ),
                            _build_provenance_panel(active_design, session_state, spec_versions),
                        ],
                    ),
                ],
            ),
        ],
    )

    inputs: list[Any] = [
        Input("theme-toggle", "value"),
        Input("version-store", "data"),
        Input("approval-store", "data"),
        Input("tab-store", "data"),
        Input("spec-versions-store", "data"),
    ]
    for column, is_date_filter in filter_columns:
        if is_date_filter:
            inputs.append(Input(f"filter-{column}", "start_date"))
            inputs.append(Input(f"filter-{column}", "end_date"))
        else:
            inputs.append(Input(f"filter-{column}", "value"))

    tab_inputs: list[Any] = []
    regenerate_inputs: list[Any] = []
    for index in range(max_visuals):
        tab_inputs.append(Input(f"tab-{index}", "n_clicks"))
        regenerate_inputs.append(Input(f"regenerate-{index}", "n_clicks"))

    regenerate_prompt_states = [State(f"regenerate-prompt-{index}", "value") for index in range(max_visuals)]

    shell_outputs = [
        Output("dashboard-root", "className"),
        Output("theme-status", "children"),
        Output("theme-badge", "children"),
    ]

    @app.callback(
        Output("approval-store", "data"),
        Output("version-store", "data"),
        Output("tab-store", "data"),
        Output("spec-versions-store", "data"),
        Output("version-select", "value"),
        Output("version-select", "options"),
        Output("workspace-status", "children"),
        Output("approve-plan", "children"),
        Output("action-status", "children"),
        Output("plan-download", "data"),
        Output("provenance-download", "data"),
        Output("trace-download", "data"),
        Input("approve-plan", "n_clicks"),
        Input("export-plan", "n_clicks"),
        Input("export-provenance", "n_clicks"),
        Input("export-trace", "n_clicks"),
        Input("version-select", "value"),
        *tab_inputs,
        *regenerate_inputs,
        State("approval-store", "data"),
        State("version-store", "data"),
        State("tab-store", "data"),
        State("session-state-store", "data"),
        State("spec-versions-store", "data"),
        *regenerate_prompt_states,
        prevent_initial_call=True,
    )
    def manage_workspace(
        approve_clicks,
        export_plan_clicks,
        export_provenance_clicks,
        export_clicks,
        selected_version,
        *args,
    ):
        state_offset = len(tab_inputs) + len(regenerate_inputs)
        approved = bool(args[state_offset])
        version_index = int(args[state_offset + 1] or 0)
        tab_index = int(args[state_offset + 2] or 0)
        payload = args[state_offset + 3] or {}
        spec_payload = args[state_offset + 4] or []
        regenerate_prompts = list(args[state_offset + 5 :])
        current_versions = _deserialize_spec_versions(spec_payload, spec_versions)
        triggered = ctx.triggered_id
        plan_download = no_update
        provenance_download = no_update
        trace_download = no_update
        status = _workspace_status(design, session_state)
        action_status = f"{len(current_versions)} tracked version(s)"
        approve_label = "Plan Approved" if approved else "Approve Plan"

        if not current_versions:
            current_versions = spec_versions
        version_index = min(max(version_index, 0), len(current_versions) - 1)
        tab_index = min(max(tab_index, 0), max(len(current_versions[version_index].visuals) - 1, 0))

        if triggered == "approve-plan":
            approved = True
            approved_spec = current_versions[version_index].model_copy(deep=True)
            approved_spec.layout = "grid"
            approved_spec.approval_status = "approved"
            approved_spec.visuals = [
                visual.model_copy(update={"status": "approved"})
                for visual in approved_spec.visuals
            ]
            current_versions.append(approved_spec)
            version_index = len(current_versions) - 1
            tab_index = 0
            status = "Plan status: dashboard canvas approved"
            action_status = "Approved plan and switched to the dashboard canvas"
            approve_label = "Plan Approved"
        elif triggered == "export-plan":
            filename = f"{payload.get('session_id', 'session')}_navigator_plan_v{version_index + 1}.md"
            plan_download = {
                "content": _plan_export_markdown(payload, current_versions, version_index),
                "filename": filename,
            }
            action_status = "Navigator plan export prepared"
        elif triggered == "export-provenance":
            filename = f"{payload.get('session_id', 'session')}_provenance_v{version_index + 1}.md"
            provenance_download = {
                "content": _provenance_export_markdown(payload, current_versions, version_index, False),
                "filename": filename,
            }
            action_status = "Provenance drawer export prepared"
        elif triggered == "export-trace":
            trace_payload = payload.get("trace", {})
            filename = f"{payload.get('session_id', 'session')}_trace.json"
            trace_download = {
                "content": json.dumps(trace_payload, indent=2, sort_keys=True),
                "filename": filename,
            }
            action_status = "Trace export prepared"
        elif triggered == "version-select":
            version_index = int(selected_version or 0)
            version_index = min(max(version_index, 0), len(current_versions) - 1)
            tab_index = min(tab_index, max(len(current_versions[version_index].visuals) - 1, 0))
            action_status = f"Viewing Version {version_index + 1}"
        elif isinstance(triggered, str) and triggered.startswith("tab-"):
            graph_index = int(triggered.split("-", maxsplit=1)[1])
            active_spec = current_versions[version_index]
            if graph_index < len(active_spec.visuals):
                tab_index = graph_index
                action_status = f"Viewing {active_spec.visuals[graph_index].title}"
        elif isinstance(triggered, str) and triggered.startswith("regenerate-"):
            graph_index = int(triggered.split("-", maxsplit=1)[1])
            active_spec = current_versions[version_index].model_copy(deep=True)
            if graph_index < len(active_spec.visuals):
                prompt = (regenerate_prompts[graph_index] or "").strip()
                regenerated = _llm_regenerate_visual_spec(df, active_spec.visuals[graph_index], prompt, llm_api_key)
                if regenerated is None:
                    regenerated = _regenerate_visual_spec(df, active_spec.visuals[graph_index], prompt)
                active_spec.visuals[graph_index] = regenerated
                active_spec.approval_status = "approved" if approved else "draft"
                current_versions.append(active_spec)
                version_index = len(current_versions) - 1
                tab_index = graph_index
                action_status = (
                    f"Regenerated {active_spec.visuals[graph_index].title} around '{prompt}'"
                    if prompt
                    else f"Regenerated chart {graph_index + 1} with a fresh analytical angle"
                )

        version_options = _version_options(current_versions)
        return (
            approved,
            version_index,
            tab_index,
            _serialize_spec_versions(current_versions),
            version_index,
            version_options,
            status,
            approve_label,
            action_status,
            plan_download,
            provenance_download,
            trace_download,
        )

    if max_visuals > 0:

        @app.callback(
            [Output(f"graph-{index}", "figure") for index in range(max_visuals)]
            + [Output(f"graph-slot-{index}", "className") for index in range(max_visuals)]
            + [Output(f"graph-card-{index}", "className") for index in range(max_visuals)]
            + [Output(f"graph-title-{index}", "children") for index in range(max_visuals)]
            + [Output(f"graph-caption-{index}", "children") for index in range(max_visuals)]
            + [Output(f"graph-kind-{index}", "children") for index in range(max_visuals)]
            + [Output(f"graph-rationale-{index}", "children") for index in range(max_visuals)]
            + [Output(f"tab-{index}", "className") for index in range(max_visuals)]
            + [Output("version-summary", "children")]
            + [Output("workspace-grid", "className")]
            + [Output("workspace-main", "className")]
            + [Output("plan-panel", "className")]
            + [Output("provenance-panel", "className")]
            + [Output("filter-panel", "className")]
            + [Output("workspace-controls", "className")]
            + [Output("visual-tabs", "className")]
            + shell_outputs,
            inputs,
        )
        def update_dashboard(
            is_dark: bool,
            version_index: int,
            approved: bool,
            tab_index: int,
            spec_payload: list[dict[str, Any]],
            *filter_args,
        ):
            theme = "dark" if is_dark else "light"
            live_versions = _deserialize_spec_versions(spec_payload, spec_versions)
            safe_version = min(max(int(version_index or 0), 0), len(live_versions) - 1)
            active_spec = live_versions[safe_version]
            visible_visuals = _visual_slots(active_spec, max_visuals)
            visual_count = len(active_spec.visuals)
            safe_tab = min(max(int(tab_index or 0), 0), max(visual_count - 1, 0))
            values = _collect_filter_values(filter_columns, filter_args)
            filtered = _apply_filters(df, active_spec.filters, values)
            figures = [_render_visual(filtered, spec, theme) for spec in visible_visuals]
            slot_classes = []
            for index, spec in enumerate(visible_visuals):
                if spec.description == "__hidden_slot__":
                    slot_classes.append("graph-slot graph-slot--hidden")
                elif approved:
                    slot_classes.append("graph-slot graph-slot--full")
                elif index == safe_tab:
                    slot_classes.append("graph-slot graph-slot--focus")
                else:
                    slot_classes.append("graph-slot graph-slot--hidden")
            card_classes = [_graph_card_class(spec, None) for spec in visible_visuals]
            titles = [spec.title for spec in visible_visuals]
            captions = [_visual_caption(spec) for spec in visible_visuals]
            kinds = [spec.chart_type for spec in visible_visuals]
            rationales = [
                spec.rationale or "Navigator rationale will appear here once the plan is available."
                for spec in visible_visuals
            ]
            tab_classes = [
                _tab_button_class(index == safe_tab, spec.description == "__hidden_slot__")
                for index, spec in enumerate(visible_visuals)
            ]
            version_summary = html.Ul(
                [html.Li(item) for item in _version_summary(live_versions, safe_version, False)],
                className="sidebar-list",
            )
            return (
                figures
                + slot_classes
                + card_classes
                + titles
                + captions
                + kinds
                + rationales
                + tab_classes
                + [
                    version_summary,
                    _workspace_grid_class(bool(approved)),
                    _workspace_main_class(bool(approved)),
                    _sidebar_panel_class("panel-card", bool(approved)),
                    _sidebar_panel_class("panel-card provenance-card", bool(approved)),
                    _filter_panel_class(bool(approved)),
                    _controls_class(bool(approved)),
                    _tab_bar_class(bool(approved), visual_count),
                    _root_class(theme),
                    _theme_status(theme),
                    _theme_badge(theme),
                ]
            )

    else:

        @app.callback(
            [
                Output("version-summary", "children"),
                Output("workspace-grid", "className"),
                Output("workspace-main", "className"),
                Output("plan-panel", "className"),
                Output("provenance-panel", "className"),
                Output("filter-panel", "className"),
                Output("workspace-controls", "className"),
            ]
            + shell_outputs,
            inputs,
        )
        def update_dashboard_shell(
            is_dark: bool,
            version_index: int,
            approved: bool,
            _tab_index,
            spec_payload: list[dict[str, Any]],
            *_filter_args,
        ):
            theme = "dark" if is_dark else "light"
            live_versions = _deserialize_spec_versions(spec_payload, spec_versions)
            safe_version = min(max(int(version_index or 0), 0), len(live_versions) - 1)
            version_summary = html.Ul(
                [html.Li(item) for item in _version_summary(live_versions, safe_version, False)],
                className="sidebar-list",
            )
            return (
                version_summary,
                _workspace_grid_class(bool(approved)),
                _workspace_main_class(bool(approved)),
                _sidebar_panel_class("panel-card", bool(approved)),
                _sidebar_panel_class("panel-card provenance-card", bool(approved)),
                _filter_panel_class(bool(approved)),
                _controls_class(bool(approved)),
                _root_class(theme),
                _theme_status(theme),
                _theme_badge(theme),
            )

    return DashboardResult(app=app, spec=design)
