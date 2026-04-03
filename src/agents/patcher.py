from __future__ import annotations

import re
from typing import Iterable

from src.models import DashboardPatch, DashboardPatchOperation, DashboardSpec, VisualSpec


_CHART_TYPES = {
    "line chart": "line",
    "line": "line",
    "bar chart": "bar",
    "bar": "bar",
    "scatter plot": "scatter",
    "scatter": "scatter",
    "histogram": "histogram",
    "box plot": "box",
    "box": "box",
    "heat map": "heatmap",
    "heatmap": "heatmap",
    "area chart": "area",
    "area": "area",
    "pie chart": "pie",
    "pie": "pie",
}

_LAYOUTS = {"grid": "grid", "tabs": "tabs", "tabbed": "tabs", "sections": "sections"}
_THEMES = {"light": "light", "dark": "dark"}
_AGGREGATIONS = {"sum": "sum", "mean": "mean", "median": "median", "count": "count"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_visual_by_hint(visuals: Iterable[VisualSpec], prompt: str) -> VisualSpec | None:
    lowered = _normalize(prompt)
    for visual in visuals:
        title = _normalize(visual.title)
        if title and title in lowered:
            return visual
    if not visuals:
        return None
    return list(visuals)[0]


def _extract_chart_type(prompt: str) -> str | None:
    lowered = _normalize(prompt)
    for token, chart_type in _CHART_TYPES.items():
        if token in lowered:
            return chart_type
    return None


def _extract_layout(prompt: str) -> str | None:
    lowered = _normalize(prompt)
    for token, layout in _LAYOUTS.items():
        if token in lowered:
            return layout
    return None


def _extract_theme(prompt: str) -> str | None:
    lowered = _normalize(prompt)
    for token, theme in _THEMES.items():
        if f"{token} theme" in lowered or f"{token} mode" in lowered or lowered.endswith(token):
            return theme
    return None


def _extract_aggregation(prompt: str) -> str | None:
    lowered = _normalize(prompt)
    for token, aggregation in _AGGREGATIONS.items():
        if f"use {token}" in lowered or f"{token} instead of" in lowered:
            return aggregation
    return None


def _extract_filter_column(prompt: str) -> str | None:
    match = re.search(r"(?:filter for|filter by|add filter for|remove filter for)\s+([a-zA-Z0-9_ ]+)", prompt, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().replace(" ", "_")


def _extract_field_remap(prompt: str) -> tuple[str, str] | None:
    match = re.search(r"(?:set|change)\s+(x|y|color)\s+(?:to|as)\s+([a-zA-Z0-9_]+)", prompt, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), match.group(2)


def _extract_visual_addition(prompt: str) -> dict[str, str] | None:
    match = re.search(
        r"add (?:a )?(line|bar|scatter|histogram|box|heatmap|area|pie)?(?: chart| plot| visualization)?(?: for)?\s*([a-zA-Z0-9_]+)?(?: by ([a-zA-Z0-9_]+))?",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    chart_type = match.group(1).lower() if match.group(1) else "bar"
    metric = match.group(2) or ""
    dimension = match.group(3) or ""
    return {"chart_type": chart_type, "metric": metric, "dimension": dimension}


def parse_update_prompt(prompt: str, spec: DashboardSpec) -> DashboardPatch:
    operations: list[DashboardPatchOperation] = []
    warnings: list[str] = []
    target_visual = _find_visual_by_hint(spec.visuals, prompt)
    chart_type = _extract_chart_type(prompt)
    layout = _extract_layout(prompt)
    theme = _extract_theme(prompt)
    aggregation = _extract_aggregation(prompt)
    field_remap = _extract_field_remap(prompt)
    filter_column = _extract_filter_column(prompt)
    lowered = _normalize(prompt)

    if "remove filter" in lowered and filter_column:
        operations.append(
            DashboardPatchOperation(
                op="remove_filter",
                value={"filter": filter_column},
                note=f"Remove filter `{filter_column}`.",
            )
        )
    elif ("add filter" in lowered or "filter by" in lowered or "filter for" in lowered) and filter_column:
        operations.append(
            DashboardPatchOperation(
                op="add_filter",
                value={"filter": filter_column},
                note=f"Add filter `{filter_column}`.",
            )
        )

    if layout:
        operations.append(
            DashboardPatchOperation(op="change_layout", value={"layout": layout}, note=f"Switch layout to `{layout}`.")
        )

    if theme:
        operations.append(
            DashboardPatchOperation(op="change_theme", value={"theme": theme}, note=f"Switch theme to `{theme}`.")
        )

    if aggregation and target_visual:
        operations.append(
            DashboardPatchOperation(
                op="change_aggregation",
                target_visual_id=target_visual.id,
                value={"aggregation": aggregation},
                note=f"Use `{aggregation}` aggregation for `{target_visual.title}`.",
            )
        )

    if field_remap and target_visual:
        target_field, column = field_remap
        operations.append(
            DashboardPatchOperation(
                op="remap_field",
                target_visual_id=target_visual.id,
                target_field=target_field,  # type: ignore[arg-type]
                value={"column": column},
                note=f"Remap `{target_field}` to `{column}`.",
            )
        )

    if chart_type and target_visual:
        operations.append(
            DashboardPatchOperation(
                op="replace_visual",
                target_visual_id=target_visual.id,
                value={"chart_type": chart_type},
                note=f"Replace `{target_visual.title}` with a `{chart_type}` chart.",
            )
        )

    if "remove visual" in lowered or "remove chart" in lowered:
        if target_visual:
            operations.append(
                DashboardPatchOperation(
                    op="remove_visual",
                    target_visual_id=target_visual.id,
                    note=f"Remove `{target_visual.title}`.",
                )
            )
        else:
            warnings.append("No visual matched the removal request.")

    if "add visual" in lowered or "add chart" in lowered or "add visualization" in lowered:
        added = _extract_visual_addition(prompt)
        if added:
            operations.append(
                DashboardPatchOperation(
                    op="add_visual",
                    value=added,
                    note="Add a new visual from the request.",
                )
            )

    if not operations:
        warnings.append("No structured patch operation could be inferred; keeping the current dashboard spec.")
    return DashboardPatch(prompt=prompt, operations=operations, warnings=warnings)


def _target_indices(spec: DashboardSpec, operation: DashboardPatchOperation) -> list[int]:
    if operation.target_visual_id:
        return [index for index, visual in enumerate(spec.visuals) if visual.id == operation.target_visual_id]
    if operation.target_visual_title:
        lowered = _normalize(operation.target_visual_title)
        return [
            index for index, visual in enumerate(spec.visuals) if lowered in _normalize(visual.title)
        ]
    return [0] if spec.visuals else []


def apply_dashboard_patch(spec: DashboardSpec, patch: DashboardPatch) -> DashboardSpec:
    result = spec.model_copy(deep=True)
    for operation in patch.operations:
        if operation.op == "change_layout":
            layout = operation.value.get("layout")
            if layout in _LAYOUTS.values():
                result.layout = layout
        elif operation.op == "change_theme":
            theme = operation.value.get("theme")
            if theme in _THEMES.values():
                result.theme = theme
        elif operation.op == "add_filter":
            filter_name = str(operation.value.get("filter", "")).strip()
            if filter_name and filter_name not in result.filters:
                result.filters.append(filter_name)
        elif operation.op == "remove_filter":
            filter_name = str(operation.value.get("filter", "")).strip()
            result.filters = [item for item in result.filters if item != filter_name]
        elif operation.op == "add_visual":
            metric = str(operation.value.get("metric", "")).strip() or None
            dimension = str(operation.value.get("dimension", "")).strip() or None
            chart_type = str(operation.value.get("chart_type", "bar")).strip() or "bar"
            result.visuals.append(
                VisualSpec(
                    title="Generated Visual" if not metric else f"{metric} by {dimension or metric}",
                    chart_type=chart_type,  # type: ignore[arg-type]
                    x=dimension,
                    y=metric if metric != dimension else None,
                    description=operation.note or "Added from dashboard patch request.",
                    rationale=operation.note,
                    status="revised",
                )
            )
        else:
            indices = _target_indices(result, operation)
            if not indices:
                continue
            for index in indices:
                visual = result.visuals[index]
                if operation.op == "remove_visual":
                    result.visuals.pop(index)
                    break
                if operation.op == "replace_visual":
                    chart_type = operation.value.get("chart_type")
                    if chart_type in _CHART_TYPES.values():
                        visual.chart_type = chart_type
                        visual.status = "revised"
                        visual.rationale = operation.note or visual.rationale
                elif operation.op == "change_aggregation":
                    aggregation = operation.value.get("aggregation")
                    if aggregation in _AGGREGATIONS.values():
                        visual.aggregation = aggregation
                        visual.status = "revised"
                        visual.rationale = operation.note or visual.rationale
                elif operation.op == "remap_field":
                    column = str(operation.value.get("column", "")).strip()
                    if column and operation.target_field:
                        setattr(visual, operation.target_field, column)
                        visual.status = "revised"
                        visual.rationale = operation.note or visual.rationale

    result.approval_status = "reviewed"
    if patch.warnings:
        result.assumptions.extend([warning for warning in patch.warnings if warning not in result.assumptions])
    return result
