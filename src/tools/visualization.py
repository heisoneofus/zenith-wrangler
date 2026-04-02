from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

from dash import Dash, html

from src.models import VisualSpec


OPTIONAL_ENCODING_FIELDS: tuple[str, ...] = (
    "color",
    "shape",
    "size",
    "symbol",
    "facet_row",
    "facet_col",
)


def _require_column(df: pd.DataFrame, column: str | None, field_name: str, chart_type: str) -> None:
    if not column:
        raise ValueError(f"{chart_type} charts require '{field_name}' to be set.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' required for {chart_type} chart is missing from dataset.")


def _infer_bar_axes(df: pd.DataFrame, spec: VisualSpec) -> VisualSpec:
    if spec.chart_type != "bar":
        return spec
    if spec.x is not None and spec.y is not None:
        return spec

    inferred_x = spec.x
    inferred_y = spec.y
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

    if inferred_y is None:
        inferred_y = next((column for column in numeric_columns if column != inferred_x), None)
    if inferred_x is None:
        inferred_x = next((column for column in df.columns if column != inferred_y), None)
    if inferred_y is None and inferred_x is not None:
        inferred_y = next((column for column in df.columns if column != inferred_x), None)

    return spec.model_copy(update={"x": inferred_x, "y": inferred_y})


def _required_fields(spec: VisualSpec) -> set[str]:
    if spec.chart_type in {"line", "scatter", "box", "area"}:
        return {"x", "y"}
    if spec.chart_type == "histogram":
        return {"x_or_y"}
    if spec.chart_type == "pie":
        return {"x"}
    return set()


def _optional_encodings(spec: VisualSpec) -> dict[str, str]:
    optional: dict[str, str] = {}
    for field_name in OPTIONAL_ENCODING_FIELDS:
        column = getattr(spec, field_name, None)
        if isinstance(column, str) and column:
            optional[field_name] = column
    return optional


def _sanitize_optional_encodings(df: pd.DataFrame, spec: VisualSpec) -> tuple[VisualSpec, list[str]]:
    updates: dict[str, None] = {}
    warnings: list[str] = []
    for field_name, column in _optional_encodings(spec).items():
        if column not in df.columns:
            updates[field_name] = None
            warnings.append(
                f"Optional encoding '{field_name}' dropped because column '{column}' is missing from dataset."
            )
    if not updates:
        return spec, warnings
    return spec.model_copy(update=updates), warnings


def _validate_spec(df: pd.DataFrame, spec: VisualSpec) -> None:
    required = _required_fields(spec)
    if {"x", "y"}.issubset(required):
        _require_column(df, spec.x, "x", spec.chart_type)
        _require_column(df, spec.y, "y", spec.chart_type)
    elif spec.chart_type == "bar":
        if spec.x is None and spec.y is None:
            raise ValueError("bar charts require either 'x' or 'y' to be set.")
        if spec.x is not None:
            _require_column(df, spec.x, "x", spec.chart_type)
        if spec.y is not None:
            _require_column(df, spec.y, "y", spec.chart_type)
    elif "x_or_y" in required:
        histogram_axis = spec.x or spec.y
        _require_column(df, histogram_axis, "x or y", spec.chart_type)
    elif spec.chart_type == "heatmap":
        if spec.x and spec.x not in df.columns:
            raise ValueError(f"Column '{spec.x}' required for heatmap chart is missing from dataset.")
        if spec.y and spec.y not in df.columns:
            raise ValueError(f"Column '{spec.y}' required for heatmap chart is missing from dataset.")
    if spec.aggregation:
        if spec.chart_type == "histogram":
            histogram_axis = spec.x or spec.y
            if not histogram_axis:
                raise ValueError("Histogram aggregation requires either 'x' or 'y' field.")
            if spec.aggregation != "count" and spec.y is None:
                raise ValueError("Histogram aggregation requires 'y' field unless aggregation is 'count'.")
        elif not (spec.x and spec.y):
            raise ValueError("Aggregation requires both 'x' and 'y' fields.")


def create_figure(df: pd.DataFrame, spec: VisualSpec):
    if df.empty:
        return error_figure(spec.title, "No data available to visualize.")

    spec = _infer_bar_axes(df, spec)
    spec, warnings = _sanitize_optional_encodings(df, spec)
    for warning in warnings:
        logger.warning(warning)
    _validate_spec(df, spec)
    data = df

    if spec.aggregation and spec.x and spec.y and spec.chart_type != "heatmap":
        group_keys = [spec.x]
        optional_grouping_columns = [
            column for column in _optional_encodings(spec).values() if column in df.columns
        ]
        for column in optional_grouping_columns:
            if column not in group_keys:
                group_keys.append(column)
        data = (
            df.groupby(group_keys, dropna=False)[spec.y]
            .agg(spec.aggregation)
            .reset_index()
        )
        if data.empty:
            return error_figure(spec.title, "Aggregation resulted in empty dataset.")

    chart = spec.chart_type
    if chart == "line":
        return px.line(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    if chart == "bar":
        return px.bar(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    if chart == "scatter":
        return px.scatter(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    if chart == "histogram":
        return px.histogram(data, x=spec.x or spec.y, color=spec.color, title=spec.title)
    if chart == "box":
        return px.box(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    if chart == "heatmap":
        if spec.x and spec.y:
            value_column = spec.color or spec.y
            if value_column not in data.columns:
                return error_figure(spec.title, f"Heatmap value column '{value_column}' is missing.")

            aggfunc = spec.aggregation or "mean"
            heatmap_data = data
            if aggfunc in {"mean", "median"} and not pd.api.types.is_numeric_dtype(heatmap_data[value_column]):
                numeric_values = pd.to_numeric(heatmap_data[value_column], errors="coerce")
                if numeric_values.notna().sum() == 0:
                    return error_figure(
                        spec.title,
                        f"No numeric value column available for heatmap aggregation '{aggfunc}'.",
                    )
                heatmap_data = heatmap_data.copy()
                heatmap_data[value_column] = numeric_values

            pivot = pd.pivot_table(
                heatmap_data,
                index=spec.y,
                columns=spec.x,
                values=value_column,
                aggfunc=aggfunc,
            )
            if pivot.empty:
                return error_figure(spec.title, "Aggregation resulted in empty dataset.")
            return px.imshow(pivot, title=spec.title)
        return px.imshow(data.select_dtypes(include="number").corr(), title=spec.title)
    if chart == "area":
        return px.area(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    if chart == "pie":
        return px.pie(data, names=spec.x, values=spec.y, color=spec.color, title=spec.title)
    return px.bar(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)


def error_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14},
                "align": "center",
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
        template="plotly_white",
    )
    return figure


def build_error_app(title: str, message: str, details: str | None = None) -> Dash:
    app = Dash(__name__)
    children: list = [html.H2(title), html.P(message)]
    if details:
        children.append(
            html.Pre(
                details,
                style={
                    "whiteSpace": "pre-wrap",
                    "backgroundColor": "#f8f9fa",
                    "padding": "12px",
                    "borderRadius": "6px",
                },
            )
        )
    app.layout = html.Div(children, style={"maxWidth": "900px", "margin": "40px auto", "padding": "0 20px"})
    return app


def export_static_html(figures: Iterable, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_parts = [
        "<html><head><meta charset='utf-8'/>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
    ]
    for fig in figures:
        html_parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path


def export_dashboard(
    output_format: str,
    output_path: Path,
    title: str,
    figures: Iterable,
    app: Dash | None = None,
    port: int = 8050,
) -> Path | None:
    normalized_format = output_format.lower()
    if normalized_format in {"server", "dash"}:
        if app is None:
            raise ValueError("Dash app is required for server output.")
        app.run(debug=False, port=port)
        return None
    if normalized_format == "html":
        return export_static_html(figures=figures, output_path=output_path, title=title)
    raise ValueError(f"Unsupported output format: {output_format}")
