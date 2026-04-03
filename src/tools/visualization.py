from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

from dash import Dash, html

from src.models import VisualSpec


DashboardTheme = Literal["light", "dark"]

OPTIONAL_ENCODING_FIELDS: tuple[str, ...] = (
    "color",
    "shape",
    "size",
    "symbol",
    "facet_row",
    "facet_col",
)

THEME_TOKENS: dict[DashboardTheme, dict[str, Any]] = {
    "light": {
        "font_family": "'Space Grotesk', 'Segoe UI', sans-serif",
        "mono_family": "'IBM Plex Mono', 'Consolas', monospace",
        "background": "#f4f7ff",
        "background_secondary": "#e9f4ff",
        "surface": "rgba(255, 255, 255, 0.82)",
        "surface_strong": "#ffffff",
        "surface_inner": "#edf3ff",
        "border": "rgba(71, 103, 190, 0.18)",
        "border_strong": "rgba(71, 103, 190, 0.34)",
        "text": "#10203d",
        "muted_text": "#586784",
        "grid": "rgba(81, 108, 174, 0.16)",
        "accent": "#246bff",
        "accent_secondary": "#11c7b3",
        "accent_tertiary": "#6d5efc",
        "shadow": "0 24px 60px rgba(72, 94, 161, 0.18)",
        "colorway": ["#246bff", "#11c7b3", "#6d5efc", "#ff7a59", "#12a6ff", "#5bd66f"],
        "hero_badge": "Photon Light Mode",
    },
    "dark": {
        "font_family": "'Space Grotesk', 'Segoe UI', sans-serif",
        "mono_family": "'IBM Plex Mono', 'Consolas', monospace",
        "background": "#050816",
        "background_secondary": "#0b1431",
        "surface": "rgba(10, 18, 42, 0.78)",
        "surface_strong": "#0f1a38",
        "surface_inner": "#132247",
        "border": "rgba(101, 140, 255, 0.20)",
        "border_strong": "rgba(80, 227, 194, 0.38)",
        "text": "#eef4ff",
        "muted_text": "#9dafd4",
        "grid": "rgba(123, 149, 214, 0.16)",
        "accent": "#5ea1ff",
        "accent_secondary": "#50e3c2",
        "accent_tertiary": "#a46bff",
        "shadow": "0 26px 80px rgba(0, 0, 0, 0.42)",
        "colorway": ["#5ea1ff", "#50e3c2", "#a46bff", "#ff8a5b", "#ffd166", "#76e0ff"],
        "hero_badge": "Nebula Dark Mode",
    },
}


def get_dashboard_theme(theme: DashboardTheme | str = "light") -> dict[str, Any]:
    normalized = "dark" if str(theme).lower() == "dark" else "light"
    return THEME_TOKENS[normalized]


def apply_figure_theme(figure, theme: DashboardTheme | str = "light") -> go.Figure:
    tokens = get_dashboard_theme(theme)
    themed = go.Figure(figure)
    themed.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=tokens["surface_inner"],
        font={"family": tokens["font_family"], "color": tokens["text"], "size": 13},
        colorway=tokens["colorway"],
        margin={"l": 28, "r": 20, "t": 72, "b": 28},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": tokens["muted_text"]},
        },
        hoverlabel={
            "bgcolor": tokens["surface_strong"],
            "bordercolor": tokens["border_strong"],
            "font": {"family": tokens["font_family"], "color": tokens["text"]},
        },
        title={
            "font": {"family": tokens["font_family"], "size": 22, "color": tokens["text"]},
            "x": 0.02,
            "xanchor": "left",
        },
    )
    themed.update_xaxes(
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=False,
        linecolor=tokens["border"],
        tickfont={"color": tokens["muted_text"]},
        title_font={"color": tokens["muted_text"]},
    )
    themed.update_yaxes(
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=False,
        linecolor=tokens["border"],
        tickfont={"color": tokens["muted_text"]},
        title_font={"color": tokens["muted_text"]},
    )
    themed.update_coloraxes(
        colorbar={"outlinecolor": tokens["border"], "tickfont": {"color": tokens["muted_text"]}}
    )
    themed.update_annotations(font={"color": tokens["text"]})

    for trace in themed.data:
        trace_type = getattr(trace, "type", "")
        if trace_type == "bar":
            trace.update(marker_line_color=tokens["surface_inner"], marker_line_width=1.2)
        elif trace_type in {"scatter", "scattergl"}:
            trace.update(marker_line_color=tokens["surface_inner"], marker_line_width=0.8)
            if getattr(trace, "mode", "") and "lines" in str(trace.mode):
                trace.update(line={"width": 3})
        elif trace_type in {"pie", "sunburst"}:
            trace.update(marker={"line": {"color": tokens["surface_inner"], "width": 2}})

    return themed


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


def _build_heatmap_data(df: pd.DataFrame, spec: VisualSpec, theme: DashboardTheme | str) -> go.Figure:
    if not (spec.x and spec.y):
        figure = px.imshow(df.select_dtypes(include="number").corr(), title=spec.title)
        return apply_figure_theme(figure, theme=theme)

    aggfunc = spec.aggregation or "mean"
    if aggfunc == "count" and not spec.color:
        grouped = df.groupby([spec.y, spec.x], dropna=False).size().reset_index(name="__count__")
        pivot = grouped.pivot(index=spec.y, columns=spec.x, values="__count__")
    else:
        value_column = spec.color or spec.y
        if value_column not in df.columns:
            return error_figure(
                spec.title,
                f"Heatmap value column '{value_column}' is missing.",
                theme=theme,
            )

        heatmap_data = df
        if aggfunc in {"mean", "median"} and not pd.api.types.is_numeric_dtype(heatmap_data[value_column]):
            numeric_values = pd.to_numeric(heatmap_data[value_column], errors="coerce")
            if numeric_values.notna().sum() == 0:
                return error_figure(
                    spec.title,
                    f"No numeric value column available for heatmap aggregation '{aggfunc}'.",
                    theme=theme,
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
        return error_figure(spec.title, "Aggregation resulted in empty dataset.", theme=theme)
    figure = px.imshow(pivot, title=spec.title)
    return apply_figure_theme(figure, theme=theme)


def create_figure(df: pd.DataFrame, spec: VisualSpec, theme: DashboardTheme | str = "light"):
    if df.empty:
        return error_figure(spec.title, "No data available to visualize.", theme=theme)

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
            return error_figure(spec.title, "Aggregation resulted in empty dataset.", theme=theme)

    chart = spec.chart_type
    if chart == "line":
        figure = px.line(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    if chart == "bar":
        figure = px.bar(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    if chart == "scatter":
        figure = px.scatter(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    if chart == "histogram":
        figure = px.histogram(data, x=spec.x or spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    if chart == "box":
        figure = px.box(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    if chart == "heatmap":
        return _build_heatmap_data(data, spec, theme)
    if chart == "area":
        figure = px.area(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    if chart == "pie":
        figure = px.pie(data, names=spec.x, values=spec.y, color=spec.color, title=spec.title)
        return apply_figure_theme(figure, theme=theme)
    figure = px.bar(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    return apply_figure_theme(figure, theme=theme)


def error_figure(title: str, message: str, theme: DashboardTheme | str = "light") -> go.Figure:
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
    )
    return apply_figure_theme(figure, theme=theme)


def build_error_app(
    title: str,
    message: str,
    details: str | None = None,
    theme: DashboardTheme | str = "dark",
) -> Dash:
    tokens = get_dashboard_theme(theme)
    app = Dash(__name__)
    children: list = [
        html.Div(
            [
                html.Span(
                    "Dashboard Runtime Alert",
                    style={
                        "display": "inline-block",
                        "padding": "8px 14px",
                        "borderRadius": "999px",
                        "fontFamily": tokens["mono_family"],
                        "fontSize": "12px",
                        "letterSpacing": "0.12em",
                        "textTransform": "uppercase",
                        "background": tokens["surface_inner"],
                        "border": f"1px solid {tokens['border_strong']}",
                        "color": tokens["accent_secondary"],
                    },
                ),
                html.H2(title, style={"margin": "18px 0 10px", "fontSize": "2rem"}),
                html.P(message, style={"margin": 0, "color": tokens["muted_text"]}),
            ]
        )
    ]
    if details:
        children.append(
            html.Pre(
                details,
                style={
                    "whiteSpace": "pre-wrap",
                    "backgroundColor": tokens["surface_inner"],
                    "padding": "18px",
                    "borderRadius": "18px",
                    "border": f"1px solid {tokens['border']}",
                    "color": tokens["text"],
                    "fontFamily": tokens["mono_family"],
                    "fontSize": "13px",
                },
            )
        )
    app.layout = html.Div(
        children,
        style={
            "maxWidth": "960px",
            "margin": "48px auto",
            "padding": "28px",
            "borderRadius": "28px",
            "background": tokens["surface"],
            "boxShadow": tokens["shadow"],
            "border": f"1px solid {tokens['border']}",
            "color": tokens["text"],
            "fontFamily": tokens["font_family"],
        },
    )
    return app


def export_static_html(figures: Iterable, output_path: Path, title: str) -> Path:
    tokens = get_dashboard_theme("light")
    themed_figures = [apply_figure_theme(fig, theme="light") for fig in figures]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_parts = [
        "<html><head><meta charset='utf-8'/>",
        f"<title>{title}</title>",
        "<link rel='preconnect' href='https://fonts.googleapis.com'>",
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>",
        "<link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap' rel='stylesheet'>",
        "<style>",
        dedent(
            f"""
            :root {{
                color-scheme: light;
            }}
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                min-height: 100vh;
                padding: 40px 22px 56px;
                background:
                    radial-gradient(circle at top left, rgba(36, 107, 255, 0.18), transparent 32%),
                    radial-gradient(circle at top right, rgba(17, 199, 179, 0.16), transparent 28%),
                    linear-gradient(180deg, {tokens["background"]} 0%, {tokens["background_secondary"]} 100%);
                color: {tokens["text"]};
                font-family: {tokens["font_family"]};
            }}
            .dashboard-export {{
                max-width: 1480px;
                margin: 0 auto;
            }}
            .hero {{
                padding: 28px;
                border-radius: 28px;
                background: {tokens["surface"]};
                border: 1px solid {tokens["border"]};
                box-shadow: {tokens["shadow"]};
                backdrop-filter: blur(16px);
                margin-bottom: 24px;
            }}
            .eyebrow {{
                display: inline-block;
                padding: 8px 12px;
                border-radius: 999px;
                background: {tokens["surface_inner"]};
                border: 1px solid {tokens["border_strong"]};
                color: {tokens["accent_secondary"]};
                font-family: {tokens["mono_family"]};
                font-size: 12px;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }}
            h1 {{
                margin: 18px 0 8px;
                font-size: clamp(2.4rem, 4vw, 4.5rem);
                line-height: 0.95;
            }}
            .hero p {{
                margin: 0;
                max-width: 760px;
                color: {tokens["muted_text"]};
                font-size: 1.02rem;
            }}
            .graph-card {{
                margin-top: 22px;
                padding: 20px;
                border-radius: 24px;
                background: {tokens["surface"]};
                border: 1px solid {tokens["border"]};
                box-shadow: {tokens["shadow"]};
                backdrop-filter: blur(14px);
            }}
            .graph-card h2 {{
                margin: 0 0 12px;
                font-size: 1.05rem;
                letter-spacing: 0.02em;
            }}
            """
        ),
        "</style>",
        "</head><body>",
        "<main class='dashboard-export'>",
        "<section class='hero'>",
        f"<span class='eyebrow'>{tokens['hero_badge']}</span>",
        f"<h1>{title}</h1>",
        "<p>Static export rendered with the upgraded futuristic dashboard theme.</p>",
        "</section>",
    ]
    for index, fig in enumerate(themed_figures):
        chart_title = getattr(fig.layout.title, "text", None) or f"Chart {index + 1}"
        html_parts.append(f"<section class='graph-card'><h2>{chart_title}</h2>")
        html_parts.append(
            pio.to_html(
                fig,
                include_plotlyjs="cdn" if index == 0 else False,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
            )
        )
        html_parts.append("</section>")
    html_parts.append("</main></body></html>")
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
