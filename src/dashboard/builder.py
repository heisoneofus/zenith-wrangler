from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from src.models import DashboardSpec
from src.tools.visualization import create_figure, error_figure


@dataclass
class DashboardResult:
    app: Dash
    spec: DashboardSpec


def _coerce_datetime_filter_series(column: str, series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return None

    parsed = pd.to_datetime(series, errors="coerce")
    threshold = 0.7 if ("date" in column.lower() or "time" in column.lower()) else 0.95
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


def _render_visual(df: pd.DataFrame, spec) -> Any:
    try:
        return create_figure(df, spec)
    except Exception as exc:
        return error_figure(spec.title, f"Unable to render chart: {exc}")


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
            components.append(
                dbc.Col(
                    [
                        html.Label(column),
                        dcc.DatePickerRange(
                            id=f"filter-{column}",
                            start_date=_serialize_date_value(min_date),
                            end_date=_serialize_date_value(max_date),
                            min_date_allowed=_serialize_date_value(min_date),
                            max_date_allowed=_serialize_date_value(max_date),
                        ),
                    ],
                    md=4,
                )
            )
        else:
            values = df[column].dropna().unique().tolist()
            options = [{"label": "All", "value": "__all__"}] + [
                {"label": str(value), "value": value} for value in values[:200]
            ]
            components.append(
                dbc.Col(
                    [
                        html.Label(column),
                        dcc.Dropdown(
                            id=f"filter-{column}",
                            options=options,
                            value="__all__",
                            clearable=False,
                        ),
                    ],
                    md=4,
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


def build_dashboard(df: pd.DataFrame, design: DashboardSpec) -> DashboardResult:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    filter_components = _build_filter_components(df, design.filters)
    graph_components = [
        dbc.Col(dcc.Graph(id=f"graph-{index}", figure=_render_visual(df, spec)), md=6)
        for index, spec in enumerate(design.visuals)
    ]
    if not graph_components:
        graph_components = [
            dbc.Col(
                dbc.Alert(
                    "No visualizations were produced for this dataset. Try regenerating with a different prompt or input file.",
                    color="warning",
                ),
                md=12,
            )
        ]

    app.layout = dbc.Container(
        [
            html.H1(design.title),
            dbc.Row(filter_components, className="mb-4") if filter_components else html.Div(),
            dbc.Row(graph_components),
        ],
        fluid=True,
    )

    if design.visuals:
        inputs = []
        for column in design.filters:
            if column not in df.columns:
                continue
            if _is_date_filter(df, column):
                inputs.append(Input(f"filter-{column}", "start_date"))
                inputs.append(Input(f"filter-{column}", "end_date"))
            else:
                inputs.append(Input(f"filter-{column}", "value"))

        if inputs:
            @app.callback(
                [Output(f"graph-{index}", "figure") for index in range(len(design.visuals))],
                inputs,
            )
            def update_graphs(*args):
                values: dict[str, Any] = {}
                arg_index = 0
                for column in design.filters:
                    if column not in df.columns:
                        continue
                    if _is_date_filter(df, column):
                        values[column] = (args[arg_index], args[arg_index + 1])
                        arg_index += 2
                    else:
                        values[column] = args[arg_index]
                        arg_index += 1
                filtered = _apply_filters(df, design.filters, values)
                return [_render_visual(filtered, spec) for spec in design.visuals]

    return DashboardResult(app=app, spec=design)
