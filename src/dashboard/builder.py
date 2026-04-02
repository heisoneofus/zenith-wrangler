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


def _build_filter_components(df: pd.DataFrame, filters: list[str]) -> list[Any]:
    components: list[Any] = []
    for column in filters:
        if column not in df.columns:
            continue
        column_series = df[column]
        is_datetime = pd.api.types.is_datetime64_any_dtype(column_series)
        parsed_dates = None
        if not is_datetime and ("date" in column.lower() or "time" in column.lower()):
            parsed_dates = pd.to_datetime(column_series, errors="coerce")
            is_datetime = parsed_dates.notna().mean() > 0.7
        if is_datetime:
            series = column_series if parsed_dates is None else parsed_dates
            min_date = series.min()
            max_date = series.max()
            components.append(
                dbc.Col(
                    [
                        html.Label(column),
                        dcc.DatePickerRange(
                            id=f"filter-{column}",
                            start_date=min_date,
                            end_date=max_date,
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
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
                series = result[column]
                if not pd.api.types.is_datetime64_any_dtype(series):
                    series = pd.to_datetime(series, errors="coerce")
                result = result[(series >= start) & (series <= end)]
        elif value != "__all__":
            result = result[result[column] == value]
    return result


def build_dashboard(df: pd.DataFrame, design: DashboardSpec) -> DashboardResult:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    filter_components = _build_filter_components(df, design.filters)
    graph_components = [
        dbc.Col(dcc.Graph(id=f"graph-{index}"), md=6) for index in range(len(design.visuals))
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
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                inputs.append(Input(f"filter-{column}", "start_date"))
                inputs.append(Input(f"filter-{column}", "end_date"))
            else:
                inputs.append(Input(f"filter-{column}", "value"))

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
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    values[column] = (args[arg_index], args[arg_index + 1])
                    arg_index += 2
                else:
                    values[column] = args[arg_index]
                    arg_index += 1
            filtered = _apply_filters(df, design.filters, values)
            figures = []
            for spec in design.visuals:
                try:
                    figures.append(create_figure(filtered, spec))
                except Exception as exc:
                    figures.append(error_figure(spec.title, f"Unable to render chart: {exc}"))
            return figures

    return DashboardResult(app=app, spec=design)
