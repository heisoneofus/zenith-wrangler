from __future__ import annotations

import pandas as pd
from dash import dcc

from src.dashboard.builder import _apply_filters, _build_filter_components, build_dashboard
from src.models import DashboardSpec, VisualSpec


def test_build_filter_components_uses_date_picker_for_date_like_string_columns() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "sales": [10, 20, 30],
        }
    )

    components = _build_filter_components(df, ["event_date"])

    assert len(components) == 1
    assert isinstance(components[0].children[1], dcc.DatePickerRange)


def test_apply_filters_treats_end_date_as_inclusive_for_date_like_string_columns() -> None:
    df = pd.DataFrame(
        {
            "event_date": [
                "2026-04-01T08:30:00",
                "2026-04-02T14:15:00",
                "2026-04-03T09:00:00",
            ],
            "sales": [10, 20, 30],
        }
    )

    filtered = _apply_filters(df, ["event_date"], {"event_date": ("2026-04-02", "2026-04-02")})

    assert filtered["sales"].tolist() == [20]


def test_build_dashboard_sets_initial_figures_without_filters() -> None:
    df = pd.DataFrame({"region": ["East", "West"], "sales": [10, 12]})
    design = DashboardSpec(
        title="Sales Dashboard",
        visuals=[VisualSpec(title="Sales by Region", chart_type="bar", x="region", y="sales")],
        filters=[],
    )

    dashboard = build_dashboard(df, design)
    graph_row = dashboard.app.layout.children[2]
    graph = graph_row.children[0].children

    assert graph.figure is not None
