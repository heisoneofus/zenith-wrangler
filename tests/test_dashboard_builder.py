from __future__ import annotations

import pandas as pd
from dash import dcc

from src.dashboard.builder import _apply_filters, _build_filter_components, _quality_score, _regenerate_visual_spec, build_dashboard
from src.models import DashboardSpec, PlanProposal, SessionState, VisualSpec


def test_build_filter_components_uses_date_picker_for_date_like_string_columns() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "sales": [10, 20, 30],
        }
    )

    components = _build_filter_components(df, ["event_date"])

    assert len(components) == 1
    assert isinstance(components[0].children.children[1], dcc.DatePickerRange)


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


def test_build_filter_components_uses_dropdown_for_non_date_string_columns() -> None:
    df = pd.DataFrame(
        {
            "shift": ["day", "night", "swing"],
            "tickets": [3, 2, 1],
        }
    )

    components = _build_filter_components(df, ["shift"])

    assert len(components) == 1
    assert isinstance(components[0].children.children[1], dcc.Dropdown)


def test_build_dashboard_sets_initial_figures_without_filters() -> None:
    df = pd.DataFrame({"region": ["East", "West"], "sales": [10, 12]})
    design = DashboardSpec(
        title="Sales Dashboard",
        visuals=[VisualSpec(title="Sales by Region", chart_type="bar", x="region", y="sales")],
        filters=[],
    )
    state = SessionState(
        session_id="session_1",
        active_spec=design,
        spec_versions=[design],
        plan=PlanProposal(summary="Review then execute", design=design),
    )

    dashboard = build_dashboard(df, design, session_state=state)
    assert dashboard.app.layout.className == "dashboard-shell theme-light"
    frame = dashboard.app.layout.children[-1]
    workspace = frame.children[1]
    plan_panel = workspace.children[0]

    assert "Navigator Plan" in str(plan_panel)
    assert "dashboard-graph" in str(dashboard.app.layout)
    assert "Export Provenance" in str(dashboard.app.layout)
    assert "Tell Navigator what this regeneration should focus on" in str(dashboard.app.layout)


def test_regenerate_visual_spec_uses_prompt_to_reframe_chart() -> None:
    df = pd.DataFrame(
        {
            "region": ["East", "West", "North"],
            "sales": [10, 12, 9],
            "profit": [2, 3, 1],
        }
    )
    visual = VisualSpec(title="Sales by Region", chart_type="bar", x="region", y="sales")

    regenerated = _regenerate_visual_spec(df, visual, "scatter profit vs sales")

    assert regenerated.chart_type == "scatter"
    assert regenerated.status == "revised"
    assert "focus on scatter profit vs sales" in regenerated.description


def test_quality_score_penalizes_severity_levels() -> None:
    design = DashboardSpec(
        data_quality_summary=[
            "missing (high) on csat_score",
            "duplicates (medium) on ticket_id",
            "types (low) on day_of_week",
        ]
    )

    assert _quality_score(design, None) == 59
