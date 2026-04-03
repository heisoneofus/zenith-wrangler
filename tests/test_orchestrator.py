from __future__ import annotations

from pathlib import Path

from src.agents.orchestrator import Orchestrator, build_registry
from src.config import AppConfig
from src.models import AnalysisReport, DashboardSpec, DataQualityAssessment, MetricsAnalysis, VisualSpec


def test_plan_execution_sanitizes_dashboard_design_visuals() -> None:
    analysis = AnalysisReport(
        metrics=MetricsAnalysis(
            primary_metrics=["sales"],
            secondary_metrics=[],
            dimensions=["region"],
            time_fields=[],
            notes="",
        ),
        quality=DataQualityAssessment(),
        design=DashboardSpec(
            visuals=[
                VisualSpec(
                    title="Bad Visual",
                    chart_type="bar",
                    x="region",
                    y="user_id",
                    color="user_id",
                )
            ]
        ),
        sampled_rows=3,
        data_schema={"region": "object", "sales": "float64", "user_id": "int64"},
    )

    orchestrator = Orchestrator(AppConfig.default(Path(".")), build_registry())
    plan = orchestrator.plan_execution(analysis, Path("dataset.csv"))
    dashboard_call = next(call for call in plan if call.tool_name == "build_dashboard")
    visual = dashboard_call.params["design"]["visuals"][0]

    assert visual["y"] == "sales"
    assert visual["color"] is None
