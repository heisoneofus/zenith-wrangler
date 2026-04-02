from __future__ import annotations

from src.models import DashboardSpec, VisualSpec


def default_dashboard(title: str = "Zenith Wrangler Dashboard") -> DashboardSpec:
    return DashboardSpec(title=title)


def single_metric_layout(metric: str, dimension: str | None = None) -> DashboardSpec:
    chart = VisualSpec(
        title=f"{metric} Overview",
        chart_type="bar",
        x=dimension,
        y=metric if dimension else None,
    )
    return DashboardSpec(title=f"{metric} Dashboard", visuals=[chart], filters=[dimension] if dimension else [])
