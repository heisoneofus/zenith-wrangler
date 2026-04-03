from __future__ import annotations

from src.agents.patcher import apply_dashboard_patch, parse_update_prompt
from src.models import DashboardSpec, VisualSpec


def test_parse_update_prompt_targets_single_visual_and_filter() -> None:
    spec = DashboardSpec(
        visuals=[
            VisualSpec(title="Sales by Region", chart_type="bar", x="region", y="sales"),
            VisualSpec(title="Margin Over Time", chart_type="line", x="date", y="margin"),
        ],
        filters=["region"],
    )

    patch = parse_update_prompt("Change Margin Over Time to a scatter chart and add filter for segment", spec)

    assert len(patch.operations) == 2
    assert any(operation.op == "replace_visual" for operation in patch.operations)
    assert any(operation.op == "add_filter" for operation in patch.operations)


def test_apply_dashboard_patch_adds_visual_and_changes_theme() -> None:
    spec = DashboardSpec(
        visuals=[VisualSpec(title="Sales", chart_type="bar", x="region", y="sales")],
        theme="light",
    )
    patch = parse_update_prompt("Add chart for margin by segment and make dark theme", spec)

    updated = apply_dashboard_patch(spec, patch)

    assert updated.theme == "dark"
    assert len(updated.visuals) == 2
    assert updated.visuals[-1].status == "revised"
