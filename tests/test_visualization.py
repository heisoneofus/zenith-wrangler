from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import plotly.express as px

from src.models import VisualSpec
from src.tools.visualization import build_error_app, create_figure, export_dashboard


class _DummyApp:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, object]] = []

    def run(self, **kwargs) -> None:
        self.run_calls.append(kwargs)


class VisualizationTests(unittest.TestCase):
    def test_create_figure_bar_infers_axes_when_missing(self) -> None:
        df = pd.DataFrame(
            {
                "segment": ["A", "B", "C"],
                "sales": [10, 12, 9],
            }
        )
        spec = VisualSpec(title="Auto Axis", chart_type="bar", x=None, y=None)

        figure = create_figure(df, spec)

        self.assertGreater(len(figure.data), 0)

    def test_create_figure_bar_allows_single_axis(self) -> None:
        df = pd.DataFrame({"region": ["EU", "US", "APAC"]})
        spec = VisualSpec(title="Counts", chart_type="bar", x="region", y=None)

        figure = create_figure(df, spec)

        self.assertGreater(len(figure.data), 0)

    def test_create_figure_validates_missing_columns(self) -> None:
        df = pd.DataFrame({"sales": [10, 12]})
        spec = VisualSpec(title="Sales by Region", chart_type="bar", x="region", y="sales")

        with self.assertRaisesRegex(ValueError, "missing"):
            create_figure(df, spec)

    def test_create_figure_drops_missing_optional_color(self) -> None:
        df = pd.DataFrame(
            {
                "ticket_date": ["2026-01-01", "2026-01-02"],
                "tickets": [10, 20],
            }
        )
        spec = VisualSpec(
            title="Tickets",
            chart_type="line",
            x="ticket_date",
            y="tickets",
            color="handled_type",
        )

        figure = create_figure(df, spec)

        self.assertEqual(len(figure.data), 1)

    def test_create_figure_aggregation_preserves_valid_color_grouping(self) -> None:
        df = pd.DataFrame(
            {
                "ticket_date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
                "handled_type": ["Celeste", "Human", "Celeste", "Human"],
                "tickets": [2, 3, 4, 5],
            }
        )
        spec = VisualSpec(
            title="Grouped",
            chart_type="line",
            x="ticket_date",
            y="tickets",
            color="handled_type",
            aggregation="sum",
        )

        figure = create_figure(df, spec)

        self.assertEqual({trace.name for trace in figure.data}, {"Celeste", "Human"})

    def test_create_figure_nonexistent_grouping_field_degrades_predictably(self) -> None:
        df = pd.DataFrame(
            {
                "ticket_date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
                "tickets": [2, 3, 4, 5],
            }
        )
        spec = VisualSpec(
            title="Grouped",
            chart_type="line",
            x="ticket_date",
            y="tickets",
            color="handled_type",
            aggregation="sum",
        )

        figure = create_figure(df, spec)

        self.assertEqual(len(figure.data), 1)
        self.assertEqual(list(figure.data[0].y), [5, 9])

    def test_create_figure_histogram_count_aggregation_with_single_axis(self) -> None:
        df = pd.DataFrame(
            {
                "sleep_duration_hrs": [6.0, 7.5, 6.0, 8.0],
                "day_type": ["weekday", "weekday", "weekend", "weekend"],
            }
        )
        spec = VisualSpec(
            title="Sleep Duration Distribution",
            chart_type="histogram",
            x="sleep_duration_hrs",
            color="day_type",
            aggregation="count",
        )

        figure = create_figure(df, spec)

        self.assertGreater(len(figure.data), 0)

    def test_create_figure_aggregation_still_requires_x_and_y_for_non_histogram(self) -> None:
        df = pd.DataFrame({"region": ["EU", "US", "APAC"]})
        spec = VisualSpec(title="Counts", chart_type="bar", x="region", aggregation="count")

        with self.assertRaisesRegex(ValueError, "Aggregation requires both 'x' and 'y' fields"):
            create_figure(df, spec)

    def test_create_figure_pie_supports_category_value_mapping(self) -> None:
        df = pd.DataFrame(
            {
                "segment": ["A", "B", "C"],
                "sales": [10, 20, 30],
            }
        )
        spec = VisualSpec(title="Mix", chart_type="pie", x="segment", y="sales")

        figure = create_figure(df, spec)

        self.assertGreater(len(figure.data), 0)

    def test_create_figure_heatmap_uses_color_metric_for_mean_aggregation(self) -> None:
        df = pd.DataFrame(
            {
                "country": ["US", "US", "CA", "CA"],
                "occupation": ["Engineer", "Nurse", "Engineer", "Nurse"],
                "sleep_quality_score": [7.0, 6.0, 8.0, 7.0],
            }
        )
        spec = VisualSpec(
            title="Heatmap",
            chart_type="heatmap",
            x="country",
            y="occupation",
            color="sleep_quality_score",
            aggregation="mean",
        )

        figure = create_figure(df, spec)

        self.assertGreater(len(figure.data), 0)

    def test_create_figure_heatmap_with_non_numeric_values_returns_error_figure(self) -> None:
        df = pd.DataFrame(
            {
                "country": ["US", "US", "CA", "CA"],
                "occupation": ["Engineer", "Nurse", "Engineer", "Nurse"],
            }
        )
        spec = VisualSpec(
            title="Heatmap",
            chart_type="heatmap",
            x="country",
            y="occupation",
            aggregation="mean",
        )

        figure = create_figure(df, spec)

        self.assertIn("No numeric value column available", str(figure.layout.annotations[0]["text"]))

    def test_export_dashboard_runs_server_app(self) -> None:
        app = _DummyApp()

        result = export_dashboard(
            output_format="server",
            output_path=Path("outputs/unused.html"),
            title="Demo",
            figures=[],
            app=app,  # type: ignore[arg-type]
            port=8123,
        )

        self.assertIsNone(result)
        self.assertEqual(len(app.run_calls), 1)
        self.assertEqual(app.run_calls[0]["port"], 8123)
        self.assertEqual(app.run_calls[0]["debug"], False)

    def test_export_dashboard_html_writes_file(self) -> None:
        figure = px.bar(pd.DataFrame({"region": ["EU", "US"], "sales": [1, 2]}), x="region", y="sales")

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "dashboard.html"
            result = export_dashboard(
                output_format="html",
                output_path=output_path,
                title="Demo",
                figures=[figure],
            )

            self.assertEqual(result, output_path)
            html = output_path.read_text(encoding="utf-8")
            self.assertIn("Demo", html)
            self.assertIn("plotly", html.lower())

    def test_build_error_app_contains_message(self) -> None:
        app = build_error_app("Dashboard Generation Failed", "Unable to build dashboard", "missing column")

        rendered = str(app.layout)
        self.assertIn("Dashboard Generation Failed", rendered)
        self.assertIn("Unable to build dashboard", rendered)
        self.assertIn("missing column", rendered)


if __name__ == "__main__":
    unittest.main()
