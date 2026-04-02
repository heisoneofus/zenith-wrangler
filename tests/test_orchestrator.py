from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.agents.orchestrator import Orchestrator, ToolCall, ToolRegistry, build_registry
from src.config import AppConfig
from src.models import AnalysisReport, DashboardSpec, DataQualityAssessment, MetricsAnalysis, VisualSpec
from src.tooling import ToolSpec


class _DummyLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str, dict]] = []

    def log_tool_call(self, index: int, tool_name: str, reasoning: str, params: dict) -> None:
        self.calls.append((index, tool_name, reasoning, params))


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ReadCsvParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sample_rows: int | None = Field(default=None, ge=1)


class OrchestratorTests(unittest.TestCase):
    def test_parse_operation_maps_flatten_like_recommendation(self) -> None:
        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), ToolRegistry())

        tool_call = orchestrator._parse_operation_to_tool("Flatten JSON-like/list-like nested columns such as genres")

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "flatten_nested")
        self.assertEqual(tool_call.params, {"max_depth": 1})

    def test_parse_operation_maps_aggregate_recommendation(self) -> None:
        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), ToolRegistry())
        analysis = AnalysisReport(
            metrics=MetricsAnalysis(primary_metrics=["sales"], dimensions=["region"]),
            quality=DataQualityAssessment(),
            design=DashboardSpec(),
            sampled_rows=3,
            data_schema={"region": "object", "sales": "float64"},
        )

        tool_call = orchestrator._parse_operation_to_tool("Aggregate grouped KPIs before charting", analysis=analysis)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "aggregate_by")
        self.assertEqual(tool_call.params["group_by"], ["region"])
        self.assertEqual(tool_call.params["metrics"], ["sales"])
        self.assertEqual(tool_call.params["dataframe_ref"], "baseline")
        self.assertEqual(tool_call.params["output_dataframe_ref"], "aggregate_sales_by_region")

    def test_parse_operation_maps_pivot_recommendation(self) -> None:
        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), ToolRegistry())
        analysis = AnalysisReport(
            metrics=MetricsAnalysis(primary_metrics=["sales"], dimensions=["region"], time_fields=["month"]),
            quality=DataQualityAssessment(),
            design=DashboardSpec(
                visuals=[
                    VisualSpec(
                        title="Sales heatmap",
                        chart_type="heatmap",
                        x="month",
                        y="region",
                        color="sales",
                    )
                ]
            ),
            sampled_rows=3,
            data_schema={"region": "object", "month": "object", "sales": "float64"},
        )

        tool_call = orchestrator._parse_operation_to_tool("Pivot data for heatmap matrix", analysis=analysis)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "pivot_data")
        self.assertEqual(tool_call.params["index"], "region")
        self.assertEqual(tool_call.params["columns"], "month")
        self.assertEqual(tool_call.params["values"], "sales")
        self.assertEqual(tool_call.params["dataframe_ref"], "baseline")
        self.assertEqual(tool_call.params["output_dataframe_ref"], "pivot_sales_by_region_month")

    def test_plan_execution_builds_expected_sequence(self) -> None:
        analysis = AnalysisReport(
            metrics=MetricsAnalysis(primary_metrics=["sales"]),
            quality=DataQualityAssessment(suggested_operations=["fill_missing"]),
            design=DashboardSpec(
                visuals=[VisualSpec(title="Sales", chart_type="bar", x="region", y="sales")]
            ),
            sampled_rows=2,
            data_schema={"sales": "int64"},
        )

        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), ToolRegistry())
        plan = orchestrator.plan_execution(analysis, Path("dataset.parquet"))

        self.assertEqual(plan[0].tool_name, "read_parquet")
        self.assertIn("fill_missing", [item.tool_name for item in plan])
        self.assertIn("create_figure", [item.tool_name for item in plan])
        self.assertEqual(plan[-1].tool_name, "build_dashboard")

    def test_plan_execution_sanitizes_identifier_axes(self) -> None:
        analysis = AnalysisReport(
            metrics=MetricsAnalysis(primary_metrics=["transaction_id"], secondary_metrics=["amount"], dimensions=["region"]),
            quality=DataQualityAssessment(),
            design=DashboardSpec(
                visuals=[
                    VisualSpec(title="Invalid ID Metric", chart_type="bar", x="region", y="transaction_id")
                ]
            ),
            sampled_rows=3,
            data_schema={"transaction_id": "int64", "amount": "float64", "region": "object"},
        )

        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), ToolRegistry())
        plan = orchestrator.plan_execution(analysis, Path("dataset.csv"))

        figure_calls = [call for call in plan if call.tool_name == "create_figure"]
        self.assertEqual(len(figure_calls), 1)
        self.assertEqual(figure_calls[0].params["spec"]["y"], "amount")

    def test_execute_plan_tracks_transformations_and_exports(self) -> None:
        registry = ToolRegistry()
        captured_paths: list[Path] = []

        def _read(ctx, path: Path, sample_rows: int | None = None) -> pd.DataFrame:
            captured_paths.append(path)
            return pd.DataFrame({"value": [1, 2]})

        registry.register(
            ToolSpec(
                name="read_csv",
                description="Read data",
                category="loader",
                input_model=_ReadCsvParams,
                output_kind="dataframe",
                produces_context=("dataframe",),
                execute=_read,
            )
        )
        registry.register(
            ToolSpec(
                name="fill_missing",
                description="Fill",
                category="cleaning",
                input_model=_NoParams,
                output_kind="dataframe",
                requires_context=("dataframe",),
                produces_context=("dataframe",),
                execute=lambda ctx: ctx["dataframe"],
            )
        )
        registry.register(
            ToolSpec(
                name="create_figure",
                description="Figure",
                category="visualization",
                input_model=_NoParams,
                output_kind="figure",
                requires_context=("dataframe",),
                produces_context=("figure",),
                execute=lambda ctx: {"figure": "ok"},
            )
        )
        registry.register(
            ToolSpec(
                name="build_dashboard",
                description="Dashboard",
                category="dashboard",
                input_model=_NoParams,
                output_kind="dashboard",
                requires_context=("dataframe",),
                produces_context=("dashboard",),
                execute=lambda ctx: SimpleNamespace(spec=DashboardSpec(title="Demo"), app=None),
            )
        )

        plan = [
            ToolCall(tool_name="read_csv", params={"path": "dataset.csv"}, reasoning="load"),
            ToolCall(tool_name="fill_missing", params={}, reasoning="clean"),
            ToolCall(tool_name="create_figure", params={}, reasoning="plot"),
            ToolCall(tool_name="build_dashboard", params={}, reasoning="build"),
        ]

        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), registry)
        logger_ctx = _DummyLogger()

        with patch(
            "src.agents.orchestrator.visualization.export_dashboard",
            return_value=Path("outputs/dashboard_test.html"),
        ) as export_mock:
            result = orchestrator.execute_plan(
                plan=plan,
                output_format="html",
                output_path=Path("outputs/dashboard_test.html"),
                port=8050,
                logger_ctx=logger_ctx,
            )

        self.assertEqual(result.transformations_applied, ["fill_missing"])
        self.assertEqual(len(result.figures), 1)
        self.assertEqual(result.output_path, Path("outputs/dashboard_test.html"))
        self.assertEqual(len(logger_ctx.calls), 4)
        self.assertTrue(captured_paths)
        self.assertIsInstance(captured_paths[0], Path)
        self.assertIsInstance(logger_ctx.calls[0][3]["path"], Path)
        export_mock.assert_called_once()

    def test_execute_plan_preserves_baseline_dataframe_for_visuals_after_reshape(self) -> None:
        registry = build_registry()
        dataset = pd.DataFrame(
            {
                "age": [20, 20, 30],
                "sleep_quality_score": [5, 7, 8],
                "sleep_disorder_risk": ["low", "high", "low"],
            }
        )

        plan = [
            ToolCall(tool_name="read_csv", params={"path": "dataset.csv"}, reasoning="load"),
            ToolCall(
                tool_name="aggregate_by",
                params={
                    "group_by": ["age"],
                    "metrics": ["sleep_quality_score"],
                    "agg": "sum",
                    "output_dataframe_ref": "sleep_by_age",
                },
                reasoning="reshape",
            ),
            ToolCall(
                tool_name="create_figure",
                params={
                    "spec": {
                        "title": "Risk Counts",
                        "chart_type": "bar",
                        "x": "sleep_disorder_risk",
                        "y": "sleep_quality_score",
                        "aggregation": "count",
                    }
                },
                reasoning="baseline chart",
            ),
            ToolCall(
                tool_name="create_figure",
                params={
                    "dataframe_ref": "sleep_by_age",
                    "spec": {
                        "title": "Sleep By Age",
                        "chart_type": "bar",
                        "x": "age",
                        "y": "sleep_quality_score",
                    },
                },
                reasoning="derived chart",
            ),
        ]

        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), registry)
        logger_ctx = _DummyLogger()

        with patch("src.agents.orchestrator.loaders.read_csv", return_value=dataset), patch(
            "src.agents.orchestrator.visualization.export_dashboard",
            return_value=Path("outputs/dashboard_test.html"),
        ):
            result = orchestrator.execute_plan(
                plan=plan,
                output_format="html",
                output_path=Path("outputs/dashboard_test.html"),
                port=8050,
                logger_ctx=logger_ctx,
            )

        self.assertEqual(len(result.figures), 2)
        baseline_chart_x = {str(value) for value in result.figures[0].data[0].x}
        derived_chart_x = {int(value) for value in result.figures[1].data[0].x}

        self.assertEqual(baseline_chart_x, {"high", "low"})
        self.assertEqual(derived_chart_x, {20, 30})
        self.assertIn("aggregate_by", result.transformations_applied)

    def test_execute_plan_allows_sequential_reshape_operations_from_baseline(self) -> None:
        registry = build_registry()
        dataset = pd.DataFrame(
            {
                "age": [20, 20, 30],
                "occupation": ["Engineer", "Engineer", "Teacher"],
                "chronotype": ["Morning", "Evening", "Morning"],
                "sleep_quality_score": [6.0, 8.0, 7.0],
            }
        )

        plan = [
            ToolCall(tool_name="read_csv", params={"path": "dataset.csv"}, reasoning="load"),
            ToolCall(
                tool_name="aggregate_by",
                params={
                    "group_by": ["age"],
                    "metrics": ["sleep_quality_score"],
                    "agg": "sum",
                    "dataframe_ref": "baseline",
                    "output_dataframe_ref": "sleep_by_age",
                },
                reasoning="reshape aggregate",
            ),
            ToolCall(
                tool_name="pivot_data",
                params={
                    "index": "occupation",
                    "columns": "chronotype",
                    "values": "sleep_quality_score",
                    "aggfunc": "mean",
                    "dataframe_ref": "baseline",
                    "output_dataframe_ref": "sleep_quality_matrix",
                },
                reasoning="reshape pivot",
            ),
        ]

        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), registry)
        logger_ctx = _DummyLogger()

        with patch("src.agents.orchestrator.loaders.read_csv", return_value=dataset), patch(
            "src.agents.orchestrator.visualization.export_dashboard",
            return_value=Path("outputs/dashboard_test.html"),
        ):
            result = orchestrator.execute_plan(
                plan=plan,
                output_format="html",
                output_path=Path("outputs/dashboard_test.html"),
                port=8050,
                logger_ctx=logger_ctx,
            )

        self.assertIn("aggregate_by", result.transformations_applied)
        self.assertIn("pivot_data", result.transformations_applied)
        self.assertIsNotNone(result.dataframe)
        self.assertIn("occupation", result.dataframe.columns)
        self.assertIn("Morning", result.dataframe.columns)

    def test_execute_plan_server_mode_returns_error_app_on_tool_failure(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="read_csv",
                description="Read data",
                category="loader",
                input_model=_ReadCsvParams,
                output_kind="dataframe",
                produces_context=("dataframe",),
                execute=lambda ctx, path, sample_rows=None: pd.DataFrame({"value": [1, 2]}),
            )
        )
        registry.register(
            ToolSpec(
                name="create_figure",
                description="Figure",
                category="visualization",
                input_model=_NoParams,
                output_kind="figure",
                requires_context=("dataframe",),
                produces_context=("figure",),
                execute=lambda ctx: (_ for _ in ()).throw(ValueError("missing column")),
            )
        )

        plan = [
            ToolCall(tool_name="read_csv", params={"path": "dataset.csv"}, reasoning="load"),
            ToolCall(tool_name="create_figure", params={}, reasoning="plot"),
        ]

        orchestrator = Orchestrator(AppConfig.default(Path.cwd()), registry)
        logger_ctx = _DummyLogger()
        logger_ctx.log_block = lambda *args, **kwargs: None  # type: ignore[attr-defined]

        with patch("src.agents.orchestrator.visualization.build_error_app", return_value=SimpleNamespace()) as app_mock, patch(
            "src.agents.orchestrator.visualization.export_dashboard",
            return_value=None,
        ) as export_mock:
            result = orchestrator.execute_plan(
                plan=plan,
                output_format="server",
                output_path=Path("outputs/dashboard_test.html"),
                port=8050,
                logger_ctx=logger_ctx,
            )

        self.assertIsNone(result.output_path)
        self.assertEqual(len(result.figures), 0)
        app_mock.assert_called_once()
        export_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
