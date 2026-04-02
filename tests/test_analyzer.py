from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.agents.analyzer import Analyzer
from src.config import AppConfig
from src.models import DashboardSpec, DataQualityAssessment, LLMAnalysisResponse, MetricsAnalysis


class _DummyLogger:
    def __init__(self) -> None:
        self.json_entries: list[tuple[str, dict]] = []
        self.blocks: list[tuple[str, str]] = []

    def log_json(self, title: str, payload: dict) -> None:
        self.json_entries.append((title, payload))

    def log_block(self, title: str, content: str) -> None:
        self.blocks.append((title, content))


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig.default(Path.cwd())
        self.df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3),
                "sales": [10, 20, 30],
                "region": ["EU", "US", "EU"],
            }
        )

    def test_run_analysis_uses_structured_parse_path(self) -> None:
        expected = LLMAnalysisResponse(
            metrics=MetricsAnalysis(primary_metrics=["sales"]),
            quality=DataQualityAssessment(),
            design=DashboardSpec(),
            sampled_rows=3,
        )
        parse_kwargs: dict[str, object] = {}

        class _FakeResponses:
            def parse(self, **kwargs):
                parse_kwargs.update(kwargs)

                class _Response:
                    output_parsed = expected
                    output_text = '{"ok": true}'

                return _Response()

        class _FakeClient:
            responses = _FakeResponses()

        class _FakeOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def __new__(cls, *args, **kwargs):
                return _FakeClient()

        with patch("src.agents.analyzer.OpenAI", _FakeOpenAI):
            analyzer = Analyzer(self.config)

        logger_ctx = _DummyLogger()
        result = analyzer.run_analysis(self.df, None, logger_ctx)

        self.assertEqual(result.metrics.primary_metrics, ["sales"])
        self.assertEqual(result.sampled_rows, 3)
        self.assertIn("sales", result.data_schema)
        self.assertIn("region", result.data_schema)
        self.assertIs(parse_kwargs.get("text_format"), LLMAnalysisResponse)
        self.assertEqual(parse_kwargs.get("model"), self.config.llm.model)
        prompt_payload = json.loads(str(parse_kwargs.get("input")))
        instructions = " ".join(prompt_payload.get("instructions", []))
        self.assertIn("Use only column names that exist in the provided schema", instructions)
        self.assertIn("optional encodings", instructions)
        self.assertIn("nested-like content", instructions)
        self.assertIn("recommend `flatten_nested`", instructions)
        self.assertIn("identifier-like fields", instructions)
        self.assertIn("recommend `aggregate_by`", instructions)
        self.assertIn("recommend `pivot_data`", instructions)
        self.assertIn("varied chart mix", instructions)
        self.assertIn("Do not invent derived field names", instructions)
        self.assertTrue(any(title == "Metrics/Data Quality Response" for title, _ in logger_ctx.blocks))

    def test_run_analysis_heuristic_excludes_identifier_metrics(self) -> None:
        with patch("src.agents.analyzer.OpenAI", None):
            analyzer = Analyzer(self.config)

        df = pd.DataFrame(
            {
                "transaction_id": [1001, 1002, 1003, 1004],
                "amount": [10.5, 12.0, 11.2, 18.1],
                "region": ["EU", "US", "EU", "US"],
            }
        )
        logger_ctx = _DummyLogger()

        result = analyzer.run_analysis(df, None, logger_ctx)

        self.assertIn("amount", result.metrics.primary_metrics + result.metrics.secondary_metrics)
        self.assertNotIn("transaction_id", result.metrics.primary_metrics)

    def test_run_analysis_falls_back_when_llm_parse_fails(self) -> None:
        class _FakeResponses:
            def parse(self, **kwargs):
                raise TypeError("invalid call")

        class _FakeClient:
            responses = _FakeResponses()

        class _FakeOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def __new__(cls, *args, **kwargs):
                return _FakeClient()

        with patch("src.agents.analyzer.OpenAI", _FakeOpenAI):
            analyzer = Analyzer(self.config)

        logger_ctx = _DummyLogger()
        result = analyzer.run_analysis(self.df, "sales dashboard", logger_ctx)

        self.assertEqual(result.sampled_rows, 3)
        self.assertIn("sales", result.metrics.primary_metrics)
        self.assertIn("region", result.data_schema)
        self.assertIn("Heuristic", result.design.notes)


if __name__ == "__main__":
    unittest.main()
