from __future__ import annotations

import unittest

from openai.lib._pydantic import to_strict_json_schema

from src.models import (
    AnalysisReport,
    DashboardSpec,
    DataQualityAssessment,
    MetricsAnalysis,
)


class AnalysisReportModelTests(unittest.TestCase):
    def test_model_dump_uses_data_schema_key(self) -> None:
        report = AnalysisReport(
            metrics=MetricsAnalysis(),
            quality=DataQualityAssessment(),
            design=DashboardSpec(),
            sampled_rows=3,
            data_schema={"value": "int64"},
        )

        dumped = report.model_dump()

        self.assertIn("data_schema", dumped)
        self.assertEqual(dumped["data_schema"], {"value": "int64"})
        self.assertNotIn("schema", dumped)

    def test_schema_alias_is_accepted_on_input(self) -> None:
        report = AnalysisReport.model_validate(
            {
                "metrics": {},
                "quality": {},
                "design": {},
                "sampled_rows": 1,
                "schema": {"created_at": "datetime64[ns]"},
            }
        )

        self.assertEqual(report.data_schema, {"created_at": "datetime64[ns]"})

    def test_analysis_report_schema_required_matches_properties(self) -> None:
        schema = to_strict_json_schema(AnalysisReport)
        required = set(schema.get("required", []))
        properties = set(schema.get("properties", {}).keys())
        self.assertSetEqual(required, properties)
        self.assertIn("data_schema", properties)
        self.assertNotIn("schema", properties)


if __name__ == "__main__":
    unittest.main()
