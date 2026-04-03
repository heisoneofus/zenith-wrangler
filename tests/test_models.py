from __future__ import annotations

import unittest

from openai.lib._pydantic import to_strict_json_schema

from src.models import (
    AnalysisReport,
    DashboardSpec,
    DataQualityAssessment,
    MetricsAnalysis,
    SessionState,
    VisualSpec,
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

    def test_visual_spec_includes_agentic_provenance_defaults(self) -> None:
        visual = VisualSpec(title="Sales", chart_type="bar")

        self.assertTrue(visual.id.startswith("visual_"))
        self.assertEqual(visual.rationale, "")
        self.assertEqual(visual.warnings, [])
        self.assertEqual(visual.status, "proposed")

    def test_session_state_tracks_spec_versions(self) -> None:
        spec = DashboardSpec(title="Demo")
        state = SessionState(session_id="session_1", active_spec=spec, spec_versions=[spec])

        self.assertEqual(state.active_spec.title, "Demo")
        self.assertEqual(len(state.spec_versions), 1)


if __name__ == "__main__":
    unittest.main()
