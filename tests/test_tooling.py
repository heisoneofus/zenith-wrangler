from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from src.agents.orchestrator import build_registry


class ToolingSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_validate_params_accepts_valid_fill_missing_payload(self) -> None:
        spec = self.registry.get("fill_missing")

        validated = spec.validate_params(
            {
                "strategy": "constant",
                "fill_value": {"region": "Unknown"},
            }
        )

        normalized = validated.model_dump(mode="python", exclude_none=True)
        self.assertEqual(normalized["strategy"], "constant")
        self.assertEqual(normalized["fill_value"], {"region": "Unknown"})

    def test_validate_params_rejects_invalid_read_csv_payload(self) -> None:
        spec = self.registry.get("read_csv")

        with self.assertRaises(ValidationError):
            spec.validate_params({"path": "data.csv", "sample_rows": 0})

    def test_export_tool_catalog_is_json_compatible(self) -> None:
        catalog = self.registry.export_tool_catalog()

        json_blob = json.dumps(catalog)
        names = {item["name"] for item in catalog}
        fill_missing = next(item for item in catalog if item["name"] == "fill_missing")

        self.assertTrue(json_blob.startswith("["))
        self.assertIn("read_csv", names)
        self.assertIn("build_dashboard", names)
        self.assertEqual(fill_missing["category"], "cleaning")
        self.assertIn("input_schema", fill_missing)


if __name__ == "__main__":
    unittest.main()
