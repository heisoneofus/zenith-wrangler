from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
from click.testing import CliRunner

from wrangler import _prepare_dataframe_for_parquet, _write_tool_catalog_artifact, main


class WranglerParquetPreparationTests(unittest.TestCase):
    def test_prepare_dataframe_for_parquet_converts_mixed_object_columns(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Table 1": [b"Crime in the United States", 1997, None],
                "Category": ["violent", "property", None],
                "Count": [1, 2, 3],
            }
        )

        prepared = _prepare_dataframe_for_parquet(dataframe)

        self.assertEqual(prepared.at[0, "Table 1"], "Crime in the United States")
        self.assertEqual(prepared.at[1, "Table 1"], "1997")
        self.assertTrue(pd.isna(prepared.at[2, "Table 1"]))
        self.assertEqual(prepared.at[0, "Category"], "violent")
        self.assertEqual(prepared.at[1, "Category"], "property")
        self.assertTrue(pd.isna(prepared.at[2, "Category"]))
        self.assertEqual(prepared["Count"].tolist(), [1, 2, 3])
        self.assertEqual(dataframe["Table 1"].tolist(), [b"Crime in the United States", 1997, None])

    def test_prepare_dataframe_for_parquet_allows_parquet_export_after_coercion(self) -> None:
        dataframe = pd.DataFrame({"Table 1": [b"Crime", 1997, None], "Count": [1, 2, 3]})

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "transformed.parquet"

            with self.assertRaises(pa.ArrowTypeError):
                dataframe.to_parquet(output_path, index=False)

            prepared = _prepare_dataframe_for_parquet(dataframe)
            prepared.to_parquet(output_path, index=False)
            loaded = pd.read_parquet(output_path)

        self.assertEqual(loaded.at[0, "Table 1"], "Crime")
        self.assertEqual(loaded.at[1, "Table 1"], "1997")
        self.assertTrue(pd.isna(loaded.at[2, "Table 1"]))


class ToolCatalogExportTests(unittest.TestCase):
    def test_write_tool_catalog_artifact_creates_versioned_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            artifact_path = _write_tool_catalog_artifact(output_dir)

            self.assertTrue(artifact_path.exists())
            self.assertRegex(artifact_path.name, r"^tool_catalog_\d{8}T\d{6}Z\.json$")

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(payload["artifact_version"], artifact_path.stem.replace("tool_catalog_", ""))
            self.assertIsInstance(payload["tools"], list)
            self.assertGreater(len(payload["tools"]), 0)

    def test_cli_dump_tool_catalog_writes_artifact_and_exits(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(main, ["--dump-tool-catalog", "--catalog-output-dir", tmp])

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Tool catalog artifact written to:", result.output)

            artifact_files = list(Path(tmp).glob("tool_catalog_*.json"))
            self.assertEqual(len(artifact_files), 1)

            payload = json.loads(artifact_files[0].read_text(encoding="utf-8"))
            self.assertIn("tools", payload)
            self.assertTrue(any(tool.get("name") == "read_csv" for tool in payload["tools"]))

    def test_cli_review_only_writes_session_state_artifacts(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "sample.csv"
            data_path.write_text("region,sales\nEU,10\nUS,20\n", encoding="utf-8")
            before = {path.name for path in Path("logs").glob("session_*.state.json")}

            result = runner.invoke(main, ["--data", str(data_path), "--review-only", "--output-format", "html"])

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Plan proposal written to:", result.output)
            after = {path.name for path in Path("logs").glob("session_*.state.json")}
            self.assertGreater(len(after - before), 0)


if __name__ == "__main__":
    unittest.main()
