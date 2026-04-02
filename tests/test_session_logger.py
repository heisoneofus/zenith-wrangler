from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.logging.session import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_log_tool_call_serializes_path_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "session.log"
            logger = SessionLogger(path=log_path)

            logger.log_tool_call(
                index=1,
                tool_name="read_csv",
                reasoning="Load source file",
                params={"path": Path("data/input.csv"), "sample_rows": None},
            )

            content = log_path.read_text(encoding="utf-8")
            self.assertIn('"path": "data\\\\input.csv"', content)
            self.assertIn('"sample_rows": null', content)

    def test_log_json_serializes_datetime_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "session.log"
            logger = SessionLogger(path=log_path)

            logger.log_json("Timestamps", {"created_at": datetime(2026, 4, 2, 0, 0, 0)})

            content = log_path.read_text(encoding="utf-8")
            self.assertIn('"created_at": "2026-04-02T00:00:00"', content)


if __name__ == "__main__":
    unittest.main()
