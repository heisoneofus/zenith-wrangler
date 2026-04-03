from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.logging.session import SessionLogger, load_execution_trace, load_session_state
from src.models import DashboardSpec, ExecutionTrace, SessionState


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

    def test_session_state_and_trace_sidecars_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "session.log"
            logger = SessionLogger(path=log_path)
            spec = DashboardSpec(title="Navigator")
            state = SessionState(session_id="session", active_spec=spec, spec_versions=[spec])
            trace = ExecutionTrace(session_id="session")

            logger.log_session_state(state)
            logger.log_execution_trace(trace)

            loaded_state = load_session_state(log_path)
            loaded_trace = load_execution_trace(log_path)

            self.assertEqual(loaded_state.active_spec.title, "Navigator")
            self.assertEqual(loaded_trace.session_id, "session")


if __name__ == "__main__":
    unittest.main()
