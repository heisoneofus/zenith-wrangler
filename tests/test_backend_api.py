from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app


def _client(tmp_path: Path) -> TestClient:
    app = create_app(root_dir=tmp_path)
    return TestClient(app)


def test_analyze_endpoint_persists_session_and_artifacts(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/analyze",
        files={"dataset": ("sales.csv", io.BytesIO(b"region,sales\nEU,10\nUS,20\n"), "text/csv")},
        data={"context_text": "Focus on regional sales trends."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("session_")
    assert payload["analysis"]["data_schema"]["region"] in {"object", "str"}
    assert payload["analysis"]["data_schema"]["sales"] == "int64"
    assert payload["dashboard_spec"]["title"]
    artifact_types = {artifact["type"] for artifact in payload["artifacts"]}
    assert {"log", "state", "trace", "dashboard_spec", "source", "context"} <= artifact_types

    session_id = payload["session_id"]
    sessions = client.get("/sessions")
    assert sessions.status_code == 200
    session_payload = sessions.json()
    assert len(session_payload["items"]) == 1
    assert session_payload["items"][0]["session_id"] == session_id
    assert session_payload["items"][0]["status"] == "planned"

    detail = client.get(f"/sessions/{session_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["session_id"] == session_id
    assert detail_payload["dashboard_spec"]["title"] == payload["dashboard_spec"]["title"]
    assert detail_payload["figures"] == []

    artifact = client.get(f"/artifacts/{session_id}/dashboard_spec")
    assert artifact.status_code == 200
    spec_payload = json.loads(artifact.text)
    assert spec_payload["title"] == payload["dashboard_spec"]["title"]


def test_artifact_endpoint_rejects_unknown_artifact_types(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/artifacts/session_20260404_000000/unknown")

    assert response.status_code == 404
    assert response.json()["code"] == "artifact_not_found"


def test_generate_and_update_endpoints_return_plotly_json(tmp_path: Path) -> None:
    client = _client(tmp_path)

    generate = client.post(
        "/generate",
        files={"dataset": ("sales.csv", io.BytesIO(b"region,sales,profit\nEU,10,4\nUS,20,9\nAPAC,15,5\n"), "text/csv")},
        data={"context_text": "Show sales and profit by region."},
    )

    assert generate.status_code == 200
    generated = generate.json()
    assert generated["session_id"].startswith("session_")
    assert generated["session_status"] in {"reviewed", "repaired", "executed"}
    assert generated["figures"]
    assert "data" in generated["figures"][0]
    assert "layout" in generated["figures"][0]
    generated_artifacts = {artifact["type"] for artifact in generated["artifacts"]}
    assert {"dashboard_spec", "figures", "state", "trace", "source"} <= generated_artifacts

    session_id = generated["session_id"]
    detail = client.get(f"/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["figures"]

    update = client.post(
        "/update",
        json={"session_id": session_id, "prompt": "Change to a scatter chart and add filter for region"},
    )

    assert update.status_code == 200
    updated = update.json()
    assert updated["session_id"] == session_id
    assert updated["figures"]
    assert any(visual["chart_type"] == "scatter" for visual in updated["dashboard_spec"]["visuals"])
    assert "region" in updated["dashboard_spec"]["filters"]

    figures_artifact = client.get(f"/artifacts/{session_id}/figures")
    assert figures_artifact.status_code == 200
    serialized_figures = json.loads(figures_artifact.text)
    assert len(serialized_figures) == len(updated["figures"])
