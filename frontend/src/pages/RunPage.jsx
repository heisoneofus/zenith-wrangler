import { startTransition, useState } from "react";
import { useNavigate } from "react-router-dom";

import { uploadDataset } from "../api";

export function RunPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [contextText, setContextText] = useState("");
  const [busyAction, setBusyAction] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(mode) {
    if (!file) {
      setError("Choose a dataset file before running analysis.");
      return;
    }

    setBusyAction(mode);
    setError("");
    try {
      const payload = await uploadDataset(mode === "analyze" ? "/analyze" : "/generate", file, contextText);
      startTransition(() => {
        navigate(`/results/${payload.session_id}`);
      });
    } catch (submissionError) {
      setError(submissionError.message || "Unable to process the dataset right now.");
    } finally {
      setBusyAction("");
    }
  }

  return (
    <section className="panel panel--feature">
      <div className="panel__intro">
        <p className="eyebrow">Run pipeline</p>
        <h2>Upload a dataset and hand it to the FastAPI backend.</h2>
        <p>
          The backend keeps the existing analyzer, orchestrator, and session artifacts intact. This UI just packages the
          upload and shows the results in a lighter workflow than the original CLI + Dash loop.
        </p>
      </div>

      <label className="field">
        <span>Dataset File</span>
        <input
          aria-label="Dataset File"
          type="file"
          accept=".csv,.xlsx,.xls,.parquet"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
      </label>

      <label className="field">
        <span>Context</span>
        <textarea
          aria-label="Context"
          rows={6}
          placeholder="Optional analyst guidance, business questions, or dashboard goals."
          value={contextText}
          onChange={(event) => setContextText(event.target.value)}
        />
      </label>

      {error ? <p className="status status--error">{error}</p> : null}

      <div className="actions">
        <button type="button" className="button button--secondary" disabled={!!busyAction} onClick={() => handleSubmit("analyze")}>
          {busyAction === "analyze" ? "Analyzing..." : "Analyze Only"}
        </button>
        <button type="button" className="button" disabled={!!busyAction} onClick={() => handleSubmit("generate")}>
          {busyAction === "generate" ? "Generating..." : "Generate Dashboard"}
        </button>
      </div>
    </section>
  );
}
