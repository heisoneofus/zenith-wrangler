import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { fetchSession, updateDashboard } from "../api";
import { PlotlyChart } from "../components/PlotlyChart";

export function UpdatePage() {
  const { sessionId: routeSessionId = "" } = useParams();
  const [sessionId, setSessionId] = useState(routeSessionId);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(Boolean(routeSessionId));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState(null);

  useEffect(() => {
    let active = true;

    async function loadSession(targetSessionId) {
      if (!targetSessionId) {
        setPayload(null);
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");
      try {
        const detail = await fetchSession(targetSessionId);
        if (active) {
          setPayload(detail);
          setSessionId(targetSessionId);
        }
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load the requested session.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadSession(routeSessionId);
    return () => {
      active = false;
    };
  }, [routeSessionId]);

  async function handleLoad() {
    setLoading(true);
    setError("");
    try {
      const detail = await fetchSession(sessionId);
      setPayload(detail);
    } catch (loadError) {
      setError(loadError.message || "Unable to load the requested session.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!sessionId.trim()) {
      setError("Enter a session id before applying an update.");
      return;
    }
    if (!prompt.trim()) {
      setError("Describe the dashboard change you want to apply.");
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      const updated = await updateDashboard(sessionId.trim(), prompt);
      setPayload((current) => ({
        ...(current || {}),
        ...updated,
        status: updated.session_status,
      }));
      setPrompt("");
    } catch (submissionError) {
      setError(submissionError.message || "Unable to apply the dashboard update.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="stack">
      <form className="panel" onSubmit={handleSubmit}>
        <div className="panel__header">
          <div>
            <p className="eyebrow">Update</p>
            <h2>Revise a previous dashboard session</h2>
          </div>
        </div>

        <label className="field">
          <span>Session ID</span>
          <div className="inline-field">
            <input aria-label="Session ID" value={sessionId} onChange={(event) => setSessionId(event.target.value)} />
            <button type="button" className="button button--ghost" onClick={handleLoad} disabled={loading}>
              {loading ? "Loading..." : "Load"}
            </button>
          </div>
        </label>

        <label className="field">
          <span>Update Prompt</span>
          <textarea
            aria-label="Update Prompt"
            rows={5}
            placeholder="Change the first chart to scatter, add a filter, update aggregation, and so on."
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>

        {error ? <p className="status status--error">{error}</p> : null}

        <button type="submit" className="button" disabled={submitting}>
          {submitting ? "Applying..." : "Apply Update"}
        </button>
      </form>

      {payload ? (
        <div className="content-grid">
          <div className="panel">
            <div className="panel__header">
              <h3>{payload.dashboard_spec?.title || "Updated dashboard"}</h3>
              <p>{payload.session_id}</p>
            </div>
            <ul className="schema-list">
              {(payload.dashboard_spec?.visuals || []).map((visual, index) => (
                <li key={`${visual.title || "visual"}-${index}`}>
                  <span>{visual.title || `Visual ${index + 1}`}</span>
                  <code>{visual.chart_type}</code>
                </li>
              ))}
            </ul>
          </div>

          <div className="panel">
            <div className="panel__header">
              <h3>Updated charts</h3>
              <p>{payload.figures?.length || 0} figure(s)</p>
            </div>
            {payload.figures?.length ? (
              <div className="charts-grid">
                {payload.figures.map((figure, index) => (
                  <article className="chart-card" key={`${payload.session_id}-update-${index}`}>
                    <PlotlyChart figure={figure} title={payload.dashboard_spec?.visuals?.[index]?.title || `Figure ${index + 1}`} />
                  </article>
                ))}
              </div>
            ) : (
              <p className="status">Load a session and apply an update to see regenerated charts here.</p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  );
}
