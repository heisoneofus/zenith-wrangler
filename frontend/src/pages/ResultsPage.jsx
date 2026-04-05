import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchSession } from "../api";
import { PlotlyChart } from "../components/PlotlyChart";

function MetricList({ label, values }) {
  return (
    <div className="stat-card">
      <p className="stat-card__label">{label}</p>
      <p className="stat-card__value">{values?.length ? values.join(", ") : "None inferred"}</p>
    </div>
  );
}

export function ResultsPage() {
  const { sessionId = "" } = useParams();
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSession() {
      setLoading(true);
      setError("");
      try {
        const session = await fetchSession(sessionId);
        if (active) {
          setPayload(session);
        }
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load this session.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    if (sessionId) {
      loadSession();
    }

    return () => {
      active = false;
    };
  }, [sessionId]);

  if (loading) {
    return <section className="panel"><p className="status">Loading session…</p></section>;
  }

  if (error) {
    return <section className="panel"><p className="status status--error">{error}</p></section>;
  }

  if (!payload) {
    return <section className="panel"><p className="status">No session data found.</p></section>;
  }

  const metrics = payload.analysis?.metrics || {};
  const quality = payload.analysis?.quality || {};
  const schemaEntries = Object.entries(payload.analysis?.data_schema || {});

  return (
    <section className="stack">
      <div className="panel panel--hero">
        <div>
          <p className="eyebrow">Results</p>
          <h2>{payload.dashboard_spec.title}</h2>
          <p className="status-chip">Session {payload.session_id}</p>
        </div>
        <div className="hero-actions">
          <Link className="button button--ghost" to="/sessions">
            Back to sessions
          </Link>
          <Link className="button" to={`/update/${payload.session_id}`}>
            Update dashboard
          </Link>
        </div>
      </div>

      <div className="stats-grid">
        <MetricList label="Primary metrics" values={metrics.primary_metrics} />
        <MetricList label="Dimensions" values={metrics.dimensions} />
        <MetricList label="Time fields" values={metrics.time_fields} />
        <MetricList
          label="Quality issues"
          values={(quality.issues || []).map((issue) => `${issue.type} (${issue.severity})`)}
        />
      </div>

      <div className="content-grid">
        <div className="panel">
          <div className="panel__header">
            <h3>Charts</h3>
            <p>{payload.figures?.length ? `${payload.figures.length} rendered figure(s)` : "Analysis-only session"}</p>
          </div>
          {payload.figures?.length ? (
            <div className="charts-grid">
              {payload.figures.map((figure, index) => (
                <article key={`${payload.session_id}-${index}`} className="chart-card">
                  <PlotlyChart figure={figure} title={payload.dashboard_spec.visuals?.[index]?.title || `Figure ${index + 1}`} />
                </article>
              ))}
            </div>
          ) : (
            <p className="status">No figures are stored for this session yet. Run the full generate flow to render charts.</p>
          )}
        </div>

        <div className="panel">
          <div className="panel__header">
            <h3>Schema</h3>
            <p>Inferred from the uploaded dataset</p>
          </div>
          <ul className="schema-list">
            {schemaEntries.map(([name, dtype]) => (
              <li key={name}>
                <span>{name}</span>
                <code>{dtype}</code>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
