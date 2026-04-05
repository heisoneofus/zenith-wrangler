import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { fetchSessions } from "../api";

export function SessionsPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSessions() {
      setLoading(true);
      setError("");
      try {
        const payload = await fetchSessions();
        if (active) {
          setItems(payload.items || []);
        }
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load sessions.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadSessions();
    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Sessions</p>
          <h2>Previous runs</h2>
        </div>
        <Link className="button" to="/">
          New run
        </Link>
      </div>

      {loading ? <p className="status">Loading sessions…</p> : null}
      {error ? <p className="status status--error">{error}</p> : null}
      {!loading && !error && !items.length ? <p className="status">No sessions have been recorded yet.</p> : null}

      <div className="session-list">
        {items.map((item) => (
          <article className="session-card" key={item.session_id}>
            <div>
              <p className="session-card__status">{item.status}</p>
              <h3>{item.title}</h3>
              <p className="session-card__meta">{item.session_id}</p>
            </div>
            <div className="session-card__actions">
              <Link className="button button--ghost" to={`/results/${item.session_id}`}>
                Inspect
              </Link>
              <Link className="button button--secondary" to={`/update/${item.session_id}`}>
                Update
              </Link>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
