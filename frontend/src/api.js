const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  const contentType = response.headers?.get?.("content-type") || "application/json";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload !== null
        ? payload.message || payload.code || "Request failed"
        : "Request failed";
    throw new Error(message);
  }

  return payload;
}

export async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  return parseResponse(response);
}

export async function uploadDataset(path, file, contextText) {
  const formData = new FormData();
  formData.append("dataset", file);
  if (contextText.trim()) {
    formData.append("context_text", contextText.trim());
  }
  return apiFetch(path, {
    method: "POST",
    body: formData,
  });
}

export async function fetchSession(sessionId) {
  return apiFetch(`/sessions/${sessionId}`);
}

export async function fetchSessions() {
  return apiFetch("/sessions");
}

export async function updateDashboard(sessionId, prompt) {
  return apiFetch("/update", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ session_id: sessionId, prompt }),
  });
}
