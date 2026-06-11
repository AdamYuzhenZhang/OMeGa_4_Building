export async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (response.ok) return response.json();

  let message = `${response.status} ${response.statusText}`;
  try {
    const errorPayload = await response.json();
    message = errorPayload.detail || message;
  } catch {
    // Keep the HTTP status message.
  }
  throw new Error(message);
}
