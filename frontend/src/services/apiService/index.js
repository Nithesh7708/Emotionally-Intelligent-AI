const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

export async function sendVoiceMessage({ audioBlob, transcript }) {
  const fd = new FormData();
  fd.append("audio", audioBlob, "input.wav");
  if (transcript) fd.append("transcript", transcript);

  const res = await fetch(`${API_BASE}/api/voice-chat`, {
    method: "POST",
    body: fd
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Backend error (${res.status})`);
  }

  return await res.json();
}

