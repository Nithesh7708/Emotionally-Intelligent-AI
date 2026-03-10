const API_BASE = "http://127.0.0.1:8000"

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

