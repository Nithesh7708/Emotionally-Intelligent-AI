export function stopSpeaking() {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
}

export function speak(text, opts = {}) {
  if (!("speechSynthesis" in window)) return;
  stopSpeaking();

  const utter = new SpeechSynthesisUtterance(String(text || ""));
  utter.rate   = typeof opts.rate   === "number" ? opts.rate   : 1.0;
  utter.pitch  = typeof opts.pitch  === "number" ? opts.pitch  : 1.0;
  utter.volume = typeof opts.volume === "number" ? opts.volume : 1.0;

  if (opts.voiceName) {
    const match = window.speechSynthesis.getVoices().find((v) => v.name === opts.voiceName);
    if (match) utter.voice = match;
  }

  if (typeof opts.onStart === "function") utter.onstart = opts.onStart;
  if (typeof opts.onEnd   === "function") utter.onend   = opts.onEnd;

  window.speechSynthesis.speak(utter);
}
