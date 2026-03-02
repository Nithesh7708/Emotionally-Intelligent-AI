import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { startRecording, stopRecording, createAnalyser } from "../../services/audioRecorderService/index.js";
import { sendVoiceMessage } from "../../services/apiService/index.js";
import { speak, stopSpeaking } from "../../services/speechPlaybackService/index.js";

const EMOTION_COLORS = {
  happy:   "#f59e0b",
  sad:     "#3b82f6",
  angry:   "#ef4444",
  fear:    "#a855f7",
  neutral: "#6366f1",
};
const EMOTION_LABELS = { happy: "Happy", sad: "Sad", angry: "Angry", fear: "Fearful", neutral: "Neutral" };
const EMOTION_EMOJI  = { happy: "😊", sad: "😢", angry: "😠", fear: "😨", neutral: "😐" };

const AI_WAVE_PEAKS  = [4,10,20,32,44,54,60,56,62,50,58,46,58,50,62,56,60,54,44,32,20,10,4,3];
const MIC_BAR_COUNT  = 24;

// ── Emotion Block ─────────────────────────────────────────────────
function EmotionBlock({ emotion, color }) {
  return (
    <div className={`vaEmotionBlock emo--${emotion}`} style={{ "--ec": color }} key={emotion}>
      {/* Radial glow */}
      <div className="vaEmoGlow" />

      {/* Particles — per emotion */}
      {emotion === "happy" && (
        <>
          <span className="emoP emoSpark es1">✦</span>
          <span className="emoP emoSpark es2">✦</span>
          <span className="emoP emoSpark es3">✦</span>
          <span className="emoP emoSpark es4">✦</span>
          <span className="emoP emoSpark es5">✦</span>
        </>
      )}

      {emotion === "sad" && (
        <>
          <div className="emoP emoTear et1" />
          <div className="emoP emoTear et2" />
          <div className="emoP emoTear et3" />
        </>
      )}

      {emotion === "angry" && (
        <>
          <span className="emoP emoFlame ef1">🔥</span>
          <span className="emoP emoFlame ef2">🔥</span>
          <span className="emoP emoFlame ef3">🔥</span>
        </>
      )}

      {emotion === "fear" && (
        <>
          <div className="emoP emoPulse ep1" />
          <div className="emoP emoPulse ep2" />
          <div className="emoP emoPulse ep3" />
          <div className="emoP emoPulse ep4" />
          <span className="emoP emoZap ez1">⚡</span>
          <span className="emoP emoZap ez2">⚡</span>
        </>
      )}

      {emotion === "neutral" && (
        <div className="emoP emoOrbitRing">
          <div className="emoOrbitDot" />
        </div>
      )}

      {/* Main emoji */}
      <span className="vaEmoIcon">{EMOTION_EMOJI[emotion] || "😐"}</span>

      {/* Label */}
      <span className="vaEmoLabel">{EMOTION_LABELS[emotion] || "Neutral"}</span>
    </div>
  );
}

// ── Speech recognition ────────────────────────────────────────────
function createSpeechRecognition() {
  const SR = window.webkitSpeechRecognition;
  if (!SR) return null;
  const rec = new SR();
  rec.continuous = false;
  rec.interimResults = false;
  rec.lang = "en-US";
  return rec;
}

// ── Main screen ───────────────────────────────────────────────────
export default function VoiceChatScreen() {
  const [isRecording,    setIsRecording]    = useState(false);
  const [isAiSpeaking,   setIsAiSpeaking]   = useState(false);
  const [isLoading,      setIsLoading]      = useState(false);
  const [emotion,        setEmotion]        = useState("neutral");
  const [aiResponse,     setAiResponse]     = useState("");
  const [userTranscript, setUserTranscript] = useState("");
  const [error,          setError]          = useState("");
  const [showSettings,   setShowSettings]   = useState(false);
  const [voices,         setVoices]         = useState([]);
  const [selectedVoice,  setSelectedVoice]  = useState("");
  const [rate,  setRate]  = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [micBars, setMicBars] = useState(() => new Array(MIC_BAR_COUNT).fill(3));

  const speechRecRef = useRef(null);
  const analyserRef  = useRef(null);
  const animFrameRef = useRef(null);

  const canSpeechRecognize = useMemo(() => Boolean(window.webkitSpeechRecognition), []);
  const emotionColor = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;

  useEffect(() => {
    function loadVoices() {
      const list = window.speechSynthesis?.getVoices() ?? [];
      setVoices(list);
      if (list.length && !selectedVoice) setSelectedVoice(list[0].name);
    }
    loadVoices();
    if ("speechSynthesis" in window) window.speechSynthesis.onvoiceschanged = loadVoices;
  }, []);

  const startMicWave = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const step = Math.floor(data.length / MIC_BAR_COUNT);
    function tick() {
      animFrameRef.current = requestAnimationFrame(tick);
      analyser.getByteFrequencyData(data);
      const half = MIC_BAR_COUNT / 2;
      const halfBars = Array.from({ length: half }, (_, i) => Math.max(3, (data[i * step] / 255) * 58));
      setMicBars([...halfBars.slice().reverse(), ...halfBars]);
    }
    tick();
  }, []);

  const stopMicWave = useCallback(() => {
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null; }
    analyserRef.current = null;
    setMicBars(new Array(MIC_BAR_COUNT).fill(3));
  }, []);

  async function onStart() {
    setError(""); setUserTranscript(""); setAiResponse("");
    stopSpeaking(); setIsAiSpeaking(false);
    try {
      await startRecording();
      setIsRecording(true);
      const analyser = createAnalyser(128);
      analyserRef.current = analyser;
      startMicWave();
      if (canSpeechRecognize) {
        const rec = createSpeechRecognition();
        speechRecRef.current = rec;
        rec.onresult = (e) => setUserTranscript(e?.results?.[0]?.[0]?.transcript || "");
        rec.onerror  = () => {};
        rec.start();
      }
    } catch (e) {
      setError(e?.message || "Could not start microphone.");
      setIsRecording(false);
    }
  }

  async function onStop() {
    setError(""); setIsRecording(false); setIsLoading(true);
    stopMicWave();
    try {
      if (speechRecRef.current) { try { speechRecRef.current.stop(); } catch { /* ignore */ } }
      const result = await stopRecording();
      if (!result?.blob) throw new Error("No audio captured.");
      const transcript = userTranscript?.trim() || null;
      const data = await sendVoiceMessage({ audioBlob: result.blob, transcript });
      setEmotion(data.emotion || "neutral");
      const text = data.response_text || "";
      setAiResponse(text);
      if (text) {
        speak(text, {
          rate, pitch, voiceName: selectedVoice,
          onStart: () => setIsAiSpeaking(true),
          onEnd:   () => setIsAiSpeaking(false),
        });
      }
    } catch (e) {
      setError(e?.message || "Request failed.");
    } finally {
      setIsLoading(false);
    }
  }

  const handleMicClick = () => {
    if (isLoading) return;
    if (isRecording) onStop(); else onStart();
  };

  function handleTestVoice() {
    speak("Hello! This is how I sound with the current voice settings.", {
      rate, pitch, voiceName: selectedVoice,
      onStart: () => setIsAiSpeaking(true),
      onEnd:   () => setIsAiSpeaking(false),
    });
  }

  const showMicWave = isRecording;
  const showAiWave  = isAiSpeaking && !isRecording;

  return (
    <div className="vaScreen">

      {/* ── Top bar ── */}
      <div className="vaTopBar">
        <div className="vaLogo">
          <span className="vaLogoIcon">🎙</span>
          <span className="vaLogoText">EmoSist</span>
        </div>
        <button className="vaSettingsBtn" onClick={() => setShowSettings((s) => !s)} aria-label="Voice settings">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>

      {/* ── Settings panel ── */}
      {showSettings && (
        <div className="vaSettings">
          <div className="vaSettingsTitle">Voice Settings</div>
          <div className="vaSettingsRow">
            <label className="vaSettingsLabel">Voice</label>
            <select className="vaSelect" value={selectedVoice} onChange={(e) => setSelectedVoice(e.target.value)}>
              {voices.length === 0 && <option value="">Default</option>}
              {voices.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
            </select>
          </div>
          <div className="vaSettingsRow">
            <label className="vaSettingsLabel">Speed <span className="vaSettingsVal">{rate.toFixed(1)}x</span></label>
            <input className="vaRange" type="range" min="0.5" max="2" step="0.1" value={rate} onChange={(e) => setRate(Number(e.target.value))} />
          </div>
          <div className="vaSettingsRow">
            <label className="vaSettingsLabel">Pitch <span className="vaSettingsVal">{pitch.toFixed(1)}</span></label>
            <input className="vaRange" type="range" min="0.5" max="2" step="0.1" value={pitch} onChange={(e) => setPitch(Number(e.target.value))} />
          </div>
          <button className="vaTestBtn" onClick={handleTestVoice} type="button">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21" /></svg>
            Test Voice
          </button>
        </div>
      )}

      {/* ── Stage ── */}
      <div className="vaStage">

        {/* Big emotion block with animations */}
        <EmotionBlock emotion={emotion} color={emotionColor} />

        {/* Live transcript while recording */}
        {isRecording && userTranscript && (
          <div className="vaTranscript">{userTranscript}</div>
        )}

        {/* Orb */}
        <div className="vaOrbArea">
          {isRecording && (
            <>
              <div className="vaRing vaRing1" style={{ borderColor: emotionColor }} />
              <div className="vaRing vaRing2" style={{ borderColor: emotionColor }} />
            </>
          )}
          {isAiSpeaking && (
            <>
              <div className="vaRing vaRing1 vaRing--ai" style={{ borderColor: emotionColor }} />
              <div className="vaRing vaRing2 vaRing--ai" style={{ borderColor: emotionColor }} />
            </>
          )}
          <button
            className={`vaOrb${isRecording ? " vaOrb--recording" : ""}${isAiSpeaking ? " vaOrb--speaking" : ""}${isLoading ? " vaOrb--loading" : ""}`}
            style={{ "--ec": emotionColor }}
            onClick={handleMicClick}
            disabled={isLoading}
            aria-label={isRecording ? "Stop recording" : "Start speaking"}
          >
            {isLoading ? (
              <svg className="vaSpinner" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            ) : isRecording ? (
              <svg width="34" height="34" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="3" />
              </svg>
            ) : (
              <svg width="34" height="34" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 1a4 4 0 0 0-4 4v7a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
                <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                <line x1="8" y1="23" x2="16" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            )}
          </button>
        </div>

        {/* Waveform zone */}
        <div className="vaWaveZone">
          {showMicWave && (
            <div className="vaWaveRow">
              {micBars.map((h, i) => (
                <div key={i} className="vaBar" style={{ height: `${h}px`, background: emotionColor, opacity: 0.5 + (h / 58) * 0.5 }} />
              ))}
            </div>
          )}
          {showAiWave && (
            <div className="vaWaveRow">
              {AI_WAVE_PEAKS.map((peak, i) => (
                <div key={i} className="vaAiBar" style={{ "--peak": `${peak}px`, "--delay": `${(i * 0.055).toFixed(3)}s`, "--dur": `${0.7 + (i % 5) * 0.08}s`, background: emotionColor }} />
              ))}
            </div>
          )}
        </div>

        {/* Status */}
        <div className="vaStatus">
          {isLoading ? "Processing…" : isRecording ? "Listening…" : isAiSpeaking ? "Speaking…" : "Tap to speak"}
        </div>

        {/* AI response — compact */}
        {aiResponse && !isLoading && (
          <div className="vaAiText">{aiResponse}</div>
        )}

        {error && <div className="vaError">{error}</div>}
        {!canSpeechRecognize && <div className="vaHint">Use Chrome or Edge for speech recognition</div>}
      </div>
    </div>
  );
}
