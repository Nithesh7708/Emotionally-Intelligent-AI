export default function MicButton({ isRecording, isLoading, onStart, onStop }) {
  const label = isLoading ? "Thinking…" : isRecording ? "Stop" : "Speak";

  const onClick = () => {
    if (isLoading) return;
    if (isRecording) onStop();
    else onStart();
  };

  return (
    <button
      className={`micButton ${isRecording ? "recording" : ""}`}
      onClick={onClick}
      disabled={isLoading}
      type="button"
      aria-pressed={isRecording}
    >
      {label}
    </button>
  );
}

