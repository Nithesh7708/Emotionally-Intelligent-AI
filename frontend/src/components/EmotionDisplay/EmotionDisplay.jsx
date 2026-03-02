const COLORS = {
  happy: "#18a34a",
  sad: "#2563eb",
  angry: "#dc2626",
  fear: "#9333ea",
  neutral: "#64748b"
};

export default function EmotionDisplay({ emotion }) {
  const color = COLORS[emotion] || COLORS.neutral;
  return (
    <div className="emotionDisplay">
      <div className="emotionDot" style={{ background: color }} />
      <div className="emotionLabel">Emotion: {emotion || "neutral"}</div>
    </div>
  );
}

