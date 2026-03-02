export default function AudioPlayer({ title, src }) {
  return (
    <div className="audioPlayer">
      <div className="audioTitle">{title}</div>
      {src ? (
        <audio controls src={src} />
      ) : (
        <div className="audioEmpty">No recording yet.</div>
      )}
    </div>
  );
}

