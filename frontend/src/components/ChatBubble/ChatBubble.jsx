export default function ChatBubble({ role, text }) {
  return (
    <div className={`bubbleRow ${role}`}>
      <div className="bubble">
        <div className="bubbleRole">{role === "user" ? "You" : "AI"}</div>
        <div className="bubbleText">{text}</div>
      </div>
    </div>
  );
}

