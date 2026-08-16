const STATUS_LABELS = {
  retrieve: "Pulling notes from the shelf…",
  web_search: "Stepping out to the library…",
  grade_document: "Checking these notes hold up…",
  generate: "Chalking up an answer…",
  llm_fallback: "Fallback...",
};

export default function ChatMessage({ message }) {
    /**
   * Render each user and assistant messages component.
   *
   */
  if (message.role === "user") {
    return (
      <div className="msg-row user">
        <div className="msg-user">{message.content}</div>
      </div>
    );
  }

  // assistant message: may still be streaming (status), mid-answer, or errored
  return (
    <div className="msg-row assistant">
      <div className="msg-assistant">
        {message.status && (
          <div className={`status-line ${message.content ? "wiping" : ""}`}>
            <span className="dot" />
            {STATUS_LABELS[message.status] || `${message.status}…`}
          </div>
        )}

        {message.error && <div className="error-text">{message.error}</div>}

        {message.content && (
          <div className="answer-text">
            {message.content}
            {message.streaming && <span className="caret" />}
          </div>
        )}
      </div>
    </div>
  );
}
