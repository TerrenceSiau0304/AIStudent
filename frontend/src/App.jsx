import { useEffect, useRef, useState } from "react";
import ChatMessage from "./components/ChatMessage.jsx";

const API_URL = `${import.meta.env.VITE_API_BASE_URL}/chat`;

function getSessionId() {
  //Browser storage, data are remember when page refresh
  let id = localStorage.getItem("studentai_session_id"); 
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("studentai_session_id", id);
  }
  return id;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const sessionId = useRef(getSessionId());
  const bodyRef = useRef(null);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]); //Change whenever add AI or user message

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;

    setInput("");
    setBusy(true);

    const userMsg = { role: "user", content: text };
    // placeholder assistant message we'll fill in as events arrive
    const assistantMsg = { role: "assistant", status: null, content: "", streaming: true, error: null };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    const updateAssistant = (patch) => {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, ...patch }; // update the last's attribute based on what given by patch
        return next;
      });
    };

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId.current }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // keep the last (possibly incomplete) chunk

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();

          if (payload === "[DONE]") {
            updateAssistant({ streaming: false });
            continue;
          }

          try {
            const event = JSON.parse(payload);
            if (event.type === "status") {
              updateAssistant({ status: event.node });
            } else if (event.type === "answer") {
              updateAssistant({ content: event.content, status: null });
            } else if (event.type === "error") {
              updateAssistant({ error: event.message, streaming: false });
            }
          } catch {
            // ignore malformed chunk, keep reading
          }
        }
      }
    } catch (err) {
      updateAssistant({ error: err.message || "Something went wrong.", streaming: false });
    } finally {
      setBusy(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app-frame">
      <div className="board">
        <div className="board-header">
          <h1>StudentAI</h1>
          <span className="session-tag">session {sessionId.current.slice(0, 8)}</span>
        </div>

        <div className="board-body" ref={bodyRef}>
          {messages.length === 0 && (
            <div className="empty-state">
              <span className="mark">Chalk one up</span>
              <p>Ask a question about your course material and I'll dig through your notes, and the web if I need to.</p>
            </div>
          )}

          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} /> // must include a key to identify each child
          ))}
        </div>

        <div className="board-tray">
          <input
            type="text"
            placeholder="Ask something…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={busy}
          />
          <button onClick={sendMessage} disabled={busy || !input.trim()}>
            {busy ? "Asking…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}
