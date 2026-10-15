import { useEffect, useRef, useState } from "react";
import ChatMessage from "./components/ChatMessage.jsx";

const API_URL = `${import.meta.env.VITE_API_BASE_URL}/chat`;
const CHATS_STORAGE_KEY = "studentai_chats";
const ACTIVE_SESSION_STORAGE_KEY = "studentai_active_session_id";

function getSessionId() {
  //Get ID per session
  //Browser storage, data are remember when page refresh
  let id = localStorage.getItem("studentai_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("studentai_session_id", id);
  }
  return id;
}

function loadSessions() {
  const storedSessions = localStorage.getItem(CHATS_STORAGE_KEY);
  if (storedSessions) {
    try {
      const parsed = JSON.parse(storedSessions);
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        parsed.every((session) =>
          typeof session.id === "string" &&
          typeof session.title === "string" &&
          Array.isArray(session.messages)
        )
      ) {
        return parsed;
      }
    } catch (error) {
      console.error("Unable to restore saved chats.", error);
    }
  }

  return [{ id: getSessionId(), title: "New chat", messages: [], updatedAt: Date.now() }];
}

/** 
* Main application component for StudentAI.
* Responsible for: * - Managing chat messages and user input. 
* - Maintaining the current chat session ID. 
* - Sending questions to the backend API. 
* - Receiving and processing Server-Sent Event (SSE) responses. 
* - Updating the assistant message as backend events arrive. * 
* - Automatically scrolling to the latest message. 
* - Handling Enter-key submission. */
export default function App() {
  const [sessions, setSessions] = useState(loadSessions);
  const [activeSessionId, setActiveSessionId] = useState(() =>
    localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY) || getSessionId()
  );
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bodyRef = useRef(null);
  const activeSession = sessions.find((session) => session.id === activeSessionId) || sessions[0];
  const messages = activeSession.messages;
  const orderedSessions = [...sessions].sort(
    (left, right) => (right.updatedAt || 0) - (left.updatedAt || 0)
  );

  useEffect(() => {
    localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    if (!sessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(sessions[0].id);
      return;
    }
    localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, activeSessionId);
  }, [activeSessionId, sessions]);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: "smooth" });
  }, [activeSessionId, messages]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;

    setInput("");
    setBusy(true);

    const sessionId = activeSession.id;
    const assistantMessageId = crypto.randomUUID();
    const userMsg = { role: "user", content: text };
    const assistantMsg = {
      id: assistantMessageId,
      role: "assistant",
      status: null,
      content: "",
      streaming: true,
      error: null,
    };
    setSessions((prev) => prev.map((session) => {
      if (session.id !== sessionId) return session;
      return {
        ...session,
        title: session.title === "New chat"
          ? (text.length > 32 ? `${text.slice(0, 32)}…` : text)
          : session.title,
        messages: [...session.messages, userMsg, assistantMsg],
        updatedAt: Date.now(),
      };
    }));

    const updateAssistant = (patch) => {
      setSessions((prev) => prev.map((session) => session.id !== sessionId
        ? session
        : {
          ...session,
          messages: session.messages.map((message) =>
            message.id === assistantMessageId ? { ...message, ...patch } : message
          ),
          updatedAt: Date.now(),
        }));
    };

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      //Use to decode the bits to readable text
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
          //Only focus on information starting with "data:"
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();

          //Stop the caret once done
          if (payload === "[DONE]") {
            updateAssistant({ streaming: false });
            continue;
          }

          //Get the information and update accordingly
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
            // Ignore malformed SSE chunks and continue reading the response.
          }
        }
      }
    } catch (err) {
      updateAssistant({ error: err.message || "Something went wrong.", streaming: false });
    } finally {
      setBusy(false);
    }
  }

  function startNewChat() {
    const session = {
      id: crypto.randomUUID(),
      title: "New chat",
      messages: [],
      updatedAt: Date.now(),
    };
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setInput("");
  }

    //Send message when enter key is pushed but without shift
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app-frame">
      <div className="chat-layout">
        <aside className="chat-sidebar" aria-label="Chat sessions">
          <div className="sidebar-heading">
            <span className="sidebar-brand">StudentAI</span>
            <button
              className="new-chat-button"
              type="button"
              onClick={startNewChat}
              disabled={busy}
              title="New chat"
              aria-label="Start a new chat"
            >
              <span aria-hidden="true">+</span>
              <span className="new-chat-label">New chat</span>
            </button>
          </div>

          <div className="session-list-label">Your chats</div>
          <nav className="session-list" aria-label="Your chats">
            {orderedSessions.map((session) => (
              <button
                className={`session-button${session.id === activeSession.id ? " active" : ""}`}
                key={session.id}
                type="button"
                onClick={() => setActiveSessionId(session.id)}
                disabled={busy}
                title={session.title}
                aria-current={session.id === activeSession.id ? "page" : undefined}
              >
                <span className="session-icon" aria-hidden="true">✦</span>
                <span className="session-title">{session.title}</span>
              </button>
            ))}
          </nav>
        </aside>

        <main className="board">
          <div className="board-header">
            <h1>StudentAI</h1>
            <span className="session-tag">session {activeSession.id.slice(0, 8)}</span>
          </div>

          <div className="board-body" ref={bodyRef}>
            {messages.length === 0 && (
              <div className="empty-state">
                <span className="mark">Chalk one up</span>
                <p>Ask a question about your course material and I'll dig through your notes, and the web if I need to.</p>
              </div>
            )}

            {messages.map((message, index) => (
              <ChatMessage key={message.id || index} message={message} />// must include a key to identify each child
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
        </main>
      </div>
    </div>
  );
}
