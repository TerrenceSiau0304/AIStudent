# StudentAI — Frontend

A React + Vite chat UI for your StudentAI FastAPI/LangGraph backend. Themed like a
chalkboard: your questions land like red-pen margin notes, and the AI's status
updates ("Retrieving…", "Grading…") get wiped away as the real answer gets
chalked in — mirroring the `status` / `answer` / `[DONE]` events your `/chat`
endpoint streams.

## Setup

```bash
cd frontend
npm install
npm run dev
```

This starts the Vite dev server at `http://localhost:5173` — the exact origin
your backend's `CORSMiddleware` already allows, so no config changes needed on
either side.

**Make sure your backend is running first**, in a separate terminal, from
`backend/`:

```bash
uvicorn api.main:app --reload
```

Then open `http://localhost:5173` in your browser.

## How it works

- `src/App.jsx` — owns chat state, sends `POST /chat`, and parses the
  `text/event-stream` response manually via `fetch` + `ReadableStream`
  (note: the browser's built-in `EventSource` API only supports `GET`
  requests, so a POST-based SSE stream like yours has to be read this way).
- `src/components/ChatMessage.jsx` — renders a single message: either a
  user bubble, or an assistant message that can carry a live `status`, final
  `content`, or an `error`.
- A `session_id` (UUID) is generated once per browser and persisted in
  `localStorage`, matching the `thread_id` your backend uses for LangGraph's
  checkpointer — so conversation memory survives a page refresh.

## Config

If your backend runs on a different host/port, update `API_URL` at the top
of `src/App.jsx`:

```js
const API_URL = "http://127.0.0.1:8000/chat";
```

## Building for production

```bash
npm run build
```

Outputs static files to `dist/`, which you can serve with any static host —
just remember to update `allow_origins` in your backend's `CORSMiddleware` to
match wherever you deploy this.
