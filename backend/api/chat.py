import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    """
    Request body received by the chat endpoint.

    Attributes:
        message (str):
            The user's question or message that will be
            processed by the StudentAI system.

        session_id (str):
            Unique identifier for the user's conversation session.
            The frontend generates and persists this ID so that
            multiple requests from the same browser session can
            share the same LangGraph conversation state.
    """
    message: str
    session_id: str   # frontend generates/persists this per browser session

@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    """
    Process a user's chat message and stream the AI response
    back to the frontend using Server-Sent Events (SSE).

    The endpoint retrieves the pre-built LangGraph from the
    FastAPI application state and executes the graph using
    the user's message.

    LangGraph events are streamed to the frontend as they occur.
    This allows the frontend to display the current processing
    status, such as document retrieval, web searching, grading,
    and answer generation.

    Args:
        request (Request):
            FastAPI request object used to access shared
            application resources stored in `app.state`.

        body (ChatRequest):
            Request body containing the user's message and
            conversation session ID.

    Returns:
        StreamingResponse:
            An SSE response that streams status updates and
            the generated answer to the frontend.
    """
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": body.session_id}}

    async def event_stream():
        """
        Execute the LangGraph and stream its events to the client.

        Each event is converted into Server-Sent Event (SSE)
        format before being sent to the frontend.

        Events are divided into two main types:

        1. Status events:
           Inform the frontend which LangGraph node is currently
           being executed.

        2. Answer events:
           Send the final generated response to the frontend
           when the `generate` or `llm_fallback` node finishes.
        """
        async for event in graph.astream_events(
            {"question": body.message},
            config=config,
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chain_start" and event["name"] in (
                "retrieve", "web_search", "grade_document", "generate", "llm_fallback"
            ):
                yield f"data: {json.dumps({'type': 'status', 'node': event['name']})}\n\n"

            if kind == "on_chain_end" and (event["name"] == "generate" or event["name"] == "llm_fallback"):
                output = event["data"]["output"]
                yield f"data: {json.dumps({'type': 'answer', 'content': output.get('generation', '')})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")