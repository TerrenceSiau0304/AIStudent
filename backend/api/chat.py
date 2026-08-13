import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str   # frontend generates/persists this per browser session

@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": body.session_id}}

    async def event_stream():
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

            if kind == "on_chain_end" and event["name"] == "generate":
                output = event["data"]["output"]
                yield f"data: {json.dumps({'type': 'answer', 'content': output.get('generation', '')})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")