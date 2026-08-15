# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# from rag.retrieve import get_retriever
from rag.graph import build_graph
from ingestion.build_vectorstore import build_vectorstore
from api.chat import router as chat_router
# from core.config import DATA_DIR
from core.config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # retriever = get_retriever()
    vs = build_vectorstore()
    async with AsyncPostgresSaver.from_conn_string(str(get_settings().database_url)) as checkpointer:
        app.state.graph = build_graph(vs.retriever, checkpointer)
        yield

app = FastAPI(title="StudentAI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://ai-student-frontend.vercel.app"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)