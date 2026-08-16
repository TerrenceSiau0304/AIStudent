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
    """
    Manage resources that need to be initialized and cleaned up
    during the FastAPI application's lifecycle.

    The lifespan function is executed when the application starts
    and remains active until the application shuts down.

    During startup:
        1. Build the vector store and retrieve its retriever.
        2. Establish a connection to the PostgreSQL database.
        3. Initialize the PostgreSQL checkpointer.
        4. Build the LangGraph application graph.
        5. Store the graph in `app.state` so it can be accessed
           by API routes during requests.

    Args:
        app (FastAPI):
            The FastAPI application instance.

    Yields:
        None:
            Control is returned to FastAPI after all required
            resources have been initialized.
    """
    # retriever = get_retriever()
    vs = build_vectorstore()
    async with AsyncPostgresSaver.from_conn_string(str(get_settings().database_url)) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(vs.retriever, checkpointer)
        yield

# Create the FastAPI application instance.
#
# The lifespan handler is registered here so that the
# vector store, database checkpointer, and LangGraph are
# initialized when the application starts.
app = FastAPI(title="StudentAI API", lifespan=lifespan)

# CORS allows the StudentAI frontend applications to send
# HTTP requests to this FastAPI backend even though the
# frontend and backend are hosted on different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://ai-student-frontend-theta.vercel.app",
        "https://ai-student-frontend-git-main-terrencesiau0304s-projects.vercel.app",], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes defined inside `chat_router` become part of the
# StudentAI API and can access shared application resources,
# such as `app.state.graph`.
app.include_router(chat_router)