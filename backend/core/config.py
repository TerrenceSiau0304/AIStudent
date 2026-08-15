from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR/"data"

class Settings(BaseSettings):
    google_api_key: str
    cohere_api_key: str
    tavily_api_key: str

    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "AIStudent"
    langsmith_endpoint: str | None = None

    google_application_credentials: str | None = None

    chroma_persist_dir: str = "./data/chroma_persist"
    docstore_dir: str = "./data/docstore"

    huggingfacehub_api_token: str | None = None
    database_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()