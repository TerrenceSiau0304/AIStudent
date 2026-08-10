# backend/core/config.py

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM / AI providers
    google_api_key: str
    cohere_api_key: str
    tavily_api_key: str

    langsmith_api_key: str | None = None
    langchain_tracing: bool = False
    langchain_project: str = "AIStudent"

    huggingfacehub_api_token: str | None = None

    chroma_persist_dir: str = "./data/chroma_persist"
    docstore_dir: str = "./data/docstore"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()