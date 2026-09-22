from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
QDRANT_PATH = DATA_DIR / "qdrant_storage"
SQLITE_PATH = DATA_DIR / "app.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # --- LLM providers (đều có thể để trống và nhập API key trực tiếp trên UI) ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    xai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None

    default_provider: str = "anthropic"
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"
    xai_model: str = "grok-4"
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Embedding (chạy local, miễn phí, đa ngôn ngữ - hỗ trợ tốt tiếng Việt) ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # --- Retrieval / Chunking ---
    default_top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Vector store ---
    qdrant_collection: str = "rag_chunks"

    # --- CORS ---
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return settings
