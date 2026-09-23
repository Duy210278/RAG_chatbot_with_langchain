from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
QDRANT_PATH = DATA_DIR / "qdrant_storage"
SQLITE_PATH = DATA_DIR / "app.db"

PROVIDER_LABELS = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "xai": "xAI Grok",
    "gemini": "Google Gemini (free tier)",
    "groq": "Groq (free tier)",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # --- LLM providers ---
    # API key CHỈ khai báo ở đây (.env) - không nhập trên UI. Provider nào không có
    # key sẽ tự động bị ẩn khỏi dropdown "Provider" trên giao diện (xem GET /api/v1/config/providers).
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    xai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None

    # Danh sách model cho từng provider, phân cách bằng dấu phẩy - model đầu tiên là mặc định.
    # UI hiển thị đúng danh sách này dưới dạng dropdown (không cho gõ tay để tránh sai định dạng model ID).
    anthropic_models: str = "claude-sonnet-5,claude-opus-5,claude-haiku-4-5-20251001,claude-fable-5-1"
    openai_models: str = "gpt-4o-mini,gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-4.1-nano,o3-mini,o1"
    xai_models: str = "grok-4,grok-4-fast"
    gemini_models: str = "gemini-2.5-flash,gemini-2.5-pro,gemini-2.5-flash-lite,gemini-2.0-flash"
    groq_models: str = "llama-3.3-70b-versatile,llama-3.1-8b-instant"

    # --- Lịch sử hội thoại làm ngữ cảnh multi-turn ---
    # Giới hạn số tin nhắn lịch sử gửi kèm cho LLM (tính cả user+assistant) để tránh
    # prompt phình quá to / tốn phí - luôn được backend cắt bớt bất kể client gửi bao nhiêu.
    max_history_messages: int = 20

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

    def api_key_for(self, provider: str) -> str | None:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "xai": self.xai_api_key,
            "gemini": self.google_api_key,
            "groq": self.groq_api_key,
        }.get(provider)

    def models_for(self, provider: str) -> list[str]:
        raw = {
            "anthropic": self.anthropic_models,
            "openai": self.openai_models,
            "xai": self.xai_models,
            "gemini": self.gemini_models,
            "groq": self.groq_models,
        }.get(provider, "")
        return [m.strip() for m in raw.split(",") if m.strip()]

    def provider_catalog(self) -> list[dict]:
        """Danh sách provider ĐÃ có API key trong .env, kèm model khả dụng -
        dùng để dựng dropdown Provider/Model trên UI (GET /api/v1/config/providers)."""
        catalog = []
        for provider_id, label in PROVIDER_LABELS.items():
            if not self.api_key_for(provider_id):
                continue
            models = self.models_for(provider_id)
            catalog.append(
                {
                    "id": provider_id,
                    "label": label,
                    "models": models,
                    "default_model": models[0] if models else None,
                }
            )
        return catalog


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return settings
