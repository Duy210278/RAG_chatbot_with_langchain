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
    "openrouter": "OpenRouter (1 key, nhiều hãng)",
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
    openrouter_api_key: str | None = None

    # Danh sách model cho từng provider, phân cách bằng dấu phẩy - model đầu tiên là mặc định.
    # UI hiển thị đúng danh sách này dưới dạng dropdown (không cho gõ tay để tránh sai định dạng model ID).
    anthropic_models: str = "claude-sonnet-5,claude-opus-5,claude-haiku-4-5-20251001,claude-fable-5-1"
    openai_models: str = "gpt-4o-mini,gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-4.1-nano,o3-mini,o1"
    xai_models: str = "grok-4,grok-4-fast"
    gemini_models: str = "gemini-3.6-flash,gemini-2.5-flash,gemini-2.5-pro,gemini-2.5-flash-lite,gemini-2.0-flash"
    groq_models: str = "llama-3.3-70b-versatile,llama-3.1-8b-instant"
    # OpenRouter là cổng trung gian: một key dùng được model của nhiều hãng, model ID luôn dạng "hãng/model".
    # Danh sách dưới đây xếp theo giá tăng dần (giá tại thời điểm thêm, USD/1 triệu token vào-ra):
    # gpt-6-luna 0.10/0.50 · llama-4-maverick 0.19/0.65 · deepseek-v4.1-flash 0.30/1.20
    # · gemini-3.8-flash 0.75/3.75 · claude-sonnet-5.5 2.00/10.00. Bản ":free" miễn phí nhưng giới hạn lượt gọi.
    openrouter_models: str = (
        "openai/gpt-6-luna,meta-llama/llama-4-maverick,deepseek/deepseek-v4.1-flash,"
        "google/gemini-3.8-flash,anthropic/claude-sonnet-5.5,nvidia/nemotron-3.5-lightning:free"
    )

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

    # --- Reranker (cross-encoder chấm lại các đoạn mà vector search tìm được) ---
    # Mặc định dùng model nhỏ đa ngôn ngữ (có tiếng Việt) vì chạy CPU: ~0.8s cho 20 đoạn.
    # BAAI/bge-reranker-v2-m3 chính xác hơn nhưng ~9s cho 20 đoạn trên CPU - chỉ nên dùng khi có GPU.
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidates: int = 30  # số đoạn lấy từ vector search để đưa vào chấm lại (đủ rộng cho quy trình nhiều bước)
    rerank_threshold: float = 0.3  # điểm liên quan tối thiểu (0-1) để 1 đoạn được gửi cho LLM

    # --- Truy hồi lai: vector dense + từ khoá BM25 (SQLite FTS5), trộn bằng RRF ---
    # Bù đúng điểm yếu của dense: truy vấn theo mã/số hiệu văn bản ("mẫu 01-HĐLĐ", "Điều 8").
    use_hybrid_search: bool = True
    lexical_candidates: int = 30  # số chunk lấy từ nhánh BM25 trước khi trộn
    rrf_k: int = 60  # hằng số RRF (giá trị chuẩn trong tài liệu gốc; càng lớn càng san bằng thứ hạng)

    # --- Gửi kèm đoạn liền kề (chunk_index ± N, cùng tài liệu) cho các đoạn đứng đầu ---
    # Bù slide bị vụn và điều khoản bị cắt đôi ở ranh giới trang, không cần nạp lại tài liệu.
    neighbor_window: int = 1  # số đoạn lấy thêm MỖI phía; 0 = tắt
    neighbor_top_n: int = 3  # chỉ mở rộng N đoạn đứng đầu, để prompt không phình theo top_k

    # --- Viết lại câu hỏi nối tiếp thành câu hỏi độc lập trước khi truy hồi ---
    # Chỉ chạy khi có lịch sử hội thoại, nên không ảnh hưởng câu hỏi đầu tiên của mỗi phiên.
    query_rewrite_enabled: bool = True
    query_rewrite_timeout: float = 8.0  # giây - quá hạn thì dùng câu hỏi gốc, không chặn lượt hỏi

    # --- Khi LLM hết quota: trả thẳng N đoạn tài liệu gốc đã tìm được thay vì chỉ báo lỗi ---
    quota_fallback_results: int = 3

    # --- Câu hỏi gợi ý (cột bên phải khung chat): FAQ soạn tay + "Hay được hỏi" từ query_logs ---
    # faq.json: danh sách {"question", "group"?, "category"?}. Đọc lại mỗi lần gọi API nên sửa file
    # là có hiệu lực ngay, không cần restart. Nằm ngoài data/ để được commit cùng mã nguồn.
    faq_path: str = str(BASE_DIR / "backend" / "faq.json")
    suggestion_limit: int = 5  # số câu hiển thị cho MỖI nhóm (FAQ / Hay được hỏi)
    popular_days: int = 30  # chỉ xét lượt hỏi gần đây để danh sách bám theo tài liệu hiện có
    # Phải được hỏi ở ít nhất N cuộc trò chuyện khác nhau: lọc kiểu hỏi đi hỏi lại trong một cuộc, và
    # tránh đưa câu hỏi mang tính cá nhân của MỘT người ra cho mọi người cùng thấy.
    popular_min_sessions: int = 2

    # --- Quan sát / vận hành ---
    log_level: str = "INFO"
    log_format: str = "pretty"  # pretty = dễ đọc trên terminal khi dev; json = cho Loki/ELK khi chạy thật
    log_quiet_access: bool = True  # ẩn access log của GET thành công (UI gọi lại liên tục mỗi lần tải lại)
    # Gửi kèm prompt đầy đủ (system + lịch sử + ngữ cảnh) về UI và lưu cùng lượt hội thoại - để kiểm tra
    # LLM thực sự được đọc gì. Tắt khi không muốn người dùng cuối thấy system prompt.
    expose_prompt: bool = True

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
            "openrouter": self.openrouter_api_key,
        }.get(provider)

    def models_for(self, provider: str) -> list[str]:
        raw = {
            "anthropic": self.anthropic_models,
            "openai": self.openai_models,
            "xai": self.xai_models,
            "gemini": self.gemini_models,
            "groq": self.groq_models,
            "openrouter": self.openrouter_models,
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
