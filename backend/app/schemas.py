from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    title: str
    category: str
    status: str
    chunk_count: int
    file_size_bytes: int
    created_at: str

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    document: DocumentOut


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    provider: str = Field(default="anthropic", pattern="^(anthropic|openai|xai|gemini|groq)$")
    model: str | None = None
    top_k: int = 12  # khi bật rerank: số đoạn TỐI ĐA gửi cho LLM (có thể ít hơn nếu ít đoạn đạt ngưỡng)
    use_rerank: bool = True
    category: str | None = None
    history: list[ChatMessage] = []
    api_key: str | None = None  # chỉ dùng khi gọi thẳng API (curl/test) - UI không còn gửi field này
    session_id: str | None = None  # có thì lượt hỏi được lưu vào hội thoại để mở lại sau


class FeedbackRequest(BaseModel):
    query_log_id: str
    rating: int = Field(ge=-1, le=1)  # 1 = hữu ích, -1 = không hữu ích
    note: str | None = None


class ChatSessionOut(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict] = []
    quality: dict | None = None
    query_log_id: str | None = None
    created_at: str


class QueryLogOut(BaseModel):
    id: str
    created_at: str
    question: str
    rewritten_query: str | None = None
    provider: str
    model: str | None = None
    n_hits: int
    top_score: float | None = None
    confidence: str | None = None
    invalid_citations: int
    no_answer: bool
    error: str | None = None
    total_ms: int
    first_token_ms: int | None = None
    latency_ms: dict = {}
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_estimated: bool
    feedback: int | None = None
    feedback_note: str | None = None


class UnansweredGroup(BaseModel):
    question: str
    count: int
    last_seen: str


class AdminStats(BaseModel):
    days: int
    total_queries: int
    no_answer_count: int
    error_count: int
    invalid_citation_count: int
    positive_feedback: int
    negative_feedback: int
    avg_total_ms: int
    p95_total_ms: int
    avg_first_token_ms: int
    stage_avg_ms: dict = {}
    by_confidence: dict = {}
    prompt_tokens: int
    completion_tokens: int
    estimated_token_share: float  # tỉ lệ lượt mà số token là ƯỚC LƯỢNG, không phải số thật từ provider


class Citation(BaseModel):
    index: int  # số thứ tự [n] mà LLM dùng để trích dẫn nguồn này
    document_id: str
    title: str
    page_number: int | None = None
    score: float
    score_type: str = "rerank"  # rerank (0-1) | hybrid_rrf | cosine - ba thang điểm KHÔNG so sánh được với nhau
    sources: list[str] = []  # nhánh tìm ra đoạn này: dense và/hoặc lexical
    used: bool = False  # câu trả lời có thực sự trích dẫn nguồn này không
    snippet: str


class ProviderInfo(BaseModel):
    id: str
    label: str
    models: list[str]
    default_model: str | None = None


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
