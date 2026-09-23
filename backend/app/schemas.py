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
    top_k: int = 5
    category: str | None = None
    history: list[ChatMessage] = []
    api_key: str | None = None  # chỉ dùng khi gọi thẳng API (curl/test) - UI không còn gửi field này


class Citation(BaseModel):
    document_id: str
    title: str
    page_number: int | None = None
    score: float
    snippet: str


class ProviderInfo(BaseModel):
    id: str
    label: str
    models: list[str]
    default_model: str | None = None


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
