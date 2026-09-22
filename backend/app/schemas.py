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


class ChatRequest(BaseModel):
    message: str
    provider: str = Field(default="anthropic", pattern="^(anthropic|openai)$")
    api_key: str | None = None
    model: str | None = None
    top_k: int = 5
    category: str | None = None


class Citation(BaseModel):
    document_id: str
    title: str
    page_number: int | None = None
    score: float
    snippet: str
