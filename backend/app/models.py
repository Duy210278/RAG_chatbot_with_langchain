"""
Bản rút gọn của lược đồ PostgreSQL đầy đủ trong tài liệu thiết kế
(mục 4.3 - Quản lý Tài liệu & Chunks), chạy trên SQLite cho MVP.
Khi lên production, migrate sang PostgreSQL dùng đúng DDL trong tài liệu thiết kế
(bổ sung RBAC, audit logs, chat history...).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str] = mapped_column(String(1000))
    file_extension: Mapped[str] = mapped_column(String(20))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)
    # SHA-256 nội dung file - chặn nạp trùng cùng một file (kể cả khi đổi tên)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    category: Mapped[str] = mapped_column(String(50), default="GENERAL")
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(50), default="PENDING")  # PENDING/PROCESSING/COMPLETED/FAILED
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # trùng point id trên Qdrant
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class ChatSession(Base):
    """Hội thoại lưu bền - trước đây lịch sử chỉ nằm trong st.session_state của Streamlit
    nên mất sạch khi tải lại trang."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(255), default="Cuộc trò chuyện mới")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # Lưu JSON thay vì bảng riêng: trích dẫn/chất lượng chỉ để hiển thị lại, không cần truy vấn theo trường.
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Trỏ sang query_logs để nút 👍/👎 vẫn dùng được khi mở lại hội thoại cũ.
    query_log_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class QueryLog(Base):
    """Một dòng cho mỗi lượt hỏi - nguồn dữ liệu để biết hệ thống đang hỏng ở đâu.

    Tách khỏi chat_messages vì hai thứ này có vòng đời khác nhau: người dùng xoá hội thoại
    (quyền riêng tư của họ) nhưng số liệu vận hành vẫn phải còn để cải thiện hệ thống.
    """

    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    top_k: Mapped[int] = mapped_column(Integer, default=0)
    use_rerank: Mapped[bool] = mapped_column(Boolean, default=True)
    hybrid: Mapped[bool] = mapped_column(Boolean, default=False)

    n_hits: Mapped[int] = mapped_column(Integer, default=0)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    invalid_citations: Mapped[int] = mapped_column(Integer, default=0)
    # "Bó tay" = không truy hồi được gì, hoặc có lỗi provider. Đánh index vì đây là truy vấn hay dùng nhất.
    no_answer: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    latency_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # thời gian từng khâu (ms)
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    first_token_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # True = ước lượng bằng tiktoken vì provider không trả số liệu, KHÔNG phải số thật để đối soát hoá đơn.
    tokens_estimated: Mapped[bool] = mapped_column(Boolean, default=False)

    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)  # 1 = hữu ích, -1 = không
    feedback_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
