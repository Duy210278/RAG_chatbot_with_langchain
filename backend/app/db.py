import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import SQLITE_PATH

engine = create_engine(f"sqlite:///{SQLITE_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401  (đảm bảo model được đăng ký trước khi create_all)

    Base.metadata.create_all(bind=engine)
    _migrate_add_content_hash()
    _migrate_add_agent_columns()
    _init_lexical_index()


def _migrate_add_agent_columns() -> None:
    """Cột của chế độ Agent trong query_logs - như _migrate_add_content_hash, create_all không tự thêm
    cột vào bảng đã có. Các lượt cũ đều chạy chế độ thường nên mặc định 0 / NULL là đúng."""
    with engine.begin() as conn:
        columns = {c["name"] for c in inspect(conn).get_columns("query_logs")}
        if "agent_mode" not in columns:
            conn.execute(text("ALTER TABLE query_logs ADD COLUMN agent_mode BOOLEAN NOT NULL DEFAULT 0"))
        if "agent_steps_json" not in columns:
            conn.execute(text("ALTER TABLE query_logs ADD COLUMN agent_steps_json TEXT"))


def _init_lexical_index() -> None:
    """Dựng chỉ mục từ khoá (FTS5) cho tìm kiếm lai. Nội dung chunk đã nằm sẵn trong SQLite
    nên tài liệu nạp trước khi có tính năng này được đánh chỉ mục lại ngay, không cần nạp lại file."""
    from . import lexical_search

    if lexical_search.ensure_schema():
        lexical_search.backfill_if_empty()


def _migrate_add_content_hash() -> None:
    """create_all không thêm cột vào bảng đã tồn tại, nên DB tạo trước khi có tính năng chống
    nạp trùng cần bổ sung cột content_hash thủ công + tính hash cho tài liệu cũ còn file gốc."""
    with engine.begin() as conn:
        columns = {c["name"] for c in inspect(conn).get_columns("documents")}
        if "content_hash" in columns:
            return
        conn.execute(text("ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64)"))
        conn.execute(text("CREATE INDEX ix_documents_content_hash ON documents (content_hash)"))
        for doc_id, file_path in conn.execute(text("SELECT id, file_path FROM documents")).all():
            path = Path(file_path)
            if path.exists():
                conn.execute(
                    text("UPDATE documents SET content_hash = :hash WHERE id = :id"),
                    {"hash": hashlib.sha256(path.read_bytes()).hexdigest(), "id": doc_id},
                )
