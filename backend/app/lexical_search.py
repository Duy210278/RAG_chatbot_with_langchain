"""Tìm kiếm từ khoá (BM25) bằng SQLite FTS5 - nhánh bổ sung cho tìm kiếm vector.

Vì sao cần: embedding dense hay trượt đúng loại truy vấn mà người dùng nội bộ hỏi nhiều nhất -
tra theo MÃ/SỐ HIỆU chính xác ("mẫu 01-HĐLĐ", "Quyết định 07/2026", "Điều 8"). Những token hiếm
đó bị nén mất trong vector 384 chiều, còn BM25 thì xử lý đúng. Hai nhánh bù nhau nên hợp nhất
lại (RRF, xem retrieval.py) cho recall tốt hơn hẳn dùng riêng từng nhánh.

Vì sao chọn SQLite FTS5 thay vì sparse vector của Qdrant: nội dung chunk vốn đã nằm sẵn trong
bảng document_chunks, FTS5 có sẵn trong Python (không thêm dependency), và quan trọng nhất là
KHÔNG phải đổi schema collection Qdrant hay embed lại toàn bộ tài liệu đã nạp trước đó.
"""

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from .db import engine

TABLE = "chunks_fts"

# remove_diacritics 2: bỏ dấu khi đánh chỉ mục VÀ khi tra. Đánh đổi có chủ ý:
# "nghỉ"/"nghĩ"/"nghi" gộp làm một (nhiễu hơn), nhưng bù lại tra được khi người dùng gõ không
# dấu ("nghi phep") - rất phổ biến trong nội bộ - và cứu được text OCR bị mất/sai dấu.
# Nhiễu thêm chấp nhận được vì BM25 ở đây chỉ đóng vai trò tăng recall, cross-encoder chấm lại sau.
_CREATE_VARIANTS = [
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5("
    "chunk_id UNINDEXED, document_id UNINDEXED, category UNINDEXED, content, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5("
    "chunk_id UNINDEXED, document_id UNINDEXED, category UNINDEXED, content, "
    "tokenize=\"unicode61 remove_diacritics 1\")",
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5("
    "chunk_id UNINDEXED, document_id UNINDEXED, category UNINDEXED, content)",
]

# Từ chức năng tiếng Việt: xuất hiện ở gần như mọi chunk nên không phân biệt được tài liệu nào,
# chỉ làm loãng điểm BM25. Chỉ lọc khi câu hỏi còn đủ từ có nghĩa (xem _build_match_query).
_STOPWORDS = {
    "là", "của", "và", "cho", "các", "những", "được", "có", "không", "thì", "mà", "với",
    "trong", "khi", "này", "đó", "một", "để", "ở", "về", "như", "nào", "gì", "bao", "nhiêu",
    "thế", "sao", "hay", "hoặc", "tôi", "bạn", "em", "anh", "chị", "ạ", "vậy", "còn", "nữa",
}

_ready: bool | None = None


def ensure_schema() -> bool:
    """Tạo bảng FTS5 nếu chưa có. Trả về False khi bản SQLite không biên dịch kèm FTS5 -
    lúc đó hệ thống tự chạy tiếp ở chế độ chỉ-vector thay vì hỏng cả luồng hỏi đáp."""
    global _ready
    if _ready is not None:
        return _ready
    for sql in _CREATE_VARIANTS:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
            _ready = True
            return True
        except Exception:  # noqa: BLE001 - thử lần lượt các biến thể tokenizer theo phiên bản SQLite
            continue
    _ready = False
    return False


def is_ready() -> bool:
    return bool(_ready)


def backfill_if_empty() -> int:
    """Nạp chỉ mục từ khoá cho các chunk đã có sẵn trong DB (tài liệu nạp trước khi có tính năng
    này). Nhờ nội dung chunk đã nằm trong SQLite nên không phải đọc lại file gốc hay embed lại."""
    if not is_ready():
        return 0
    with engine.begin() as conn:
        if conn.execute(text(f"SELECT count(*) FROM {TABLE}")).scalar_one():
            return 0
        total = conn.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()
        if not total:
            return 0
        conn.execute(
            text(
                f"INSERT INTO {TABLE} (chunk_id, document_id, category, content) "
                "SELECT c.id, c.document_id, d.category, c.content "
                "FROM document_chunks c JOIN documents d ON d.id = c.document_id"
            )
        )
        return int(total)


def index_chunks(db: Session, rows: list[dict]) -> None:
    """rows: [{chunk_id, document_id, category, content}] - gọi ngay sau khi ghi chunk vào SQLite."""
    if not is_ready() or not rows:
        return
    db.execute(
        text(
            f"INSERT INTO {TABLE} (chunk_id, document_id, category, content) "
            "VALUES (:chunk_id, :document_id, :category, :content)"
        ),
        rows,
    )
    db.commit()


def remove_document(db: Session, document_id: str) -> None:
    if not is_ready():
        return
    db.execute(text(f"DELETE FROM {TABLE} WHERE document_id = :doc_id"), {"doc_id": document_id})
    db.commit()


def _build_match_query(question: str) -> str:
    """Chuyển câu hỏi tự do thành cú pháp MATCH hợp lệ của FTS5.

    Bắt buộc phải làm sạch: dấu ?, ngoặc, dấu ngang... đều là toán tử trong cú pháp FTS5 nên
    truyền thẳng câu hỏi người dùng vào sẽ ném OperationalError. Nối bằng OR (không phải AND)
    vì đây là nhánh tăng recall - tài liệu chứa nhiều từ hiếm hơn sẽ tự động được BM25 xếp trên.
    """
    tokens = [t for t in re.findall(r"\w+", question.lower(), flags=re.UNICODE) if len(t) > 1]
    meaningful = [t for t in tokens if t not in _STOPWORDS]
    # Câu quá ngắn (vd "nghỉ lễ?") có thể bị lọc sạch - khi đó giữ lại nguyên bản còn hơn không tra gì.
    chosen = meaningful if len(meaningful) >= 2 else tokens
    if not chosen:
        return ""
    return " OR ".join(f'"{t}"' for t in chosen[:24])


def search(db: Session, question: str, limit: int, category: str | None = None) -> list[tuple[str, float]]:
    """Trả về [(chunk_id, điểm bm25)] đã xếp hạng sẵn, tốt nhất đứng đầu.

    Lưu ý: bm25() của SQLite trả về số ÂM, càng nhỏ càng khớp - nên ORDER BY tăng dần là đúng.
    Chỉ dùng thứ hạng để hợp nhất RRF nên không cần chuẩn hoá thang điểm này."""
    if not is_ready():
        return []
    match_query = _build_match_query(question)
    if not match_query:
        return []

    sql = f"SELECT chunk_id, bm25({TABLE}) AS score FROM {TABLE} WHERE {TABLE} MATCH :q"
    params: dict = {"q": match_query, "limit": limit}
    if category and category != "ALL":
        sql += " AND category = :cat"
        params["cat"] = category
    sql += " ORDER BY score LIMIT :limit"

    try:
        rows = db.execute(text(sql), params).all()
    except Exception:  # noqa: BLE001 - truy vấn lạ không được phép làm hỏng cả lượt hỏi đáp
        return []
    return [(row[0], float(row[1])) for row in rows]
