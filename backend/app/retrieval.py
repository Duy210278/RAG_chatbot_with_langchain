"""Truy hồi lai (hybrid): vector dense + từ khoá BM25, hợp nhất bằng RRF.

Trước đây chỉ có một nhánh vector cosine. Vấn đề: câu hỏi tra theo mã/số hiệu văn bản
("mẫu 01-HĐLĐ", "Quyết định 07/2026") bị trượt vì token hiếm không sống sót qua embedding.
Nay chạy song song hai nhánh rồi trộn thứ hạng.

Vì sao trộn bằng RRF (Reciprocal Rank Fusion) chứ không cộng điểm: điểm cosine (0-1) và điểm
BM25 (số âm, không giới hạn) không cùng thang, chuẩn hoá lại luôn phải hiệu chỉnh tay mỗi khi
đổi model. RRF chỉ dùng THỨ HẠNG nên miễn nhiễm với chuyện đó.
"""

import asyncio

from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import lexical_search, models
from .config import get_settings
from .embeddings import get_embedder
from .vector_store import get_vector_store


class Hit(BaseModel):
    """Kiểu chung cho kết quả của cả hai nhánh, để reranker và rag.py không phải biết
    kết quả đến từ Qdrant hay từ SQLite FTS5."""

    id: str
    score: float
    payload: dict
    sources: list[str] = []


def _from_qdrant(points) -> list[Hit]:
    return [Hit(id=str(p.id), score=float(p.score), payload=dict(p.payload), sources=["dense"]) for p in points]


def _from_lexical(db: Session, ranked: list[tuple[str, float]]) -> list[Hit]:
    """Dựng Hit từ chunk_id mà FTS5 trả về - payload lấy trong SQLite để khớp đúng
    cấu trúc payload của nhánh Qdrant (content/title/page_number/document_id/category)."""
    if not ranked:
        return []
    chunk_ids = [chunk_id for chunk_id, _ in ranked]
    chunks = {
        c.id: c
        for c in db.query(models.DocumentChunk).filter(models.DocumentChunk.id.in_(chunk_ids)).all()
    }
    doc_ids = {c.document_id for c in chunks.values()}
    docs = {d.id: d for d in db.query(models.Document).filter(models.Document.id.in_(doc_ids)).all()}

    hits: list[Hit] = []
    for chunk_id, score in ranked:
        chunk = chunks.get(chunk_id)
        if chunk is None:  # chunk đã bị xoá nhưng chỉ mục chưa kịp dọn
            continue
        doc = docs.get(chunk.document_id)
        hits.append(
            Hit(
                id=chunk_id,
                score=score,
                payload={
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "category": doc.category if doc else "GENERAL",
                    "title": doc.title if doc else "Không rõ",
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "is_public": doc.is_public if doc else True,
                },
                sources=["lexical"],
            )
        )
    return hits


def rrf_fuse(dense: list[Hit], lexical: list[Hit], k: int, limit: int) -> list[Hit]:
    """score = Σ 1/(k + thứ_hạng) trên từng nhánh. Chunk xuất hiện ở CẢ HAI nhánh được cộng dồn
    nên tự động nổi lên đầu - đó chính là tín hiệu "vừa khớp ngữ nghĩa, vừa khớp từ khoá"."""
    fused: dict[str, Hit] = {}
    rrf: dict[str, float] = {}

    for ranked in (dense, lexical):
        for rank, hit in enumerate(ranked, start=1):
            rrf[hit.id] = rrf.get(hit.id, 0.0) + 1.0 / (k + rank)
            if hit.id in fused:
                fused[hit.id].sources = sorted(set(fused[hit.id].sources) | set(hit.sources))
            else:
                fused[hit.id] = hit.model_copy(deep=True)

    ordered = sorted(fused.values(), key=lambda h: rrf[h.id], reverse=True)
    for hit in ordered:
        hit.score = rrf[hit.id]

    # Bỏ trùng nội dung (cùng một đoạn nằm ở 2 file khác nhau) để không chiếm chỗ trong top-k.
    unique, seen = [], set()
    for hit in ordered:
        key = " ".join(hit.payload.get("content", "").split())
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique[:limit]


async def retrieve(
    db: Session,
    query_text: str,
    limit: int,
    category: str | None,
    use_hybrid: bool = True,
) -> tuple[list[Hit], bool]:
    """Trả về (danh sách Hit, có_dùng_hybrid_không).

    Embed chạy trong thread riêng vì model CPU-bound sẽ chặn event loop của FastAPI
    (cùng lý do với reranker) - một câu hỏi đang embed không được làm nghẽn các request khác."""
    settings = get_settings()
    embedder = get_embedder()
    store = get_vector_store()

    query_vector = await asyncio.to_thread(embedder.embed_query, query_text)
    dense = _from_qdrant(store.search(query_vector, top_k=limit, category=category))

    hybrid_used = use_hybrid and settings.use_hybrid_search and lexical_search.is_ready()
    if not hybrid_used:
        return dense[:limit], False

    lexical = _from_lexical(db, lexical_search.search(db, query_text, settings.lexical_candidates, category))
    if not lexical:
        # Không có kết quả từ khoá nào (vd câu hỏi toàn từ chức năng) - dùng nguyên nhánh vector,
        # giữ nguyên điểm cosine thay vì thay bằng điểm RRF vô nghĩa khi chỉ có một nhánh.
        return dense[:limit], False

    return rrf_fuse(dense, lexical, settings.rrf_k, limit), True
