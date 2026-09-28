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
from sqlalchemy import and_, or_
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


_MIN_OVERLAP_CHARS = 10


def _join_chunks(a: str, b: str, same_page: bool) -> str:
    """Nối hai đoạn liền nhau, bỏ phần chồng lấn (chunk_overlap) nếu có.

    Chỉ đoạn CÙNG TRANG mới có chồng lấn (mỗi trang được cắt riêng), nên chỉ tìm chồng lấn khi cùng
    trang - khác trang mà tình cờ trùng chữ (vd tiêu đề slide lặp lại) thì vẫn giữ nguyên cả hai.
    Tìm phần cuối DÀI NHẤT của a trùng phần đầu của b: duyệt vị trí xuất hiện của vài ký tự đầu b
    từ trái sang, vị trí đầu tiên khớp trọn chính là phần chồng lấn dài nhất."""
    probe = b[:_MIN_OVERLAP_CHARS]
    start = a.find(probe) if same_page and len(probe) == _MIN_OVERLAP_CHARS else -1
    while start != -1:
        if b.startswith(a[start:]):
            return a + b[len(a) - start :]
        start = a.find(probe, start + 1)
    return f"{a}\n\n{b}"


def expand_with_neighbors(db: Session, hits: list[Hit], top_n: int, window: int) -> list[Hit]:
    """Gửi kèm các đoạn liền trước/liền sau (chunk_index ± window, cùng tài liệu) cho top_n đoạn đứng đầu.

    Bù hai điểm yếu của cách cắt hiện tại mà không phải nạp lại tài liệu: slide bị vụn (một ý trải
    trên vài slide liên tiếp) và điều khoản bị cắt đôi ở ranh giới trang (chunk không vượt qua trang).

    Phần ghép nằm ở payload["context"], CHỈ dùng làm ngữ cảnh cho LLM; payload["content"] vẫn là đúng
    đoạn đã khớp nên trích dẫn/snippet không lệch. Rerank đã chấm trên đoạn gốc từ trước - điểm và thứ
    hạng giữ nguyên. Đoạn liền kề đã là một nguồn riêng, hoặc đã được ghép cho đoạn xếp trên, thì bỏ qua
    để cùng một nội dung không bị gửi hai lần.
    """
    if top_n <= 0 or window <= 0:
        return hits

    def key(hit: Hit) -> tuple:
        return hit.payload.get("document_id"), hit.payload.get("chunk_index")

    wanted: dict[str, set[int]] = {}
    for hit in hits[:top_n]:
        doc_id, idx = key(hit)
        if doc_id is not None and idx is not None:
            wanted.setdefault(doc_id, set()).update(idx + o for o in range(-window, window + 1) if o)
    if not wanted:
        return hits

    chunk = models.DocumentChunk
    rows = (
        db.query(chunk.document_id, chunk.chunk_index, chunk.content, chunk.page_number)
        .filter(or_(*(and_(chunk.document_id == d, chunk.chunk_index.in_(idx)) for d, idx in wanted.items())))
        .all()
    )
    by_key = {(r.document_id, r.chunk_index): r for r in rows}
    taken = {key(h) for h in hits}

    def grab(doc_id: str, idx: int, step: int) -> list:
        """Lấy liên tiếp theo một hướng, dừng ở đoạn đầu tiên không lấy được để phần ghép luôn liền mạch."""
        out = []
        for n in range(1, window + 1):
            k = (doc_id, idx + step * n)
            if k in taken or k not in by_key:
                break
            taken.add(k)
            out.append(by_key[k])
        return out

    expanded: list[Hit] = []
    for rank, hit in enumerate(hits):
        doc_id, idx = key(hit)
        if rank >= top_n or doc_id is None or idx is None:
            expanded.append(hit)
            continue
        before = grab(doc_id, idx, -1)[::-1]
        after = grab(doc_id, idx, 1)
        if not before and not after:
            expanded.append(hit)
            continue
        parts = (
            [(r.content, r.page_number) for r in before]
            + [(hit.payload["content"], hit.payload.get("page_number"))]
            + [(r.content, r.page_number) for r in after]
        )
        context = parts[0][0]
        for (_, prev_page), (text, page) in zip(parts, parts[1:]):
            context = _join_chunks(context, text, same_page=page == prev_page)
        payload = {**hit.payload, "context": context, "neighbor_pages": [r.page_number for r in before + after]}
        expanded.append(hit.model_copy(update={"payload": payload}))
    return expanded
