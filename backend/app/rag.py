import asyncio
import json
from collections.abc import AsyncIterator

from sqlalchemy.orm import Session

from . import answer_check
from .config import PROVIDER_LABELS, get_settings
from .llm import generate_answer_stream, is_quota_error
from .observability import StageTimer, persist_turn, record_query, resolve_usage
from .query_rewrite import rewrite_for_retrieval
from .reranker import rerank_hits
from .retrieval import retrieve

# Câu trả lời cố định khi không truy hồi được gì - dùng lại ở nhiều chỗ nên đặt thành hằng.
NO_CONTEXT_ANSWER = "Không tìm thấy tài liệu liên quan trong hệ thống để trả lời câu hỏi này."

# Đủ dài để đọc được ý chính của một đoạn, đủ ngắn để vài đoạn không chiếm hết màn hình.
_FALLBACK_EXCERPT_CHARS = 700


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _excerpt(text: str, limit: int = _FALLBACK_EXCERPT_CHARS) -> str:
    # Gộp xuống dòng của PDF/OCR thành một đoạn: vừa gọn, vừa tránh dòng bắt đầu bằng "#"/"-"
    # bị markdown hiểu thành tiêu đề/danh sách.
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def quota_fallback_answer(provider: str, model: str | None, hits: list, n: int) -> str:
    """Câu trả lời thay thế khi LLM hết quota: nói rõ lý do, rồi đưa nguyên văn n đoạn tìm được.
    Đánh số [n] khớp với danh sách nguồn để người dùng đối chiếu được với phần trích dẫn."""
    name = PROVIDER_LABELS.get(provider, provider) + (f" · {model}" if model else "")
    parts = [
        f"⚠️ **{name} đã hết quota** (hoặc vượt giới hạn số lượt gọi) nên chưa soạn được câu trả lời. "
        "Hãy thử lại sau ít phút, hoặc chọn provider/model khác ở thanh bên trái."
    ]
    if n:
        parts.append(
            f"Dưới đây là {n} đoạn tài liệu gốc liên quan nhất đã tìm được — **chưa qua LLM tổng hợp**, "
            "bạn cần tự đọc và đối chiếu:"
        )
    for i, hit in enumerate(hits[:n], start=1):
        page = hit.payload.get("page_number")
        where = f" — trang {page}" if page else ""
        parts.append(f"**[{i}] {hit.payload.get('title', 'Không rõ')}{where}**\n\n> {_excerpt(hit.payload['content'])}")
    return "\n\n".join(parts)


async def answer_question_stream(
    db: Session,
    question: str,
    provider: str,
    api_key: str,
    model: str | None,
    top_k: int,
    category: str | None,
    history: list[dict] | None = None,
    use_rerank: bool = True,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """Viết lại truy vấn -> Truy hồi lai -> Rerank -> Sinh câu trả lời -> Kiểm tra trích dẫn.

    Phát Server-Sent Events ở từng mốc để UI hiển thị tiến trình thay vì màn hình trắng.
    Câu hỏi đi tới bước TÌM KIẾM là bản đã viết lại (độc lập, đủ ngữ cảnh), còn câu hỏi đưa cho
    LLM SINH câu trả lời vẫn là câu gốc - vì LLM đã có sẵn lịch sử hội thoại và nên trả lời đúng
    câu người dùng thực sự hỏi.

    Mọi lượt hỏi đều được đo thời gian từng khâu và ghi vào query_logs, kể cả lượt hỏng.
    LLM hết quota thì phát sự kiện 'fallback' chứa các đoạn tài liệu gốc thay cho sự kiện 'error'.
    """
    settings = get_settings()
    history = history or []
    timer = StageTimer()

    base_log = {
        "session_id": session_id,
        "question": question,
        "provider": provider,
        "model": model,
        "category": category,
        "top_k": top_k,
        "use_rerank": use_rerank,
    }

    search_query, rewritten = question, False
    if history and settings.query_rewrite_enabled:
        yield _sse("status", {"stage": "rewriting", "message": "🧭 Đang đọc lại ngữ cảnh hội thoại..."})
        with timer.stage("rewrite"):
            search_query, rewritten = await rewrite_for_retrieval(provider, api_key, model, question, history)

    yield _sse("status", {"stage": "retrieving", "message": "🔍 Đang tìm kiếm tài liệu liên quan..."})

    # Bật rerank thì lấy rộng rồi để cross-encoder chọn lại; top_k lúc đó là số đoạn TỐI ĐA gửi cho LLM.
    limit = max(settings.rerank_candidates, top_k) if use_rerank else top_k
    with timer.stage("retrieve"):
        hits, hybrid_used = await retrieve(db, search_query, limit=limit, category=category)

    if use_rerank and hits:
        yield _sse("status", {"stage": "reranking", "message": "⚖️ Đang chấm lại mức độ liên quan..."})
        with timer.stage("rerank"):
            # chạy ở thread riêng để model (CPU nặng) không chặn event loop / các request khác
            hits = await asyncio.to_thread(rerank_hits, search_query, hits, top_k)
    else:
        hits = hits[:top_k]

    if not hits:
        quality = {
            "confidence": "thap",
            "reason": "Không truy hồi được đoạn tài liệu nào.",
            "invalid_citations": [],
            "rewritten_query": search_query if rewritten else None,
            "hybrid": hybrid_used,
            "latency_ms": timer.stages,
        }
        log_id = record_query(
            **base_log,
            rewritten_query=search_query if rewritten else None,
            hybrid=hybrid_used,
            n_hits=0,
            confidence="thap",
            no_answer=True,
            latency=timer.stages,
            total_ms=timer.total_ms,
        )
        persist_turn(session_id, question, NO_CONTEXT_ANSWER, [], quality, log_id)
        yield _sse("status", {"stage": "done", "message": "Không tìm thấy tài liệu liên quan"})
        yield _sse("token", {"text": NO_CONTEXT_ANSWER})
        yield _sse("citations", {"citations": []})
        yield _sse("quality", {**quality, "query_log_id": log_id})
        yield _sse("done", {"query_log_id": log_id})
        return

    # Thang điểm khác nhau tuỳ đường đi, nên gắn nhãn để UI không hiển thị hai loại điểm
    # không so sánh được dưới cùng một cái tên "score".
    score_type = "rerank" if use_rerank else ("hybrid_rrf" if hybrid_used else "cosine")
    top_score = max(h.score for h in hits)

    context_blocks = [hit.payload["content"] for hit in hits]
    citations = [
        {
            "index": i + 1,  # khớp với ký hiệu [n] mà LLM được yêu cầu trích dẫn
            "document_id": hit.payload["document_id"],
            "title": hit.payload.get("title", "Không rõ"),
            "page_number": hit.payload.get("page_number"),
            "score": round(hit.score, 4),
            "score_type": score_type,
            "sources": hit.sources,
            "used": False,
            "snippet": hit.payload["content"][:300],
        }
        for i, hit in enumerate(hits)
    ]

    yield _sse(
        "status",
        {
            "stage": "generating",
            "message": f"📄 Đã tìm thấy {len(hits)} đoạn liên quan - đang soạn câu trả lời...",
        },
    )

    answer_parts: list[str] = []
    usage_sink: dict = {}
    prompt_sink: list[str] = []
    first_token_ms: int | None = None
    try:
        with timer.stage("generate"):
            async for token in generate_answer_stream(
                provider,
                api_key,
                model,
                question,
                context_blocks,
                history=history,
                usage_sink=usage_sink,
                prompt_sink=prompt_sink,
            ):
                if first_token_ms is None:
                    first_token_ms = timer.elapsed_ms()
                answer_parts.append(token)
                yield _sse("token", {"text": token})
    except Exception as exc:  # noqa: BLE001 - lỗi gọi LLM (key sai, hết quota...) cần báo về UI thay vì crash SSE
        log_id = record_query(
            **base_log,
            rewritten_query=search_query if rewritten else None,
            hybrid=hybrid_used,
            n_hits=len(hits),
            top_score=top_score,
            no_answer=True,
            error=str(exc)[:2000],
            latency=timer.stages,
            total_ms=timer.total_ms,
        )
        if not is_quota_error(exc):
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {"query_log_id": log_id})
            return

        # Hết quota nhưng kết quả truy hồi vẫn còn nguyên trong tay - đưa ra cho người dùng tự đọc
        # còn hơn một dòng báo lỗi trống trơn.
        n_shown = min(settings.quota_fallback_results, len(hits))
        answer = quota_fallback_answer(provider, model, hits, n_shown)
        for citation in citations:
            citation["used"] = citation["index"] <= n_shown
        quality = {
            "confidence": "khong_do_duoc",
            "reason": "LLM hết quota — các đoạn trên là tài liệu gốc, chưa được tổng hợp thành câu trả lời.",
            "fallback": "quota",  # UI dựa vào cờ này để không gửi lượt này làm lịch sử cho LLM
            "invalid_citations": [],
            "rewritten_query": search_query if rewritten else None,
            "hybrid": hybrid_used,
            "latency_ms": timer.stages,
            "total_ms": timer.total_ms,
        }
        persist_turn(session_id, question, answer, citations, quality, log_id)
        yield _sse("fallback", {"reason": "quota", "text": answer})
        yield _sse("citations", {"citations": citations})
        yield _sse("quality", {**quality, "query_log_id": log_id})
        yield _sse("done", {"query_log_id": log_id})
        return

    answer = "".join(answer_parts)

    # Chỉ kiểm tra được sau khi có câu trả lời đầy đủ, nên nằm sau vòng streaming.
    used, invalid = answer_check.check_citations(answer, len(citations))
    for citation in citations:
        citation["used"] = citation["index"] in used
    confidence = answer_check.estimate_confidence(hits, used, invalid, use_rerank, settings.rerank_threshold)
    usage = resolve_usage(usage_sink, prompt_sink[0] if prompt_sink else "", answer)

    quality = {
        "confidence": confidence["level"],
        "reason": confidence["reason"],
        "invalid_citations": sorted(invalid),
        "rewritten_query": search_query if rewritten else None,
        "hybrid": hybrid_used,
        "latency_ms": timer.stages,
        "first_token_ms": first_token_ms,
        "total_ms": timer.total_ms,
        "tokens": usage,
    }

    log_id = record_query(
        **base_log,
        rewritten_query=search_query if rewritten else None,
        hybrid=hybrid_used,
        n_hits=len(hits),
        top_score=top_score,
        confidence=confidence["level"],
        invalid_citations=len(invalid),
        # "Bó tay" tính cả trường hợp LLM trả lời nhưng không dựa được vào tài liệu nào.
        no_answer=not used,
        latency=timer.stages,
        total_ms=timer.total_ms,
        first_token_ms=first_token_ms,
        **usage,
    )
    persist_turn(session_id, question, answer, citations, quality, log_id)

    yield _sse("citations", {"citations": citations})
    yield _sse("quality", {**quality, "query_log_id": log_id})
    yield _sse("done", {"query_log_id": log_id})
