import json
from collections.abc import AsyncIterator

from sqlalchemy.orm import Session

from .embeddings import get_embedder
from .llm import generate_answer_stream
from .vector_store import get_vector_store


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def answer_question_stream(
    db: Session,
    question: str,
    provider: str,
    api_key: str,
    model: str | None,
    top_k: int,
    category: str | None,
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """Retrieval -> Context Assembly -> Generation, phát theo Server-Sent Events (mục 6.2)."""
    embedder = get_embedder()
    store = get_vector_store()

    query_vector = embedder.embed_query(question)
    hits = store.search(query_vector, top_k=top_k, category=category)

    if not hits:
        yield _sse("token", {"text": "Không tìm thấy tài liệu liên quan trong hệ thống để trả lời câu hỏi này."})
        yield _sse("citations", {"citations": []})
        yield _sse("done", {})
        return

    context_blocks = [hit.payload["content"] for hit in hits]
    citations = [
        {
            "document_id": hit.payload["document_id"],
            "title": hit.payload.get("title", "Không rõ"),
            "page_number": hit.payload.get("page_number"),
            "score": round(hit.score, 4),
            "snippet": hit.payload["content"][:300],
        }
        for hit in hits
    ]

    try:
        async for token in generate_answer_stream(
            provider, api_key, model, question, context_blocks, history=history
        ):
            yield _sse("token", {"text": token})
    except Exception as exc:  # noqa: BLE001 - lỗi gọi LLM (key sai, hết quota...) cần báo về UI thay vì crash SSE
        yield _sse("error", {"message": str(exc)})
        yield _sse("done", {})
        return

    yield _sse("citations", {"citations": citations})
    yield _sse("done", {})
