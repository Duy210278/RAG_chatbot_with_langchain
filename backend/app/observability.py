"""Đo thời gian từng khâu, đếm token, và ghi lại mỗi lượt hỏi vào DB.

Mục tiêu: trả lời được những câu mà trước đây không có dữ liệu để trả lời -
"chậm ở khâu nào?", "câu nào hệ thống bó tay?", "tháng này tiêu bao nhiêu token?".
"""

import json
import time
from contextlib import contextmanager

from .db import SessionLocal
from .logging_setup import get_logger

logger = get_logger("rag.query")


class StageTimer:
    """Đo thời gian từng khâu trong một lượt hỏi.

    Đo riêng từng khâu (viết lại / embed+tìm kiếm / rerank / chờ token đầu tiên / sinh câu trả lời)
    thay vì chỉ đo tổng, vì mỗi khâu có cách khắc phục hoàn toàn khác nhau khi chậm.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.stages: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = int((time.perf_counter() - started) * 1000)

    def elapsed_ms(self) -> int:
        """Thời gian tính từ lúc bắt đầu lượt hỏi - dùng cho mốc 'token đầu tiên'.

        Cố ý KHÔNG ghi vào self.stages: stages chỉ chứa KHOẢNG thời gian của từng khâu, trộn
        thêm mốc tích luỹ vào đó thì tổng các giá trị trong stages sẽ vô nghĩa.
        """
        return int((time.perf_counter() - self._start) * 1000)

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)


def estimate_tokens(text: str) -> int:
    """Ước lượng bằng tiktoken (đã có sẵn trong dự án cho bước chunking).

    Chỉ dùng khi provider không trả về số liệu thật. Đây là số XẤP XỈ: tokenizer của
    Claude/Gemini khác cl100k_base, nên đừng dùng con số này để đối soát hoá đơn.
    """
    from .ingestion import count_tokens

    try:
        return count_tokens(text)
    except Exception:  # noqa: BLE001 - đếm token hỏng không được phép làm hỏng lượt trả lời
        return 0


def resolve_usage(usage: dict | None, prompt_text: str, answer_text: str) -> dict:
    """Ưu tiên số liệu thật từ provider, thiếu thì mới ước lượng - và đánh dấu rõ là ước lượng."""
    if usage and usage.get("prompt_tokens") is not None:
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "tokens_estimated": False,
        }
    return {
        "prompt_tokens": estimate_tokens(prompt_text),
        "completion_tokens": estimate_tokens(answer_text),
        "tokens_estimated": True,
    }


def _summary(fields: dict, latency: dict) -> str:
    """Một dòng cho người đọc terminal (định dạng pretty) - bản JSON vẫn giữ đủ từng trường riêng."""
    parts = [f"{fields.get('provider')}/{fields.get('model') or 'mặc định'}"]
    # Bỏ các khâu gần như tức thì (<50ms) - chỉ làm dài dòng mà không nói lên điều gì.
    stages = ", ".join(f"{name} {ms / 1000:.1f}s" for name, ms in latency.items() if ms >= 50)
    parts.append(f"{(fields.get('total_ms') or 0) / 1000:.1f}s" + (f" ({stages})" if stages else ""))
    if fields.get("error"):
        parts.append("LỖI " + " ".join(str(fields["error"]).split())[:120])
    else:
        parts.append(f"{fields.get('n_hits', 0)} đoạn")
        if fields.get("no_answer"):
            parts.append("không trả lời được")
        if fields.get("confidence"):
            parts.append(f"tin cậy {fields['confidence']}")
        if fields.get("prompt_tokens") is not None:
            parts.append(f"{fields['prompt_tokens']}+{fields.get('completion_tokens') or 0} tok")
    question = " ".join((fields.get("question") or "").split())
    parts.append(f'"{question[:80]}{"…" if len(question) > 80 else ""}"')
    return " · ".join(parts)


def record_query(**fields) -> str | None:
    """Ghi một dòng query_logs + một dòng log JSON. Trả về id để UI gắn phản hồi 👍/👎.

    Dùng session DB riêng: hàm này chạy ở cuối luồng SSE, không nên phụ thuộc vào vòng đời
    session của request. Mọi lỗi ghi log đều bị nuốt - ghi nhận vận hành không bao giờ
    được phép làm hỏng câu trả lời đã sinh ra cho người dùng.
    """
    from . import models

    latency = fields.pop("latency", None) or {}
    try:
        with SessionLocal() as db:
            row = models.QueryLog(
                latency_json=json.dumps(latency, ensure_ascii=False),
                **fields,
            )
            db.add(row)
            db.commit()
            log_id = row.id
    except Exception:  # noqa: BLE001
        logger.warning("Không ghi được query_log", exc_info=True)
        return None

    # Lượt lỗi (hết quota, sai model...) in ở mức WARNING để nổi bật giữa các lượt bình thường.
    log = logger.warning if fields.get("error") else logger.info
    log(
        "query",
        extra={
            "pretty": _summary(fields, latency),
            "query_log_id": log_id,
            "session_id": fields.get("session_id"),
            "provider": fields.get("provider"),
            "model": fields.get("model"),
            "n_hits": fields.get("n_hits"),
            "top_score": fields.get("top_score"),
            "confidence": fields.get("confidence"),
            "no_answer": fields.get("no_answer"),
            "error": fields.get("error"),
            "total_ms": fields.get("total_ms"),
            "first_token_ms": fields.get("first_token_ms"),
            "latency_ms": latency,
            "prompt_tokens": fields.get("prompt_tokens"),
            "completion_tokens": fields.get("completion_tokens"),
            "tokens_estimated": fields.get("tokens_estimated"),
            "question": (fields.get("question") or "")[:200],
        },
    )
    return log_id


def persist_turn(
    session_id: str | None,
    question: str,
    answer: str,
    citations: list[dict] | None,
    quality: dict | None,
    query_log_id: str | None = None,
) -> str | None:
    """Lưu một lượt (câu hỏi + câu trả lời) vào hội thoại, để mở lại được sau khi tải lại trang."""
    from . import models

    if not session_id:
        return None
    try:
        with SessionLocal() as db:
            session = db.get(models.ChatSession, session_id)
            if session is None:
                return None
            db.add(models.ChatMessage(session_id=session_id, role="user", content=question))
            assistant = models.ChatMessage(
                session_id=session_id,
                role="assistant",
                content=answer,
                citations_json=json.dumps(citations or [], ensure_ascii=False),
                quality_json=json.dumps(quality, ensure_ascii=False) if quality else None,
                query_log_id=query_log_id,
            )
            db.add(assistant)
            # Tiêu đề hội thoại lấy từ câu hỏi đầu tiên - đủ để nhận ra trong danh sách.
            if session.title == "Cuộc trò chuyện mới":
                session.title = question[:120]
            from datetime import datetime, timezone

            session.updated_at = datetime.now(timezone.utc)
            db.commit()
            return assistant.id
    except Exception:  # noqa: BLE001
        logger.warning("Không lưu được lượt hội thoại", exc_info=True)
        return None
