"""Thống kê vận hành: hệ thống đang chậm ở đâu, câu nào bó tay, tiêu bao nhiêu token.

Đây là phần trả lời cho câu hỏi "làm sao biết hệ thống đang tốt lên hay xấu đi" - trước đây
không có dữ liệu nào để trả lời.

CẢNH BÁO: các endpoint này chưa có xác thực, giống mọi endpoint khác của MVP. Chúng để lộ
nội dung câu hỏi của người dùng, nên phải đặt sau SSO trước khi mở ra mạng nội bộ.
"""

import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


@router.get("/stats", response_model=schemas.AdminStats)
def stats(days: int = 7, db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.query(models.QueryLog).filter(models.QueryLog.created_at >= since).all()

    durations = [r.total_ms for r in rows if r.total_ms]
    first_tokens = [r.first_token_ms for r in rows if r.first_token_ms]

    # Trung bình từng khâu: gộp theo tên khâu để thấy khâu nào ăn hết thời gian.
    stage_totals: dict[str, list[int]] = {}
    for row in rows:
        for name, ms in json.loads(row.latency_json or "{}").items():
            stage_totals.setdefault(name, []).append(ms)

    by_confidence: dict[str, int] = {}
    for row in rows:
        key = row.confidence or "khong_ro"
        by_confidence[key] = by_confidence.get(key, 0) + 1

    return schemas.AdminStats(
        days=days,
        total_queries=len(rows),
        no_answer_count=sum(1 for r in rows if r.no_answer),
        error_count=sum(1 for r in rows if r.error),
        invalid_citation_count=sum(1 for r in rows if r.invalid_citations),
        positive_feedback=sum(1 for r in rows if r.feedback == 1),
        negative_feedback=sum(1 for r in rows if r.feedback == -1),
        avg_total_ms=int(sum(durations) / len(durations)) if durations else 0,
        p95_total_ms=_percentile(durations, 0.95),
        avg_first_token_ms=int(sum(first_tokens) / len(first_tokens)) if first_tokens else 0,
        stage_avg_ms={k: int(sum(v) / len(v)) for k, v in sorted(stage_totals.items())},
        by_confidence=by_confidence,
        prompt_tokens=sum(r.prompt_tokens or 0 for r in rows),
        completion_tokens=sum(r.completion_tokens or 0 for r in rows),
        estimated_token_share=round(
            sum(1 for r in rows if r.tokens_estimated) / len(rows), 2
        ) if rows else 0.0,
        by_mode=_by_mode(rows),
    )


def _mode_stats(rows: list) -> dict:
    """Số liệu để so sánh hai chế độ. Lượt lỗi provider (hết quota, sai model...) đứng riêng ở 'errors':
    chúng không nói gì về chất lượng của chế độ, gộp vào thì tỉ lệ bó tay bị đội lên vô cớ."""
    ok = [r for r in rows if not r.error]
    durations = [r.total_ms for r in ok if r.total_ms]
    tokens = [(r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in ok if r.prompt_tokens is not None]
    return {
        "queries": len(rows),
        "errors": len(rows) - len(ok),
        "no_answer_rate": round(sum(1 for r in ok if r.no_answer) / len(ok), 3) if ok else 0.0,
        "avg_total_ms": int(sum(durations) / len(durations)) if durations else 0,
        "p95_total_ms": _percentile(durations, 0.95),
        "avg_tokens": int(sum(tokens) / len(tokens)) if tokens else 0,
        "positive_feedback": sum(1 for r in ok if r.feedback == 1),
        "negative_feedback": sum(1 for r in ok if r.feedback == -1),
    }


def _by_mode(rows: list) -> dict:
    """Rỗng khi chưa ai bật Agent. Lượt bật Agent nhưng phải quay về luồng thường được tính vào 'thuong'
    (đúng là luồng đã chạy) và đếm riêng ở agent.fallbacks."""
    if not any(r.agent_steps_json for r in rows):
        return {}
    return {
        "thuong": _mode_stats([r for r in rows if not r.agent_mode]),
        "agent": {
            **_mode_stats([r for r in rows if r.agent_mode]),
            "fallbacks": sum(1 for r in rows if r.agent_steps_json and not r.agent_mode),
        },
    }


@router.get("/queries", response_model=list[schemas.QueryLogOut])
def queries(only_problems: bool = True, days: int = 30, limit: int = 100, db: Session = Depends(get_db)):
    """Danh sách lượt hỏi. only_problems = những lượt đáng xem lại nhất:
    không truy hồi được gì, bị người dùng chấm 👎, có lỗi provider, hoặc trích dẫn hỏng."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    q = db.query(models.QueryLog).filter(models.QueryLog.created_at >= since)
    if only_problems:
        q = q.filter(
            (models.QueryLog.no_answer.is_(True))
            | (models.QueryLog.feedback == -1)
            | (models.QueryLog.error.isnot(None))
            | (models.QueryLog.invalid_citations > 0)
        )
    rows = q.order_by(models.QueryLog.created_at.desc()).limit(limit).all()
    return [_to_out(r) for r in rows]


@router.get("/unanswered", response_model=list[schemas.UnansweredGroup])
def unanswered(days: int = 30, limit: int = 30, db: Session = Depends(get_db)):
    """Gom các câu hỏi mà hệ thống bó tay theo nội dung câu hỏi - câu nào lặp lại nhiều lần
    chính là khoảng trống tài liệu cần bổ sung trước."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        db.query(
            models.QueryLog.question,
            func.count(models.QueryLog.id).label("n"),
            func.max(models.QueryLog.created_at).label("last_seen"),
        )
        .filter(models.QueryLog.created_at >= since, models.QueryLog.no_answer.is_(True))
        .group_by(models.QueryLog.question)
        .order_by(func.count(models.QueryLog.id).desc())
        .limit(limit)
        .all()
    )
    return [
        schemas.UnansweredGroup(question=r.question, count=r.n, last_seen=r.last_seen.isoformat())
        for r in rows
    ]


def _to_out(row: models.QueryLog) -> schemas.QueryLogOut:
    return schemas.QueryLogOut(
        id=row.id,
        created_at=row.created_at.isoformat(),
        question=row.question,
        rewritten_query=row.rewritten_query,
        provider=row.provider,
        model=row.model,
        n_hits=row.n_hits,
        top_score=row.top_score,
        confidence=row.confidence,
        invalid_citations=row.invalid_citations,
        no_answer=row.no_answer,
        error=row.error,
        total_ms=row.total_ms,
        first_token_ms=row.first_token_ms,
        latency_ms=json.loads(row.latency_json or "{}"),
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        tokens_estimated=row.tokens_estimated,
        feedback=row.feedback,
        feedback_note=row.feedback_note,
        agent_mode=row.agent_mode,
        agent=json.loads(row.agent_steps_json) if row.agent_steps_json else None,
    )
