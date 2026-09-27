"""Hội thoại lưu bền: tạo / liệt kê / mở lại / xoá.

Trước đây lịch sử chỉ nằm trong st.session_state của Streamlit nên mất sạch khi tải lại trang.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db

router = APIRouter(prefix="/api/v1/chat/sessions", tags=["chat-sessions"])


@router.post("", response_model=schemas.ChatSessionOut)
def create_session(db: Session = Depends(get_db)):
    session = models.ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return _to_out(session, message_count=0)


@router.get("", response_model=list[schemas.ChatSessionOut])
def list_sessions(limit: int = 50, db: Session = Depends(get_db)):
    sessions = (
        db.query(models.ChatSession).order_by(models.ChatSession.updated_at.desc()).limit(limit).all()
    )
    # Đếm bằng một truy vấn gộp thay vì chạm relationship của từng session (tránh N+1 query).
    counts = dict(
        db.query(models.ChatMessage.session_id, func.count(models.ChatMessage.id))
        .group_by(models.ChatMessage.session_id)
        .all()
    )
    return [_to_out(s, counts.get(s.id, 0)) for s in sessions]


@router.get("/{session_id}/messages", response_model=list[schemas.ChatMessageOut])
def get_messages(session_id: str, db: Session = Depends(get_db)):
    session = db.get(models.ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Không tìm thấy cuộc trò chuyện.")
    return [
        schemas.ChatMessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            citations=json.loads(m.citations_json) if m.citations_json else [],
            quality=json.loads(m.quality_json) if m.quality_json else None,
            query_log_id=m.query_log_id,
            created_at=m.created_at.isoformat(),
        )
        for m in session.messages
    ]


@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(models.ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Không tìm thấy cuộc trò chuyện.")
    # Cố ý KHÔNG xoá query_logs: người dùng xoá hội thoại của họ, nhưng số liệu vận hành
    # (câu nào hệ thống bó tay, chậm ở đâu) vẫn phải giữ để cải thiện hệ thống.
    db.delete(session)
    db.commit()
    return {"status": "deleted"}


def _to_out(session: models.ChatSession, message_count: int) -> schemas.ChatSessionOut:
    return schemas.ChatSessionOut(
        id=session.id,
        title=session.title,
        message_count=message_count,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )
