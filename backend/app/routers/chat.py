from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models, schemas
from ..config import get_settings
from ..db import get_db
from ..rag import answer_question_stream

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/completions")
async def chat_completions(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    # api_key trên payload chỉ còn dùng cho gọi API trực tiếp (curl/test) - UI không gửi field này nữa,
    # luôn lấy từ .env theo provider đã chọn.
    api_key = payload.api_key or settings.api_key_for(payload.provider)
    if not api_key:
        raise HTTPException(
            400,
            f"Chưa cấu hình API key cho provider '{payload.provider}' trong file .env "
            f"(biến {payload.provider.upper()}_API_KEY hoặc GOOGLE_API_KEY với Gemini). "
            "Khởi động lại backend sau khi thêm key.",
        )

    generator = answer_question_stream(
        db=db,
        question=payload.message,
        provider=payload.provider,
        api_key=api_key,
        model=payload.model,
        top_k=payload.top_k,
        category=payload.category,
        history=[h.model_dump() for h in payload.history],
        use_rerank=payload.use_rerank,
        session_id=payload.session_id,
    )
    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("/feedback")
def submit_feedback(payload: schemas.FeedbackRequest, db: Session = Depends(get_db)):
    """Chấm 👍/👎 cho một lượt trả lời. query_log_id lấy từ sự kiện SSE 'done'."""
    row = db.get(models.QueryLog, payload.query_log_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy lượt hỏi tương ứng.")
    row.feedback = payload.rating
    row.feedback_note = payload.note
    db.commit()
    return {"status": "ok"}
