from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import schemas
from ..config import get_settings
from ..db import get_db
from ..rag import answer_question_stream

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/completions")
async def chat_completions(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    default_keys = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "xai": settings.xai_api_key,
        "gemini": settings.google_api_key,
        "groq": settings.groq_api_key,
    }
    api_key = payload.api_key or default_keys.get(payload.provider)
    if not api_key:
        raise HTTPException(
            400,
            f"Chưa cấu hình API key cho provider '{payload.provider}'. "
            "Nhập API key ở thanh bên trái giao diện, hoặc đặt biến môi trường tương ứng trong file .env.",
        )

    generator = answer_question_stream(
        db=db,
        question=payload.message,
        provider=payload.provider,
        api_key=api_key,
        model=payload.model,
        top_k=payload.top_k,
        category=payload.category,
    )
    return StreamingResponse(generator, media_type="text/event-stream")
