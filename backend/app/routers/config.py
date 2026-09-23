from fastapi import APIRouter

from .. import schemas
from ..config import get_settings

router = APIRouter(prefix="/api/v1/config", tags=["config"])


@router.get("/providers", response_model=schemas.ProvidersResponse)
def list_providers():
    """Trả về danh sách provider ĐÃ có API key cấu hình trong .env, kèm model khả dụng.
    UI dùng endpoint này để dựng dropdown Provider/Model - không còn cho nhập API key trên UI."""
    settings = get_settings()
    return schemas.ProvidersResponse(providers=settings.provider_catalog())
