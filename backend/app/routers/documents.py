import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from qdrant_client.http import models as qmodels
from sqlalchemy.orm import Session

from .. import models, schemas
from ..config import UPLOAD_DIR
from ..db import get_db
from ..embeddings import get_embedder
from ..ingestion import CHUNKERS, chunk_document
from ..vector_store import get_vector_store

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Các định dạng khác trong Ma trận Ingestion (mục 2) sẽ được thêm dần - mỗi định dạng
# chỉ cần thêm 1 hàm chunk_xxx() và đăng ký vào CHUNKERS trong ingestion.py.
ALLOWED_EXTENSIONS = set(CHUNKERS.keys())


@router.post("/upload", response_model=schemas.UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("GENERAL"),
    is_public: bool = Form(True),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(400, f"Định dạng {ext} chưa được hỗ trợ. Hiện chỉ hỗ trợ: {supported}.")

    doc_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{doc_id}{ext}"
    with saved_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    document = models.Document(
        id=doc_id,
        title=file.filename,
        file_path=str(saved_path),
        file_extension=ext,
        file_size_bytes=saved_path.stat().st_size,
        category=category,
        is_public=is_public,
        status="PROCESSING",
    )
    db.add(document)
    db.commit()

    try:
        chunks = chunk_document(str(saved_path), ext)
        if not chunks:
            raise ValueError("Không trích xuất được nội dung văn bản nào từ file.")

        embedder = get_embedder()
        vectors = embedder.embed_passages([c["content"] for c in chunks])

        points = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = str(uuid.uuid4())
            db.add(
                models.DocumentChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    token_count=chunk["token_count"],
                    page_number=chunk["page_number"],
                )
            )
            points.append(
                qmodels.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "document_id": doc_id,
                        "chunk_index": chunk["chunk_index"],
                        "category": category,
                        "title": file.filename,
                        "page_number": chunk["page_number"],
                        "content": chunk["content"],
                        "is_public": is_public,
                    },
                )
            )

        get_vector_store().upsert_chunks(points)

        document.status = "COMPLETED"
        document.chunk_count = len(chunks)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        document.status = "FAILED"
        document.error_message = str(exc)
        db.commit()
        raise HTTPException(500, f"Lỗi khi xử lý tài liệu: {exc}") from exc

    db.refresh(document)
    return schemas.UploadResponse(document=_to_document_out(document))


@router.get("", response_model=list[schemas.DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(models.Document).order_by(models.Document.created_at.desc()).all()
    return [_to_document_out(d) for d in docs]


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(models.Document, document_id)
    if not document:
        raise HTTPException(404, "Không tìm thấy tài liệu.")
    get_vector_store().delete_document(document_id)
    Path(document.file_path).unlink(missing_ok=True)
    db.delete(document)
    db.commit()
    return {"status": "deleted"}


def _to_document_out(document: models.Document) -> schemas.DocumentOut:
    return schemas.DocumentOut(
        id=document.id,
        title=document.title,
        category=document.category,
        status=document.status,
        chunk_count=document.chunk_count,
        file_size_bytes=document.file_size_bytes,
        created_at=document.created_at.isoformat(),
    )
