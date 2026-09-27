from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .embeddings import get_embedder
from .logging_setup import get_logger, setup_logging
from .reranker import get_reranker
from .routers import admin, chat, config, documents, sessions
from .vector_store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(get_settings().log_level)
    logger = get_logger("rag.startup")
    init_db()
    get_embedder()  # tải model embedding trước để request đầu tiên không bị chậm
    get_reranker()  # tương tự cho model rerank (lần chạy đầu sẽ tải model từ HuggingFace)
    get_vector_store()  # khởi tạo Qdrant local + collection
    logger.info("Backend sẵn sàng", extra={"embedding_model": get_settings().embedding_model})
    yield


settings = get_settings()

app = FastAPI(title="RAG Chatbot v2 - MVP", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(config.router)
app.include_router(admin.router)


@app.get("/health")
def health():
    return {"status": "ok"}
