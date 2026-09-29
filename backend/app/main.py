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


settings = get_settings()

# Cấu hình log ngay lúc import app chứ không đợi tới lifespan: uvicorn import app TRƯỚC khi in
# "Started server process"/"Waiting for application startup", nên mọi dòng từ đây trở đi đều cùng
# một định dạng (trước đây các dòng đó vẫn theo định dạng mặc định của uvicorn, lẫn với JSON).
setup_logging(settings.log_level, settings.log_format, settings.log_quiet_access)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_logger("rag.startup")
    init_db()
    get_embedder()  # tải model embedding trước để request đầu tiên không bị chậm
    get_reranker()  # tương tự cho model rerank (lần chạy đầu sẽ tải model từ HuggingFace)
    get_vector_store()  # khởi tạo Qdrant local + collection
    logger.info("Backend sẵn sàng", extra={"embedding_model": get_settings().embedding_model})
    yield


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
