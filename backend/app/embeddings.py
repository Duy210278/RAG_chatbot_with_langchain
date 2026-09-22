from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import get_settings


class Embedder:
    """Model embedding chạy local (miễn phí, không cần API key), đa ngôn ngữ - hỗ trợ tốt tiếng Việt.

    Model họ "e5" cần prefix "query: " / "passage: " để đạt chất lượng truy hồi tốt nhất
    (theo khuyến nghị của tác giả model intfloat/multilingual-e5-*).
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(settings.embedding_model)
