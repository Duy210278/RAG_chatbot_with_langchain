from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import QDRANT_PATH, get_settings


class VectorStore:
    """Wrapper quanh Qdrant chạy ở local mode (lưu file, không cần server/Docker).

    Tương ứng mục 5 của tài liệu thiết kế (payload + filtering), nhưng rút gọn:
    MVP chưa có RBAC nhiều lớp, chỉ lọc theo `category` và `is_public`.
    """

    def __init__(self):
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self._ensure_collection(settings.embedding_dim)

    def _ensure_collection(self, dim: int) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

    def upsert_chunks(self, points: list[qmodels.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int, category: str | None = None):
        """Trả về danh sách điểm (mỗi điểm có .score, .payload) - dùng query_points
        vì .search() đã bị loại bỏ khỏi qdrant-client >= 1.11."""
        query_filter = None
        if category and category != "ALL":
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=category))]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return response.points

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                )
            ),
        )


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
