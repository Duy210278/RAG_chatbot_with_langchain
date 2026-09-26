from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from .config import get_settings


class Reranker:
    """Cross-encoder chấm lại từng cặp (câu hỏi, đoạn) - chính xác hơn so khớp vector vì model
    đọc câu hỏi và đoạn văn cùng lúc, nên điểm của đoạn đúng và đoạn nhiễu tách biệt rõ
    (mục Re-ranking trong tài liệu thiết kế)."""

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, question: str, passages: list[str]) -> list[float]:
        # Ép sigmoid để điểm luôn nằm trong 0-1 dù model trả logit thô (mmarco) hay đã sigmoid (bge),
        # nhờ vậy dùng chung được rerank_threshold khi đổi model.
        # batch nhỏ = ít padding hơn, nhanh hơn rõ trên CPU (đo với bge-v2-m3: 8.8s batch 4 so với 14.4s batch 16).
        scores = self.model.predict(
            [(question, p) for p in passages], batch_size=4, activation_fn=torch.nn.Sigmoid()
        )
        return scores.tolist()


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings().reranker_model)


def rerank_hits(question: str, hits: list, max_k: int) -> list:
    """Chấm lại + K động: chỉ giữ các đoạn đạt ngưỡng liên quan (tối đa max_k), thay cho
    việc luôn lấy đúng K đoạn. Điểm của mỗi hit được thay bằng điểm rerank.

    Nếu không đoạn nào đạt ngưỡng (câu hỏi mơ hồ kiểu "liệt kê tất cả...", hoặc tài liệu không
    có câu trả lời) thì vẫn lấy max_k đoạn điểm cao nhất như top-k thường, để LLM tự đánh giá
    thay vì mất hết ngữ cảnh."""
    if not hits:
        return []
    scores = get_reranker().score(question, [h.payload["content"] for h in hits])
    ranked = sorted(
        (h.model_copy(update={"score": s}) for h, s in zip(hits, scores)),
        key=lambda h: h.score,
        reverse=True,
    )
    threshold = get_settings().rerank_threshold
    relevant = [h for h in ranked if h.score >= threshold]
    return (relevant or ranked)[:max_k]
