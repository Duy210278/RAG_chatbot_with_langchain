"""Kiểm tra câu trả lời sau khi sinh: trích dẫn [n] có trỏ tới nguồn có thật không,
và ước lượng độ tin cậy của lượt trả lời đó.

Vì sao cần: system prompt yêu cầu LLM trích [n], nhưng trước đây không có gì kiểm chứng.
Model hoàn toàn có thể viết [5] trong khi chỉ có 3 nguồn được gửi - người đọc thấy có trích dẫn
thì tin, mà trích dẫn đó trỏ vào hư không. Đây là kiểu lỗi nguy hiểm nhất với tài liệu nội bộ
vì nó trông *giống* một câu trả lời có căn cứ.
"""

import re

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

# Mức tin cậy được suy ra TƯƠNG ĐỐI so với rerank_threshold đã cấu hình, không phải từ hằng số
# tuyệt đối. Lý do: thang điểm tuyệt đối của mỗi model cross-encoder rất khác nhau - đo thực tế
# trên corpus tiếng Việt với mmarco-mMiniLMv2, đoạn ĐÚNG chỉ đạt 0.04-0.25, nên mọi ngưỡng cứng
# kiểu 0.8 sẽ báo "độ tin cậy thấp" cho 100% câu hỏi và người dùng sẽ bỏ qua chỉ báo này ngay.
# Neo vào rerank_threshold - vốn là mức "thế nào là liên quan" mà người vận hành đã tự chọn -
# giúp chỉ báo tự đi theo khi đổi model hoặc chỉnh lại ngưỡng.
_HIGH_RATIO = 1.0  # đạt/vượt ngưỡng liên quan
_MEDIUM_RATIO = 0.5  # còn một nửa ngưỡng


def extract_citation_indices(answer: str) -> set[int]:
    """Bắt cả [1], [2,3] lẫn [2; 3] - các kiểu LLM hay viết."""
    found: set[int] = set()
    for group in _CITE_RE.findall(answer):
        for part in re.split(r"[,;]", group):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return found


def check_citations(answer: str, n_sources: int) -> tuple[set[int], set[int]]:
    """Trả về (chỉ_số_hợp_lệ_được_dùng, chỉ_số_không_tồn_tại)."""
    cited = extract_citation_indices(answer)
    used = {i for i in cited if 1 <= i <= n_sources}
    invalid = cited - used
    return used, invalid


def estimate_confidence(
    hits: list,
    used: set[int],
    invalid: set[int],
    use_rerank: bool,
    threshold: float,
) -> dict:
    """Ước lượng độ tin cậy từ BẰNG CHỨNG TRUY HỒI - không phải phép đo hallucination.

    Cố ý chỉ tính khi bật rerank: lúc đó .score là điểm cross-encoder (0-1, đã sigmoid) có
    ý nghĩa tuyệt đối. Khi tắt rerank, .score là cosine hoặc điểm RRF - hai thang hoàn toàn
    khác, đem so với cùng một ngưỡng sẽ cho ra con số tự tin một cách vô căn cứ.
    """
    if not hits:
        return {"level": "thap", "reason": "Không truy hồi được đoạn tài liệu nào."}

    if not use_rerank:
        return {
            "level": "khong_do_duoc",
            "reason": "Đang tắt Reranker nên không có thang điểm liên quan để đánh giá.",
        }

    top_score = max(h.score for h in hits)
    strong = sum(1 for h in hits if h.score >= threshold)
    reasons = [f"đoạn khớp nhất đạt {top_score:.2f}", f"{strong}/{len(hits)} đoạn đạt ngưỡng {threshold}"]

    if top_score >= threshold * _HIGH_RATIO:
        level = "cao"
    elif top_score >= threshold * _MEDIUM_RATIO:
        level = "trung_binh"
    else:
        level = "thap"

    # Trả lời mà không trích nguồn nào: có thể model đang nói từ kiến thức nền thay vì từ tài liệu.
    # Chỉ hạ TRẦN xuống "trung bình", không kéo tụt mức vốn đã thấp.
    if not used and _rank(level) > _rank("trung_binh"):
        level = "trung_binh"
    if not used:
        reasons.append("câu trả lời không trích dẫn nguồn nào")

    # Trích dẫn trỏ vào nguồn không tồn tại là dấu hiệu đỏ rõ ràng, hạ thẳng xuống mức thấp.
    if invalid:
        level = "thap"
        reasons.append(f"có trích dẫn không hợp lệ: {sorted(invalid)}")

    return {"level": level, "reason": "; ".join(reasons)}


def _rank(level: str) -> int:
    return {"thap": 0, "trung_binh": 1, "cao": 2}.get(level, 0)
