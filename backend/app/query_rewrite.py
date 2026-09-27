"""Viết lại câu hỏi nối tiếp thành câu hỏi độc lập trước khi đem đi truy hồi.

Lỗi được sửa ở đây: trước đây vector truy vấn tính thẳng từ câu hỏi thô. Người dùng hỏi
"Nghỉ thai sản được bao nhiêu ngày?" rồi hỏi tiếp "Thế còn nghỉ ốm thì sao?" thì bước tìm kiếm
chỉ nhìn thấy đúng chuỗi "Thế còn nghỉ ốm thì sao?" - mất sạch ngữ cảnh, truy hồi ra tài liệu
vu vơ. Lịch sử hội thoại vốn CHỈ được đưa cho LLM ở bước sinh câu trả lời, không tới được
bước truy hồi. Với giao diện chat thì đây là kiểu hỏi rất phổ biến.

Đánh đổi: tốn thêm một lượt gọi LLM (ngắn) trước khi tìm kiếm. Chỉ chạy khi CÓ lịch sử hội thoại,
nên câu hỏi đầu tiên của mỗi phiên không bị ảnh hưởng. Mọi lỗi/timeout đều rơi về câu hỏi gốc.
"""

import asyncio

from .config import get_settings
from .llm import complete_text

# Bản prompt đầu tiên chỉ nói chung chung "viết lại thành câu độc lập" và cho ra kết quả hỏng:
# "Thế còn ưu điểm thì sao?" -> "Ưu điểm gồm những gì?" - vẫn mất sạch chủ thể nên truy hồi vẫn
# trượt y như cũ. Phải nói thẳng yêu cầu "chép lại CHỦ THỂ từ lịch sử" kèm ví dụ mẫu thì model
# mới làm đúng.
REWRITE_SYSTEM = (
    "Bạn là bộ viết lại truy vấn cho hệ thống tìm kiếm tài liệu nội bộ. "
    "Dựa vào lịch sử hội thoại, hãy viết lại CÂU HỎI MỚI NHẤT thành một câu hỏi độc lập, "
    "đầy đủ ngữ cảnh, hiểu được mà không cần đọc lịch sử.\n"
    "QUAN TRỌNG NHẤT: phải chép lại rõ ràng CHỦ THỂ/ĐỐI TƯỢNG đang được nói tới trong lịch sử "
    "(tên tài liệu, tên quy trình, đối tượng áp dụng...) vào câu hỏi mới. "
    "Câu hỏi viết lại phải tự nó nêu đủ chủ thể, vì bộ tìm kiếm không nhìn thấy lịch sử hội thoại.\n"
    "Giữ nguyên mọi thuật ngữ, mã số, tên văn bản xuất hiện trong câu gốc. "
    "Nếu câu hỏi đã độc lập sẵn thì trả lại đúng như cũ.\n"
    "Ví dụ — lịch sử đang nói về chế độ nghỉ thai sản trong quy chế nhân sự, "
    'câu hỏi mới "Thế còn nghỉ ốm thì sao?" phải viết lại thành '
    '"Chế độ nghỉ ốm trong quy chế nhân sự được quy định như thế nào?".\n'
    "CHỈ trả về một dòng duy nhất là câu hỏi đã viết lại, không giải thích, không thêm ngoặc kép."
)

_MAX_HISTORY_TURNS = 4
_MAX_HISTORY_CHARS = 400


def _build_prompt(question: str, history: list[dict]) -> str:
    recent = history[-_MAX_HISTORY_TURNS:]
    lines = [
        f"{'Người dùng' if h['role'] == 'user' else 'Trợ lý'}: {h['content'][:_MAX_HISTORY_CHARS]}"
        for h in recent
    ]
    return (
        "### Lịch sử hội thoại:\n" + "\n".join(lines) + "\n\n"
        f"### Câu hỏi mới nhất:\n{question}\n\n"
        "### Câu hỏi độc lập để đi tìm kiếm:"
    )


def _is_sane(rewritten: str, original: str) -> bool:
    """Chặn các kiểu đầu ra hỏng: rỗng, model trả lời câu hỏi thay vì viết lại, hoặc lan man.
    Thà tìm kiếm bằng câu gốc còn hơn tìm bằng một truy vấn đã bị bóp méo."""
    if not rewritten:
        return False
    return len(rewritten) <= max(200, len(original) * 3)


async def rewrite_for_retrieval(
    provider: str,
    api_key: str,
    model: str | None,
    question: str,
    history: list[dict] | None,
) -> tuple[str, bool]:
    """Trả về (truy_vấn_để_tìm_kiếm, đã_viết_lại_hay_chưa)."""
    settings = get_settings()
    if not settings.query_rewrite_enabled or not history:
        return question, False

    try:
        raw = await asyncio.wait_for(
            complete_text(provider, api_key, model, REWRITE_SYSTEM, _build_prompt(question, history)),
            timeout=settings.query_rewrite_timeout,
        )
    except Exception:  # noqa: BLE001 - gồm cả TimeoutError: viết lại là tuỳ chọn, hỏng thì bỏ qua
        return question, False

    rewritten = raw.strip().splitlines()[0].strip().strip('"').strip() if raw.strip() else ""
    if not _is_sane(rewritten, question) or rewritten == question:
        return question, False
    return rewritten, True
