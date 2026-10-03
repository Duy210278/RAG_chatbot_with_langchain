"""Câu hỏi gợi ý cho cột bên phải khung chat, từ hai nguồn bù cho nhau:

  - FAQ: admin soạn tay trong faq.json - chủ động chọn câu, nhưng dễ lỗi thời so với tài liệu đã nạp.
  - Hay được hỏi: lấy từ query_logs - tự bám theo thực tế sử dụng, nhưng CHỈ giữ những câu hệ thống
    đã trả lời tốt. Gợi ý một câu mà bấm vào ra "không tìm thấy" còn tệ hơn không gợi ý gì.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from . import models, schemas
from .config import get_settings

_FAQ_LIST = TypeAdapter(list[schemas.FaqItem])


def normalize_question(text: str) -> str:
    """Khoá để so trùng: bỏ khác biệt hoa/thường, khoảng trắng thừa và dấu câu cuối câu."""
    return " ".join(text.lower().split()).rstrip(" ?.!")


def load_faq(category: str | None) -> list[schemas.FaqItem]:
    """Đọc faq.json mỗi lần gọi (file nhỏ) để admin sửa là thấy ngay, không cần restart.
    Không có file thì trả rỗng - FAQ là tuỳ chọn. File hỏng thì ném ValueError
    (JSONDecodeError và ValidationError đều là ValueError)."""
    path = Path(get_settings().faq_path)
    if not path.exists():
        return []
    items = _FAQ_LIST.validate_python(json.loads(path.read_text(encoding="utf-8")))
    # Câu gắn với một loại tài liệu chỉ hiện khi phạm vi đang chọn bao gồm loại đó - hỏi câu
    # về hợp đồng trong khi đang lọc LEGAL_PDF thì chắc chắn nhận về "không tìm thấy".
    return [i for i in items if category in (None, "ALL") or i.category in (None, category)]


def _answered_well(row) -> bool:
    return not row.no_answer and not row.invalid_citations and row.confidence != "thap"


def popular_questions(db: Session, category: str | None, exclude: set[str], limit: int) -> list[schemas.PopularQuestion]:
    """Câu được hỏi ở nhiều cuộc trò chuyện nhất mà LẦN HỎI GẦN NHẤT được trả lời tốt.

    - Hiển thị câu đã viết lại (rewritten_query) nếu có: câu gốc kiểu "nếu thiếu hồ sơ thì sao"
      chỉ có nghĩa trong hội thoại cũ, còn bản viết lại vốn được sinh ra để đứng độc lập.
    - Xét lần hỏi gần nhất chứ không phải bất kỳ lần nào: câu từng trả lời được nhưng tài liệu
      liên quan đã bị xoá thì phải rơi khỏi danh sách.
    - Bỏ hẳn các lượt lỗi provider (hết quota, sai model...): chúng không nói gì về việc tài liệu
      có trả lời được câu hỏi hay không.
    - Một lần bị chấm 👎 là loại luôn, không đề xuất câu người dùng đã báo là trả lời sai.
    - Bỏ câu nối tiếp hỏi ở chế độ Agent: chế độ đó không viết lại câu hỏi (agent tự đọc lịch sử),
      nên không có bản đứng độc lập để hiển thị.
    """
    settings = get_settings()
    since = datetime.now(timezone.utc) - timedelta(days=settings.popular_days)
    log = models.QueryLog
    q = db.query(
        log.id,
        log.session_id,
        log.question,
        log.rewritten_query,
        log.no_answer,
        log.invalid_citations,
        log.confidence,
        log.feedback,
        log.created_at,
        log.agent_mode,
        log.agent_steps_json,
    ).filter(log.created_at >= since, log.error.is_(None))
    if category and category != "ALL":
        # Câu hỏi ở phạm vi ALL có thể được trả lời từ loại tài liệu khác - không chắc đúng ở phạm vi hẹp.
        q = q.filter(log.category == category)

    groups: dict[str, dict] = {}
    for row in q.order_by(log.created_at).all():
        if row.agent_mode and json.loads(row.agent_steps_json or "{}").get("follow_up"):
            continue
        text = (row.rewritten_query or row.question).strip()
        key = normalize_question(text)
        if not key or key in exclude:
            continue
        g = groups.setdefault(key, {"sessions": set(), "downvoted": False})
        g["sessions"].add(row.session_id or row.id)  # lượt không gắn hội thoại tính là một hội thoại riêng
        g["downvoted"] |= row.feedback == -1
        g["latest"], g["text"] = row, text  # duyệt theo thời gian tăng dần nên đây luôn là lần gần nhất

    qualified = [
        g
        for g in groups.values()
        if len(g["sessions"]) >= settings.popular_min_sessions and not g["downvoted"] and _answered_well(g["latest"])
    ]
    qualified.sort(key=lambda g: (len(g["sessions"]), g["latest"].created_at), reverse=True)
    return [
        schemas.PopularQuestion(question=g["text"], asked_in_sessions=len(g["sessions"])) for g in qualified[:limit]
    ]
