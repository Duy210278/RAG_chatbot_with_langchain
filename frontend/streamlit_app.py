"""
Giao diện Streamlit MVP - tương ứng mục 6.1 của tài liệu thiết kế, rút gọn còn
các phần cốt lõi cho MVP: Chat & Citations, Nạp dữ liệu, Danh sách tài liệu, Giám sát.

Mỗi phần là một TRANG riêng (st.navigation, thanh điều hướng phía trên) thay vì tab: với tab,
ô chat bị ghim ở đáy MỌI tab (gõ câu hỏi ở tab Giám sát thì câu trả lời hiện ở tab khác), và
sidebar cấu hình model vẫn hiện cả khi đang nạp tài liệu.

API key KHÔNG nhập trên UI - chỉ khai báo trong file .env của backend (xem README).
Sidebar trang Hỏi đáp chỉ hiển thị Provider/Model đã có key cấu hình sẵn (lấy từ
GET /api/v1/config/providers), cùng Top-K và bật/tắt Reranker.
"""

import json
import mimetypes
import os

import requests
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot Nội bộ", page_icon="🤖", layout="wide")

# Mã phân loại giữ nguyên như backend, chỉ đổi cách HIỂN THỊ cho người dùng đọc được.
CATEGORY_LABELS = {
    "ALL": "Tất cả tài liệu",
    "GENERAL": "Chung",
    "LEGAL_PDF": "Văn bản pháp lý",
    "CONTRACT": "Hợp đồng",
    "TECH_SPEC": "Tài liệu kỹ thuật",
}
DOC_CATEGORIES = [c for c in CATEGORY_LABELS if c != "ALL"]

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/smart_toy:"

# Nội dung thay cho lượt lỗi khi gửi lại làm lịch sử hội thoại cho LLM.
ERROR_PLACEHOLDER = "(Không trả lời được do lỗi hệ thống ở lượt này.)"

_CONFIDENCE_BADGE = {
    "cao": ("🟢", "Độ tin cậy cao"),
    "trung_binh": ("🟡", "Độ tin cậy trung bình"),
    "thap": ("🔴", "Độ tin cậy thấp"),
    "khong_do_duoc": ("⚪", "Chưa đo được độ tin cậy"),
}

# Ba thang điểm hoàn toàn khác nhau - phải gọi đúng tên, nếu không người đọc sẽ so sánh
# điểm 0.87 của lượt này với 0.87 của lượt khác trong khi chúng không cùng đơn vị.
_SCORE_LABEL = {
    "rerank": "điểm liên quan",
    "hybrid_rrf": "điểm trộn RRF",
    "cosine": "điểm cosine",
}

_STATUS_LABEL = {
    "COMPLETED": "✅ Hoàn tất",
    "PROCESSING": "⏳ Đang xử lý",
    "PENDING": "⏳ Chờ xử lý",
    "FAILED": "❌ Lỗi",
}

# CSS tối thiểu, chỉ nhắm vào container có `key` (Streamlit gắn lớp .st-key-<key> - cách tuỳ biến
# được hỗ trợ chính thức) thay vì class nội bộ dễ đổi tên giữa các phiên bản.
_CSS = """
<style>
/* Lịch sử trò chuyện + câu hỏi gợi ý là DANH SÁCH, nên căn trái thay vì căn giữa như nút bấm */
.st-key-sessions button, .st-key-suggestions button { justify-content: flex-start; text-align: left; }
/* Câu hỏi gợi ý hiện đủ nội dung (xuống dòng) thay vì bị cắt "…" giữa chừng */
.st-key-suggestions button div, .st-key-suggestions button p {
    white-space: normal; overflow: visible; text-overflow: clip; text-align: left;
}
</style>
"""


# ---------------------------------------------------------------------------
# Gọi backend
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_providers() -> list[dict]:
    """Danh sách provider ĐÃ có API key trong .env của backend, kèm model khả dụng.
    Cache 30s để không gọi backend liên tục mỗi lần Streamlit rerun."""
    resp = requests.get(f"{API_BASE}/api/v1/config/providers", timeout=10)
    resp.raise_for_status()
    return resp.json()["providers"]


@st.cache_data(ttl=30)
def fetch_suggestions(category: str) -> dict:
    """FAQ (backend/faq.json) + câu hay được hỏi (từ query_logs), đã lọc theo phạm vi tài liệu."""
    resp = requests.get(f"{API_BASE}/api/v1/chat/suggestions", params={"category": category}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_get(path: str, **params):
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_session() -> str | None:
    try:
        resp = requests.post(f"{API_BASE}/api/v1/chat/sessions", timeout=15)
        resp.raise_for_status()
        return resp.json()["id"]
    except requests.exceptions.RequestException:
        # Không tạo được hội thoại thì vẫn cho hỏi bình thường, chỉ là lượt này không được lưu.
        return None


def load_session(session_id: str) -> None:
    """Mở lại một hội thoại cũ từ DB vào khung chat."""
    try:
        messages = api_get(f"/api/v1/chat/sessions/{session_id}/messages")
    except requests.exceptions.RequestException as exc:
        st.error(f"Không tải được cuộc trò chuyện: {exc}")
        return
    st.session_state.session_id = session_id
    st.session_state.messages = [
        {
            "role": m["role"],
            "content": m["content"],
            "citations": m.get("citations") or [],
            "quality": m.get("quality"),
            "query_log_id": m.get("query_log_id"),
        }
        for m in messages
    ]


def send_feedback(query_log_id: str, rating: int) -> None:
    try:
        resp = requests.post(
            f"{API_BASE}/api/v1/chat/feedback",
            json={"query_log_id": query_log_id, "rating": rating},
            timeout=15,
        )
        resp.raise_for_status()
        st.session_state.rated.add(query_log_id)
        st.toast("Đã ghi nhận phản hồi, cảm ơn bạn.", icon=":material/check_circle:")
    except requests.exceptions.RequestException as exc:
        st.toast(f"Không gửi được phản hồi: {exc}", icon=":material/error:")


def format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ---------------------------------------------------------------------------
# Hiển thị một lượt hội thoại
# ---------------------------------------------------------------------------
def _on_feedback(query_log_id: str) -> None:
    value = st.session_state.get(f"fb_{query_log_id}")
    if value is not None:  # st.feedback "thumbs": 1 = 👍, 0 = 👎, None = bỏ chọn
        send_feedback(query_log_id, 1 if value == 1 else -1)


def render_feedback(query_log_id: str | None) -> None:
    """👍/👎 - nguồn dữ liệu duy nhất cho biết câu trả lời có thật sự dùng được hay không."""
    if not query_log_id:
        return
    st.feedback(
        "thumbs",
        key=f"fb_{query_log_id}",
        on_change=_on_feedback,
        args=(query_log_id,),
        disabled=query_log_id in st.session_state.rated,
    )


def render_answer(content: str, quality: dict | None) -> None:
    if content.startswith("❌"):
        st.error(content.removeprefix("❌").strip(), icon=":material/error:")
    elif (quality or {}).get("fallback"):
        # Câu báo hết quota (đoạn đầu tiên) tách thành khung cảnh báo riêng, để không lẫn với
        # nội dung tài liệu gốc bên dưới - người đọc phải thấy ngay đây KHÔNG phải câu trả lời của LLM.
        notice, _, rest = content.partition("\n\n")
        st.warning(notice.removeprefix("⚠️").strip(), icon=":material/warning:")
        st.markdown(rest)
    else:
        st.markdown(content)


def _tech_details(quality: dict) -> str:
    bits = []
    if quality.get("total_ms"):
        bits.append(f"tổng {quality['total_ms'] / 1000:.1f}s")
    if quality.get("first_token_ms"):
        bits.append(f"chữ đầu tiên {quality['first_token_ms'] / 1000:.1f}s")
    for name, ms in (quality.get("latency_ms") or {}).items():
        bits.append(f"{name} {ms / 1000:.1f}s")
    tokens = quality.get("tokens") or {}
    if tokens.get("prompt_tokens") is not None:
        suffix = " (ước lượng)" if tokens.get("tokens_estimated") else ""
        bits.append(f"token {tokens['prompt_tokens']}+{tokens.get('completion_tokens') or 0}{suffix}")
    return " · ".join(bits)


def render_details(quality: dict | None, citations: list[dict]) -> None:
    """Dưới mỗi câu trả lời: cảnh báo quan trọng + độ tin cậy luôn hiện; nguồn và số liệu kỹ thuật
    (thời gian từng khâu, token, truy vấn đã viết lại) gom vào một expander cho gọn."""
    if quality and quality.get("invalid_citations"):
        st.warning(
            f"Câu trả lời có trích dẫn trỏ tới nguồn không tồn tại: {quality['invalid_citations']}. "
            "Hãy kiểm chứng lại nội dung trước khi sử dụng.",
            icon=":material/report:",
        )
    if quality:
        icon, label = _CONFIDENCE_BADGE.get(quality.get("confidence"), ("⚪", "Không rõ"))
        st.caption(f"{icon} **{label}** — {quality.get('reason', '')}")

    tech = _tech_details(quality) if quality else ""
    rewritten = (quality or {}).get("rewritten_query")
    if not citations and not tech and not rewritten:
        return

    used = sum(1 for c in citations if c.get("used"))
    label = f"Nguồn tham khảo · {used}/{len(citations)} được trích dẫn" if citations else "Chi tiết xử lý"
    with st.expander(label, icon=":material/attach_file:" if citations else ":material/info:"):
        for c in citations:
            mark = "✅" if c.get("used") else "▫️"
            score_label = _SCORE_LABEL.get(c.get("score_type", ""), "điểm")
            page = c.get("page_number") or "—"
            st.markdown(f"{mark} **[{c.get('index', '?')}] {c['title']}** — trang {page} ({score_label}: {c['score']})")
            st.caption(c["snippet"])
        if rewritten or tech:
            if citations:
                st.divider()
            if rewritten:
                st.caption(f"🧭 Truy vấn thực tế dùng để tìm kiếm: _{rewritten}_")
            if tech:
                st.caption(f"⏱️ {tech}")


def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        render_answer(msg["content"], msg.get("quality"))
        render_details(msg.get("quality"), msg.get("citations") or [])
        render_feedback(msg.get("query_log_id"))


def render_welcome() -> None:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(
            "Xin chào 👋 Mình trả lời **dựa trên tài liệu nội bộ đã nạp**, kèm trích dẫn nguồn [n] để bạn kiểm chứng.\n\n"
            "Hãy đặt câu hỏi ở ô bên dưới, hoặc chọn một câu gợi ý ở cột bên phải."
        )


def stream_answer(question: str, cfg: dict) -> None:
    """Gửi câu hỏi, stream câu trả lời vào khung chat, rồi rerun: lượt vừa hỏi được vẽ lại y hệt
    các lượt cũ, và danh sách hội thoại ở sidebar cập nhật luôn tiêu đề của cuộc trò chuyện mới."""
    # Lấy lịch sử hội thoại TRƯỚC khi thêm câu hỏi mới vào session_state, để làm
    # ngữ cảnh multi-turn gửi cho LLM (chỉ lấy role/content, bỏ citations).
    # Lưu ý: KHÔNG được xoá hẳn tin nhắn lỗi khỏi danh sách - Anthropic API bắt buộc
    # user/assistant phải xen kẽ đúng thứ tự, xoá 1 tin sẽ làm lệch thứ tự và gây lỗi 400.
    # Thay vào đó thay nội dung lỗi bằng 1 placeholder ngắn để giữ đúng số lượt.
    # Lượt LLM hết quota cũng thay như vậy: đó là đoạn tài liệu thô chứ không phải câu trả lời,
    # gửi lại làm lịch sử chỉ tốn token và khiến LLM tưởng mình từng trả lời như thế.
    history_payload = [
        {
            "role": m["role"],
            "content": ERROR_PLACEHOLDER
            if m["content"].startswith("❌") or (m.get("quality") or {}).get("fallback")
            else m["content"],
        }
        for m in st.session_state.messages
    ]

    # Tạo hội thoại ngay ở câu hỏi đầu tiên để backend lưu được cả lượt này.
    if st.session_state.session_id is None:
        st.session_state.session_id = create_session()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    reply = {"role": "assistant", "content": "", "citations": [], "quality": None, "query_log_id": None}
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        # Khung trạng thái kiểu ChatGPT/Claude: hiện "Đang tìm tài liệu..." -> "Đã tìm
        # thấy N đoạn, đang soạn..." trước khi có token đầu tiên, để không bị màn hình trắng.
        status_box = st.status("Đang xử lý câu hỏi...", state="running")
        placeholder = st.empty()
        first_token = True
        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/chat/completions",
                json={
                    "message": question,
                    "provider": cfg["provider"],
                    "model": cfg["model"],
                    "top_k": cfg["top_k"],
                    "use_rerank": cfg["use_rerank"],
                    "category": cfg["category"],
                    "history": history_payload,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            event_name = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = json.loads(line.split(":", 1)[1].strip())
                if event_name == "status":
                    status_box.update(label=data["message"], state="running")
                elif event_name == "token":
                    if first_token:
                        status_box.update(label="Đã có câu trả lời", state="complete")
                        first_token = False
                    reply["content"] += data["text"]
                    placeholder.markdown(reply["content"] + "▌")
                elif event_name == "fallback":  # LLM hết quota -> đoạn tài liệu gốc
                    status_box.update(label="LLM hết quota — hiển thị tài liệu gốc", state="error")
                    reply["content"] = data["text"]
                elif event_name == "citations":
                    reply["citations"] = data["citations"]
                elif event_name == "quality":
                    reply["quality"] = data
                elif event_name == "done":
                    reply["query_log_id"] = data.get("query_log_id")
                elif event_name == "error":
                    status_box.update(label="Có lỗi xảy ra", state="error")
                    reply["content"] = f"❌ Lỗi từ LLM provider: {data['message']}"
        except requests.exceptions.RequestException as exc:
            reply["content"] = f"❌ Lỗi kết nối tới backend: {exc}"

    st.session_state.messages.append(reply)
    st.rerun()


# ---------------------------------------------------------------------------
# Cột câu hỏi gợi ý
# ---------------------------------------------------------------------------
def queue_question(question: str) -> None:
    """on_click của nút câu hỏi gợi ý - chạy TRƯỚC lần rerun kế tiếp, nên câu hỏi được xử lý ngay trong
    lần rerun đó y như vừa gõ vào ô chat."""
    st.session_state.pending_question = question


def render_suggestions(category: str, enabled: bool) -> None:
    with st.container(border=True, key="suggestions"):
        try:
            data = fetch_suggestions(category)
        except requests.exceptions.RequestException as exc:
            st.caption(f"⚠️ Không tải được câu hỏi gợi ý: {exc}")
            return

        st.markdown("**💡 Câu hỏi thường gặp**")
        if data.get("faq_error"):
            st.caption(f"⚠️ {data['faq_error']}")
        elif not data["faq"]:
            st.caption("Chưa có câu hỏi gợi ý cho phạm vi tài liệu này.")
        group = None
        for i, item in enumerate(data["faq"]):
            if item.get("group") and item["group"] != group:  # tiêu đề nhóm theo đúng thứ tự trong faq.json
                group = item["group"]
                st.caption(group)
            q = item["question"]
            st.button(q, key=f"faq_{i}", on_click=queue_question, args=(q,), disabled=not enabled, width="stretch")

        st.divider()
        st.markdown("**🔥 Hay được hỏi**")
        if not data["popular"]:
            st.caption(
                "Chưa đủ dữ liệu. Câu được hỏi ở nhiều cuộc trò chuyện và được trả lời tốt sẽ tự xuất hiện ở đây."
            )
        for i, item in enumerate(data["popular"]):
            q = item["question"]
            st.button(
                q,
                key=f"pop_{i}",
                on_click=queue_question,
                args=(q,),
                disabled=not enabled,
                width="stretch",
                help=f"Đã được hỏi ở {item['asked_in_sessions']} cuộc trò chuyện",
            )


# ---------------------------------------------------------------------------
# Sidebar trang Hỏi đáp: cấu hình trả lời + lịch sử trò chuyện
# ---------------------------------------------------------------------------
def render_session_list() -> None:
    try:
        sessions = api_get("/api/v1/chat/sessions", limit=30)
    except requests.exceptions.RequestException:
        st.caption("⚠️ Không tải được danh sách hội thoại.")
        return

    current = st.session_state.session_id
    # Hội thoại rỗng (đã tạo nhưng lượt đầu lỗi nên không lưu được gì) chỉ làm rối danh sách.
    sessions = [s for s in sessions if s["message_count"] or s["id"] == current]
    if not sessions:
        st.caption("Chưa có cuộc trò chuyện nào.")
        return

    with st.container(key="sessions"):
        for s in sessions:
            is_current = s["id"] == current
            open_col, del_col = st.columns([6, 1], vertical_alignment="center", gap="small")
            if open_col.button(
                s["title"],
                key=f"open_{s['id']}",
                type="secondary" if is_current else "tertiary",  # cuộc đang mở có viền, còn lại dạng danh sách
                width="stretch",
                help=s["title"],
            ):
                load_session(s["id"])
                st.rerun()
            if del_col.button(":material/delete:", key=f"delses_{s['id']}", type="tertiary", help="Xoá cuộc trò chuyện này"):
                try:
                    requests.delete(f"{API_BASE}/api/v1/chat/sessions/{s['id']}", timeout=15).raise_for_status()
                except requests.exceptions.RequestException as exc:
                    st.error(f"Không xoá được: {exc}")
                else:
                    if is_current:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                    st.rerun()


def render_chat_sidebar() -> dict:
    """Trả về cấu hình đang chọn: provider/model/phạm vi/rerank/top_k."""
    with st.sidebar:
        head, refresh = st.columns([5, 1], vertical_alignment="center")
        head.subheader("⚙️ Cấu hình", anchor=False)
        if refresh.button(
            ":material/refresh:",
            type="tertiary",
            help="Tải lại danh sách Provider/Model từ backend (sau khi sửa .env + restart backend)",
        ):
            fetch_providers.clear()

        try:
            providers = fetch_providers()
        except requests.exceptions.RequestException as exc:
            providers = []
            st.error(f"Không kết nối được backend ({API_BASE}): {exc}", icon=":material/cloud_off:")

        provider, model, provider_label = None, None, ""
        if not providers:
            st.warning(
                "Chưa có provider nào được cấu hình.\n\n"
                "Thêm API key vào file `.env` ở thư mục gốc dự án "
                "(vd: `ANTHROPIC_API_KEY=...` hoặc `GOOGLE_API_KEY=...`), "
                "rồi khởi động lại backend."
            )
        else:
            labels = {p["id"]: p["label"] for p in providers}
            provider = st.selectbox("Provider", list(labels), format_func=labels.get)
            provider_label = labels[provider]
            model = st.selectbox("Model", next(p for p in providers if p["id"] == provider)["models"])

        category = st.selectbox("Phạm vi tài liệu", list(CATEGORY_LABELS), format_func=CATEGORY_LABELS.get)

        # Rerank/Top-K ít khi cần đổi - gom lại để phần cấu hình chính gọn, không đẩy lịch sử trò chuyện xuống.
        with st.expander("Tuỳ chọn truy hồi nâng cao"):
            use_rerank = st.toggle(
                "Bật Reranker",
                value=True,
                help="Chấm lại các đoạn tìm được bằng model cross-encoder và chỉ gửi cho LLM những đoạn thật sự liên quan.",
            )
            top_k = st.slider(
                "Top-K: số đoạn tối đa gửi cho LLM",
                min_value=1,
                max_value=30,
                value=12,
                help=(
                    "Khi bật Reranker: chỉ gửi các đoạn đạt ngưỡng liên quan (có thể ít hơn K). "
                    "Nếu không đoạn nào đạt, gửi K đoạn điểm cao nhất. Khi tắt: luôn gửi đúng K đoạn."
                ),
            )

        st.divider()
        st.subheader("💬 Trò chuyện", anchor=False)
        if st.button("Cuộc trò chuyện mới", icon=":material/add:", width="stretch"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
        render_session_list()

        st.divider()
        st.caption(f"Backend: {API_BASE}")

    return {
        "provider": provider,
        "provider_label": provider_label,
        "model": model,
        "category": category,
        "use_rerank": use_rerank,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Trang 1: Hỏi đáp Chatbot & Citation
# ---------------------------------------------------------------------------
def chat_page() -> None:
    cfg = render_chat_sidebar()

    # st.chat_input CHỈ tự ghim cố định xuống đáy màn hình khi được gọi ở thân trang (ngoài mọi
    # container/cột) - đặt trong cột thì nó sẽ nằm giữa dòng thay vì cố định.
    question = st.chat_input("Đặt câu hỏi về tài liệu nội bộ...", disabled=cfg["provider"] is None)
    question = question or st.session_state.pop("pending_question", None)  # câu chọn từ cột gợi ý

    chat_col, suggest_col = st.columns([7, 3], gap="large")
    # Vẽ cột gợi ý TRƯỚC khung chat: script dừng ở vòng streaming câu trả lời, vẽ sau thì
    # cột này chỉ hiện ra khi đã trả lời xong.
    with suggest_col:
        render_suggestions(cfg["category"], enabled=cfg["provider"] is not None)

    with chat_col:
        st.subheader("Trợ lý tài liệu nội bộ", anchor=False)
        if cfg["provider"]:
            # Luôn cho thấy câu trả lời SẮP TỚI dùng model nào - lịch sử có thể chứa lượt của model khác.
            st.caption(
                f"{cfg['provider_label']} · **{cfg['model']}** · Phạm vi: {CATEGORY_LABELS[cfg['category']]} · "
                f"Reranker {'bật' if cfg['use_rerank'] else 'tắt'} · Top-K {cfg['top_k']}"
            )

        if not st.session_state.messages and not question:
            render_welcome()
        for msg in st.session_state.messages:
            render_message(msg)

        if question:
            if cfg["provider"] is None:
                st.warning("Chưa có provider nào được cấu hình trong .env - xem hướng dẫn ở thanh bên trái.")
            else:
                stream_answer(question, cfg)


# ---------------------------------------------------------------------------
# Trang 2: Nạp dữ liệu (Ingestion Dashboard, rút gọn)
# ---------------------------------------------------------------------------
def upload_page() -> None:
    st.header("📤 Nạp tài liệu", anchor=False)
    st.caption(
        "Hỗ trợ PDF (kể cả bản scan), DOCX, PPTX, Markdown, HTML, TXT và ảnh. "
        "Ảnh và PDF scan được OCR (vie+eng) nên xử lý lâu hơn."
    )

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Chọn file",
            type=["pdf", "docx", "pptx", "md", "html", "htm", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            help=(
                "Ảnh (png/jpg/jpeg/bmp/tiff/webp) sẽ được OCR toàn bộ bằng Tesseract (vie+eng). "
                "HTML (trang wiki nội bộ) sẽ tự loại bỏ menu/footer, giữ nội dung theo cấu trúc heading. "
                "PPTX: mỗi slide 1 trang, gồm text/bảng, OCR ảnh trong slide, dữ liệu chart thật, ghi chú thuyết trình."
            ),
        )
        cat_col, public_col = st.columns(2, vertical_alignment="bottom")
        up_category = cat_col.selectbox("Phân loại tài liệu", DOC_CATEGORIES, format_func=CATEGORY_LABELS.get)
        is_public = public_col.toggle("Tài liệu công khai", value=True, help="Trường is_public của tài liệu.")
        submitted = st.form_submit_button("Nạp tài liệu", type="primary", icon=":material/upload:")

    if not submitted:
        return
    if uploaded_file is None:
        st.warning("Hãy chọn một file trước khi nạp.")
        return
    with st.spinner("Đang trích xuất, phân mảnh và tạo embedding... (ảnh/PDF scan cần OCR nên có thể chậm hơn)"):
        try:
            mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
            data = {"category": up_category, "is_public": str(is_public)}
            resp = requests.post(f"{API_BASE}/api/v1/documents/upload", files=files, data=data, timeout=300)
            if resp.status_code == 409:  # file trùng nội dung với tài liệu đã nạp
                st.warning(resp.json()["detail"])
            else:
                resp.raise_for_status()
                result = resp.json()["document"]
                st.success(f"Đã nạp '{result['title']}' — {result['chunk_count']} chunks.", icon=":material/check_circle:")
        except requests.exceptions.RequestException as exc:
            st.error(f"Lỗi: {exc}")


# ---------------------------------------------------------------------------
# Trang 3: Danh sách tài liệu
# ---------------------------------------------------------------------------
@st.dialog("Xác nhận xóa tài liệu")
def confirm_delete_dialog(doc: dict) -> None:
    """Hộp thoại xác nhận trước khi xóa, vì xóa là vĩnh viễn (file gốc, DB, toàn bộ chunk trên Qdrant)."""
    st.markdown(f"Xóa **{doc['title']}** ({doc['chunk_count']} chunks, nạp lúc {doc['created_at'][:19]})?")
    st.warning("File gốc, bản ghi và toàn bộ chunk trên Qdrant sẽ bị xóa vĩnh viễn, không thể khôi phục.")
    delete_col, cancel_col = st.columns(2)
    if delete_col.button("Xóa vĩnh viễn", type="primary", icon=":material/delete_forever:", width="stretch"):
        try:
            resp = requests.delete(f"{API_BASE}/api/v1/documents/{doc['id']}", timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            st.error(f"Không xóa được: {exc}")
            return
        # st.rerun() đóng dialog - lưu thông báo vào session_state để hiện sau khi tải lại
        st.session_state.flash = f"Đã xóa '{doc['title']}' ({doc['id'][:8]})."
        st.rerun()
    if cancel_col.button("Hủy", width="stretch"):
        st.rerun()


def documents_page() -> None:
    head, refresh = st.columns([5, 1], vertical_alignment="bottom")
    head.header("📄 Tài liệu đã nạp", anchor=False)
    if refresh.button("Tải lại", icon=":material/refresh:", width="stretch"):
        st.rerun()
    if flash := st.session_state.pop("flash", None):
        st.success(flash, icon=":material/check_circle:")

    try:
        docs = api_get("/api/v1/documents")
    except requests.exceptions.RequestException as exc:
        st.error(f"Không thể tải danh sách tài liệu: {exc}")
        return
    if not docs:
        st.info("Chưa có tài liệu nào được nạp. Vào trang **Nạp tài liệu** để thêm.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Số tài liệu", len(docs), border=True)
    m2.metric("Tổng số đoạn (chunks)", f"{sum(d['chunk_count'] for d in docs):,}", border=True)
    m3.metric("Dung lượng", format_size(sum(d["file_size_bytes"] for d in docs)), border=True)

    st.caption("Chọn một dòng trong bảng để xoá tài liệu đó.")
    event = st.dataframe(
        [
            {
                "Tên tài liệu": d["title"],
                "Phân loại": CATEGORY_LABELS.get(d["category"], d["category"]),
                "Trạng thái": _STATUS_LABEL.get(d["status"], d["status"]),
                "Số đoạn": d["chunk_count"],
                "Dung lượng": format_size(d["file_size_bytes"]),
                "Ngày nạp": d["created_at"][:16].replace("T", " "),
                "ID": d["id"][:8],  # phân biệt các tài liệu trùng tên
            }
            for d in docs
        ],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="docs_table",
    )
    selected = event.selection.rows
    if st.button("Xoá tài liệu đã chọn", icon=":material/delete:", disabled=not selected):
        confirm_delete_dialog(docs[selected[0]])


# ---------------------------------------------------------------------------
# Trang 4: Giám sát vận hành
# ---------------------------------------------------------------------------
def monitor_page() -> None:
    head, period_col = st.columns([3, 1], vertical_alignment="bottom")
    head.header("📊 Giám sát vận hành", anchor=False)
    period = period_col.selectbox(
        "Khoảng thời gian", [1, 7, 30, 90], index=1, format_func=lambda d: f"{d} ngày gần nhất"
    )
    st.caption(
        "Số liệu lấy từ bảng query_logs - mỗi lượt hỏi ghi một dòng, kể cả lượt hỏng. "
        "Dùng để biết hệ thống đang chậm ở khâu nào và câu hỏi nào đang bó tay."
    )

    try:
        data = api_get("/api/v1/admin/stats", days=period)
    except requests.exceptions.RequestException as exc:
        st.error(f"Không tải được thống kê: {exc}")
        data = None

    if data and not data["total_queries"]:
        st.info("Chưa có lượt hỏi nào trong khoảng thời gian này.")
    elif data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng lượt hỏi", data["total_queries"], border=True)
        c2.metric(
            "Không trả lời được",
            data["no_answer_count"],
            delta=f"{data['no_answer_count'] / data['total_queries'] * 100:.0f}%",
            delta_color="inverse",
            border=True,
        )
        c3.metric("👎 Phản hồi tiêu cực", data["negative_feedback"], border=True)
        c4.metric("Lỗi provider", data["error_count"], border=True)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Thời gian TB", f"{data['avg_total_ms'] / 1000:.1f}s", border=True)
        c6.metric("P95", f"{data['p95_total_ms'] / 1000:.1f}s", border=True)
        c7.metric("Chữ đầu tiên (TB)", f"{data['avg_first_token_ms'] / 1000:.1f}s", border=True)
        c8.metric("👍 Phản hồi tích cực", data["positive_feedback"], border=True)

        left, right = st.columns(2)
        with left:
            st.subheader("⏱️ Thời gian trung bình từng khâu", anchor=False)
            if data["stage_avg_ms"]:
                st.bar_chart({"ms": data["stage_avg_ms"]})
            else:
                st.caption("Chưa có dữ liệu.")
        with right:
            st.subheader("🎯 Phân bố độ tin cậy", anchor=False)
            if data["by_confidence"]:
                st.bar_chart({"số lượt": data["by_confidence"]})
            else:
                st.caption("Chưa có dữ liệu.")

        st.subheader("🔢 Token đã dùng", anchor=False)
        t1, t2, t3 = st.columns(3)
        t1.metric("Token đầu vào", f"{data['prompt_tokens']:,}", border=True)
        t2.metric("Token đầu ra", f"{data['completion_tokens']:,}", border=True)
        t3.metric("Tỉ lệ là ước lượng", f"{data['estimated_token_share'] * 100:.0f}%", border=True)
        if data["estimated_token_share"]:
            st.caption(
                "⚠️ Phần ước lượng được tính bằng tiktoken vì provider không trả số liệu - "
                "dùng để theo dõi xu hướng, KHÔNG dùng để đối soát hoá đơn."
            )

    st.divider()
    st.subheader("🕳️ Câu hỏi hệ thống bó tay (gom theo nội dung)", anchor=False)
    st.caption("Câu nào lặp lại nhiều lần chính là khoảng trống tài liệu nên bổ sung trước.")
    try:
        gaps = api_get("/api/v1/admin/unanswered", days=period, limit=30)
        if gaps:
            st.dataframe(gaps, hide_index=True, width="stretch")
        else:
            st.success("Không có câu hỏi nào bị bỏ trống.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Không tải được danh sách: {exc}")

    st.divider()
    st.subheader("🔍 Các lượt hỏi cần xem lại", anchor=False)
    only_problems = st.toggle("Chỉ hiện lượt có vấn đề", value=True)
    try:
        rows = api_get("/api/v1/admin/queries", only_problems=only_problems, days=period, limit=100)
        if not rows:
            st.success("Không có lượt hỏi nào cần xem lại.")
        else:
            st.dataframe(
                [
                    {
                        "Lúc": r["created_at"][:19].replace("T", " "),
                        "Câu hỏi": r["question"][:80],
                        "Viết lại": (r["rewritten_query"] or "")[:60],
                        "Hits": r["n_hits"],
                        "Điểm cao nhất": round(r["top_score"], 3) if r["top_score"] is not None else None,
                        "Tin cậy": r["confidence"],
                        "Trích dẫn hỏng": r["invalid_citations"],
                        "Bó tay": r["no_answer"],
                        "Phản hồi": {1: "👍", -1: "👎"}.get(r["feedback"], ""),
                        "Tổng (s)": round(r["total_ms"] / 1000, 1),
                        "Lỗi": (r["error"] or "")[:60],
                    }
                    for r in rows
                ],
                hide_index=True,
                width="stretch",
            )
    except requests.exceptions.RequestException as exc:
        st.error(f"Không tải được danh sách: {exc}")


# ---------------------------------------------------------------------------
# Khởi tạo + điều hướng
# ---------------------------------------------------------------------------
# Khởi tạo state trước mọi thao tác đọc, vì nhiều trang cùng dùng tới.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("rated", set())

st.markdown(_CSS, unsafe_allow_html=True)

st.navigation(
    [
        st.Page(chat_page, title="Hỏi đáp", icon=":material/chat:", default=True),
        st.Page(upload_page, title="Nạp tài liệu", icon=":material/upload_file:", url_path="upload"),
        st.Page(documents_page, title="Tài liệu", icon=":material/description:", url_path="documents"),
        st.Page(monitor_page, title="Giám sát", icon=":material/monitoring:", url_path="monitor"),
    ],
    position="top",
).run()
