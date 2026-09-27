"""
Giao diện Streamlit MVP - tương ứng mục 6.1 của tài liệu thiết kế, rút gọn còn
các phần cốt lõi cho MVP: Chat & Citations, Nạp dữ liệu, Danh sách tài liệu.

API key KHÔNG nhập trên UI - chỉ khai báo trong file .env của backend (xem README).
Sidebar chỉ hiển thị Provider/Model đã có key cấu hình sẵn (lấy từ
GET /api/v1/config/providers), cùng Top-K và bật/tắt Reranker.
"""

import json
import mimetypes
import os

import requests
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot Nội bộ", page_icon="🤖", layout="wide")


@st.cache_data(ttl=30)
def fetch_providers() -> list[dict]:
    """Danh sách provider ĐÃ có API key trong .env của backend, kèm model khả dụng.
    Cache 30s để không gọi backend liên tục mỗi lần Streamlit rerun."""
    resp = requests.get(f"{API_BASE}/api/v1/config/providers", timeout=10)
    resp.raise_for_status()
    return resp.json()["providers"]


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
    except requests.exceptions.RequestException as exc:
        st.error(f"Không gửi được phản hồi: {exc}")


def render_feedback(query_log_id: str | None) -> None:
    """Nút 👍/👎 - nguồn dữ liệu duy nhất cho biết câu trả lời có thật sự dùng được hay không."""
    if not query_log_id:
        return
    if query_log_id in st.session_state.rated:
        st.caption("✔️ Đã ghi nhận phản hồi, cảm ơn bạn.")
        return
    up, down, _ = st.columns([1, 1, 10])
    if up.button("👍", key=f"up_{query_log_id}", help="Câu trả lời này hữu ích"):
        send_feedback(query_log_id, 1)
        st.rerun()
    if down.button("👎", key=f"down_{query_log_id}", help="Câu trả lời này chưa dùng được"):
        send_feedback(query_log_id, -1)
        st.rerun()


def render_quality(quality: dict | None) -> None:
    """Hiển thị chỉ báo độ tin cậy + cảnh báo trích dẫn hỏng cho một lượt trả lời."""
    if not quality:
        return
    icon, label = _CONFIDENCE_BADGE.get(quality.get("confidence"), ("⚪", "Không rõ"))
    st.caption(f"{icon} **{label}** — {quality.get('reason', '')}")
    if quality.get("invalid_citations"):
        st.warning(
            f"⚠️ Câu trả lời có trích dẫn trỏ tới nguồn không tồn tại: "
            f"{quality['invalid_citations']}. Hãy kiểm chứng lại nội dung trước khi sử dụng."
        )
    if quality.get("rewritten_query"):
        st.caption(f"🧭 Truy vấn thực tế dùng để tìm kiếm: _{quality['rewritten_query']}_")

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
    if bits:
        st.caption("⏱️ " + " · ".join(bits))


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    used = sum(1 for c in citations if c.get("used"))
    with st.expander(f"📎 {len(citations)} nguồn đã gửi cho LLM — {used} nguồn được trích dẫn"):
        for c in citations:
            mark = "✅" if c.get("used") else "▫️"
            score_label = _SCORE_LABEL.get(c.get("score_type", ""), "điểm")
            index = c.get("index", "?")
            page = c.get("page_number") or "—"
            st.markdown(f"{mark} **[{index}] {c['title']}** — trang {page} ({score_label}: {c['score']})")
            st.caption(c["snippet"])


@st.dialog("Xác nhận xóa tài liệu")
def confirm_delete_dialog(doc: dict) -> None:
    """Hộp thoại xác nhận trước khi xóa, vì xóa là vĩnh viễn (file gốc, DB, toàn bộ chunk trên Qdrant)."""
    st.markdown(f"Xóa **{doc['title']}** ({doc['chunk_count']} chunks, nạp lúc {doc['created_at'][:19]})?")
    st.warning("File gốc, bản ghi và toàn bộ chunk trên Qdrant sẽ bị xóa vĩnh viễn, không thể khôi phục.")
    delete_col, cancel_col = st.columns(2)
    if delete_col.button("🗑️ Xóa vĩnh viễn", type="primary"):
        try:
            resp = requests.delete(f"{API_BASE}/api/v1/documents/{doc['id']}", timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            st.error(f"❌ Không xóa được: {exc}")
            return
        # st.rerun() đóng dialog - lưu thông báo vào session_state để hiện sau khi tải lại
        st.session_state.flash = f"🗑️ Đã xóa '{doc['title']}' ({doc['id'][:8]})."
        st.rerun()
    if cancel_col.button("Hủy"):
        st.rerun()


# Khởi tạo state trước mọi thao tác đọc, vì sidebar và tab đều dùng tới.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("rated", set())


# ---------- Sidebar: chọn Provider/Model đã cấu hình sẵn trong .env + Top-K ----------
with st.sidebar:
    header_col, refresh_col = st.columns([4, 1])
    header_col.header("⚙️ Cấu hình Model")
    if refresh_col.button("🔄", help="Tải lại danh sách Provider/Model từ backend (sau khi sửa .env + restart backend)"):
        fetch_providers.clear()

    try:
        providers = fetch_providers()
    except requests.exceptions.RequestException as exc:
        providers = []
        st.error(f"❌ Không kết nối được backend ({API_BASE}): {exc}")

    if not providers:
        st.warning(
            "⚠️ Chưa có provider nào được cấu hình.\n\n"
            "Thêm API key vào file `.env` ở thư mục gốc dự án "
            "(vd: `ANTHROPIC_API_KEY=...` hoặc `GOOGLE_API_KEY=...`), "
            "rồi khởi động lại backend."
        )
        provider = None
        model = None
    else:
        provider_ids = [p["id"] for p in providers]
        provider_labels = {p["id"]: p["label"] for p in providers}
        provider = st.selectbox("Provider", provider_ids, format_func=lambda p: provider_labels[p])

        selected = next(p for p in providers if p["id"] == provider)
        model = st.selectbox("Model", selected["models"])

    use_rerank = st.checkbox(
        "⚖️ Bật Reranker",
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
    category = st.selectbox("Phạm vi tài liệu", ["ALL", "GENERAL", "LEGAL_PDF", "CONTRACT", "TECH_SPEC"])

    # ---------- Hội thoại đã lưu ----------
    st.divider()
    st.header("💬 Cuộc trò chuyện")
    if st.button("➕ Cuộc trò chuyện mới", width="stretch"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    try:
        saved_sessions = api_get("/api/v1/chat/sessions", limit=30)
    except requests.exceptions.RequestException:
        saved_sessions = []
        st.caption("⚠️ Không tải được danh sách hội thoại.")

    for s in saved_sessions:
        is_current = s["id"] == st.session_state.session_id
        open_col, del_col = st.columns([6, 1])
        label = f"{'▶ ' if is_current else ''}{s['title'][:38]}"
        if open_col.button(label, key=f"open_{s['id']}", width="stretch", help=s["title"]):
            load_session(s["id"])
            st.rerun()
        if del_col.button("🗑️", key=f"delses_{s['id']}", help="Xoá cuộc trò chuyện này"):
            try:
                requests.delete(f"{API_BASE}/api/v1/chat/sessions/{s['id']}", timeout=15).raise_for_status()
            except requests.exceptions.RequestException as exc:
                st.error(f"Không xoá được: {exc}")
            else:
                if is_current:
                    st.session_state.session_id = None
                    st.session_state.messages = []
                st.rerun()

    st.divider()
    st.caption(f"Backend: {API_BASE}")

tab_chat, tab_upload, tab_docs, tab_monitor = st.tabs(
    ["💬 Hỏi đáp", "📤 Nạp tài liệu", "📄 Danh sách tài liệu", "📊 Giám sát"]
)

# st.chat_input CHỈ tự ghim cố định xuống đáy màn hình khi được gọi ở cấp ngoài cùng
# của script - nếu đặt bên trong `with tab_chat:` (1 container) nó sẽ render inline
# (nằm giữa dòng) thay vì cố định. Vì vậy đặt ở đây, ngoài mọi tab/container.
question = st.chat_input("Đặt câu hỏi về tài liệu nội bộ...", disabled=provider is None)

# ---------- Tab 1: Hỏi đáp Chatbot & Citation ----------
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_quality(msg.get("quality"))
            render_citations(msg.get("citations") or [])
            if msg["role"] == "assistant":
                render_feedback(msg.get("query_log_id"))

    if question:
        if provider is None:
            st.warning("⚠️ Chưa có provider nào được cấu hình trong .env - xem hướng dẫn ở thanh bên trái.")
        else:
            # Lấy lịch sử hội thoại TRƯỚC khi thêm câu hỏi mới vào session_state, để làm
            # ngữ cảnh multi-turn gửi cho LLM (chỉ lấy role/content, bỏ citations).
            # Lưu ý: KHÔNG được xoá hẳn tin nhắn lỗi khỏi danh sách - Anthropic API bắt buộc
            # user/assistant phải xen kẽ đúng thứ tự, xoá 1 tin sẽ làm lệch thứ tự và gây lỗi 400.
            # Thay vào đó thay nội dung lỗi bằng 1 placeholder ngắn để giữ đúng số lượt.
            history_payload = [
                {
                    "role": m["role"],
                    "content": "(Không trả lời được do lỗi hệ thống ở lượt này.)"
                    if m["content"].startswith("❌")
                    else m["content"],
                }
                for m in st.session_state.messages
            ]

            # Tạo hội thoại ngay ở câu hỏi đầu tiên để backend lưu được cả lượt này.
            if st.session_state.session_id is None:
                st.session_state.session_id = create_session()

            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # Khung trạng thái kiểu ChatGPT/Claude: hiện "Đang tìm tài liệu..." -> "Đã tìm
                # thấy N đoạn, đang soạn..." trước khi có token đầu tiên, để không bị màn hình
                # trắng trong lúc chờ. Tự thu gọn khi có câu trả lời (state="complete").
                status_box = st.status("Đang xử lý câu hỏi...", state="running")
                placeholder = st.empty()
                full_text = ""
                citations = []
                quality = None
                query_log_id = None
                first_token = True
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/chat/completions",
                        json={
                            "message": question,
                            "provider": provider,
                            "model": model,
                            "top_k": top_k,
                            "use_rerank": use_rerank,
                            "category": category,
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
                        elif line.startswith("data:"):
                            data = json.loads(line.split(":", 1)[1].strip())
                            if event_name == "status":
                                status_box.update(label=data["message"], state="running")
                            elif event_name == "token":
                                if first_token:
                                    status_box.update(label="✅ Đã có câu trả lời", state="complete")
                                    first_token = False
                                full_text += data["text"]
                                placeholder.markdown(full_text + "▌")
                            elif event_name == "citations":
                                citations = data["citations"]
                            elif event_name == "quality":
                                quality = data
                            elif event_name == "done":
                                query_log_id = data.get("query_log_id")
                            elif event_name == "error":
                                status_box.update(label="❌ Có lỗi xảy ra", state="error")
                                full_text = f"❌ Lỗi từ LLM provider: {data['message']}"
                    placeholder.markdown(full_text)
                    render_quality(quality)
                    render_citations(citations)
                    render_feedback(query_log_id)
                except requests.exceptions.RequestException as exc:
                    status_box.update(label="❌ Mất kết nối backend", state="error")
                    full_text = f"❌ Lỗi kết nối tới backend: {exc}"
                    placeholder.markdown(full_text)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_text,
                    "citations": citations,
                    "quality": quality,
                    "query_log_id": query_log_id,
                }
            )

# ---------- Tab 2: Nạp dữ liệu (Ingestion Dashboard, rút gọn) ----------
with tab_upload:
    st.subheader("Nạp tài liệu vào hệ thống")
    uploaded_file = st.file_uploader(
        "Chọn file (PDF, DOCX, PPTX, Markdown, HTML, TXT, hoặc ảnh)",
        type=["pdf", "docx", "pptx", "md", "html", "htm", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help=(
            "Ảnh (png/jpg/jpeg/bmp/tiff/webp) sẽ được OCR toàn bộ bằng Tesseract (vie+eng). "
            "HTML (trang wiki nội bộ) sẽ tự loại bỏ menu/footer, giữ nội dung theo cấu trúc heading. "
            "PPTX: mỗi slide 1 trang, gồm text/bảng, OCR ảnh trong slide, dữ liệu chart thật, ghi chú thuyết trình."
        ),
    )
    up_category = st.selectbox("Phân loại tài liệu", ["GENERAL", "LEGAL_PDF", "CONTRACT", "TECH_SPEC"], key="up_cat")
    is_public = st.checkbox("Tài liệu công khai (is_public)", value=True)

    if st.button("🚀 Nạp tài liệu", disabled=uploaded_file is None):
        with st.spinner("Đang trích xuất, phân mảnh và tạo embedding... (ảnh/PDF scan cần OCR nên có thể chậm hơn)"):
            try:
                mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
                data = {"category": up_category, "is_public": str(is_public)}
                resp = requests.post(f"{API_BASE}/api/v1/documents/upload", files=files, data=data, timeout=300)
                if resp.status_code == 409:  # file trùng nội dung với tài liệu đã nạp
                    st.warning(f"⚠️ {resp.json()['detail']}")
                else:
                    resp.raise_for_status()
                    result = resp.json()["document"]
                    st.success(f"✅ Đã nạp '{result['title']}' — {result['chunk_count']} chunks.")
            except requests.exceptions.RequestException as exc:
                st.error(f"❌ Lỗi: {exc}")

# ---------- Tab 3: Danh sách tài liệu ----------
with tab_docs:
    st.subheader("Tài liệu đã nạp")
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    if st.button("🔄 Tải lại danh sách"):
        st.rerun()
    try:
        resp = requests.get(f"{API_BASE}/api/v1/documents", timeout=30)
        resp.raise_for_status()
        docs = resp.json()
        if not docs:
            st.info("Chưa có tài liệu nào được nạp.")
        else:
            st.dataframe(docs, use_container_width=True)

            st.markdown("##### 🗑️ Xóa tài liệu")
            # Kèm thời gian nạp + id rút gọn để phân biệt các tài liệu trùng tên
            doc_to_delete = st.selectbox(
                "Chọn tài liệu cần xóa",
                docs,
                format_func=lambda d: f"{d['title']} — {d['chunk_count']} chunks — {d['created_at'][:19]} ({d['id'][:8]})",
                index=None,
                placeholder="Chọn tài liệu...",
            )
            if st.button("🗑️ Xóa tài liệu", disabled=doc_to_delete is None):
                confirm_delete_dialog(doc_to_delete)
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ Không thể tải danh sách tài liệu: {exc}")

# ---------- Tab 4: Giám sát vận hành ----------
with tab_monitor:
    st.subheader("Giám sát vận hành")
    st.caption(
        "Số liệu lấy từ bảng query_logs - mỗi lượt hỏi ghi một dòng, kể cả lượt hỏng. "
        "Dùng để biết hệ thống đang chậm ở khâu nào và câu hỏi nào đang bó tay."
    )
    period = st.selectbox("Khoảng thời gian", [1, 7, 30, 90], index=1, format_func=lambda d: f"{d} ngày gần nhất")

    try:
        data = api_get("/api/v1/admin/stats", days=period)
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ Không tải được thống kê: {exc}")
        data = None

    if data:
        if not data["total_queries"]:
            st.info("Chưa có lượt hỏi nào trong khoảng thời gian này.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng lượt hỏi", data["total_queries"])
            c2.metric(
                "Không trả lời được",
                data["no_answer_count"],
                delta=f"{data['no_answer_count'] / data['total_queries'] * 100:.0f}%",
                delta_color="inverse",
            )
            c3.metric("👎 Phản hồi tiêu cực", data["negative_feedback"])
            c4.metric("Lỗi provider", data["error_count"])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Thời gian TB", f"{data['avg_total_ms'] / 1000:.1f}s")
            c6.metric("P95", f"{data['p95_total_ms'] / 1000:.1f}s")
            c7.metric("Chữ đầu tiên (TB)", f"{data['avg_first_token_ms'] / 1000:.1f}s")
            c8.metric("👍 Phản hồi tích cực", data["positive_feedback"])

            left, right = st.columns(2)
            with left:
                st.markdown("##### ⏱️ Thời gian trung bình từng khâu")
                if data["stage_avg_ms"]:
                    st.bar_chart({"ms": data["stage_avg_ms"]})
                else:
                    st.caption("Chưa có dữ liệu.")
            with right:
                st.markdown("##### 🎯 Phân bố độ tin cậy")
                if data["by_confidence"]:
                    st.bar_chart({"số lượt": data["by_confidence"]})
                else:
                    st.caption("Chưa có dữ liệu.")

            st.markdown("##### 🔢 Token đã dùng")
            t1, t2, t3 = st.columns(3)
            t1.metric("Token đầu vào", f"{data['prompt_tokens']:,}")
            t2.metric("Token đầu ra", f"{data['completion_tokens']:,}")
            t3.metric("Tỉ lệ là ước lượng", f"{data['estimated_token_share'] * 100:.0f}%")
            if data["estimated_token_share"]:
                st.caption(
                    "⚠️ Phần ước lượng được tính bằng tiktoken vì provider không trả số liệu - "
                    "dùng để theo dõi xu hướng, KHÔNG dùng để đối soát hoá đơn."
                )

    st.divider()
    st.markdown("##### 🕳️ Câu hỏi hệ thống bó tay (gom theo nội dung)")
    st.caption("Câu nào lặp lại nhiều lần chính là khoảng trống tài liệu nên bổ sung trước.")
    try:
        gaps = api_get("/api/v1/admin/unanswered", days=period, limit=30)
        if gaps:
            st.dataframe(gaps, width="stretch")
        else:
            st.success("Không có câu hỏi nào bị bỏ trống.")
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ Không tải được danh sách: {exc}")

    st.divider()
    st.markdown("##### 🔍 Các lượt hỏi cần xem lại")
    only_problems = st.checkbox("Chỉ hiện lượt có vấn đề", value=True)
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
                width="stretch",
            )
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ Không tải được danh sách: {exc}")
