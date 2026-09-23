"""
Giao diện Streamlit MVP - tương ứng mục 6.1 của tài liệu thiết kế, rút gọn còn
các phần cốt lõi cho MVP: Chat & Citations, Nạp dữ liệu, Danh sách tài liệu.

API key KHÔNG nhập trên UI - chỉ khai báo trong file .env của backend (xem README).
Sidebar chỉ hiển thị Provider/Model đã có key cấu hình sẵn (lấy từ
GET /api/v1/config/providers), cùng Top-K.
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

    top_k = st.slider("Top-K tài liệu truy hồi", min_value=1, max_value=10, value=5)
    category = st.selectbox("Phạm vi tài liệu", ["ALL", "GENERAL", "LEGAL_PDF", "CONTRACT", "TECH_SPEC"])
    st.divider()
    st.caption(f"Backend: {API_BASE}")

tab_chat, tab_upload, tab_docs = st.tabs(["💬 Hỏi đáp", "📤 Nạp tài liệu", "📄 Danh sách tài liệu"])

# st.chat_input CHỈ tự ghim cố định xuống đáy màn hình khi được gọi ở cấp ngoài cùng
# của script - nếu đặt bên trong `with tab_chat:` (1 container) nó sẽ render inline
# (nằm giữa dòng) thay vì cố định. Vì vậy đặt ở đây, ngoài mọi tab/container.
question = st.chat_input("Đặt câu hỏi về tài liệu nội bộ...", disabled=provider is None)

# ---------- Tab 1: Hỏi đáp Chatbot & Citation ----------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📎 {len(msg['citations'])} trích dẫn nguồn"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c['title']}** — trang {c.get('page_number', '—')} (score: {c['score']})")
                        st.caption(c["snippet"])

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

            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_text = ""
                citations = []
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/chat/completions",
                        json={
                            "message": question,
                            "provider": provider,
                            "model": model,
                            "top_k": top_k,
                            "category": category,
                            "history": history_payload,
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
                            if event_name == "token":
                                full_text += data["text"]
                                placeholder.markdown(full_text + "▌")
                            elif event_name == "citations":
                                citations = data["citations"]
                            elif event_name == "error":
                                full_text = f"❌ Lỗi từ LLM provider: {data['message']}"
                    placeholder.markdown(full_text)
                    if citations:
                        with st.expander(f"📎 {len(citations)} trích dẫn nguồn"):
                            for c in citations:
                                st.markdown(
                                    f"**{c['title']}** — trang {c.get('page_number', '—')} (score: {c['score']})"
                                )
                                st.caption(c["snippet"])
                except requests.exceptions.RequestException as exc:
                    full_text = f"❌ Lỗi kết nối tới backend: {exc}"
                    placeholder.markdown(full_text)

            st.session_state.messages.append({"role": "assistant", "content": full_text, "citations": citations})

# ---------- Tab 2: Nạp dữ liệu (Ingestion Dashboard, rút gọn) ----------
with tab_upload:
    st.subheader("Nạp tài liệu vào hệ thống")
    uploaded_file = st.file_uploader("Chọn file (PDF, DOCX, Markdown, TXT)", type=["pdf", "docx", "md", "txt"])
    up_category = st.selectbox("Phân loại tài liệu", ["GENERAL", "LEGAL_PDF", "CONTRACT", "TECH_SPEC"], key="up_cat")
    is_public = st.checkbox("Tài liệu công khai (is_public)", value=True)

    if st.button("🚀 Nạp tài liệu", disabled=uploaded_file is None):
        with st.spinner("Đang trích xuất, phân mảnh và tạo embedding... (PDF scan cần OCR nên có thể chậm hơn)"):
            try:
                mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
                data = {"category": up_category, "is_public": str(is_public)}
                resp = requests.post(f"{API_BASE}/api/v1/documents/upload", files=files, data=data, timeout=300)
                resp.raise_for_status()
                result = resp.json()["document"]
                st.success(f"✅ Đã nạp '{result['title']}' — {result['chunk_count']} chunks.")
            except requests.exceptions.RequestException as exc:
                st.error(f"❌ Lỗi: {exc}")

# ---------- Tab 3: Danh sách tài liệu ----------
with tab_docs:
    st.subheader("Tài liệu đã nạp")
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
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ Không thể tải danh sách tài liệu: {exc}")
