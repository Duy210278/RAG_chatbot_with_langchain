"""
Giao diện Streamlit MVP - tương ứng mục 6.1 của tài liệu thiết kế, rút gọn còn
các phần cốt lõi cho MVP: Chat & Citations, Nạp dữ liệu, Danh sách tài liệu.
Tab "Cấu hình Model & API Key" được gộp vào sidebar cho gọn.
"""

import json
import os

import requests
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot Nội bộ", page_icon="🤖", layout="wide")

# ---------- Sidebar: cấu hình model (tương ứng Tab 3 trong thiết kế) ----------
PROVIDER_LABELS = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "xai": "xAI Grok",
    "gemini": "Google Gemini",
}
PROVIDER_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    provider = st.selectbox(
        "Provider",
        list(PROVIDER_LABELS.keys()),
        format_func=lambda p: PROVIDER_LABELS[p],
    )
    api_key = st.text_input(
        PROVIDER_ENV_VAR[provider],
        type="password",
        help="Để trống nếu server đã cấu hình sẵn key qua biến môi trường (.env).",
    )
    model_override = st.text_input(
        "Model (tuỳ chọn)", placeholder="vd: claude-sonnet-5 / gpt-4o-mini / grok-4 / gemini-2.5-flash"
    )
    top_k = st.slider("Top-K tài liệu truy hồi", min_value=1, max_value=10, value=5)
    category = st.selectbox("Phạm vi tài liệu", ["ALL", "GENERAL", "LEGAL_PDF", "CONTRACT", "TECH_SPEC"])
    st.divider()
    st.caption(f"Backend: {API_BASE}")

tab_chat, tab_upload, tab_docs = st.tabs(["💬 Hỏi đáp", "📤 Nạp tài liệu", "📄 Danh sách tài liệu"])

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

    question = st.chat_input("Đặt câu hỏi về tài liệu nội bộ...")
    if question:
        if not api_key:
            st.warning("⚠️ Vui lòng nhập API Key ở thanh bên trái, hoặc cấu hình sẵn trên server (.env).")
        else:
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
                            "api_key": api_key or None,
                            "model": model_override or None,
                            "top_k": top_k,
                            "category": category,
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
    st.subheader("Nạp tài liệu PDF vào hệ thống")
    uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])
    up_category = st.selectbox("Phân loại tài liệu", ["GENERAL", "LEGAL_PDF", "CONTRACT", "TECH_SPEC"], key="up_cat")
    is_public = st.checkbox("Tài liệu công khai (is_public)", value=True)

    if st.button("🚀 Nạp tài liệu", disabled=uploaded_file is None):
        with st.spinner("Đang trích xuất, phân mảnh và tạo embedding..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
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
