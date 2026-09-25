# RAG Chatbot v2 — MVP

MVP triển khai từ [tai_lieu_thiet_ke_chatbot_v2.md](tai_lieu_thiet_ke_chatbot_v2.md): pipeline Ingestion (PDF/DOCX/Markdown/HTML/TXT/Ảnh) →
Chunking → Embedding (local) → Vector Search (Qdrant) → RAG Generation (Claude/OpenAI/Grok/Gemini/Groq) → Streamlit UI.

## Kiến trúc MVP so với thiết kế đầy đủ

| Thành phần | Thiết kế đầy đủ | MVP hiện tại |
| :--- | :--- | :--- |
| Loại tài liệu | 11 loại (PDF, OCR, Excel, Code, Email...) | **PDF (kể cả scan/OCR), DOCX, Markdown, HTML, TXT, Ảnh (OCR)** (kiến trúc dễ mở rộng thêm) |
| Vector DB | Qdrant/Milvus server, hybrid search | **Qdrant local mode** (file-based, không cần Docker) |
| Metadata DB | PostgreSQL (RBAC, audit logs đầy đủ) | **SQLite** (documents + chunks, chưa RBAC/audit) |
| Embedding | Chưa chỉ định | **Local, miễn phí**: `intfloat/multilingual-e5-small` (đa ngôn ngữ, hỗ trợ tiếng Việt) |
| LLM | OpenAI/Anthropic/Gemini/Ollama | **Anthropic Claude, OpenAI, xAI Grok, Google Gemini, Groq**, chọn ở UI |
| Auth/RBAC | JWT, phân quyền role/department | Chưa có (single-user MVP) |
| Streaming | SSE | Có, qua `StreamingResponse` |
| API key | Lưu mã hoá trong PostgreSQL, nhập qua UI (Tab Config) | **Chỉ khai báo trong `.env`** (không nhập trên UI) - UI tự lấy danh sách provider/model đã cấu hình qua `GET /api/v1/config/providers` |
| Hội thoại | Lưu `chat_sessions`/`chat_messages` trong PostgreSQL | **Multi-turn trong phiên hiện tại**: lịch sử hội thoại (session_state phía UI) được gửi kèm mỗi request làm ngữ cảnh cho LLM, giới hạn `MAX_HISTORY_MESSAGES` tin nhắn gần nhất - chưa lưu persistent qua các lần mở lại app |

## Cài đặt

Yêu cầu Python 3.11+.

```bash
cd /Users/a/Documents/duydo/RAG_Chatbot_v2

# 1. Tạo virtualenv dùng chung cho backend + frontend
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 3. Cấu hình API key - BẮT BUỘC, giao diện không còn ô nhập API key nữa
cp .env.example .env
# rồi mở .env, điền API key của (các) provider bạn muốn dùng - provider nào có key
# sẽ tự động hiện trong dropdown "Provider" trên UI, không cần điền hết cả 5.
```

### Cài Tesseract (bắt buộc để nạp PDF scan/ảnh — bỏ qua nếu chỉ dùng PDF có text layer)

```bash
# macOS
brew install tesseract
# Tải thêm gói ngôn ngữ tiếng Việt (bản "fast", nhẹ hơn bản đầy đủ)
curl -L -o "$(brew --prefix tesseract)/share/tessdata/vie.traineddata" \
  https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata
```

Nếu không cài Tesseract, upload PDF text bình thường vẫn hoạt động — chỉ PDF scan/ảnh (không có text layer) mới báo lỗi.

### Lấy API key miễn phí (để test nhanh, không tốn phí)

| Provider | Lấy key ở đâu | Ghi chú |
|---|---|---|
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier thật, chất lượng tốt (Gemini 2.5 Flash) |
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | Free tier, chạy Llama 3.3 70B rất nhanh |

Anthropic Claude / OpenAI / xAI Grok không có free tier lâu dài (chỉ credit dùng thử ban đầu cho tài khoản mới).

## Chạy

Mở 2 terminal riêng (cùng kích hoạt `.venv`):

```bash
# Terminal 1 — Backend API
cd backend
uvicorn app.main:app --reload --port 8000
```

```bash
# Terminal 2 — Giao diện Streamlit
cd frontend
streamlit run streamlit_app.py
```

Truy cập http://localhost:8501, vào Tab **📤 Nạp tài liệu** để upload một file PDF, sau đó hỏi đáp ở Tab **💬 Hỏi đáp**.

Lần chạy đầu tiên sẽ tải model embedding (~470MB) từ HuggingFace nên hơi chậm; các lần sau sẽ nhanh vì đã cache.

## Kiểm thử nhanh bằng API (không cần UI)

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/file.pdf" -F "category=GENERAL" -F "is_public=true"

curl -N -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Nội dung tài liệu nói gì?",
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "history": [
      {"role": "user", "content": "Câu hỏi trước đó"},
      {"role": "assistant", "content": "Câu trả lời trước đó"}
    ]
  }'
# (api_key trong .env sẽ tự được dùng - chỉ cần truyền "api_key" trong body nếu muốn override khi test)

curl http://localhost:8000/api/v1/config/providers  # xem provider nào đang có key hợp lệ
```

## Hướng mở rộng tiếp theo (theo tài liệu thiết kế)

1. **Thêm loại tài liệu**: viết thêm hàm `chunk_xxx()` trong `backend/app/ingestion.py` rồi đăng ký vào
   dict `CHUNKERS` (đã có sẵn PDF/DOCX/Markdown/HTML/TXT/Ảnh theo cách này) cho các loại còn lại
   (Excel/CSV, Email, Source Code, DB...) theo mục 2-3 của tài liệu thiết kế.
2. **Migrate sang PostgreSQL**: dùng đúng DDL ở mục 4 của tài liệu thiết kế, thêm Auth (JWT) + RBAC.
3. **Migrate Qdrant local → Qdrant server**: chỉ cần đổi `QdrantClient(path=...)` thành
   `QdrantClient(url=...)`, bổ sung payload `security.*` và `build_rbac_filter()` theo mục 5.2.
4. **Reranking**: thêm bước rerank (cross-encoder) sau khi lấy top-k từ vector search.
5. **Hybrid search**: kết hợp BM25/sparse vectors với dense vector search hiện tại.
6. **Docker Compose**: đóng gói Postgres + Qdrant server + backend + frontend khi cần triển khai thật.
7. **Thêm LLM provider khác** (DeepSeek, Mistral, OpenRouter, Ollama/vLLM local...): hầu hết tương thích chuẩn
   OpenAI Chat Completions, chỉ cần gọi `_stream_openai_compatible(api_key, model, system, user, base_url=...)`
   có sẵn trong `backend/app/llm.py` với `base_url` riêng của provider đó — không cần viết hàm mới
   (xem cách đã làm với xAI Grok).
