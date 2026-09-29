# RAG Chatbot v2 — MVP

MVP triển khai từ [tai_lieu_thiet_ke_chatbot_v2.md](tai_lieu_thiet_ke_chatbot_v2.md): pipeline Ingestion (PDF/DOCX/PPTX/Markdown/HTML/TXT/Ảnh) →
Chunking → Embedding (local) → Truy hồi lai (vector Qdrant + BM25) → Rerank → RAG Generation (Claude/OpenAI/Grok/Gemini/Groq) → Streamlit UI.

Luồng hỏi đáp đầy đủ:

1. **Viết lại câu hỏi** nối tiếp thành câu độc lập (chỉ khi có lịch sử hội thoại) — nếu không, "Thế còn nghỉ ốm thì sao?"
   sẽ đi tìm đúng chuỗi đó mà không mang theo chủ đề.
2. **Truy hồi lai**: vector dense (Qdrant) chạy song song BM25 (SQLite FTS5), trộn bằng RRF. BM25 bắt được mã số/số hiệu
   văn bản mà embedding hay đánh rơi, và tra được cả khi gõ không dấu.
3. **Rerank** bằng cross-encoder, cắt còn tối đa Top-K đoạn (mặc định 12, chỉnh bằng thanh trượt trên UI).
4. **Mở rộng đoạn lân cận** cho `NEIGHBOR_TOP_N` đoạn tốt nhất (chunk kề trong cùng tài liệu) để vá các điều khoản/slide
   bị cắt ngang.
5. **Sinh câu trả lời** kèm trích dẫn `[n]`, rồi **kiểm chứng**: mọi `[n]` có trỏ tới nguồn có thật không, nguồn nào thực
   sự được trích, và chấm một chỉ báo độ tin cậy (dựa trên bằng chứng truy hồi, *không phải* phép đo hallucination).
6. **Ghi lại toàn bộ** vào `query_logs` — kể cả lượt hỏng — kèm thời gian từng khâu và số token.

## Kiến trúc MVP so với thiết kế đầy đủ

| Thành phần | Thiết kế đầy đủ | MVP hiện tại |
| :--- | :--- | :--- |
| Loại tài liệu | 11 loại (PDF, OCR, Excel, Code, Email...) | **PDF (kể cả scan/OCR), DOCX, PPTX (kể cả chart+ảnh trong slide), Markdown, HTML, TXT, Ảnh (OCR)** (kiến trúc dễ mở rộng thêm) |
| Vector DB | Qdrant/Milvus server, hybrid search | **Qdrant local mode** (file-based, không cần Docker) + **hybrid search đã có**: BM25 chạy bằng SQLite FTS5 (bảng `chunks_fts`), trộn với vector bằng RRF — không cần sparse vector của Qdrant |
| Metadata DB | PostgreSQL (RBAC, audit logs đầy đủ) | **SQLite**: documents + chunks, `chunks_fts` (chỉ mục BM25), `chat_sessions`/`chat_messages` (hội thoại), `query_logs` (nhật ký vận hành) — chưa RBAC |
| Embedding | Chưa chỉ định | **Local, miễn phí**: `intfloat/multilingual-e5-small` (đa ngôn ngữ, hỗ trợ tiếng Việt) |
| LLM | OpenAI/Anthropic/Gemini/Ollama | **Anthropic Claude, OpenAI, xAI Grok, Google Gemini, Groq**, chọn ở UI |
| Auth/RBAC | JWT, phân quyền role/department | **Chưa có.** Lưu ý: từ khi hội thoại được lưu bền, `GET /api/v1/chat/sessions` **không lọc theo người dùng** nên mọi người đều thấy cuộc trò chuyện của tất cả. Các endpoint `/api/v1/admin/*` cũng để lộ nguyên văn câu hỏi. An toàn khi chạy `localhost`, **không an toàn khi mở ra mạng nội bộ** |
| Streaming | SSE | Có, qua `StreamingResponse` |
| API key | Lưu mã hoá trong PostgreSQL, nhập qua UI (Tab Config) | **Chỉ khai báo trong `.env`** (không nhập trên UI) - UI tự lấy danh sách provider/model đã cấu hình qua `GET /api/v1/config/providers` |
| Hội thoại | Lưu `chat_sessions`/`chat_messages` trong PostgreSQL | **Đã lưu bền** trong SQLite (`chat_sessions`/`chat_messages`), kèm cả trích dẫn lẫn chỉ báo tin cậy — mở lại app vẫn còn. Lịch sử gửi cho LLM giới hạn `MAX_HISTORY_MESSAGES` tin nhắn gần nhất. Xoá hội thoại **không** xoá `query_logs` |
| Quan sát vận hành | Audit logs trong PostgreSQL | **Log ra stdout**, mặc định dạng dễ đọc cho terminal, đổi sang JSON bằng `LOG_FORMAT=json` khi cần đẩy vào Loki/ELK; `query_logs` ghi mỗi lượt hỏi (thời gian từng khâu, token, độ tin cậy, lỗi provider, 👍/👎); trang **Giám sát** tổng hợp lại kèm danh sách "câu hỏi hệ thống bó tay" |

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

Truy cập http://localhost:8501. Giao diện gồm 4 trang trên thanh điều hướng: **Hỏi đáp**, **Nạp tài liệu**, **Tài liệu**,
**Giám sát**. Vào **Nạp tài liệu** upload một file PDF, rồi sang **Hỏi đáp**.

Lần chạy đầu tiên sẽ tải model embedding (~470MB) từ HuggingFace nên hơi chậm; các lần sau sẽ nhanh vì đã cache.

**Log backend** mặc định dạng dễ đọc (`LOG_FORMAT=pretty`), mỗi lượt hỏi một dòng:

```
09:31:10 INFO  query    gemini/gemini-3.6-flash · 69.7s (rewrite 3.7s, rerank 0.9s, generate 65.1s) · 6 đoạn · tin cậy cao · 2760+172 tok · "tìm nhân sự phụ trách lương"
```

Lượt lỗi (hết quota, sai model...) in ở mức WARNING. Access log của các GET thành công bị ẩn (`LOG_QUIET_ACCESS`) vì
Streamlit gọi lại chúng mỗi lần tải lại trang. Khi triển khai thật, đặt `LOG_FORMAT=json` để đẩy vào Loki/ELK.
Riêng vài dòng `WatchFiles detected changes...` của tiến trình `--reload` vẫn theo định dạng của uvicorn, vì tiến trình
đó không import app.

### Câu hỏi gợi ý: FAQ + Hay được hỏi

Cột bên phải trang Hỏi đáp có hai nhóm, mỗi nhóm tối đa 5 câu (`SUGGESTION_LIMIT`); bấm một câu là gửi luôn cho chatbot.
Dữ liệu lấy từ `GET /api/v1/chat/suggestions?category=...`.

**🔥 Hay được hỏi** - tự sinh từ `query_logs`, không cần soạn. Một câu chỉ lên danh sách khi:

- được hỏi ở ít nhất `POPULAR_MIN_SESSIONS` (mặc định 2) cuộc trò chuyện khác nhau trong `POPULAR_DAYS` (mặc định 30) ngày;
- lần hỏi GẦN NHẤT được trả lời có trích dẫn, không trích dẫn hỏng, độ tin cậy không thấp, và chưa từng bị 👎;
- chưa có trong FAQ. Câu hỏi nối tiếp được hiển thị bằng bản đã viết lại (đứng độc lập được).

Lượt lỗi provider (hết quota, sai model...) bị bỏ qua. Lưu ý: danh sách hiển thị câu hỏi của người dùng này cho người
dùng khác - ngưỡng nhiều cuộc trò chuyện giúp hạn chế lộ câu hỏi mang tính cá nhân, nhưng không thay được phân quyền thật.

**💡 Câu hỏi thường gặp** - soạn tay trong [backend/faq.json](backend/faq.json), sửa file là có hiệu lực sau tối đa
30 giây (không cần restart):

```json
[{"group": "Nhân sự & nội quy", "question": "Người lao động được nghỉ phép năm bao nhiêu ngày?", "category": "GENERAL"}]
```

- `group` (tuỳ chọn): tiêu đề nhóm trên UI.
- `category` (tuỳ chọn): câu chỉ hiện khi "Phạm vi tài liệu" là `ALL` hoặc đúng loại này; bỏ trống = luôn hiện.
- Chỉ nên đưa vào những câu mà tài liệu đã nạp thực sự trả lời được - câu FAQ ra "không tìm thấy" còn tệ hơn không có FAQ.
  Nguồn gợi ý tốt: các lượt độ tin cậy cao / được 👍 ở trang **Giám sát**.

### Khi LLM hết quota

Provider trả lỗi hết quota/hết credit/vượt giới hạn gọi (HTTP 429, Gemini `RESOURCE_EXHAUSTED`, Anthropic
"credit balance is too low"...) thì chatbot không chỉ báo lỗi: nó nói rõ provider/model nào hết quota, rồi trả nguyên
văn `QUOTA_FALLBACK_RESULTS` (mặc định 3) đoạn tài liệu liên quan nhất đã tìm được, đánh số khớp danh sách nguồn.
Qua API, nội dung này đến bằng sự kiện SSE `fallback` (`{"reason": "quota", "text": ...}`) thay cho `error`.
Các lỗi khác (key sai, sai tên model...) vẫn báo lỗi như cũ.

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

Hội thoại, phản hồi và giám sát:

```bash
# Tạo hội thoại rồi truyền "session_id" trong body /chat/completions để lượt hỏi được lưu lại
curl -X POST http://localhost:8000/api/v1/chat/sessions
curl http://localhost:8000/api/v1/chat/sessions                      # danh sách hội thoại
curl http://localhost:8000/api/v1/chat/sessions/<id>/messages        # mở lại tin nhắn cũ

# Chấm 👍/👎 - query_log_id lấy từ sự kiện SSE "done" của lượt trả lời đó
curl -X POST http://localhost:8000/api/v1/chat/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_log_id": "<id>", "rating": -1, "note": "thiếu bước cuối"}'

curl "http://localhost:8000/api/v1/admin/stats?days=7"               # latency, token, tỉ lệ bó tay
curl "http://localhost:8000/api/v1/admin/unanswered?days=30"         # câu hỏi bó tay, gom theo nội dung
curl "http://localhost:8000/api/v1/admin/queries?only_problems=true" # các lượt cần xem lại
```

## Hướng mở rộng tiếp theo (theo tài liệu thiết kế)

Theo thứ tự ưu tiên thực tế, không theo thứ tự trong tài liệu thiết kế:

1. **Auth + lọc hội thoại theo người dùng** — việc cấp bách nhất. Hội thoại đang được lưu bền nhưng chưa lọc theo
   người dùng, nên hiện ai cũng đọc được của tất cả (xem dòng Auth/RBAC ở bảng trên). Cần làm trước khi mở ra
   ngoài `localhost`.
2. **Bộ câu hỏi chuẩn để đo chất lượng**: 20–30 câu hỏi thật kèm tài liệu nguồn đúng, đo recall@k mỗi lần đổi
   model/ngưỡng. Nguyên liệu đã có sẵn (`query_logs`, 👍/👎, tỉ lệ bó tay) nhưng chưa có phép đo — nên hiện chưa
   trả lời được câu "hệ thống trả lời đúng bao nhiêu phần trăm".
3. **Hiệu chỉnh `RERANK_THRESHOLD`**: đo trên tài liệu thật cho thấy đoạn khớp đúng nhất chỉ đạt ~0.25, tức ngưỡng
   mặc định 0.3 gần như không bao giờ kích hoạt — cơ chế "K động" đang luôn rơi vào nhánh dự phòng.
4. **Thêm loại tài liệu**: viết thêm hàm `chunk_xxx()` trong `backend/app/ingestion.py` rồi đăng ký vào
   dict `CHUNKERS` (đã có sẵn PDF/DOCX/PPTX/Markdown/HTML/TXT/Ảnh theo cách này) cho các loại còn lại
   (Excel/CSV, Email, Source Code, DB...) theo mục 2-3 của tài liệu thiết kế. Với dữ liệu nhân sự thì Excel là
   thiếu sót đáng kể nhất.
5. **Ngày hiệu lực / phiên bản tài liệu**: nạp quy chế bản mới mà chưa xoá bản cũ thì hệ thống trả lời bằng cả hai
   mà không cảnh báo gì.
6. **Migrate sang PostgreSQL**: dùng đúng DDL ở mục 4 của tài liệu thiết kế, thêm Auth (JWT) + RBAC.
7. **Migrate Qdrant local → Qdrant server**: chỉ cần đổi `QdrantClient(path=...)` thành
   `QdrantClient(url=...)`, bổ sung payload `security.*` và `build_rbac_filter()` theo mục 5.2. Cũng là cách gỡ
   giới hạn "chỉ 1 tiến trình mở được kho vector" của chế độ nhúng hiện tại.
8. **Docker Compose**: đóng gói Postgres + Qdrant server + backend + frontend khi cần triển khai thật.
9. **Thêm LLM provider khác** (DeepSeek, Mistral, OpenRouter, Ollama/vLLM local...): hầu hết tương thích chuẩn
   OpenAI Chat Completions, chỉ cần gọi `_stream_openai_compatible(api_key, model, system, user, base_url=...)`
   có sẵn trong `backend/app/llm.py` với `base_url` riêng của provider đó — không cần viết hàm mới
   (xem cách đã làm với xAI Grok).

> **Hybrid search** (mục 4 của bản lộ trình cũ) đã làm xong — BM25 bằng SQLite FTS5 trộn với vector bằng RRF.
