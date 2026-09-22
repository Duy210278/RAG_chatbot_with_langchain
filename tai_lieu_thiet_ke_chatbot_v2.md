# TÀI LIỆU THIẾT KẾ KỸ THUẬT HỆ THỐNG TRỢ LÝ RAG NỘI BỘ DOANH NGHIỆP
**(ENTERPRISE RETRIEVAL-AUGMENTED GENERATION SYSTEM DESIGN SPECIFICATION)**

* **Quy mô đối tượng:** ~1.000 cán bộ nhân viên (~100 - 200 truy vấn đồng thời giờ cao điểm).
* **Công nghệ chủ đạo:** Python, LangChain, FastAPI, Streamlit, PostgreSQL, Qdrant / Milvus, MinIO / S3.

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG (HIGH-LEVEL ARCHITECTURE)

Hệ thống được tổ chức theo kiến trúc phân tầng (Multi-tier Modular Architecture):

```
[ Giao diện Người dùng (Streamlit UI) ]
   ├── Tab 1: Hỏi đáp Chatbot & Hiển thị Citation
   ├── Tab 2: Quản trị & Nạp dữ liệu (Data Ingestion & Monitoring)
   └── Tab 3: Cấu hình Model & API Key (Config Manager)
          │ (REST API / SSE Streaming)
          ▼
[ Backend API (FastAPI) ]
   ├── Auth & Role-Based Access Control (RBAC Module)
   ├── Routing Agent / Query Planner
   ├── RAG Engine (LangChain: Retrieval, Re-ranking, Context Assembly)
   └── Async Task Worker (Celery/Redis hoặc BackgroundTasks)
          │
          ├── Vector Store: Qdrant / Milvus (Hybrid Search + Metadata Filtering)
          ├── BM25 / Sparse Search: Qdrant Sparse Vectors / Elasticsearch
          ├── Metadata & Config DB: PostgreSQL (Identity, Config, History, Audit)
          └── Object Storage: MinIO / S3 (Lưu raw files: PDF, scan, code, excel...)
```

---

## 2. MA TRẬN XỬ LÝ & BÓC TÁCH DỮ LIỆU ĐA NGUỒN (INGESTION MATRIX)

| Phân loại Dữ liệu | Thư viện & Công cụ Trích xuất | Quy trình & Tiền xử lý |
| :--- | :--- | :--- |
| **PDF Pháp lý & Hợp đồng** | `pdfplumber`, `PyMuPDF` | Bảo toàn cấu trúc phân cấp (Điều, Khoản, Điểm); bóc tách phụ lục và bảng cam kết; trích xuất metadata số hiệu văn bản, ngày ban hành, thời hạn hiệu lực. |
| **OCR (Ảnh / PDF scan)** | `PaddleOCR`, `Tesseract` (hỗ trợ Tiếng Việt), `Surya OCR` | Khử nhiễu, xoay thẳng trang (deskew), nhận diện chữ tiếng Việt có dấu, bóc tách hóa đơn, hợp đồng scan có dấu đỏ. |
| **Tài liệu Kỹ thuật** | `LangChain Unstructured`, `PyPDF` | Bảo toàn code block nội tuyến, sơ đồ luồng dạng text, trích xuất bảng thông số kỹ thuật. |
| **Markdown / HTML** | `MarkdownHeaderTextSplitter`, `BeautifulSoup4` | Loại bỏ thẻ CSS/JS/Navigation thừa, giữ nguyên cấu trúc phân cấp tiêu đề (`#`, `##`, `###`). |
| **Email (EML, MSG)** | `mail-parser`, `email` | Chuẩn hóa Header (`From`, `To`, `Date`, `Subject`), trích xuất Body, tách file đính kèm đưa vào hàng đợi riêng. |
| **Cơ sở dữ liệu (Database)** | `SQLAlchemy`, `DuckDB` | Áp dụng Text-to-SQL cho số liệu biến động; lưu trữ Data Dictionary (Schema + Chú thích bảng/cột) phục vụ RAG giải thích nghiệp vụ. |
| **FAQ nội bộ** | `pandas`, `json` | Định dạng cặp Question - Answer kèm metadata phòng ban (HR, IT, Tài chính). |
| **Source Code** | `LangChain Language Parser` (AST) | Giữ nguyên cấu trúc Scope (Class, Method/Function, Docstring, Comment). |
| **Bảng tính Excel / CSV** | `openpyxl`, `pandas` | Serialize từng hàng (Row-by-row) dạng markdown table hoặc JSON kèm Context Header của cột để Vector Search hiểu được ngữ cảnh hàng ngang. |
| **Log hệ thống** | Regex Parser, Logstash | Nhóm log theo Session ID, Transaction ID, Error stack trace; loại bỏ thông điệp heartbeat vô nghĩa. |
| **Tài liệu Tiếng Việt** | `pyvi`, `underthesea` | Chuẩn hóa Unicode dựng sẵn (NFC), chuẩn hóa dấu Telex/VNI, gán nhãn token Tiếng Việt. |

---

## 3. CHIẾN LƯỢC PHÂN MẢNH (CHUNKING STRATEGIES)

1. **Structure-Aware Chunking (Văn bản có cấu trúc):** Áp dụng cho Hợp đồng, Pháp lý, Markdown. Phân tách theo Điều, Khoản, Tiêu đề để bảo toàn tính toàn vẹn của một điều khoản.
2. **Code AST Hierarchy Chunking:** Áp dụng cho Source Code. Cắt tách theo Syntax Tree của từng ngôn ngữ (Python, Java, Go, JS), đảm bảo một hàm/lớp không bị đứt đoạn.
3. **Semantic / Sentence-Window Chunking:** Áp dụng cho Email, FAQ, Log. Embed một câu hoặc đoạn ngắn (150–200 tokens) để tối ưu tính chính xác khi so khớp tương đồng, nhưng tự động mở rộng cửa sổ ngữ cảnh (lấy thêm 3 câu trước và sau) khi nạp vào Prompt cho LLM.
4. **Row-level & Schema Chunking:** Áp dụng cho Excel, CSV. Mỗi hàng được gắn kèm tiêu đề cột: `[Cột A: Giá trị] | [Cột B: Giá trị]`.
5. **Recursive Character Chunking (Dự phòng):** Áp dụng cho văn bản phi cấu trúc. Kích thước $512 - 800$ tokens, overlap $100 - 150$ tokens.

---

## 4. CHI TIẾT LƯỢC ĐỒ CƠ SỞ DỮ LIỆU POSTGRESQL (RBAC & AUDIT DDL)

Cài đặt Extension:
```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

### 4.1. Định danh & Phân quyền (RBAC)

```sql
-- 1. Bảng Phòng ban
CREATE TABLE departments (
    id VARCHAR(50) PRIMARY KEY, -- 'LEGAL', 'TECH', 'HR', 'FINANCE'
    name VARCHAR(255) NOT NULL,
    parent_id VARCHAR(50) REFERENCES departments(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng Vai trò
CREATE TABLE roles (
    id VARCHAR(50) PRIMARY KEY, -- 'ROLE_ADMIN', 'ROLE_LEGAL_MEMBER', 'ROLE_DEV', 'ROLE_DIRECTOR'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Bảng Người dùng
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    department_id VARCHAR(50) REFERENCES departments(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Bảng liên kết Người dùng - Vai trò
CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id VARCHAR(50) REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id)
);
```

### 4.2. Cấu hình Model & Khóa API Bảo mật

```sql
CREATE TABLE llm_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE, -- NULL nghĩa là default hệ thống
    provider VARCHAR(50) NOT NULL,                       -- 'openai', 'anthropic', 'google_gemini', 'ollama'
    model_name VARCHAR(100) NOT NULL,                    -- 'gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-pro'
    encrypted_api_key BYTEA,                             -- Mã hóa bằng pgp_sym_encrypt(api_key, master_key)
    base_url VARCHAR(255),                               -- Cho Local vLLM/Ollama hoặc Proxy
    temperature NUMERIC(3, 2) DEFAULT 0.20,
    top_k INT DEFAULT 5,
    enable_rerank BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3. Quản lý Tài liệu & Chunks

```sql
CREATE TYPE document_category AS ENUM (
    'LEGAL_PDF', 'CONTRACT', 'TECH_SPEC', 'MARKDOWN', 
    'HTML', 'EMAIL', 'DB_SCHEMA', 'FAQ', 'SOURCE_CODE', 
    'EXCEL_CSV', 'SYSTEM_LOG', 'OCR_SCAN'
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_extension VARCHAR(20) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    mime_type VARCHAR(100),
    category document_category NOT NULL,
    
    -- RBAC Metadata
    is_public BOOLEAN DEFAULT FALSE,
    allowed_roles VARCHAR(50)[] DEFAULT ARRAY[]::VARCHAR(50)[],
    allowed_departments VARCHAR(50)[] DEFAULT ARRAY[]::VARCHAR(50)[],
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    status VARCHAR(50) DEFAULT 'PENDING',               -- 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
    error_message TEXT,
    chunk_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),      -- Trùng point_id trên Qdrant
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    context_window TEXT,
    token_count INT NOT NULL,
    chunk_strategy VARCHAR(50) NOT NULL,
    page_number INT,
    section_heading VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_doc_id ON document_chunks(document_id);
```

### 4.4. Lịch sử Hội thoại & Nhật ký Truy vết (Audit Logs)

```sql
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'Cuộc trò chuyện mới',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,                           -- 'user', 'assistant'
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::JSONB,
    retrieval_latency_ms INT,
    llm_latency_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rag_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids UUID[],
    user_roles VARCHAR(50)[],
    user_department VARCHAR(50),
    ip_address VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

## 5. CẤU TRÚC PAYLOAD & FILTERING TRÊN VECTOR DATABASE (QDRANT)

### 5.1. Cấu trúc Payload JSON của mỗi Vector Point

```json
{
  "id": "e9b21f37-1845-4a57-8b54-9343df891c71",
  "vector": [0.0125, -0.0481, 0.0892, "... 1024 dims ..."],
  "payload": {
    "document_id": "8bfa5dc2-5407-4e92-9657-3f360e5cfce2",
    "chunk_index": 4,
    "category": "CONTRACT",
    "title": "Hợp đồng Cung cấp Dịch vụ CNTT số 2026/IT-VN",
    "page_number": 3,
    "section_heading": "Điều 8: Điều khoản Bảo mật và Trách nhiệm Bồi thường",
    "content": "Bên B cam kết bảo mật toàn bộ dữ liệu máy chủ nội bộ. Mọi vi phạm sẽ chịu phạt 100% giá trị hợp đồng...",
    
    "security": {
      "is_public": false,
      "owner_id": "4c94432c-7b0a-4a65-8b3d-1a89c9261a01",
      "allowed_roles": ["ROLE_DIRECTOR", "ROLE_LEGAL_MEMBER", "ROLE_FINANCE_MEMBER"],
      "allowed_departments": ["LEGAL", "FINANCE", "BOD"]
    },
    
    "metadata_extra": {
      "source_file_type": ".pdf",
      "language": "vi",
      "created_timestamp": 1789985200
    }
  }
}
```

### 5.2. Hàm Xây dựng Bộ lọc Qdrant Filter (Pre-filtering RBAC)

```python
from qdrant_client.http import models

def build_rbac_filter(
    user_id: str, 
    user_roles: list[str], 
    user_dept: str, 
    selected_category: str | None = None
) -> models.Filter:
    """
    Xây dựng điều kiện lọc an toàn tuyệt đối trước khi vector similarity search:
    Truy cập = is_public OR owner_id == user_id OR (role in allowed_roles AND dept in allowed_departments)
    """
    security_condition = models.Filter(
        should=[
            # 1. Tài liệu dùng chung
            models.FieldCondition(
                key="security.is_public", 
                match=models.MatchValue(value=True)
            ),
            # 2. Người tạo tài liệu
            models.FieldCondition(
                key="security.owner_id", 
                match=models.MatchValue(value=user_id)
            ),
            # 3. Phù hợp Role và Department
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="security.allowed_roles",
                        match=models.MatchAny(any=user_roles)
                    ),
                    models.Filter(
                        should=[
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(key="security.allowed_departments")
                            ),
                            models.FieldCondition(
                                key="security.allowed_departments",
                                match=models.MatchValue(value=user_dept)
                            )
                        ]
                    )
                ]
            )
        ]
    )

    if selected_category and selected_category != "ALL":
        return models.Filter(
            must=[
                security_condition,
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=selected_category)
                )
            ]
        )
        
    return security_condition
```

---

## 6. THIẾT KẾ GIAO DIỆN STREAMLIT & TÍCH HỢP FASTAPI

### 6.1. Bố cục 3 Tab Streamlit

* **Tab 1: Hỏi đáp Chatbot (Chat & Citations):**
  * Giao diện hội thoại đa lượt, streaming phản hồi từng token.
  * Dropdown chọn phạm vi: *Tất cả, Pháp lý, Hợp đồng, Kỹ thuật, Code, FAQ*.
  * Hộp trích dẫn (Accordion Citations): Tên văn bản, trang, điểm số tương đồng cosine, đoạn văn bản gốc.
* **Tab 2: Nạp dữ liệu (Ingestion Dashboard):**
  * Kéo thả file đa định dạng (PDF, DOCX, XLSX, MD, EML, ZIP).
  * Bộ chọn: Phân loại tài liệu, quyền truy cập (`is_public`, `allowed_roles`, `allowed_departments`).
  * Bảng theo dõi tiến độ nạp file thời gian thực (Status, Chunk Count, Error log).
* **Tab 3: Cấu hình Model & API Key (Config Manager):**
  * Provider: OpenAI, Anthropic, Google Gemini, Ollama/vLLM.
  * Nhập API Key (ẩn password), Custom Base URL.
  * Thanh trượt tham số: Temperature, Top-K, Bật/Tắt Reranker.

### 6.2. Danh mục API FastAPI lõi

* `POST /api/v1/auth/login`: Xác thực và trả JWT token chứa thông tin `user_id`, `department`, `roles`.
* `POST /api/v1/chat/completions`: Tiếp nhận câu hỏi, áp dụng bộ lọc RBAC tự động, trả về Server-Sent Events (SSE) kèm Citation.
* `POST /api/v1/documents/upload`: Tiếp nhận file, kiểm tra định dạng và gán RBAC metadata.
* `POST /api/v1/config/apply`: Cập nhật và mã hóa API Key lưu trữ an toàn trong cơ sở dữ liệu.