"""
MVP: chỉ xử lý PDF, dùng chiến lược "Recursive Character Chunking (Dự phòng)"
theo mục 3.5 của tài liệu thiết kế (512-800 tokens, overlap 100-150 tokens),
có bảo toàn page_number. Các loại tài liệu khác (OCR, Excel, Code, Email...)
và các chiến lược chunking chuyên biệt hơn (structure-aware, AST...) sẽ bổ
sung ở các bước tiếp theo, theo Ma trận Ingestion mục 2.
"""

import fitz  # PyMuPDF
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def extract_pdf_pages(file_path: str) -> list[tuple[int, str]]:
    """Trả về danh sách (số trang, nội dung text), số trang bắt đầu từ 1."""
    doc = fitz.open(file_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def chunk_pdf(file_path: str) -> list[dict]:
    """Trích xuất + phân mảnh một file PDF, giữ page_number cho từng chunk."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    idx = 0
    for page_number, text in extract_pdf_pages(file_path):
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_index": idx,
                    "content": piece,
                    "page_number": page_number,
                    "token_count": count_tokens(piece),
                }
            )
            idx += 1
    return chunks
