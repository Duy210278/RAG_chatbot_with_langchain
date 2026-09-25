"""
Ma trận Ingestion (mục 2 tài liệu thiết kế) - hiện đã hỗ trợ:
  - PDF có text layer (PyMuPDF)
  - PDF scan / ảnh, không có text layer (tự động fallback sang OCR Tesseract, hỗ trợ tiếng Việt)
  - Word (.docx, python-docx)
  - Markdown (.md) - structure-aware theo heading (mục 3.1 tài liệu thiết kế)
  - Text thuần (.txt)
  - Ảnh (.png/.jpg/.jpeg/.bmp/.tiff/.webp) - OCR trực tiếp bằng Tesseract, cùng hạ tầng dùng
    cho PDF scan bên trên
  - HTML (.html/.htm) - wiki/tài liệu đào tạo nội bộ: loại bỏ CSS/JS/navigation, quy về dạng
    markdown theo heading rồi tái dùng chiến lược structure-aware của Markdown

Các loại còn lại (Excel/CSV, Email, Source Code, DB, Log, PDF pháp lý structure-aware) chưa
làm theo yêu cầu hiện tại - sẽ bổ sung ở các bước tiếp theo nếu cần,
mỗi loại chỉ cần thêm 1 hàm `chunk_xxx()` rồi đăng ký vào `chunk_document()` + `ALLOWED_EXTENSIONS`
trong routers/documents.py.
"""

import io

import fitz  # PyMuPDF
import tiktoken
from docx import Document as DocxDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import get_settings

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _chunk_pages(pages: list[tuple[int | None, str]]) -> list[dict]:
    """Cắt chunk (Recursive Character, dự phòng) từ danh sách (page_number, text) -
    dùng chung cho mọi loại tài liệu. page_number = None với các định dạng không có khái niệm trang."""
    splitter = _recursive_splitter()
    chunks: list[dict] = []
    idx = 0
    for page_number, text in pages:
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


# ---------------------------------------------------------------------------
# PDF (text layer)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# OCR dùng chung (Tesseract) - cho cả PDF scan và ảnh upload trực tiếp
# ---------------------------------------------------------------------------
def _ocr_image(image, lang: str = "vie+eng") -> str:
    """OCR 1 ảnh (PIL Image) bằng Tesseract, yêu cầu đã cài `tesseract` + gói ngôn ngữ 'vie'."""
    import pytesseract

    return pytesseract.image_to_string(image, lang=lang).strip()


def extract_pdf_pages_ocr(file_path: str, lang: str = "vie+eng") -> list[tuple[int, str]]:
    """OCR từng trang PDF scan bằng Tesseract. Render trang thành ảnh độ phân giải cao rồi OCR."""
    from PIL import Image

    doc = fitz.open(file_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=250)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = _ocr_image(image, lang=lang)
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def chunk_pdf(file_path: str) -> list[dict]:
    """Trích xuất + chunk PDF. Nếu không tìm thấy text layer nào (PDF scan/ảnh),
    tự động fallback sang OCR trước khi báo lỗi."""
    pages = extract_pdf_pages(file_path)
    if not pages:
        pages = extract_pdf_pages_ocr(file_path)
    return _chunk_pages(pages)


# ---------------------------------------------------------------------------
# Ảnh upload trực tiếp (.png/.jpg/.jpeg/.bmp/.tiff/.webp) -> OCR toàn bộ ảnh
# ---------------------------------------------------------------------------
def chunk_image(file_path: str) -> list[dict]:
    """OCR 1 file ảnh độc lập (không nhúng trong PDF) - dùng chung hàm _ocr_image()
    với nhánh PDF scan bên trên. Không có khái niệm trang nên page_number = None."""
    from PIL import Image

    with Image.open(file_path) as image:
        text = _ocr_image(image)
    if not text:
        return []
    return _chunk_pages([(None, text)])


# ---------------------------------------------------------------------------
# Word (.docx)
# ---------------------------------------------------------------------------
def extract_docx_text(file_path: str) -> str:
    doc = DocxDocument(file_path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def chunk_docx(file_path: str) -> list[dict]:
    text = extract_docx_text(file_path)
    return _chunk_pages([(None, text)])


# ---------------------------------------------------------------------------
# Markdown (.md) - structure-aware theo heading trước, recursive trong từng section
# ---------------------------------------------------------------------------
def _chunk_markdown_text(content: str) -> list[dict]:
    """Cắt structure-aware theo heading H1-H3 - dùng chung cho .md và .html
    (HTML được quy về dạng markdown trước khi gọi hàm này)."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    sections = md_splitter.split_text(content)

    if not sections:
        return _chunk_pages([(None, content)])
    return _chunk_pages([(None, section.page_content) for section in sections])


def chunk_markdown(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    return _chunk_markdown_text(content)


# ---------------------------------------------------------------------------
# HTML (.html/.htm) - wiki/tài liệu đào tạo nội bộ
# ---------------------------------------------------------------------------
def _html_to_markdown_like(html: str) -> str:
    """Loại bỏ CSS/JS/navigation/footer, quy nội dung còn lại về dạng markdown
    (heading -> '#'..'######', các thẻ nội dung -> đoạn văn) để tái dùng chiến
    lược structure-aware của Markdown thay vì viết logic cắt HTML riêng."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # Lưu ý: KHÔNG xoá cả thẻ <header> - nhiều trang wiki/CMS đặt tiêu đề chính
    # (h1) bên trong <header> (vd <header><h1>...</h1></header>), xoá cả khối sẽ
    # mất luôn tiêu đề. Chỉ xoá <nav> bên trong (nếu có) - đó mới thực sự là noise.
    for tag in soup(["script", "style", "nav", "footer", "aside", "form", "iframe", "svg", "noscript"]):
        tag.decompose()

    lines: list[str] = []
    content_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "td", "th"]
    for el in soup.find_all(content_tags):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name[0] == "h" and el.name[1:].isdigit():
            lines.append(f"{'#' * int(el.name[1])} {text}")
        else:
            lines.append(text)
        # Xoá khỏi cây ngay sau khi lấy text, tránh bị lặp nội dung khi thẻ cha
        # (vd <li>) cũng khớp content_tags và chứa thẻ con (vd <p>) bên trong.
        el.decompose()
    return "\n\n".join(lines)


def chunk_html(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        raw_html = f.read()
    content = _html_to_markdown_like(raw_html)
    return _chunk_markdown_text(content)


# ---------------------------------------------------------------------------
# Text thuần (.txt)
# ---------------------------------------------------------------------------
def chunk_txt(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    return _chunk_pages([(None, content)])


# ---------------------------------------------------------------------------
# Điều phối theo phần mở rộng file
# ---------------------------------------------------------------------------
CHUNKERS = {
    ".pdf": chunk_pdf,
    ".docx": chunk_docx,
    ".md": chunk_markdown,
    ".txt": chunk_txt,
    ".html": chunk_html,
    ".htm": chunk_html,
    ".png": chunk_image,
    ".jpg": chunk_image,
    ".jpeg": chunk_image,
    ".bmp": chunk_image,
    ".tiff": chunk_image,
    ".webp": chunk_image,
}


def chunk_document(file_path: str, ext: str) -> list[dict]:
    chunker = CHUNKERS.get(ext.lower())
    if chunker is None:
        raise ValueError(f"Định dạng {ext} chưa được hỗ trợ.")
    return chunker(file_path)
