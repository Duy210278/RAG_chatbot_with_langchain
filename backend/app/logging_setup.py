"""Log có cấu trúc ra stdout, hai định dạng (LOG_FORMAT):

  - pretty (mặc định): mỗi dòng ngắn, có màu, dành cho người đọc terminal khi phát triển.
  - json: mỗi dòng một object JSON, để đẩy thẳng vào Loki/ELK/CloudWatch mà không phải viết parser.

Trước đây chỉ có JSON, và terminal khi dev rất khó đọc: một lượt hỏi là một dòng JSON ~20 trường,
lẫn với dòng định dạng mặc định của uvicorn, access log của mọi request Streamlit gọi mỗi lần tải
lại, và thanh tiến trình tải model. Module này dọn cả mấy nguồn đó.

Dùng logging của thư viện chuẩn, không thêm dependency.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

_CONFIGURED = False

# Các trường mặc định của LogRecord - phải loại ra để chỉ còn lại trường do mình thêm vào.
# color_message: uvicorn tự gắn vào một số dòng (chứa mã màu ANSI) - in ra JSON chỉ thành rác.
# pretty: bản tóm tắt dành riêng cho định dạng pretty (xem PrettyFormatter).
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName", "color_message", "pretty"}

# GET thành công tới API là UI đọc dữ liệu (Streamlit gọi lại danh sách hội thoại, provider, câu gợi ý...
# mỗi lần rerun) - không mang thông tin gì mà chiếm gần hết terminal. Request lỗi (>= 400) và mọi
# POST/DELETE (hỏi đáp, nạp/xoá tài liệu, phản hồi) vẫn được in.
_QUIET_GET_PREFIXES = ("/health", "/api/v1/")


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


def _access_status(record: logging.LogRecord) -> int | None:
    """Mã trạng thái HTTP của một dòng uvicorn.access - args = (client, method, path, http_version, status)."""
    args = record.args
    if record.name == "uvicorn.access" and isinstance(args, tuple) and len(args) == 5:
        try:
            return int(args[4])
        except (TypeError, ValueError):
            return None
    return None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PrettyFormatter(logging.Formatter):
    """`09:31:10 INFO  query    <nội dung>` - bản ghi có thuộc tính `pretty` thì in đúng bản tóm tắt
    đó thay vì liệt kê hết các trường; bản ghi khác in thêm các trường phụ dạng key=value."""

    _LEVEL_COLOR = {"DEBUG": "2", "INFO": "32", "WARNING": "33", "ERROR": "31", "CRITICAL": "1;31"}
    _LEVEL_SHORT = {"WARNING": "WARN", "CRITICAL": "CRIT"}  # giữ cột thẳng hàng (5 ký tự)
    _NAMES = {"uvicorn.error": "uvicorn", "uvicorn.access": "http", "py.warnings": "warning"}

    def __init__(self, color: bool):
        super().__init__()
        self.color = color

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        short = self._LEVEL_SHORT.get(record.levelname, record.levelname)
        level = self._paint(f"{short:<5}", self._LEVEL_COLOR.get(record.levelname, "0"))
        name = self._NAMES.get(record.name, record.name.removeprefix("rag."))
        message = getattr(record, "pretty", None) or record.getMessage()
        if not hasattr(record, "pretty"):
            extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
            if extras:
                message = f"{message} {self._paint(extras, '2')}"
        status = _access_status(record)
        if status and status >= 400:  # uvicorn luôn ghi access log ở mức INFO - tô màu để request lỗi nổi lên
            message = self._paint(message, "31" if status >= 500 else "33")
        line = f"{self._paint(ts, '2')} {level} {name:<8} {message}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _QuietAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        status = _access_status(record)
        if status is None:
            return True
        _, method, path, _, _ = record.args
        return not (method == "GET" and status < 400 and str(path).startswith(_QUIET_GET_PREFIXES))


def _quiet_third_party() -> None:
    """Thanh tiến trình "Loading weights: 100%|███|" mỗi lần tải model (lặp lại sau mỗi lần --reload)
    và cảnh báo lặt vặt của transformers - không đi qua logging nên phải tắt ở từng thư viện."""
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass


def setup_logging(level: str = "INFO", fmt: str = "pretty", quiet_access: bool = True) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        handler.setFormatter(PrettyFormatter(color=color))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn tự cấu hình handler riêng -> bỏ đi để mọi dòng log đi qua cùng một định dạng.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    if quiet_access:
        logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())

    # Thư viện HTTP nói quá nhiều ở mức INFO, che mất log của chính mình.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel("WARNING")

    logging.captureWarnings(True)  # warnings.warn(...) của thư viện -> cùng định dạng, thay vì in thẳng stderr
    _quiet_third_party()
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
