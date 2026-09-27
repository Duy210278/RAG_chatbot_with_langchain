"""Log có cấu trúc (JSON mỗi dòng) ra stdout.

Trước đây backend không có MỘT dòng log nào - không biết ai hỏi gì, câu nào hỏng, chậm ở khâu
nào. Chọn JSON mỗi dòng thay vì log văn xuôi để về sau đẩy thẳng vào bất kỳ công cụ thu thập
log nào (Loki/ELK/CloudWatch) mà không phải viết parser.

Dùng logging của thư viện chuẩn, không thêm dependency.
"""

import json
import logging
import sys
from datetime import datetime, timezone

_CONFIGURED = False

# Các trường mặc định của LogRecord - phải loại ra để chỉ còn lại trường do mình thêm vào.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn tự cấu hình handler riêng -> bỏ đi để mọi dòng log đi qua cùng một định dạng.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # Thư viện HTTP nói quá nhiều ở mức INFO, che mất log của chính mình.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel("WARNING")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
