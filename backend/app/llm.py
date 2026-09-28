"""Lớp trừu tượng hoá LLM: hỗ trợ Anthropic Claude, OpenAI, xAI Grok, Google Gemini, Groq.
API key chỉ đọc từ .env (xem config.py) - UI chỉ chọn provider/model đã cấu hình sẵn.
Hỗ trợ multi-turn: `history` là các lượt hội thoại trước đó, dùng làm ngữ cảnh cho LLM."""

from collections.abc import AsyncIterator

from .config import get_settings

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI nội bộ doanh nghiệp. Hãy trả lời câu hỏi của người dùng CHỈ dựa trên "
    "các đoạn ngữ cảnh (context) được cung cấp trong tin nhắn mới nhất, kết hợp với lịch sử hội thoại "
    "phía trên để hiểu đúng ngữ cảnh (ví dụ câu hỏi nối tiếp/nhắc lại điều vừa hỏi). "
    "Nếu ngữ cảnh không chứa thông tin liên quan, hãy nói rõ là bạn không tìm thấy thông tin trong "
    "tài liệu, không được bịa đặt. Khi trả lời, hãy trích dẫn nguồn bằng ký hiệu [n] tương ứng với "
    "số thứ tự nguồn được liệt kê."
)


def build_user_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {block}" for i, block in enumerate(context_blocks))
    return (
        f"### Ngữ cảnh:\n{context}\n\n"
        f"### Câu hỏi:\n{question}\n\n"
        f"### Trả lời (tiếng Việt, có trích dẫn [n]):"
    )


def build_messages(history: list[dict], question: str, context_blocks: list[str]) -> list[dict]:
    """Ghép lịch sử hội thoại (role: user/assistant) với câu hỏi mới (kèm ngữ cảnh RAG vừa truy hồi)
    thành 1 danh sách message chuẩn OpenAI-style, dùng chung cho mọi provider."""
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": build_user_prompt(question, context_blocks)})
    return messages


async def _stream_anthropic(
    api_key: str, model: str, system: str, messages: list[dict], usage_sink: dict | None = None
) -> AsyncIterator[str]:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text
        if usage_sink is not None:
            try:
                usage = (await stream.get_final_message()).usage
                usage_sink["prompt_tokens"] = usage.input_tokens
                usage_sink["completion_tokens"] = usage.output_tokens
            except Exception:  # noqa: BLE001 - đếm token hỏng không được làm hỏng câu trả lời đã stream xong
                pass


async def _stream_openai_compatible(
    api_key: str,
    model: str,
    system: str,
    messages: list[dict],
    base_url: str | None = None,
    usage_sink: dict | None = None,
) -> AsyncIterator[str]:
    """Dùng chung cho OpenAI và mọi provider tương thích chuẩn OpenAI Chat Completions
    (xAI Grok, Groq, DeepSeek, Mistral, Ollama/vLLM local...) - chỉ khác nhau ở base_url."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
    }
    try:
        # include_usage: chunk CUỐI trả về số token thật (chunk đó không có choices).
        stream = await client.chat.completions.create(**payload, stream_options={"include_usage": True})
    except Exception:  # noqa: BLE001
        # Không phải provider "tương thích OpenAI" nào cũng nhận stream_options. Lỗi xảy ra ngay
        # lúc tạo stream, chưa phát token nào, nên gọi lại không kèm tham số này là an toàn.
        stream = await client.chat.completions.create(**payload)

    async for chunk in stream:
        if usage_sink is not None and getattr(chunk, "usage", None):
            usage_sink["prompt_tokens"] = chunk.usage.prompt_tokens
            usage_sink["completion_tokens"] = chunk.usage.completion_tokens
        if not chunk.choices:  # chunk chỉ chứa usage
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _stream_gemini(
    api_key: str, model: str, system: str, messages: list[dict], usage_sink: dict | None = None
) -> AsyncIterator[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    # Gemini dùng role "model" thay vì "assistant"
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    async for chunk in stream:
        meta = getattr(chunk, "usage_metadata", None)
        if usage_sink is not None and meta:
            usage_sink["prompt_tokens"] = getattr(meta, "prompt_token_count", None)
            usage_sink["completion_tokens"] = getattr(meta, "candidates_token_count", None)
        if chunk.text:
            yield chunk.text


_BASE_URLS = {
    "xai": "https://api.x.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Dấu hiệu hết quota/hết tiền nằm trong nội dung lỗi - cho các trường hợp không trả HTTP 429.
_QUOTA_MARKERS = ("insufficient_quota", "billing_not_active", "resource_exhausted", "credit balance is too low")


def is_quota_error(exc: Exception) -> bool:
    """Lỗi do hết quota / hết credit / vượt giới hạn số lượt gọi - khác với key sai, sai tên model...

    Nhận diện qua thuộc tính chung thay vì import lớp lỗi của từng SDK:
      - OpenAI, xAI, Groq (cùng SDK openai) và Anthropic: exc.status_code == 429
      - Gemini (google-genai): exc.code == 429, exc.status == "RESOURCE_EXHAUSTED"
      - Anthropic hết credit lại trả 400 "credit balance is too low" - chỉ bắt được qua nội dung lỗi.
    """
    status = getattr(exc, "status_code", None)
    if status is None and isinstance(getattr(exc, "code", None), int):  # openai dùng .code cho mã lỗi dạng chuỗi
        status = exc.code
    if status == 429:
        return True
    text = f"{getattr(exc, 'status', '')} {getattr(exc, 'code', '')} {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    available = get_settings().models_for(provider)
    if not available:
        raise ValueError(f"Chưa cấu hình model nào cho provider '{provider}' (xem *_MODELS trong .env).")
    return available[0]


def _stream_for(
    provider: str,
    api_key: str,
    model: str,
    system: str,
    messages: list[dict],
    usage_sink: dict | None = None,
) -> AsyncIterator[str]:
    if provider == "anthropic":
        return _stream_anthropic(api_key, model, system, messages, usage_sink)
    if provider in ("openai", "xai", "groq"):
        return _stream_openai_compatible(
            api_key, model, system, messages, base_url=_BASE_URLS.get(provider), usage_sink=usage_sink
        )
    if provider == "gemini":
        return _stream_gemini(api_key, model, system, messages, usage_sink)
    raise ValueError(f"Provider không hợp lệ: {provider}")


async def complete_text(provider: str, api_key: str, model: str | None, system: str, user: str) -> str:
    """Gọi LLM lấy một câu trả lời ngắn, không streaming - dùng cho các bước phụ trợ nội bộ
    (hiện tại: viết lại truy vấn). Tái dùng luôn các hàm stream sẵn có rồi ghép token lại,
    để không phải viết thêm một nhánh gọi API riêng cho từng provider."""
    parts: list[str] = []
    async for token in _stream_for(provider, api_key, _resolve_model(provider, model), system, [{"role": "user", "content": user}]):
        parts.append(token)
    return "".join(parts)


async def generate_answer_stream(
    provider: str,
    api_key: str,
    model: str | None,
    question: str,
    context_blocks: list[str],
    history: list[dict] | None = None,
    usage_sink: dict | None = None,
    prompt_sink: list[str] | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()
    history = history or []
    if settings.max_history_messages > 0:
        history = history[-settings.max_history_messages :]
    # Anthropic bắt buộc messages phải xen kẽ đúng, bắt đầu bằng "user" - nếu bị cắt lệch
    # (ví dụ MAX_HISTORY_MESSAGES là số lẻ) thì bỏ phần tử đầu để tránh lỗi 400 từ provider.
    if history and history[0]["role"] != "user":
        history = history[1:]
    messages = build_messages(history, question, context_blocks)
    if prompt_sink is not None:
        # Để bên gọi ước lượng được số token đầu vào khi provider không trả số liệu thật.
        prompt_sink.append(SYSTEM_PROMPT + "\n" + "\n".join(m["content"] for m in messages))
    async for token in _stream_for(
        provider, api_key, _resolve_model(provider, model), SYSTEM_PROMPT, messages, usage_sink
    ):
        yield token
