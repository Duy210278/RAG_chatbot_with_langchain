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


async def _stream_anthropic(api_key: str, model: str, system: str, messages: list[dict]) -> AsyncIterator[str]:
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


async def _stream_openai_compatible(
    api_key: str, model: str, system: str, messages: list[dict], base_url: str | None = None
) -> AsyncIterator[str]:
    """Dùng chung cho OpenAI và mọi provider tương thích chuẩn OpenAI Chat Completions
    (xAI Grok, Groq, DeepSeek, Mistral, Ollama/vLLM local...) - chỉ khác nhau ở base_url."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _stream_gemini(api_key: str, model: str, system: str, messages: list[dict]) -> AsyncIterator[str]:
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
        if chunk.text:
            yield chunk.text


_BASE_URLS = {
    "xai": "https://api.x.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
}


async def generate_answer_stream(
    provider: str,
    api_key: str,
    model: str | None,
    question: str,
    context_blocks: list[str],
    history: list[dict] | None = None,
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

    if not model:
        available = settings.models_for(provider)
        model = available[0] if available else None
    if not model:
        raise ValueError(f"Chưa cấu hình model nào cho provider '{provider}' (xem *_MODELS trong .env).")

    if provider == "anthropic":
        async for token in _stream_anthropic(api_key, model, SYSTEM_PROMPT, messages):
            yield token
    elif provider in ("openai", "xai", "groq"):
        async for token in _stream_openai_compatible(
            api_key, model, SYSTEM_PROMPT, messages, base_url=_BASE_URLS.get(provider)
        ):
            yield token
    elif provider == "gemini":
        async for token in _stream_gemini(api_key, model, SYSTEM_PROMPT, messages):
            yield token
    else:
        raise ValueError(f"Provider không hợp lệ: {provider}")
