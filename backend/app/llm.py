"""Lớp trừu tượng hoá LLM: hỗ trợ Anthropic Claude và OpenAI, người dùng chọn ở UI (Tab Cấu hình)."""

from collections.abc import AsyncIterator

from .config import get_settings

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI nội bộ doanh nghiệp. Hãy trả lời câu hỏi của người dùng CHỈ dựa trên "
    "các đoạn ngữ cảnh (context) được cung cấp bên dưới. Nếu ngữ cảnh không chứa thông tin liên quan, "
    "hãy nói rõ là bạn không tìm thấy thông tin trong tài liệu, không được bịa đặt. "
    "Khi trả lời, hãy trích dẫn nguồn bằng ký hiệu [n] tương ứng với số thứ tự nguồn được liệt kê."
)


def build_user_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {block}" for i, block in enumerate(context_blocks))
    return (
        f"### Ngữ cảnh:\n{context}\n\n"
        f"### Câu hỏi:\n{question}\n\n"
        f"### Trả lời (tiếng Việt, có trích dẫn [n]):"
    )


async def _stream_anthropic(api_key: str, model: str, system: str, user: str) -> AsyncIterator[str]:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_openai(api_key: str, model: str, system: str, user: str) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate_answer_stream(
    provider: str,
    api_key: str,
    model: str | None,
    question: str,
    context_blocks: list[str],
) -> AsyncIterator[str]:
    settings = get_settings()
    user_prompt = build_user_prompt(question, context_blocks)

    if provider == "anthropic":
        model = model or settings.anthropic_model
        async for token in _stream_anthropic(api_key, model, SYSTEM_PROMPT, user_prompt):
            yield token
    elif provider == "openai":
        model = model or settings.openai_model
        async for token in _stream_openai(api_key, model, SYSTEM_PROMPT, user_prompt):
            yield token
    else:
        raise ValueError(f"Provider không hợp lệ: {provider}")
