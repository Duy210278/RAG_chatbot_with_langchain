"""Lớp trừu tượng hoá LLM: hỗ trợ Anthropic Claude, OpenAI, xAI Grok, Google Gemini, Groq.
API key chỉ đọc từ .env (xem config.py) - UI chỉ chọn provider/model đã cấu hình sẵn.
Hỗ trợ multi-turn: `history` là các lượt hội thoại trước đó, dùng làm ngữ cảnh cho LLM.
Chế độ Agent dùng thêm phần gọi công cụ (tool calling) ở cuối file - xem start_tool_chat()."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
    "openrouter": "https://openrouter.ai/api/v1",
}
_OPENAI_COMPATIBLE = ("openai", "xai", "groq", "openrouter")

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
    if provider in _OPENAI_COMPATIBLE:
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


def trim_history(history: list[dict], max_messages: int) -> list[dict]:
    """Giữ max_messages tin nhắn gần nhất (<= 0: giữ hết).
    Anthropic bắt buộc messages phải xen kẽ đúng, bắt đầu bằng "user" - nếu bị cắt lệch
    (ví dụ MAX_HISTORY_MESSAGES là số lẻ) thì bỏ phần tử đầu để tránh lỗi 400 từ provider."""
    if max_messages > 0:
        history = history[-max_messages:]
    if history and history[0]["role"] != "user":
        history = history[1:]
    return history


async def generate_answer_stream(
    provider: str,
    api_key: str,
    model: str | None,
    question: str,
    context_blocks: list[str],
    history: list[dict] | None = None,
    usage_sink: dict | None = None,
    prompt_sink: list[dict] | None = None,
) -> AsyncIterator[str]:
    history = trim_history(history or [], get_settings().max_history_messages)
    messages = build_messages(history, question, context_blocks)
    resolved_model = _resolve_model(provider, model)
    if prompt_sink is not None:
        # Đúng những gì gửi đi (sau khi đã cắt lịch sử): để UI hiển thị prompt, và để bên gọi
        # ước lượng token đầu vào khi provider không trả số liệu thật (xem flatten_prompt).
        prompt_sink.append({"model": resolved_model, "system": SYSTEM_PROMPT, "messages": messages})
    async for token in _stream_for(provider, api_key, resolved_model, SYSTEM_PROMPT, messages, usage_sink):
        yield token


def flatten_prompt(prompt: dict | None) -> str:
    """Toàn bộ prompt thành một chuỗi - dùng để ước lượng số token đầu vào."""
    if not prompt:
        return ""
    return prompt["system"] + "\n" + "\n".join(m["content"] for m in prompt["messages"])


# ---------------------------------------------------------------------------
# Gọi LLM có công cụ (tool calling) - dùng cho chế độ Agent (agent.py)
# ---------------------------------------------------------------------------
# Ba họ API mô tả công cụ và trả lời gọi công cụ theo ba định dạng khác nhau, nên mỗi họ một lớp nhỏ
# giữ hội thoại ở đúng định dạng gốc của nó. agent.py chỉ thấy step() / add_results(), không cần
# biết đang nói chuyện với provider nào. Không stream: các lượt này chỉ là quyết định nội bộ
# (tìm gì tiếp), người dùng thấy tiến trình qua sự kiện "status" chứ không qua token.


@dataclass
class ToolCall:
    id: str  # Gemini có thể không cấp id - khi đó là chuỗi rỗng
    name: str
    args: dict


@dataclass
class ToolStep:
    """Một lượt gọi LLM có công cụ: không có lời gọi công cụ nào nghĩa là model đã dừng."""

    text: str
    calls: list[ToolCall]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _AnthropicToolChat:
    def __init__(self, api_key: str, model: str, system: str, messages: list[dict], tools: list[dict]):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model, self._system = model, system
        self._messages = list(messages)
        self._tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

    async def step(self) -> ToolStep:
        resp = await self._client.messages.create(
            model=self._model, max_tokens=1024, system=self._system, messages=self._messages, tools=self._tools
        )
        self._messages.append({"role": "assistant", "content": resp.content})
        return ToolStep(
            text="".join(b.text for b in resp.content if b.type == "text"),
            calls=[ToolCall(b.id, b.name, dict(b.input or {})) for b in resp.content if b.type == "tool_use"],
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

    def add_results(self, results: list[tuple[ToolCall, str]]) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call.id, "content": out} for call, out in results],
            }
        )


class _OpenAIToolChat:
    """OpenAI và mọi provider tương thích (xAI, Groq, OpenRouter) - như _stream_openai_compatible."""

    def __init__(
        self, api_key: str, model: str, system: str, messages: list[dict], tools: list[dict], base_url: str | None
    ):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._messages = [{"role": "system", "content": system}, *messages]
        self._tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools
        ]

    async def step(self) -> ToolStep:
        resp = await self._client.chat.completions.create(model=self._model, messages=self._messages, tools=self._tools)
        msg = resp.choices[0].message
        raw_calls = [tc for tc in (msg.tool_calls or []) if getattr(tc, "type", "function") == "function"]
        calls = []
        for tc in raw_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}  # agent.py báo thiếu tham số cho model, model tự gọi lại
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {}))

        assistant: dict = {"role": "assistant", "content": msg.content or ""}
        if raw_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in raw_calls
            ]
        self._messages.append(assistant)
        usage = resp.usage
        return ToolStep(
            text=msg.content or "",
            calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    def add_results(self, results: list[tuple[ToolCall, str]]) -> None:
        for call, out in results:
            self._messages.append({"role": "tool", "tool_call_id": call.id, "content": out})


class _GeminiToolChat:
    def __init__(self, api_key: str, model: str, system: str, messages: list[dict], tools: list[dict]):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        self._config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
                        )
                        for t in tools
                    ]
                )
            ],
            # Agent tự thực thi công cụ - không để SDK tự gọi hàm thay.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def step(self) -> ToolStep:
        resp = await self._client.aio.models.generate_content(
            model=self._model, contents=self._contents, config=self._config
        )
        content = resp.candidates[0].content if resp.candidates else None
        parts = (content.parts if content else None) or []
        if content is not None:
            # Gửi lại NGUYÊN nội dung model trả về: model có "thinking" gắn thought_signature vào phần gọi
            # hàm, thiếu nó thì lượt sau bị Gemini từ chối.
            self._contents.append(content)
        meta = resp.usage_metadata
        return ToolStep(
            # Không dùng resp.text: SDK in cảnh báo mỗi khi phản hồi có phần không phải chữ (gọi hàm).
            text="".join(p.text for p in parts if p.text and not p.thought),
            calls=[
                ToolCall(p.function_call.id or "", p.function_call.name, dict(p.function_call.args or {}))
                for p in parts
                if p.function_call
            ],
            prompt_tokens=getattr(meta, "prompt_token_count", None),
            completion_tokens=getattr(meta, "candidates_token_count", None),
        )

    def add_results(self, results: list[tuple[ToolCall, str]]) -> None:
        types = self._types
        self._contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(id=call.id or None, name=call.name, response={"result": out})
                    )
                    for call, out in results
                ],
            )
        )


def start_tool_chat(
    provider: str, api_key: str, model: str | None, system: str, messages: list[dict], tools: list[dict]
):
    """Mở một hội thoại có công cụ. tools: [{name, description, parameters (JSON Schema)}].

    Trả về đối tượng có:
      - await step() -> ToolStep: gọi LLM một lượt
      - add_results([(ToolCall, chuỗi kết quả)]): gửi kết quả công cụ cho lượt kế tiếp. Phải trả lời
        ĐỦ mọi lời gọi của lượt trước - Anthropic và OpenAI đều báo lỗi 400 nếu thiếu.
    """
    resolved = _resolve_model(provider, model)
    if provider == "anthropic":
        return _AnthropicToolChat(api_key, resolved, system, messages, tools)
    if provider in _OPENAI_COMPATIBLE:
        return _OpenAIToolChat(api_key, resolved, system, messages, tools, base_url=_BASE_URLS.get(provider))
    if provider == "gemini":
        return _GeminiToolChat(api_key, resolved, system, messages, tools)
    raise ValueError(f"Provider không hợp lệ: {provider}")
