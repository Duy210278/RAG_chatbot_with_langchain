"""Chế độ Agent: LLM tự quyết tìm gì, tìm mấy lần, rồi mới đưa tài liệu cho bước soạn câu trả lời.

Chế độ thường chạy một luồng cố định: viết lại câu hỏi -> tìm MỘT lần -> soạn. Lần tìm đó trượt thì
trả lời "không tìm thấy"; câu hỏi nhiều ý ("so sánh nghỉ ốm và nghỉ thai sản") bị dồn vào một truy vấn
nên dễ thiếu một vế. Ở đây LLM được gọi công cụ nhiều lượt (tối đa AGENT_MAX_STEPS): tách câu hỏi,
đọc kết quả, đổi cách diễn đạt khi tìm trượt.

Agent CHỈ thu thập tài liệu. Câu trả lời vẫn do generate_answer_stream() viết y như chế độ thường, nên
trích dẫn [n], kiểm chứng trích dẫn, chỉ báo tin cậy và fallback khi hết quota giữ nguyên - chỉ khác
danh sách đoạn tài liệu đưa vào bước đó.

Lỗi ở lượt đầu (model không gọi được công cụ - một số model ":free" trên OpenRouter, model nhỏ; hết
quota; quá hạn) thì chưa có gì trong tay: đặt fallback_reason để rag.py chạy lại bằng chế độ thường.
Lỗi ở các lượt sau thì dùng luôn những gì đã tìm được.
"""

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from . import models
from .config import get_settings
from .llm import ToolCall, is_quota_error, start_tool_chat, trim_history
from .logging_setup import get_logger
from .observability import StageTimer
from .reranker import get_reranker, rerank_hits
from .retrieval import Hit, hits_for_chunks, retrieve

logger = get_logger("rag.agent")

# Agent chỉ cần đọc đủ để biết đã tìm trúng chưa - nội dung đầy đủ vẫn đi tới bước soạn câu trả lời.
# Bản đầu chỉ hiện 400 ký tự: con số cần tìm thường nằm sâu trong đoạn, agent không thấy nên cứ tìm lại
# đến hết lượt (đo thực tế: tìm trúng ngay lượt 1 nhưng vẫn tìm thêm 4 lần, tốn ~13s).
_PREVIEW_HITS = 6
_PREVIEW_CHARS = 1500
_READ_MORE_CHUNKS = 2
_READ_MORE_CHARS = 1500
_MAX_CALLS_PER_ROUND = 4
_MAX_LISTED_DOCS = 50
# Lịch sử đưa cho agent chỉ để hiểu câu hỏi nối tiếp nói về gì - không cần nguyên văn câu trả lời dài.
_HISTORY_MESSAGES = 6
_HISTORY_CHARS = 600

# Mã tham chiếu một đoạn: TL<số thứ tự tài liệu>#<chunk_index>. Ngắn để model chép lại không sai,
# thay vì bắt nó chép nguyên UUID của tài liệu.
_REF_RE = re.compile(r"\s*\[?\s*(TL\d+)\s*#\s*(\d+)\s*\]?\s*", re.IGNORECASE)

# Không đưa điểm rerank cho agent xem: với mmarco trên tài liệu tiếng Việt, đoạn ĐÚNG cũng chỉ đạt
# 0.04-0.25 (xem answer_check.py), agent thấy điểm thấp sẽ tìm lại mãi. Để nó phán đoán bằng nội dung.
SYSTEM_PROMPT = """Bạn là bộ phận TÌM TÀI LIỆU của trợ lý nội bộ doanh nghiệp. Việc của bạn là gom đủ các \
đoạn tài liệu để một bước khác viết câu trả lời. Bạn KHÔNG viết câu trả lời cho người dùng.

Cách làm:
1. Luôn gọi search_documents ít nhất một lần. Truy vấn phải tự đứng được: chép rõ chủ thể từ lịch sử hội \
thoại vào (đang nói về quy chế nhân sự mà người dùng hỏi "thế còn nghỉ ốm?" thì tìm "chế độ nghỉ ốm \
trong quy chế nhân sự").
2. Câu hỏi có nhiều ý (so sánh, nhiều đối tượng, nhiều bước) thì tách ra, mỗi ý một lần tìm. Các lần tìm \
không phụ thuộc nhau thì gọi cùng lúc trong một lượt.
3. Đọc kết quả. Kết quả chỉ hiện phần đầu của mỗi đoạn; toàn bộ đoạn sẽ được chuyển cho bước viết câu trả \
lời. Vì vậy đoạn tìm được đúng chủ đề thì coi như đã có, KHÔNG tìm lại chỉ vì phần hiện ra chưa có đủ con số \
hay chi tiết. Chỉ tìm lại khi các đoạn nói về chủ đề khác: dùng cách diễn đạt khác (từ đồng nghĩa, thuật \
ngữ hay dùng trong văn bản, tên quy trình, mã số). Không lặp lại truy vấn đã dùng.
4. Giữ nguyên thuật ngữ, mã số, số hiệu văn bản mà người dùng nêu.
5. list_documents: khi cần biết hệ thống đang có những tài liệu nào.
6. read_more: chỉ khi một đoạn đã tìm thấy bị cắt ngang và phần tiếp theo cần cho câu trả lời.
7. Đã đủ thông tin, hoặc đã thử vài cách diễn đạt mà vẫn không thấy, thì DỪNG gọi công cụ và chỉ viết \
một dòng ngắn nói đã tìm được gì.

Bạn có tối đa {max_steps} lượt."""


@dataclass
class _Batch:
    """Kết quả của một lần gọi công cụ, giữ nguyên thứ hạng trong lần đó."""

    hits: list[Hit]
    forced: bool = False  # agent chủ động đọc thêm, không phải một lần tìm


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def _describe_error(exc: Exception, timeout: float | None = None) -> str:
    # asyncio.TimeoutError chỉ trùng với TimeoutError built-in từ Python 3.11; venv hiện chạy 3.10.
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return f"quá {timeout:g}s không phản hồi" if timeout else "quá thời gian chờ"
    if is_quota_error(exc):
        return "LLM hết quota hoặc vượt giới hạn số lượt gọi"
    message = " ".join(str(exc).split())[:200]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class AgentRun:
    """Một lượt hỏi ở chế độ Agent. Dùng:

        run = AgentRun(...)
        async for status in run.run():   # {stage, message} - rag.py chuyển thành SSE "status"
            ...
        if run.fallback_reason: ...      # chưa thu được gì -> chạy chế độ thường
        hits = run.merged_hits()
    """

    def __init__(
        self,
        db: Session,
        timer: StageTimer,
        *,
        provider: str,
        api_key: str,
        model: str | None,
        question: str,
        history: list[dict],
        category: str | None,
        top_k: int,
        use_rerank: bool,
    ):
        self.db, self.timer = db, timer
        self.provider, self.api_key, self.model = provider, api_key, model
        self.question = question
        self.history = history
        # Người dùng đã chọn phạm vi cụ thể thì agent không được tìm ra ngoài phạm vi đó.
        self.category = category if category and category != "ALL" else None
        self.top_k, self.use_rerank = top_k, use_rerank

        self.fallback_reason: str | None = None
        self.steps: list[dict] = []
        self.rounds = 0
        self.stop: str | None = None  # done | no_new | max_steps | error
        self.error: str | None = None
        self.hybrid_used = False
        self.score_types: dict[str, str] = {}  # hit.id -> thang điểm (đoạn đọc thêm có thể khác thang)
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self._batches: list[_Batch] = []
        self._seen: set[str] = set()
        self._aliases: dict[str, str] = {}  # document_id -> "TL1"
        self._alias_docs: dict[str, str] = {}  # "TL1" -> document_id
        self._categories: list[str] = []

    # ------------------------------------------------------------------ vòng lặp chính
    async def run(self) -> AsyncIterator[dict]:
        """stage "agent" = đang làm gì (UI chỉ đổi nhãn); "agent_step" = một bước đã xong (UI ghi thành dòng)."""
        settings = get_settings()
        max_steps = max(1, settings.agent_max_steps)
        if not self.category:
            self._categories = self._available_categories()
        try:
            chat = start_tool_chat(
                self.provider,
                self.api_key,
                self.model,
                SYSTEM_PROMPT.format(max_steps=max_steps),
                self._messages(),
                self._tool_specs(),
            )
        except Exception as exc:  # noqa: BLE001 - vd chưa cấu hình model nào cho provider
            self.fallback_reason = _describe_error(exc)
            return

        for round_no in range(1, max_steps + 1):
            yield {
                "stage": "agent",
                "message": "🤖 Agent đang phân tích câu hỏi..."
                if round_no == 1
                else f"🤖 [{round_no}/{max_steps}] Agent đang xem kết quả...",
            }
            try:
                with self.timer.stage("agent"):
                    step = await asyncio.wait_for(chat.step(), timeout=settings.agent_step_timeout)
            except Exception as exc:  # noqa: BLE001 - gồm cả TimeoutError
                reason = _describe_error(exc, settings.agent_step_timeout)
                logger.warning("Agent lỗi ở lượt %d: %s", round_no, reason)
                if not self._batches:
                    self.fallback_reason = reason
                    return
                self.stop, self.error = "error", reason
                break

            self.rounds = round_no
            self.prompt_tokens += step.prompt_tokens or 0
            self.completion_tokens += step.completion_tokens or 0
            if not step.calls:
                self.stop = "done"
                break

            seen_before = len(self._seen)
            searched = False
            results: list[tuple[ToolCall, str]] = []
            for i, call in enumerate(step.calls):
                if i >= _MAX_CALLS_PER_ROUND:
                    # Vẫn phải trả lời lời gọi này: Anthropic/OpenAI báo lỗi nếu thiếu kết quả của bất kỳ lời gọi nào.
                    results.append((call, f"Bỏ qua: mỗi lượt chỉ được gọi tối đa {_MAX_CALLS_PER_ROUND} công cụ."))
                    continue
                yield {"stage": "agent", "message": f"[{round_no}/{max_steps}] {self._describe_call(call)}"}
                output, label = await self._execute(call)
                results.append((call, output))
                searched |= call.name in ("search_documents", "read_more") and label is not None
                if label:
                    yield {"stage": "agent_step", "message": f"[{round_no}/{max_steps}] {label}"}

            if searched and len(self._seen) == seen_before:
                # Cả lượt không ra thêm đoạn nào: agent đang tìm lại đúng những gì đã có. Hỏi thêm chỉ tốn
                # một lượt LLM + một lần rerank mỗi truy vấn mà không đổi gì ở câu trả lời.
                self.stop = "no_new"
                break
            if round_no == max_steps:
                # Kết quả của lượt cuối vẫn được dùng, chỉ là không hỏi agent thêm lượt nào nữa.
                self.stop = "max_steps"
                break
            chat.add_results(results)

        if not any(not b.forced for b in self._batches):
            # Agent dừng mà chưa tìm lần nào (vd coi câu hỏi là lời chào): vẫn tìm bằng câu hỏi gốc,
            # để câu trả lời luôn có căn cứ như ở chế độ thường.
            _, n, _ = await self._search(self.question, self.category)
            label = f'🔍 Tìm "{self.question}" (tự động, vì agent không tìm lần nào) → {n} đoạn'
            self.steps.append({"tool": "search_documents", "query": self.question, "auto": True, "n_hits": n, "label": label})
            yield {"stage": "agent_step", "message": label}

    def merged_hits(self) -> list[Hit]:
        """Gộp các lần tìm thành danh sách đưa cho bước soạn câu trả lời, tối đa top_k đoạn.

        Xen kẽ theo thứ hạng (hạng 1 của mỗi lần tìm, rồi hạng 2...) thay vì xếp chung theo điểm: điểm
        rerank của mỗi lần tìm được chấm theo một truy vấn khác nhau, xếp chung thì vế câu hỏi có điểm cao
        sẽ chiếm hết chỗ của vế kia.

        Cố ý KHÔNG bỏ lần tìm nào dù không đoạn nào đạt RERANK_THRESHOLD. Bản đầu có bỏ, và câu "công tác
        phí bao nhiêu, đồng phục mấy bộ" mất trắng vế đồng phục: lần tìm đó chỉ đạt dưới ngưỡng nhưng lại là
        nguồn duy nhất của vế này. Với mmarco trên tiếng Việt, ngưỡng không phân biệt được tìm trượt.
        """
        batches = self._batches
        merged: list[Hit] = []
        seen: set[str] = set()
        for rank in range(max((len(b.hits) for b in batches), default=0)):
            for batch in batches:
                if rank < len(batch.hits) and batch.hits[rank].id not in seen:
                    seen.add(batch.hits[rank].id)
                    merged.append(batch.hits[rank])
        return merged[: self.top_k]

    def summary(self) -> dict:
        """Lưu cùng lượt hỏi (quality + query_logs) để UI và trang Giám sát hiển thị lại các bước."""
        max_steps = max(1, get_settings().agent_max_steps)
        stop_label = {
            "done": "agent tự dừng khi đã đủ thông tin",
            "max_steps": f"dừng vì đã dùng hết {max_steps} lượt",
            "no_new": "dừng vì lượt tìm gần nhất không ra thêm đoạn nào mới",
            "error": f"dừng sớm do lỗi: {self.error}",
        }.get(self.stop or "", "")
        return {
            "rounds": self.rounds,
            "searches": sum(1 for s in self.steps if s["tool"] == "search_documents"),
            "steps": self.steps,
            "stop": self.stop,
            "stop_label": stop_label,
            # Câu hỏi nối tiếp: ở chế độ này không có bản viết lại đứng độc lập (xem suggestions.py).
            "follow_up": bool(self.history),
            "tokens": {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens},
        }

    # ------------------------------------------------------------------ dựng prompt
    def _messages(self) -> list[dict]:
        history = trim_history(self.history[-_HISTORY_MESSAGES:], 0)
        messages = [{"role": h["role"], "content": _clip(h["content"], _HISTORY_CHARS)} for h in history]
        scope = f"\n(Người dùng đang giới hạn phạm vi: chỉ tài liệu loại {self.category}.)" if self.category else ""
        messages.append({"role": "user", "content": f"Câu hỏi cần tìm tài liệu: {self.question}{scope}"})
        return messages

    def _tool_specs(self) -> list[dict]:
        search_props: dict = {
            "query": {
                "type": "string",
                "description": "Truy vấn tìm kiếm, tự đứng được (đã chép đủ chủ thể từ lịch sử hội thoại).",
            }
        }
        if self._categories:  # chỉ khi người dùng để "Tất cả tài liệu" và hệ thống có từ 2 loại trở lên
            search_props["category"] = {
                "type": "string",
                "enum": self._categories,
                "description": "Chỉ tìm trong một loại tài liệu. Bỏ trống để tìm trong tất cả.",
            }
        return [
            {
                "name": "search_documents",
                "description": (
                    "Tìm các đoạn tài liệu nội bộ liên quan tới truy vấn (kết hợp tìm theo ngữ nghĩa và theo từ khoá). "
                    "Trả về các đoạn khớp nhất, mỗi đoạn kèm mã tham chiếu dạng TL1#12."
                ),
                "parameters": {"type": "object", "properties": search_props, "required": ["query"]},
            },
            {
                "name": "list_documents",
                "description": "Liệt kê các tài liệu đang có trong hệ thống: tên, loại, số đoạn.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "read_more",
                "description": "Đọc các đoạn liền sau (hoặc liền trước) một đoạn đã tìm thấy, khi đoạn đó bị cắt ngang.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string", "description": "Mã tham chiếu của đoạn, ví dụ TL1#12."},
                        "direction": {
                            "type": "string",
                            "enum": ["after", "before"],
                            "description": "after = đọc phần phía sau (mặc định), before = đọc phần phía trước.",
                        },
                    },
                    "required": ["ref"],
                },
            },
        ]

    # ------------------------------------------------------------------ công cụ
    async def _execute(self, call: ToolCall) -> tuple[str, str | None]:
        """Trả về (kết quả gửi lại cho agent, nhãn hiển thị cho người dùng - None nếu gọi hỏng).
        Mọi lỗi đều trả thành chữ để agent tự sửa lời gọi, không làm hỏng cả lượt hỏi."""
        try:
            if call.name == "search_documents":
                return await self._tool_search(call.args)
            if call.name == "list_documents":
                return self._tool_list_documents()
            if call.name == "read_more":
                return await self._tool_read_more(call.args)
            return f"Lỗi: không có công cụ tên '{call.name}'.", None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Công cụ %s lỗi", call.name, exc_info=True)
            return f"Lỗi khi chạy công cụ {call.name}: {_describe_error(exc)}", None

    @staticmethod
    def _describe_call(call: ToolCall) -> str:
        if call.name == "search_documents":
            return f'🔍 Đang tìm: "{call.args.get("query", "")}"...'
        if call.name == "list_documents":
            return "📚 Đang xem danh sách tài liệu..."
        if call.name == "read_more":
            return f"📖 Đang đọc thêm {call.args.get('ref', '')}..."
        return f"🛠️ {call.name}..."

    async def _tool_search(self, args: dict) -> tuple[str, str | None]:
        query = str(args.get("query") or "").strip()
        if not query:
            return "Lỗi: thiếu tham số query.", None
        requested = args.get("category")
        category = self.category or (requested if requested in self._categories else None)
        output, n, n_new = await self._search(query, category)
        where = f" (loại {category})" if category and not self.category else ""
        label = f'🔍 Tìm "{query}"{where} → {n} đoạn' + (f" ({n_new} mới)" if n_new != n else "")
        self.steps.append(
            {"tool": "search_documents", "query": query, "category": category, "n_hits": n, "n_new": n_new, "label": label}
        )
        return output, label

    async def _search(self, query: str, category: str | None) -> tuple[str, int, int]:
        """Một lần tìm giống hệt chế độ thường (truy hồi lai -> rerank), chỉ khác truy vấn do agent viết."""
        settings = get_settings()
        limit = max(settings.rerank_candidates, self.top_k) if self.use_rerank else self.top_k
        with self.timer.stage("retrieve"):
            hits, hybrid = await retrieve(self.db, query, limit=limit, category=category)
        self.hybrid_used |= hybrid
        if self.use_rerank and hits:
            with self.timer.stage("rerank"):
                hits = await asyncio.to_thread(rerank_hits, query, hits, self.top_k)
        else:
            hits = hits[: self.top_k]

        score_type = "rerank" if self.use_rerank else ("hybrid_rrf" if hybrid else "cosine")
        new_ids = {h.id for h in hits} - self._seen
        for hit in hits:
            self.score_types.setdefault(hit.id, score_type)
        self._seen |= new_ids
        self._batches.append(_Batch(hits))
        return self._format_search(query, hits, new_ids), len(hits), len(new_ids)

    def _format_search(self, query: str, hits: list[Hit], new_ids: set[str]) -> str:
        if not hits:
            return f'Không tìm thấy đoạn nào cho "{query}". Hãy thử cách diễn đạt khác.'
        lines = [f'Kết quả cho "{query}": {len(hits)} đoạn ({len(new_ids)} đoạn mới).']
        for hit in hits[:_PREVIEW_HITS]:
            head = f"[{self._ref(hit)}] {hit.payload.get('title', 'Không rõ')}"
            if hit.payload.get("page_number"):
                head += f" — trang {hit.payload['page_number']}"
            if hit.id in new_ids:
                lines.append(f"{head}\n{_clip(hit.payload['content'], _PREVIEW_CHARS)}")
            else:
                lines.append(f"{head} (đã có ở lần tìm trước)")
        if len(hits) > _PREVIEW_HITS:
            lines.append(f"(Còn {len(hits) - _PREVIEW_HITS} đoạn xếp hạng thấp hơn không hiện ở đây nhưng vẫn được giữ lại.)")
        return "\n\n".join(lines)

    def _tool_list_documents(self) -> tuple[str, str]:
        doc = models.Document
        q = self.db.query(doc).filter(doc.status == "COMPLETED")
        if self.category:
            q = q.filter(doc.category == self.category)
        docs = q.order_by(doc.created_at).limit(_MAX_LISTED_DOCS + 1).all()
        shown = docs[:_MAX_LISTED_DOCS]
        label = f"📚 Xem danh sách tài liệu → {len(shown)} tài liệu"
        self.steps.append({"tool": "list_documents", "n": len(shown), "label": label})
        if not shown:
            return "Hệ thống chưa có tài liệu nào trong phạm vi này.", label
        lines = [f"Có {len(shown)} tài liệu" + (" (chỉ liệt kê một phần):" if len(docs) > len(shown) else ":")]
        lines += [f"[{self._alias(d.id)}] {d.title} — loại {d.category}, {d.chunk_count} đoạn" for d in shown]
        return "\n".join(lines), label

    async def _tool_read_more(self, args: dict) -> tuple[str, str | None]:
        ref = str(args.get("ref") or "")
        match = _REF_RE.fullmatch(ref)
        doc_id = self._alias_docs.get(match.group(1).upper()) if match else None
        if not doc_id:
            return f"Lỗi: mã tham chiếu '{ref}' không hợp lệ - cần dạng TL1#12 lấy từ kết quả tìm kiếm.", None
        idx = int(match.group(2))
        before = args.get("direction") == "before"
        indices = range(max(0, idx - _READ_MORE_CHUNKS), idx) if before else range(idx + 1, idx + 1 + _READ_MORE_CHUNKS)

        chunk = models.DocumentChunk
        rows = (
            self.db.query(chunk.id)
            .filter(chunk.document_id == doc_id, chunk.chunk_index.in_(list(indices)))
            .order_by(chunk.chunk_index)
            .all()
        )
        hits = hits_for_chunks(self.db, [r.id for r in rows], source="agent_read")
        if hits and self.use_rerank:
            # Chấm theo câu hỏi gốc để đoạn đọc thêm có cùng thang điểm với các đoạn tìm được.
            with self.timer.stage("rerank"):
                scores = await asyncio.to_thread(get_reranker().score, self.question, [h.payload["content"] for h in hits])
            hits = [h.model_copy(update={"score": s}) for h, s in zip(hits, scores)]
        score_type = "rerank" if self.use_rerank else "read"
        for hit in hits:
            self.score_types.setdefault(hit.id, score_type)
        self._seen |= {h.id for h in hits}
        self._batches.append(_Batch(hits, forced=True))

        title = hits[0].payload.get("title", "tài liệu") if hits else ref
        side = "trước" if before else "sau"
        label = f"📖 Đọc thêm {title} (phía {side} đoạn #{idx}) → {len(hits)} đoạn"
        self.steps.append({"tool": "read_more", "ref": ref, "direction": "before" if before else "after", "n_hits": len(hits), "label": label})
        if not hits:
            return f"Không còn đoạn nào phía {side} {ref}.", label
        return "\n\n".join(f"[{self._ref(h)}]\n{_clip(h.payload['content'], _READ_MORE_CHARS)}" for h in hits), label

    # ------------------------------------------------------------------ tiện ích
    def _alias(self, document_id: str) -> str:
        if document_id not in self._aliases:
            alias = f"TL{len(self._aliases) + 1}"
            self._aliases[document_id] = alias
            self._alias_docs[alias] = document_id
        return self._aliases[document_id]

    def _ref(self, hit: Hit) -> str:
        return f"{self._alias(hit.payload['document_id'])}#{hit.payload.get('chunk_index')}"

    def _available_categories(self) -> list[str]:
        doc = models.Document
        rows = self.db.query(doc.category).filter(doc.status == "COMPLETED").distinct().all()
        categories = sorted({r[0] for r in rows if r[0]})
        return categories if len(categories) >= 2 else []
