"""Streaming OpenAI-compatible chat client with tool-call accumulation."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

from src.i18n import normalize_locale, tr

from .providers import ModelSpec, ProviderRegistry

logger = logging.getLogger("hu.llm")
TRAFFIC = os.environ.get("HU_LLM_LOG_TRAFFIC", "").strip().lower() in ("1", "true", "yes", "on")

_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{6,}"),
    re.compile(r"\bsk-[A-Za-z0-9._\-]{6,}\b"),
    re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|secret)\b[\"'\s:=]+)[A-Za-z0-9._\-]{8,}"),
)


def redact_secrets(text: str, api_key: str = "") -> str:
    """Remove credentials from provider error text before it leaves the client."""
    cleaned = str(text or "")
    if api_key and len(api_key) >= 4:
        cleaned = cleaned.replace(api_key, "[REDACTED]")

    def replacement(match: re.Match[str]) -> str:
        prefix = match.group(1) if match.groups() else ""
        return f"{prefix}[REDACTED]"

    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


class LLMError(Exception):
    """Typed error surfaced to the UI."""

    def __init__(self, kind: str, detail: str, status: int = 0, api_key: str = "") -> None:
        detail = redact_secrets(detail, api_key)
        super().__init__(f"[{kind}] {detail}")
        self.kind = kind
        self.detail = detail
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> Optional[dict]:
        text = (self.arguments or "").strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    estimated_usage: bool = False

    def assistant_message(self, echo_reasoning: bool) -> dict:
        message: Dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if echo_reasoning and self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments or "{}"},
                }
                for call in self.tool_calls
            ]
        return message

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("input_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("output_tokens", 0))


def estimate_tokens(text: str) -> int:
    """Cheap bilingual token estimate (CJK chars count as one token)."""
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk + other // 4)


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for part in value:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part.get("content"), str):
                    chunks.append(part["content"])
        return "".join(chunks)
    return ""


def _usage_from(payload: dict) -> dict:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = (
        prompt_details.get("cached_tokens")
        if isinstance(prompt_details, dict)
        else None
    )
    if cached_tokens is None:
        cached_tokens = usage.get("prompt_cache_hit_tokens")
    if cached_tokens is None:
        cached_tokens = usage.get("cache_read_input_tokens")
    if cached_tokens is None:
        cached_tokens = usage.get("cached_tokens")
    cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
    if cache_miss_tokens is None:
        cache_miss_tokens = max(0, int(input_tokens or 0) - int(cached_tokens or 0))
    if not input_tokens and not output_tokens:
        return {}
    return {
        "input_tokens": int(input_tokens or 0),
        "cached_input_tokens": int(cached_tokens or 0),
        "cache_miss_input_tokens": int(cache_miss_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _classify_text(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    if "invalid api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "auth"
    if (
        "rate limit" in lowered
        or "quota" in lowered
        or "too many requests" in lowered
        or "concurrency limit" in lowered
        or "concurrent" in lowered
        or "retry later" in lowered
    ):
        return "rate_limit"
    if any(phrase in lowered for phrase in (
        "context length",
        "context_length",
        "maximum context",
        "too many tokens",
        "prompt is too long",
        "input is too long",
    )):
        return "context_length"
    if "stream" in lowered and any(
        phrase in lowered for phrase in ("not support", "unsupported", "not enabled", "disabled")
    ):
        return "stream_unsupported"
    return None


def classify_http(status: int, body: str) -> str:
    """Classify an HTTP failure, preferring the status code over body text."""
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status == 408:
        return "timeout"
    if status >= 500:
        return "server"
    text_kind = _classify_text(body)
    if text_kind:
        return text_kind
    if status in (400, 404, 415, 422) or "invalid parameter" in (body or "").lower():
        return "param"
    return "http"


class LLMClient:
    """Thin client around ``/chat/completions`` with SSE streaming."""

    def __init__(
        self,
        registry: ProviderRegistry,
        connect_timeout: float = 20.0,
        read_timeout: float = 300.0,
        locale: str = "zh-CN",
        user_id: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.locale = normalize_locale(locale, "zh-CN") or "zh-CN"
        # DeepSeek uses user_id as a KV-cache isolation key.  Other providers
        # may ignore it; only send it when explicitly configured.
        self.user_id = str(user_id or "").strip() or None
        self._local = threading.local()
        self._sessions: List[requests.Session] = []
        self._session_lock = threading.Lock()
        # Optional fields a provider has rejected for this model. Kept across
        # requests so we do not pay for a failed request on every LLM turn.
        self._degraded: Dict[str, Set[str]] = {}
        self._degraded_lock = threading.Lock()

    def _get_session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
            with self._session_lock:
                self._sessions.append(session)
        return session

    def _degraded_fields(self, spec: ModelSpec) -> Set[str]:
        with self._degraded_lock:
            return set(self._degraded.get(f"{spec.provider_id}:{spec.id}", set()))

    def _remember_degraded(self, spec: ModelSpec, fields: Set[str]) -> None:
        if not fields:
            return
        with self._degraded_lock:
            self._degraded.setdefault(f"{spec.provider_id}:{spec.id}", set()).update(fields)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(
        self,
        spec: ModelSpec,
        messages: List[dict],
        tools: Optional[List[dict]],
        params: dict,
        on_event: Callable[[dict], None],
        cancel: Optional[Callable[[], bool]] = None,
        tool_choice: Optional[str] = None,
    ) -> ChatResult:
        api_key = ProviderRegistry.get_api_key(spec, locale=self.locale)
        payload = self.registry.build_payload(
            spec, messages, tools, params, stream=True, tool_choice=tool_choice
        )
        if self.user_id:
            payload["user_id"] = self.user_id
        dropped = self._degraded_fields(spec)
        if dropped:
            payload = _apply_payload_drops(payload, dropped)
        self._log_request(spec, payload)

        attempts = 0
        last_error: Optional[LLMError] = None
        while attempts < 3:
            if cancel and cancel():
                raise LLMError("cancelled", tr(self.locale, "client.cancelled"))
            attempts += 1
            try:
                result = self._stream_once(spec, api_key, payload, on_event, cancel)
                self._log_response(spec, result)
                return result
            except LLMError as exc:
                last_error = exc
                if exc.kind == "cancelled":
                    raise
                if exc.kind == "rate_limit" and attempts == 1:
                    time.sleep(1.5)
                    continue
                if exc.kind == "param" and attempts < 3:
                    reduced = self._reduce_payload(payload, exc.detail)
                    if reduced is not None:
                        payload, removed = reduced
                        self._remember_degraded(spec, removed)
                        on_event(
                            {
                                "type": "note",
                                "level": "warn",
                                "text": tr(self.locale, "client.param_degraded"),
                            }
                        )
                        continue
                raise
            except requests.Timeout as exc:
                raise LLMError(
                    "timeout",
                    tr(self.locale, "client.timeout", error=exc),
                    api_key=api_key,
                ) from exc
            except requests.ConnectionError as exc:
                raise LLMError(
                    "network",
                    tr(self.locale, "client.network", error=exc),
                    api_key=api_key,
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise LLMError(
                    "unknown",
                    tr(self.locale, "client.unknown_detail", error=f"{type(exc).__name__}: {exc}"),
                    api_key=api_key,
                ) from exc
        if last_error:
            raise last_error
        raise LLMError("unknown", tr(self.locale, "client.unknown"))

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream_once(
        self,
        spec: ModelSpec,
        api_key: str,
        payload: dict,
        on_event: Callable[[dict], None],
        cancel: Optional[Callable[[], bool]] = None,
    ) -> ChatResult:
        start = time.perf_counter()
        response = self._get_session().post(
            self.registry.endpoint(spec),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json=payload,
            stream=True,
            timeout=(self.connect_timeout, self.read_timeout),
        )
        if response.status_code >= 400:
            body = response.text[:4000]
            kind = classify_http(response.status_code, body)
            response.close()
            if kind == "stream_unsupported":
                return self._non_stream_once(spec, api_key, payload, start, on_event, cancel)
            raise LLMError(
                kind,
                f"HTTP {response.status_code}: {body}",
                response.status_code,
                api_key=api_key,
            )

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError(
                    "server",
                    tr(self.locale, "client.invalid_json", error=exc),
                    api_key=api_key,
                ) from exc
            finally:
                response.close()
            return self._parse_completion(api_key, payload, data, start, on_event)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: Dict[int, dict] = {}
        id_map: Dict[str, int] = {}
        index_map: Dict[int, int] = {}
        finish_reason = ""
        usage: dict = {}
        saw_event = False
        bad_lines = 0
        stream_error: Optional[requests.RequestException] = None

        try:
            for raw_line in response.iter_lines(decode_unicode=False):
                if cancel and cancel():
                    raise LLMError("cancelled", tr(self.locale, "client.cancelled"))
                if raw_line is None:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_text = line[5:].strip()
                elif line.startswith("{"):
                    data_text = line
                else:
                    continue
                if data_text == "[DONE]":
                    break
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                saw_event = True
                if isinstance(event.get("error"), dict):
                    error = event["error"]
                    detail = error.get("message") or json.dumps(error, ensure_ascii=False)
                    code = error.get("code") or error.get("status")
                    try:
                        status = int(code)
                    except (TypeError, ValueError):
                        status = 0
                    raise LLMError(
                        classify_http(status, str(detail)),
                        str(detail),
                        status,
                        api_key=api_key,
                    )

                chunk_usage = _usage_from(event)
                if chunk_usage:
                    usage = chunk_usage

                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or choice.get("message") or {}

                reasoning = _normalize_text(delta.get("reasoning_content") or delta.get("reasoning"))
                if reasoning:
                    reasoning_parts.append(reasoning)
                    on_event({"type": "reasoning_delta", "delta": reasoning})

                text = _normalize_text(delta.get("content"))
                if text:
                    content_parts.append(text)
                    on_event({"type": "content_delta", "delta": text})

                calls = delta.get("tool_calls")
                if isinstance(calls, list):
                    for raw_call in calls:
                        self._accumulate_tool_call(tool_calls, raw_call, on_event, id_map, index_map)
                legacy = delta.get("function_call")
                if isinstance(legacy, dict):
                    self._accumulate_tool_call(
                        tool_calls,
                        {"index": 0, "function": legacy},
                        on_event,
                        id_map,
                        index_map,
                    )
        except LLMError:
            raise
        except requests.RequestException as exc:
            stream_error = exc
        finally:
            response.close()

        if cancel and cancel():
            raise LLMError("cancelled", tr(self.locale, "client.cancelled"))
        latency_ms = (time.perf_counter() - start) * 1000.0

        if not saw_event:
            if bad_lines:
                on_event(
                    {
                        "type": "note",
                        "level": "warn",
                        "text": tr(self.locale, "client.stream_format"),
                    }
                )
            # Some gateways silently return an empty SSE stream or pretty JSON.
            return self._non_stream_once(spec, api_key, payload, start, on_event, cancel)

        if stream_error is not None:
            if not tool_calls and not content_parts and not reasoning_parts:
                raise LLMError(
                    "network",
                    tr(self.locale, "client.stream_interrupted", error=stream_error),
                    api_key=api_key,
                )
            on_event(
                {
                    "type": "note",
                    "level": "warn",
                    "text": tr(self.locale, "client.stream_interrupted_keep"),
                }
            )

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        content, extra_reasoning = _split_think_tags(content)
        reasoning = (reasoning + extra_reasoning).strip()
        content = content.strip()

        if not usage:
            prompt_text = json.dumps(payload.get("messages", []), ensure_ascii=False)
            usage = {
                "input_tokens": estimate_tokens(prompt_text),
                "cached_input_tokens": 0,
                "cache_miss_input_tokens": 0,
                "output_tokens": estimate_tokens(content + reasoning),
                "reasoning_tokens": 0,
                "total_tokens": 0,
            }
            estimated = True
        else:
            estimated = False

        return ChatResult(
            content=content,
            reasoning=reasoning,
            tool_calls=self._finalize_tool_calls(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            estimated_usage=estimated,
        )

    def _non_stream_once(
        self,
        spec: ModelSpec,
        api_key: str,
        payload: dict,
        start: float,
        on_event: Callable[[dict], None],
        cancel: Optional[Callable[[], bool]] = None,
    ) -> ChatResult:
        if cancel and cancel():
            raise LLMError("cancelled", tr(self.locale, "client.cancelled"))
        fallback = dict(payload)
        fallback["stream"] = False
        fallback.pop("stream_options", None)
        fallback.pop("tool_stream", None)
        response = self._get_session().post(
            self.registry.endpoint(spec),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=fallback,
            timeout=(self.connect_timeout, self.read_timeout),
        )
        if response.status_code >= 400:
            body = response.text[:4000]
            kind = classify_http(response.status_code, body)
            response.close()
            raise LLMError(
                kind,
                f"HTTP {response.status_code}: {body}",
                response.status_code,
                api_key=api_key,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                "server",
                tr(self.locale, "client.invalid_json", error=exc),
                api_key=api_key,
            ) from exc
        finally:
            response.close()
        if not isinstance(data, dict):
            raise LLMError(
                "server",
                tr(self.locale, "client.non_object_json"),
                api_key=api_key,
            )
        return self._parse_completion(api_key, payload, data, start, on_event)

    def _parse_completion(
        self,
        api_key: str,
        payload: dict,
        data: dict,
        start: float,
        on_event: Callable[[dict], None],
    ) -> ChatResult:
        if not isinstance(data, dict):
            raise LLMError(
                "server",
                tr(self.locale, "client.non_object_json"),
                api_key=api_key,
            )
        if isinstance(data.get("error"), dict):
            error = data["error"]
            detail = error.get("message") or json.dumps(error, ensure_ascii=False)
            code = error.get("code") or error.get("status")
            try:
                status = int(code)
            except (TypeError, ValueError):
                status = 0
            raise LLMError(classify_http(status, str(detail)), str(detail), status, api_key=api_key)
        choice = (data.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        content = _normalize_text(message.get("content"))
        reasoning = _normalize_text(message.get("reasoning_content") or message.get("reasoning"))
        content, extra = _split_think_tags(content)
        reasoning = (reasoning + extra).strip()
        if content:
            on_event({"type": "content_delta", "delta": content})
        if reasoning:
            on_event({"type": "reasoning_delta", "delta": reasoning})
        tool_calls: Dict[int, dict] = {}
        id_map: Dict[str, int] = {}
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            self._accumulate_tool_call(tool_calls, {**raw_call, "index": index}, on_event, id_map)
        usage = _usage_from(data)
        estimated = not bool(usage)
        if not usage:
            usage = {
                "input_tokens": estimate_tokens(json.dumps(payload.get("messages", []), ensure_ascii=False)),
                "cached_input_tokens": 0,
                "cache_miss_input_tokens": 0,
                "output_tokens": estimate_tokens(content + reasoning),
                "reasoning_tokens": 0,
                "total_tokens": 0,
            }
        return ChatResult(
            content=content.strip(),
            reasoning=reasoning,
            tool_calls=self._finalize_tool_calls(tool_calls),
            finish_reason=str(choice.get("finish_reason", "")),
            usage=usage,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            estimated_usage=estimated,
        )

    # ------------------------------------------------------------------
    # Tool call fragments
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_tool_call(
        store: Dict[int, dict],
        raw_call: dict,
        on_event: Callable[[dict], None],
        id_map: Optional[Dict[str, int]] = None,
        index_map: Optional[Dict[int, int]] = None,
    ) -> None:
        if not isinstance(raw_call, dict):
            return
        id_map = id_map if id_map is not None else {}
        index_map = index_map if index_map is not None else {}
        call_id = str(raw_call.get("id") or "")
        raw_index = raw_call.get("index")
        if raw_index is None:
            if call_id and call_id in id_map:
                provider_index = id_map[call_id]
            elif call_id:
                provider_index = (max(index_map) + 1) if index_map else 0
            else:
                provider_index = max(index_map) if index_map else 0
        else:
            try:
                provider_index = int(raw_index)
            except (TypeError, ValueError):
                provider_index = 0
        if call_id:
            id_map[call_id] = provider_index
        index = index_map.setdefault(provider_index, len(index_map))
        entry = store.setdefault(
            index,
            {"id": "", "name": "", "arguments": "", "started": False},
        )
        if call_id:
            entry["id"] = call_id
        function = raw_call.get("function") or {}
        name_fragment = _normalize_text(function.get("name"))
        if name_fragment:
            current = entry["name"]
            if not current:
                entry["name"] = name_fragment
            elif name_fragment != current and not current.endswith(name_fragment):
                entry["name"] = current + name_fragment
        if not entry["started"] and (entry["name"] or entry["id"]):
            entry["started"] = True
            on_event(
                {
                    "type": "tool_call_start",
                    "index": index,
                    "call_id": entry["id"] or f"call_{index}",
                    "name": entry["name"],
                }
            )
        args = function.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        if isinstance(args, str) and args:
            entry["arguments"] += args
            on_event(
                {
                    "type": "tool_call_args_delta",
                    "index": index,
                    "call_id": entry["id"] or f"call_{index}",
                    "delta": args,
                }
            )

    @staticmethod
    def _finalize_tool_calls(store: Dict[int, dict]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        seen_ids: Set[str] = set()
        for index in sorted(store):
            entry = store[index]
            if not entry.get("name"):
                continue
            call_id = entry.get("id") or f"call_{index}"
            if call_id in seen_ids:
                suffix = 2
                while f"{call_id}_{suffix}" in seen_ids:
                    suffix += 1
                call_id = f"{call_id}_{suffix}"
            seen_ids.add(call_id)
            calls.append(
                ToolCall(
                    id=call_id,
                    name=entry["name"],
                    arguments=entry.get("arguments") or "{}",
                )
            )
        return calls

    # ------------------------------------------------------------------
    # Payload degradation on 400
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_payload(payload: dict, detail: str) -> Optional[Tuple[dict, Set[str]]]:
        """Drop optional vendor fields named in a provider rejection."""
        reduced = json.loads(json.dumps(payload))
        text = (detail or "").lower()
        removed: Set[str] = set()

        if "max_tokens" in text and reduced.get("max_tokens", 0) > 8192:
            reduced["max_tokens"] = 8192
            removed.add("max_tokens")

        for key in ("stream_options", "tool_stream", "preserve_thinking", "thinking", "reasoning_effort", "enable_thinking"):
            if key in text and key in reduced:
                reduced.pop(key, None)
                removed.add(key)

        if "reasoning" in text:
            messages = reduced.get("messages", [])
            for message in messages:
                if isinstance(message, dict) and "reasoning_content" in message:
                    message.pop("reasoning_content", None)
                    removed.add("reasoning_content")
        if not removed:
            return None
        return reduced, removed

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_request(spec: ModelSpec, payload: dict) -> None:
        if not TRAFFIC:
            return
        summary = {
            "provider": spec.provider_id,
            "model": spec.id,
            "stream": payload.get("stream"),
            "max_tokens": payload.get("max_tokens"),
            "thinking": payload.get("thinking") or payload.get("enable_thinking"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "messages": len(payload.get("messages", [])),
        }
        logger.info("LLM request %s", json.dumps(summary, ensure_ascii=False))

    @staticmethod
    def _log_response(spec: ModelSpec, result: ChatResult) -> None:
        if not TRAFFIC:
            return
        logger.info(
            "LLM response provider=%s model=%s latency=%.0fms usage=%s tools=%d content=%dchars reasoning=%dchars",
            spec.provider_id,
            spec.id,
            result.latency_ms,
            result.usage,
            len(result.tool_calls),
            len(result.content),
            len(result.reasoning),
        )

    def close(self) -> None:
        """Close every HTTP connection pool created by this client."""
        with self._session_lock:
            sessions = list(self._sessions)
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass


def _apply_payload_drops(payload: dict, dropped: Set[str]) -> dict:
    reduced = json.loads(json.dumps(payload))
    for key in dropped:
        if key == "reasoning_content":
            for message in reduced.get("messages", []):
                if isinstance(message, dict):
                    message.pop("reasoning_content", None)
        elif key == "max_tokens":
            if reduced.get("max_tokens", 0) > 8192:
                reduced["max_tokens"] = 8192
        else:
            reduced.pop(key, None)
    return reduced


def _split_think_tags(content: str) -> tuple[str, str]:
    """Handle providers that embed reasoning in ``<think>`` tags."""
    if "<think" not in content:
        return content, ""
    import re

    pattern = r"<think(?:ing)?[^>]*>(.*?)(?:</think(?:ing)?>|<｜end▁of▁thinking｜>)"
    found = re.findall(pattern, content, flags=re.S | re.I)
    cleaned = re.sub(pattern, "", content, flags=re.S | re.I)
    return cleaned.strip(), "\n".join(part.strip() for part in found if part.strip())


__all__ = ["ChatResult", "LLMClient", "LLMError", "ToolCall", "estimate_tokens"]
