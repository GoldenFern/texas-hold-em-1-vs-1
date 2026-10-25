"""LLM client tests with fake HTTP responses; no network calls."""

from __future__ import annotations

import json
from typing import Iterable, List, Optional

import pytest
import requests

from src.llm.client import LLMClient, LLMError, classify_http, redact_secrets
from src.llm.providers import ProviderRegistry


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123456")
    yield


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        body: str = "",
        lines: Optional[Iterable[bytes]] = None,
        json_data: Optional[dict] = None,
        headers: Optional[dict] = None,
        error_after_lines: Optional[Exception] = None,
    ) -> None:
        self.status_code = status
        self.text = body
        self._lines = list(lines or [])
        self._json = json_data
        self.headers = headers or {}
        self.error_after_lines = error_after_lines
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line
        if self.error_after_lines is not None:
            raise self.error_after_lines

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: List[dict] = []
        self.closed = False

    def post(self, url, headers=None, json=None, stream=False, timeout=None):
        self.calls.append({"url": url, "json": json, "stream": stream, "timeout": timeout})
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def sse(*events: dict) -> List[bytes]:
    lines: List[bytes] = []
    for event in events:
        lines.append(("data: " + json.dumps(event, ensure_ascii=False)).encode("utf-8"))
        lines.append(b"")
    lines.append(b"data: [DONE]")
    return lines


def make_client(
    responses: List[FakeResponse],
    locale: str = "zh-CN",
    user_id: Optional[str] = None,
) -> tuple[LLMClient, FakeSession]:
    registry = ProviderRegistry()
    client = LLMClient(
        registry,
        connect_timeout=1,
        read_timeout=1,
        locale=locale,
        user_id=user_id,
    )
    session = FakeSession(responses)
    client._get_session = lambda: session  # type: ignore[method-assign]
    return client, session


class TimeoutSession(FakeSession):
    def post(self, url, headers=None, json=None, stream=False, timeout=None):
        raise requests.Timeout("simulated timeout")


def spec():
    return ProviderRegistry().resolve("deepseek", "deepseek-flash")


def messages_with_reasoning() -> List[dict]:
    return [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": "thinking",
            "reasoning_content": "secret chain of thought",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "apply_action", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "{}"},
    ]


def test_redact_secrets_removes_key_and_bearer_token() -> None:
    text = 'HTTP 401: {"error":"bad key sk-testSECRET1234"} Authorization: Bearer abc.def.123'
    cleaned = redact_secrets(text, "sk-testSECRET1234")
    assert "sk-testSECRET1234" not in cleaned
    assert "abc.def.123" not in cleaned
    assert "[REDACTED]" in cleaned


def test_client_errors_follow_locale() -> None:
    registry = ProviderRegistry()
    client = LLMClient(registry, connect_timeout=1, read_timeout=1, locale="en-US")
    client._get_session = lambda: TimeoutSession([])  # type: ignore[method-assign]
    with pytest.raises(LLMError) as excinfo:
        client.stream(
            spec(),
            [{"role": "user", "content": "hello"}],
            None,
            {},
            lambda _event: None,
        )
    assert "timed out" in excinfo.value.detail


def test_stream_assembles_text_and_tool_call() -> None:
    client, _ = make_client(
        [
            FakeResponse(
                lines=sse(
                    {"choices": [{"delta": {"reasoning_content": "think "}}]},
                    {"choices": [{"delta": {"content": "hello "}}]},
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_0",
                                            "function": {
                                                "name": "apply_action",
                                                "arguments": '{"action":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '"check"}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                )
            )
        ]
    )
    events: List[dict] = []
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, events.append)
    assert result.content == "hello"
    assert result.reasoning == "think"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "apply_action"
    assert result.tool_calls[0].arguments == '{"action":"check"}'
    assert result.usage["input_tokens"] == 10
    assert any(event.get("type") == "content_delta" for event in events)


def test_parallel_tool_calls_without_index_are_not_merged() -> None:
    client, _ = make_client(
        [
            FakeResponse(
                lines=sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": "call_a",
                                            "function": {
                                                "name": "get_random_number",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": "call_b",
                                            "function": {
                                                "name": "apply_action",
                                                "arguments": '{"action":"check"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                )
            )
        ]
    )
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert [call.name for call in result.tool_calls] == [
        "get_random_number",
        "apply_action",
    ]


def test_disconnect_keeps_complete_tool_call() -> None:
    client, _ = make_client(
        [
            FakeResponse(
                lines=sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_0",
                                            "function": {
                                                "name": "apply_action",
                                                "arguments": '{"action":"check"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                error_after_lines=requests.exceptions.ChunkedEncodingError("cut"),
            )
        ]
    )
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert result.tool_calls[0].name == "apply_action"


def test_stream_unsupported_falls_back_to_non_stream() -> None:
    client, session = make_client(
        [
            FakeResponse(status=400, body="stream is not supported"),
            FakeResponse(
                json_data={
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            ),
        ]
    )
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert result.content == "ok"
    assert [call["stream"] for call in session.calls] == [True, False]


def test_reasoning_rejection_is_remembered() -> None:
    first = FakeResponse(status=400, body="reasoning_content is not supported")
    success = FakeResponse(
        lines=sse({"choices": [{"delta": {"content": "ok"}}]})
    )
    client, session = make_client([first, success, FakeResponse(
        lines=sse({"choices": [{"delta": {"content": "again"}}]})
    )])
    client.stream(spec(), messages_with_reasoning(), None, {}, lambda e: None)
    client.stream(spec(), messages_with_reasoning(), None, {}, lambda e: None)
    assert len(session.calls) == 3
    assert "reasoning_content" not in session.calls[-1]["json"]["messages"][1]


def test_http_error_body_is_redacted() -> None:
    client, _ = make_client(
        [FakeResponse(status=401, body='{"error":"bad key sk-testSECRET1234"}')]
    )
    with pytest.raises(LLMError) as excinfo:
        client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert "sk-testSECRET1234" not in excinfo.value.detail
    assert "[REDACTED]" in excinfo.value.detail


def test_cancel_stops_before_request() -> None:
    client, session = make_client([])
    with pytest.raises(LLMError) as excinfo:
        client.stream(
            spec(),
            [{"role": "user", "content": "hi"}],
            None,
            {},
            lambda e: None,
            cancel=lambda: True,
        )
    assert excinfo.value.kind == "cancelled"
    assert session.calls == []


def test_context_length_is_classified_for_trimming() -> None:
    assert classify_http(400, "maximum context length exceeded") == "context_length"


def test_concurrency_limit_is_classified_as_rate_limit() -> None:
    assert (
        classify_http(400, "Concurrency limit exceeded for user, please retry later")
        == "rate_limit"
    )


def test_duplicate_tool_ids_are_made_unique() -> None:
    client, _ = make_client(
        [
            FakeResponse(
                lines=sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "same",
                                            "function": {
                                                "name": "get_random_number",
                                                "arguments": "{}",
                                            },
                                        },
                                        {
                                            "index": 1,
                                            "id": "same",
                                            "function": {
                                                "name": "apply_action",
                                                "arguments": '{"action":"check"}',
                                            },
                                        },
                                    ]
                                }
                            }
                        ]
                    }
                )
            )
        ]
    )
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert [call.id for call in result.tool_calls] == ["same", "same_2"]

def test_stream_usage_parses_deepseek_cache_fields() -> None:
    client, _ = make_client(
        [
            FakeResponse(
                lines=sse(
                    {"choices": [{"delta": {"content": "ok"}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 10,
                            "prompt_cache_hit_tokens": 80,
                            "prompt_cache_miss_tokens": 20,
                            "prompt_tokens_details": {"cached_tokens": 80},
                            "completion_tokens_details": {"reasoning_tokens": 4},
                        },
                    },
                )
            )
        ]
    )
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert result.usage["input_tokens"] == 100
    assert result.usage["cached_input_tokens"] == 80
    assert result.usage["cache_miss_input_tokens"] == 20
    assert result.usage["reasoning_tokens"] == 4


def test_stream_usage_computes_cache_miss_for_openai_shape() -> None:
    client, _ = make_client(
        [
            FakeResponse(
                lines=sse(
                    {"choices": [{"delta": {"content": "ok"}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 5,
                            "prompt_tokens_details": {"cached_tokens": 12},
                        },
                    },
                )
            )
        ]
    )
    result = client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert result.usage["cached_input_tokens"] == 12
    assert result.usage["cache_miss_input_tokens"] == 38

def test_user_id_is_sent_for_cache_isolation() -> None:
    client, session = make_client(
        [
            FakeResponse(
                lines=sse(
                    {"choices": [{"delta": {"content": "ok"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                )
            )
        ],
        user_id="ctx-preview-legacy-2026",
    )
    client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert session.calls[0]["json"]["user_id"] == "ctx-preview-legacy-2026"


def test_user_id_is_omitted_when_not_configured() -> None:
    client, session = make_client(
        [
            FakeResponse(
                lines=sse(
                    {"choices": [{"delta": {"content": "ok"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                )
            )
        ]
    )
    client.stream(spec(), [{"role": "user", "content": "hi"}], None, {}, lambda e: None)
    assert "user_id" not in session.calls[0]["json"]
