"""Final table-talk context tests with a fake streaming client."""

from __future__ import annotations

from typing import List

from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.llm.agent import LLMAgent
from src.llm.client import ChatResult, LLMError, ToolCall
from src.llm.context import MatchStats
from src.llm.prompt import AGENT_TOOL_NAMES
from src.llm.providers import ProviderRegistry
from src.llm.tools import select_tools


class CaptureClient:
    def __init__(self) -> None:
        self.captured: List[List[dict]] = []

    def stream(self, spec, messages, tools, params, on_event, cancel=None):
        self.captured.append([dict(item) for item in messages])
        call = ToolCall(
            id="call_talk",
            name="send_table_message",
            arguments='{"text":"good game"}',
        )
        return ChatResult(
            content="",
            reasoning="say gg",
            tool_calls=[call],
            finish_reason="tool_calls",
            usage={
                "input_tokens": 10,
                "cached_input_tokens": 6,
                "cache_miss_input_tokens": 4,
                "output_tokens": 2,
                "reasoning_tokens": 1,
            },
            latency_ms=1.0,
        )


class FailingClient:
    def stream(self, spec, messages, tools, params, on_event, cancel=None):
        raise LLMError("network", "connection reset")


class NoToolClient:
    def stream(self, spec, messages, tools, params, on_event, cancel=None):
        return ChatResult(
            content="no tool call",
            finish_reason="stop",
            usage={"input_tokens": 5, "output_tokens": 1},
            latency_ms=2.0,
        )


def result() -> dict:
    return {
        "hand_id": 3,
        "winner_seats": [0],
        "pot": 200,
        "reason": "showdown",
        "board": ["Ah", "Kd", "7c", "3s", "2h"],
        "players": [
            {"seat": 0, "net": 100, "chips_after": 1100},
            {"seat": 1, "net": -100, "chips_after": 900},
        ],
        "showdown": [
            {"seat": 0, "category": "pair", "category_name": "一对"},
            {"seat": 1, "category": "high_card", "category_name": "高牌"},
        ],
    }


def make_agent(client: CaptureClient, mode: str) -> LLMAgent:
    registry = ProviderRegistry()
    spec = registry.resolve("deepseek", "deepseek-flash")
    return LLMAgent(
        seat=0,
        name="DeepSeek",
        spec=spec,
        params=spec.normalize_params({"reasoning_effort": "low"}),
        client=client,
        analyzer=HandDistributionAnalyzer(),
        validate_action=lambda action: None,
        runtime={
            "final_talk_context": mode,
            "table_talk_max_chars": 160,
            "table_talk_max_per_turn": 1,
        },
        tools=select_tools(AGENT_TOOL_NAMES, locale="zh-CN"),
        system_prompt="full system prompt",
        submit_table_talk=lambda text: {"ok": True},
        locale="zh-CN",
    )


def test_fresh_final_talk_uses_short_context() -> None:
    client = CaptureClient()
    agent = make_agent(client, "fresh")
    agent.messages = [
        {"role": "system", "content": "full system prompt"},
        {"role": "user", "content": "old history one"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent.match_stats = MatchStats.from_results([result()], 0)
    submitted: List[str] = []
    events: List[dict] = []

    def submit(text: str) -> dict:
        submitted.append(text)
        return {"ok": True}

    agent.final_table_talk(result(), submit, events.append)

    assert submitted == ["good game"]
    assert len(client.captured) == 1
    messages = client.captured[0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "send_table_message" in messages[0]["content"]
    assert "不要调用 apply_action" in messages[0]["content"]
    assert "【整场结果】共 1 手" in messages[1]["content"]
    assert "盈利" in messages[1]["content"]
    assert "【赛后喊话】" in messages[1]["content"]
    assert "下一手即将开始" not in messages[1]["content"]
    response_event = next(
        event for event in events if event["type"] == "final_talk_response"
    )
    assert response_event["mode"] == "fresh"
    assert response_event["usage"]["input_tokens"] == 10
    assert response_event["latency_ms"] == 1.0
    assert response_event["message_count"] >= 3
    end_event = next(
        event for event in events if event["type"] == "final_talk_end"
    )
    assert end_event["mode"] == "fresh"
    assert end_event["ok"] is True
    assert end_event["usage"] == {
        "input_tokens": 10,
        "cached_input_tokens": 6,
        "cache_miss_input_tokens": 4,
        "output_tokens": 2,
        "reasoning_tokens": 1,
    }
    # The fresh call must not replace the agent's long history.
    assert len(agent.messages) == 3


def test_full_final_talk_keeps_history() -> None:
    client = CaptureClient()
    agent = make_agent(client, "full")
    agent.messages = [
        {"role": "system", "content": "full system prompt"},
        {"role": "user", "content": "old history one"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent.match_stats = MatchStats.from_results([result()], 0)
    events: List[dict] = []

    agent.final_table_talk(result(), lambda text: {"ok": True}, events.append)

    assert len(client.captured) == 1
    messages = client.captured[0]
    assert len(messages) >= 4
    assert any(item.get("content") == "old history one" for item in messages)
    assert len(agent.messages) >= 4
    response_event = next(
        event for event in events if event["type"] == "final_talk_response"
    )
    assert response_event["mode"] == "full"
    assert response_event["usage"]["input_tokens"] == 10
    assert response_event["message_count"] >= 4
    end_event = next(
        event for event in events if event["type"] == "final_talk_end"
    )
    assert end_event["mode"] == "full"
    assert end_event["ok"] is True
    assert end_event["usage"]["input_tokens"] == 10
    assert end_event["message_count"] >= 4


def test_final_talk_emits_end_event_on_failure() -> None:
    agent = make_agent(FailingClient(), "fresh")
    agent.match_stats = MatchStats.from_results([result()], 0)
    events: List[dict] = []

    agent.final_table_talk(result(), lambda text: {"ok": True}, events.append)

    assert not any(event["type"] == "final_talk_response" for event in events)
    end_event = next(
        event for event in events if event["type"] == "final_talk_end"
    )
    assert end_event["ok"] is False
    assert end_event["reason"] == "llm_error"
    assert end_event["error"] == "network"
    assert end_event["usage"]["input_tokens"] == 0


def test_final_talk_emits_end_event_when_no_result() -> None:
    agent = make_agent(NoToolClient(), "fresh")
    agent.match_stats = MatchStats.from_results([result()], 0)
    events: List[dict] = []

    agent.final_table_talk(result(), lambda text: {"ok": True}, events.append)

    responses = [
        event for event in events if event["type"] == "final_talk_response"
    ]
    assert len(responses) == 2
    assert all(event["usage"]["input_tokens"] == 5 for event in responses)
    end_event = next(
        event for event in events if event["type"] == "final_talk_end"
    )
    assert end_event["ok"] is False
    assert end_event["reason"] == "no_result"
    assert end_event["usage"]["input_tokens"] == 10
