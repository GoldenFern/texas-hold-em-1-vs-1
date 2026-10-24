"""Final table-talk context tests with a fake streaming client."""

from __future__ import annotations

from typing import List

from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.llm.agent import LLMAgent
from src.llm.client import ChatResult, ToolCall
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
    agent.hand_results = [result()]
    submitted: List[str] = []

    def submit(text: str) -> dict:
        submitted.append(text)
        return {"ok": True}

    agent.final_table_talk(result(), submit, lambda event: None)

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
    agent.hand_results = [result()]

    agent.final_table_talk(result(), lambda text: {"ok": True}, lambda event: None)

    assert len(client.captured) == 1
    messages = client.captured[0]
    assert len(messages) >= 4
    assert any(item.get("content") == "old history one" for item in messages)
    assert len(agent.messages) >= 4
