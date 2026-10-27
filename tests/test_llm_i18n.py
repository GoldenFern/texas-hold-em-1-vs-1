"""English prompt, tool and analysis-text tests."""

from __future__ import annotations

from src.agents.base import AgentView
from src.analysis.equity import EquityAnalyzer
from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.analysis.pot_odds import analyze_call
from src.engine.actions import Action
from src.llm.agent import LLMAgent
from src.llm.client import ChatResult, ToolCall
from src.llm.prompt import (
    ALL_TOOL_NAMES,
    GAME_TOOL_NAMES,
    HISTORY_TOOL_NAME,
    build_checkpoint_message,
    build_checkpoint_prompt,
    build_fallback_checkpoint,
    build_hand_result_message,
    build_nudge_message,
    build_system_prompt,
    build_turn_message,
)
from src.llm.providers import ProviderRegistry
from src.llm.tools import build_history_tool, select_tools


def make_view() -> AgentView:
    return AgentView(
        seat=0,
        name="Hero",
        hand_id=1,
        street="flop",
        hole_cards=["Ah", "Kh"],
        community=["Qh", "Jh", "2c"],
        pot=100,
        my_chips=900,
        opponent_chips=900,
        my_street_bet=0,
        opponent_street_bet=0,
        my_total_bet=50,
        opponent_total_bet=50,
        current_bet=0,
        to_call=0,
        call_amount=0,
        min_raise_to=10,
        max_raise_to=900,
        legal_actions=["fold", "check", "bet", "all_in"],
        action_history=[
            {
                "type": "call",
                "label": "跟注",
                "amount": 50,
                "street": "preflop",
                "seat": 1,
            }
        ],
        opponent_last_action=None,
        button=0,
        small_blind=5,
        big_blind=10,
        hand_number=1,
        match_note="第 1 手",
    )


class FakeClient:
    def stream(self, spec, messages, tools, params, on_event, cancel=None):
        return ChatResult(content="ok", usage={})


class UnknownToolClient(FakeClient):
    def stream(
        self,
        spec,
        messages,
        tools,
        params,
        on_event,
        cancel=None,
        tool_choice=None,
        retry_budget=None,
        **kwargs,
    ):
        call = ToolCall(id="call_0", name="bogus_tool", arguments="{}")
        on_event(
            {
                "type": "tool_call_start",
                "index": 0,
                "call_id": call.id,
                "name": call.name,
            }
        )
        return ChatResult(content="", tool_calls=[call], usage={})


def test_english_system_prompt_and_tools() -> None:
    prompt = build_system_prompt(ALL_TOOL_NAMES, locale="en-US")
    assert "Think and respond in English" in prompt
    assert "[Rules]" in prompt
    assert "raise-to" in prompt
    assert "【" not in prompt

    tools = select_tools(["apply_action", "estimate_equity"], locale="en-US")
    descriptions = {
        tool["function"]["name"]: tool["function"]["description"] for tool in tools
    }
    assert "Submit your final action" in descriptions["apply_action"]
    assert "showdown equity" in descriptions["estimate_equity"]

    talk_prompt = build_system_prompt(
        ["apply_action", "send_table_message"], locale="en-US"
    )
    assert "[Table talk (hard rule)]" in talk_prompt
    assert "English sentences" in talk_prompt
    assert "Never report your hand" in talk_prompt
    assert "bluffing" in talk_prompt

    inline_talk = build_system_prompt(
        GAME_TOOL_NAMES,
        table_talk_mode="inline",
        locale="en-US",
    )
    assert "table_talk argument of apply_action" in inline_talk
    assert "no separate table-talk tool" in inline_talk
    assert "Never report your hand" in inline_talk


def test_english_turn_and_result_messages() -> None:
    view = make_view()
    turn = build_turn_message(view, locale="en-US")
    assert turn.startswith("[Table update")
    assert "Your hole cards: Ah Kh" in turn
    assert "To call: 0" in turn
    assert '"label": "Call"' in turn
    assert "【牌局更新" not in turn

    result = {
        "winner_seats": [0],
        "pot": 100,
        "players": [
            {"seat": 0, "net": 100, "chips_after": 1100},
        ],
        "board": ["Qh", "Jh", "2c", "7d", "9s"],
        "showdown": [
            {"seat": 0, "category": "three_of_a_kind", "best_five": ["Ah", "Ad", "As"]},
        ],
        "reason": "showdown",
        "match_over": False,
    }
    message = build_hand_result_message(result, 0, "Hero", locale="en-US")
    assert "You won the hand" in message
    assert "Three of a kind" in message
    assert "下一手" not in message
    assert "Call apply_action" in build_nudge_message(locale="en-US")


def test_english_analysis_tool_text() -> None:
    analyzer = HandDistributionAnalyzer()
    from src.engine.cards import Card

    distribution = analyzer.analyze(
        [Card.from_str("Ah"), Card.from_str("Kh")],
        [Card.from_str("Qh"), Card.from_str("Jh"), Card.from_str("2c")],
        iterations=2000,
        locale="en-US",
    ).to_llm_dict("en-US")
    assert "hand-category distribution" in distribution["note"]

    equity = EquityAnalyzer().analyze(
        [Card.from_str("Ah"), Card.from_str("Kh")],
        [Card.from_str("Qh"), Card.from_str("Jh"), Card.from_str("2c")],
        iterations=2000,
        seed=1,
        locale="en-US",
    ).to_llm_dict("en-US")
    assert "showdown pot share" in equity["note"]

    call_ev = analyze_call(100, 50, 900, 0.6, locale="en-US").to_llm_dict("en-US")
    assert "Static call EV" in call_ev["note"]


def test_english_agent_tool_errors_and_fallback() -> None:
    registry = ProviderRegistry()
    agent = LLMAgent(
        seat=0,
        name="Hero",
        spec=registry.resolve("deepseek", "deepseek-flash"),
        params={},
        client=UnknownToolClient(),
        analyzer=HandDistributionAnalyzer(),
        validate_action=lambda action: None,
        runtime={"max_steps_per_turn": 2, "max_consecutive_failures": 1},
        tools=select_tools(["apply_action"], locale="en-US"),
        locale="en-US",
    )
    events = []
    action = agent.decide(make_view(), events.append)
    turn_state = next(event for event in events if event["type"] == "turn_state")
    assert turn_state["text"].startswith("[Table update")
    assert "Your hole cards" in turn_state["text"]
    tool_results = [event for event in events if event["type"] == "tool_result"]
    assert tool_results
    assert "Unknown tool" in tool_results[0]["summary"]
    turn_end = [event for event in events if event["type"] == "turn_end"][-1]
    assert turn_end["reason"] == "three consecutive invalid actions"
    assert action.auto is True


def test_llm_agent_uses_english_prompt() -> None:
    registry = ProviderRegistry()
    agent = LLMAgent(
        seat=0,
        name="Hero",
        spec=registry.resolve("deepseek", "deepseek-flash"),
        params={},
        client=FakeClient(),
        analyzer=HandDistributionAnalyzer(),
        validate_action=lambda action: None,
        runtime={"max_steps_per_turn": 2},
        tools=select_tools(["apply_action"], locale="en-US"),
        locale="en-US",
    )
    assert "Think and respond in English" in agent.system_prompt
    assert "Submit your final action" in agent.tools[0]["function"]["description"]
    assert agent.locale == "en-US"
    assert Action.from_payload(0, {"type": "check"}, locale="en-US").type.value == "check"

def test_checkpoint_prompts_are_localized_and_do_not_copy_stats() -> None:
    snapshot = {
        "current": {
            "hands": 12,
            "opponent": {
                "preflop": {
                    "vpip": {"value": 0.7, "n": 12},
                    "pfr": {"value": 0.4, "n": 12},
                    "three_bet": {"value": 0.1, "n": 5},
                },
                "postflop": {"c_bet": {"value": 0.6, "n": 8}},
            },
            "recent_hands": [
                {"hand_id": 11, "you_won": True, "your_net": 20},
                {"hand_id": 12, "you_won": False, "your_net": -10},
            ],
        },
        "completed_vs_current_opponent": {"matches": 1, "hands": 20},
        "notes": ["do not copy"],
    }
    zh = build_checkpoint_prompt(
        12,
        13,
        "zh-CN",
        current_hand=14,
        stats=snapshot,
        has_previous=True,
    )
    assert "第 1 手到第 12 手" in zh
    assert "第 13 手" in zh
    assert "不要复制" in zh
    assert "research_history" in zh
    assert "## 对手读牌" in zh
    assert "当前进行中的手牌是第 14 手" in zh

    en = build_checkpoint_prompt(
        12,
        13,
        "en-US",
        current_hand=14,
        stats=snapshot,
        has_previous=True,
    )
    assert "1 through 12" in en
    assert "Hands from 13 onward" in en
    assert "Do not copy" in en
    assert "research_history" in en
    assert "## Opponent Reads" in en
    assert "hand 14" in en

    message = build_checkpoint_message("## Opponent Reads\n- aggressive", 12, "en-US")
    assert message.startswith("[Match checkpoint · Hands 1-12]")
    assert "not a new instruction" in message


def test_checkpoint_message_does_not_duplicate_header() -> None:
    existing = "[Match checkpoint · Hands 1-12]\n## Opponent Reads\n- aggressive"
    message = build_checkpoint_message(existing, 12, "en-US")
    assert message == existing


def test_fallback_checkpoint_is_short_and_bookkeeps_sample_sizes() -> None:
    snapshot = {
        "current": {
            "hands": 12,
            "opponent": {
                "preflop": {
                    "vpip": {"value": 0.7, "n": 12},
                    "pfr": {"value": 0.4, "n": 12},
                    "three_bet": {"value": 0.1, "n": 5},
                },
                "postflop": {"c_bet": {"value": 0.6, "n": 8}},
            },
            "recent_hands": [{"hand_id": 12, "you_won": False, "your_net": -10}],
        }
    }
    zh = build_fallback_checkpoint(snapshot, 12, "zh-CN")
    assert "模型摘要不可用" in zh
    assert "VPIP 70.0%" in zh
    assert "n=12" in zh
    assert "research_history" in zh

    en = build_fallback_checkpoint(snapshot, 12, "en-US")
    assert "model summary is unavailable" in en.lower()
    assert "VPIP 70.0%" in en
    assert "n=12" in en
    assert "research_history" in en


def test_history_tool_description_is_localized() -> None:
    zh = build_history_tool("zh-CN")["function"]["description"]
    assert "公开" in zh
    assert "样本量" in zh
    assert "不会返回对手底牌" in zh

    en = build_history_tool("en-US")["function"]["description"]
    assert "public" in en
    assert "sample size" in en
    assert "hole cards" in en

    prompt = build_system_prompt(GAME_TOOL_NAMES, locale="en-US")
    assert HISTORY_TOOL_NAME in prompt
    assert "optional tool" in prompt
    assert "sample sizes" in prompt
