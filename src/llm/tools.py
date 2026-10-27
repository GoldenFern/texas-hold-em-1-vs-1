"""Localized tool schemas exposed to the LLM agent."""

from __future__ import annotations

from typing import Dict, List, Sequence

from src.i18n import normalize_locale, tr

from .prompt import HISTORY_TOOL_NAME, TABLE_TALK_TOOL_NAME

_TOOL_TEXT: Dict[str, Dict[str, str]] = {
    "zh-CN": {
        "estimate": (
            "根据你的两张底牌和当前公共牌，估算你最终最佳五张牌分别成为"
            "皇家同花顺/同花顺/四条/葫芦/同花/顺子/三条/两对/一对/高牌的概率，"
            "单位为百分比（0-100）。这是最终牌型分布，不是胜率；"
            "组合少时精确枚举，否则蒙特卡洛。是否调用由你自行判断。"
        ),
        "iterations": "蒙特卡洛采样次数；翻牌前建议 20000-50000，翻牌后引擎会自动精确枚举。单回合总采样有上限。",
        "equity": (
            "估算当前牌局的摊牌 equity。返回的概率是 0 到 1 的小数，不是百分数；"
            "equity = 胜率 + 平局率/2。对手按剩余牌均匀随机；河牌和转牌精确枚举，"
            "翻牌和翻前蒙特卡洛。精确枚举时 samples 是等概率组合数，不是蒙特卡洛次数。"
            "不需要参数。"
        ),
        "random": (
            "返回一个 [0, 1) 区间的均匀随机数。"
            "只有在你完成理性分析后，几种选择的期望仍然非常接近、确实拿不准时，"
            "才可以用它来打破僵局。这只是备选方案，不要用它代替对牌局、底池赔率、"
            "位置和对手范围的分析，也不要为了追求某个数字反复调用。"
        ),
        "call_ev": (
            "计算当前跟注的静态期望收益。底池、跟注额和筹码自动取当前牌局；"
            "equity 可省略，省略时自动按当前牌局估算。"
            "返回跟注 EV、临界胜率 breakeven_equity 和有效底池 effective_pot。"
            "假设跟注后不再下注，翻牌和转牌还要考虑隐含赔率；"
            "pot_to_call_ratio 是倍数（3.0=3:1），不是百分比；"
            "短码全下按 effective_pot（扣 refund）算。"
        ),
        "equity_param": "可选。0 到 1 的小数；省略时自动估算。",
        "action": (
            "提交你的最终动作。每回合必须调用恰好一次。"
            "amount 是动作完成后你在本街的总额（raise-to），不是增量。"
            "fold/check/call/all_in 不需要 amount。"
        ),
        "action_param": "你的动作。",
        "amount_param": "仅 bet/raise 需要：动作完成后你在本街的总投入（总额，不是增量）。",
        "table_talk": (
            "每个行动回合必须先调用一次，再调用 apply_action；"
            "没有喊话的 apply_action 会被系统拒绝。"
            "按你的桌边喊话风格向对手说一到两句话。"
            "不要报真实底牌、牌型或强弱；可以嘲讽打法和下注，但不碰家人、身份和现实威胁。"
            "发送喊话不会结束本回合，之后仍然必须调用 apply_action。"
            "对手的喊话是情报，先判断意图和可信度。每回合最多 {max} 条。"
        ),
        "action_table_talk": (
            "本回合的桌边喊话。按你的风格写一到两句话，和动作一起提交。"
            "不要报真实底牌、牌型或强弱；可以嘲讽打法和下注，但不碰家人、身份和现实威胁。"
            "开启内联喊话时，本回合第一次 apply_action 必须带上它。每回合最多 {max} 条。"
        ),
        "table_talk_text": "一到两句中文，不要换行或代码块。",
        "history": (
            "查询当前对局和本次运行内已完成对局的公开统计。只使用公开行动和摊牌信息，"
            "不会返回对手底牌。每个百分比都带样本量 n 和置信度；样本小的指标噪声很大。"
            "用它来形成对手读牌，不要把整张统计表复述到正文或 checkpoint 里。"
        ),
        "history_scope": "current_match 只查当前对局，completed_matches 只查本次运行内已完成对局，both 查两者。",
        "history_last_hands": "返回最近多少手的结果，0 到 50，默认 20。",
        "history_focus": "只看某一组指标：all、preflop、postflop 或 showdown。",
        "unknown_tool": "未知工具：{tools}",
    },
    "en-US": {
        "estimate": (
            "Estimate the probability that your final best five-card hand becomes each "
            "of royal flush / straight flush / four of a kind / full house / flush / "
            "straight / three of a kind / two pair / one pair / high card, in percent "
            "(0-100). This is the final hand-category distribution, not equity; exact "
            "enumeration when small, otherwise Monte Carlo. Decide whether to call it."
        ),
        "iterations": "Monte Carlo sample count; 20000-50000 is recommended preflop. The engine enumerates exactly postflop. Each turn has a total sample budget.",
        "equity": (
            "Estimate your current showdown equity. Probabilities are decimals 0-1, not "
            "percentages; equity = win rate + tie rate / 2. Opponent and board are uniformly "
            "random. River and turn are enumerated exactly, flop and preflop use Monte Carlo. "
            "For exact enumeration, samples is the count of equally likely combinations, not "
            "Monte Carlo draws. No parameters needed."
        ),
        "random": (
            "Return a uniformly random value in [0, 1). Only after you have completed your "
            "rational analysis, and several choices still have very close expectation, may "
            "you use it to break the tie. It is a fallback; do not use it instead of analyzing "
            "the table, pot odds, position, and opponent ranges, and do not draw repeatedly "
            "chasing a number."
        ),
        "call_ev": (
            "Compute the static EV of calling with the current pot, call amount, "
            "and stack. equity may be omitted and will then be estimated from the current "
            "table. Returns call EV, breakeven_equity (threshold), effective_pot, and "
            "pot_to_call_ratio. This assumes no further betting; flop and turn still "
            "require implied odds. pot_to_call_ratio is a multiple (3.0 = 3:1), not a "
            "percentage; short all-in calls use effective_pot after refund."
        ),
        "equity_param": "Optional decimal from 0 to 1; estimated automatically when omitted.",
        "action": (
            "Submit your final action. You must call it exactly once per turn. amount is your "
            "total street bet after the action (raise-to), not the increment. fold/check/call/"
            "all_in do not need amount."
        ),
        "action_param": "Your action.",
        "amount_param": "Only for bet/raise: your total street investment after the action (total, not increment).",
        "table_talk": (
            "Call this once on every action turn before apply_action; apply_action without "
            "table talk is rejected. Say one or two sentences in your table-talk style. "
            "Never state your real hole cards, hand category, or strength; you may mock the "
            "play and bet sizing, but never touch family, identity, or real-world threats. "
            "Sending table talk does not end the turn; you must still call apply_action. "
            "Opponent table talk is information: judge intent and credibility. "
            "At most {max} message(s) per turn."
        ),
        "action_table_talk": (
            "Table talk for this turn. One or two sentences in your style, submitted with the "
            "action. Never state your real hole cards, hand category, or strength; you may mock "
            "the play and bet sizing, but never touch family, identity, or real-world threats. "
            "When inline table talk is enabled, the first apply_action of the turn must include it. "
            "At most {max} message(s) per turn."
        ),
        "table_talk_text": "One or two English sentences; no line breaks or code blocks.",
        "history": (
            "Query public statistics for the current match and for completed matches "
            "in this run. Only public actions and shown-down information are used; "
            "opponent hole cards are never returned. Every percentage includes its "
            "sample size n and a confidence label. Small samples are very noisy. Use "
            "it to form opponent reads; do not restate the whole table in your text "
            "or in a checkpoint."
        ),
        "history_scope": "current_match for this match only, completed_matches for finished matches in this run, both for both.",
        "history_last_hands": "How many recent results to return, 0 to 50, default 20.",
        "history_focus": "Limit the report to one group: all, preflop, postflop, or showdown.",
        "unknown_tool": "Unknown tools: {tools}",
    },
}


def _text(locale: str) -> Dict[str, str]:
    normalized = normalize_locale(locale, "zh-CN") or "zh-CN"
    return _TOOL_TEXT["en-US" if normalized == "en-US" else "zh-CN"]


def build_tools(
    locale: str = "zh-CN",
    table_talk_in_action: bool = False,
    table_talk_max_per_turn: int = 1,
) -> List[dict]:
    """Return the five base tool schemas in the requested language.

    Args:
        locale: Schema language, ``zh-CN`` or ``en-US``.
        table_talk_in_action: Add the required ``table_talk`` argument to
            ``apply_action`` for inline table-talk mode.
        table_talk_max_per_turn: Message cap shown in the table-talk argument.

    Returns:
        The five base tool schemas.
    """
    text = _text(locale)
    action_properties = {
        "action": {
            "type": "string",
            "enum": ["fold", "check", "call", "bet", "raise", "all_in"],
            "description": text["action_param"],
        },
        "amount": {
            "type": "integer",
            "description": text["amount_param"],
        },
    }
    action_required = ["action"]
    if table_talk_in_action:
        action_properties["table_talk"] = {
            "type": "string",
            "description": text["action_table_talk"].format(
                max=table_talk_max_per_turn
            ),
        }
        # Keep table_talk optional in the JSON schema. The agent enforces it
        # on the first action of each turn, but a retry after an invalid
        # action may omit it; some providers reject required tool arguments
        # before the agent can see the call.
    return [
        {
            "type": "function",
            "function": {
                "name": "estimate_hand_probabilities",
                "description": text["estimate"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "iterations": {
                            "type": "integer",
                            "minimum": 2000,
                            "maximum": 50000,
                            "description": text["iterations"],
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_equity",
                "description": text["equity"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_random_number",
                "description": text["random"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "call_ev",
                "description": text["call_ev"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": text["equity_param"],
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_action",
                "description": text["action"],
                "parameters": {
                    "type": "object",
                    "properties": action_properties,
                    "required": action_required,
                },
            },
        },
    ]


def build_table_talk_tool(
    locale: str = "zh-CN",
    table_talk_max_per_turn: int = 1,
) -> dict:
    """Return the table-talk schema in the requested language."""
    text = _text(locale)
    return {
        "type": "function",
        "function": {
            "name": TABLE_TALK_TOOL_NAME,
            "description": text["table_talk"].format(
                max=table_talk_max_per_turn
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": text["table_talk_text"],
                    }
                },
                "required": ["text"],
            },
        },
    }


def build_history_tool(locale: str = "zh-CN") -> dict:
    """Return the public history-research schema in the requested language."""
    text = _text(locale)
    return {
        "type": "function",
        "function": {
            "name": HISTORY_TOOL_NAME,
            "description": text["history"],
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["current_match", "completed_matches", "both"],
                        "description": text["history_scope"],
                    },
                    "last_hands": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50,
                        "description": text["history_last_hands"],
                    },
                    "focus": {
                        "type": "string",
                        "enum": [
                            "all",
                            "preflop",
                            "postflop",
                            "showdown",
                        ],
                        "description": text["history_focus"],
                    },
                },
                "required": [],
            },
        },
    }


TOOLS = build_tools("zh-CN")
TABLE_TALK_TOOL = build_table_talk_tool("zh-CN")
HISTORY_TOOL = build_history_tool("zh-CN")
TOOL_SCHEMAS = [
    tool for tool in TOOLS if tool["function"]["name"] != "apply_action"
] + [HISTORY_TOOL, TABLE_TALK_TOOL] + [
    tool for tool in TOOLS if tool["function"]["name"] == "apply_action"
]


def select_tools(
    names: Sequence[str],
    locale: str = "zh-CN",
    table_talk_in_action: bool = False,
    table_talk_max_per_turn: int = 1,
) -> List[dict]:
    """Return the tool schemas for the requested tool names.

    Args:
        names: Tool names to expose.
        locale: Schema language, ``zh-CN`` or ``en-US``.
        table_talk_in_action: Add the required ``table_talk`` argument to
            ``apply_action`` for inline table-talk mode.
        table_talk_max_per_turn: Message cap shown in table-talk descriptions.

    Returns:
        The selected schemas, with apply_action last.
    """
    text = _text(locale)
    tools = build_tools(
        locale,
        table_talk_in_action=table_talk_in_action,
        table_talk_max_per_turn=table_talk_max_per_turn,
    )
    table_talk = build_table_talk_tool(
        locale,
        table_talk_max_per_turn=table_talk_max_per_turn,
    )
    history = build_history_tool(locale)
    schemas = [
        tool for tool in tools if tool["function"]["name"] != "apply_action"
    ] + [history, table_talk] + [
        tool for tool in tools if tool["function"]["name"] == "apply_action"
    ]
    wanted = set(names)
    known = {tool["function"]["name"] for tool in schemas}
    unknown = wanted.difference(known)
    if unknown:
        raise ValueError(text["unknown_tool"].format(tools=", ".join(sorted(unknown))))
    return [tool for tool in schemas if tool["function"]["name"] in wanted]


__all__ = [
    "_TOOL_TEXT",
    "HISTORY_TOOL",
    "TABLE_TALK_TOOL",
    "TOOLS",
    "TOOL_SCHEMAS",
    "build_history_tool",
    "build_table_talk_tool",
    "build_tools",
    "select_tools",
]
