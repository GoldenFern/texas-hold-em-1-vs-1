"""Prompts for the heads-up poker agent."""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from src.agents.base import AgentView
from src.engine.actions import action_label
from src.engine.evaluator import hand_category_name
from src.i18n import normalize_locale, tr
from src.llm.context import MatchStats
from src.llm.prompt_texts import prompt_text

ALL_TOOL_NAMES = (
    "estimate_hand_probabilities",
    "estimate_equity",
    "get_random_number",
    "call_ev",
    "apply_action",
)

TABLE_TALK_TOOL_NAME = "send_table_message"
AGENT_TOOL_NAMES = ALL_TOOL_NAMES + (TABLE_TALK_TOOL_NAME,)


def _locale(locale: Optional[str]) -> str:
    return normalize_locale(locale, "zh-CN") or "zh-CN"


def _join(items: Sequence[str], locale: str) -> str:
    separator = "、" if _locale(locale) == "zh-CN" else ", "
    return separator.join(items)


def build_table_talk_section(
    max_chars: int = 160,
    max_per_turn: int = 1,
    locale: str = "zh-CN",
) -> str:
    """Build the table-talk rules inserted into the system prompt."""
    text = prompt_text(_locale(locale))
    return text["table_talk_section"].format(
        tool=TABLE_TALK_TOOL_NAME,
        max_chars=max_chars,
        max_per_turn=max_per_turn,
    )


def build_system_prompt(
    tools: Sequence[str] = ALL_TOOL_NAMES,
    *,
    require_estimate_each_turn: bool = False,
    extra_instructions: str = "",
    table_talk_max_chars: int = 160,
    table_talk_max_per_turn: int = 1,
    locale: str = "zh-CN",
) -> str:
    """Build a system prompt matching the tools available to one seat.

    Args:
        tools: Tool names exposed to the model for this seat.
        require_estimate_each_turn: Require an estimate before every action
            until the board is complete, then stop requiring it.
        extra_instructions: Optional experiment-specific instructions inserted
            before the output-style section. The default prompt stays neutral
            about whether auxiliary tools should be used.
        table_talk_max_chars: Maximum characters in one table-talk message.
        table_talk_max_per_turn: Maximum table-talk messages per turn.
        locale: Prompt language, ``zh-CN`` or ``en-US``.

    Returns:
        The full system prompt.

    Raises:
        ValueError: If a requested tool name is unknown.
    """
    locale = _locale(locale)
    text = prompt_text(locale)
    names = set(tools)
    unknown = names.difference(AGENT_TOOL_NAMES)
    if unknown:
        raise ValueError(text["unknown_tool"].format(tools=", ".join(sorted(unknown))))
    has_estimate = "estimate_hand_probabilities" in names
    has_equity = "estimate_equity" in names
    has_random = "get_random_number" in names
    has_call_ev = "call_ev" in names
    has_table_talk = TABLE_TALK_TOOL_NAME in names

    steps: List[str] = [text["step_state"]]
    if has_estimate:
        steps.append(
            text["step_estimate_required"]
            if require_estimate_each_turn
            else text["step_estimate_optional"]
        )
    if has_equity:
        steps.append(text["step_equity"])
    if has_call_ev:
        steps.append(text["step_call_ev"])
    if has_random:
        steps.append(text["step_random"])
    if has_table_talk:
        steps.append(
            text["step_table_talk"].format(
                tool=TABLE_TALK_TOOL_NAME,
                max=table_talk_max_per_turn,
            )
        )
    steps.append(text["step_action_once"])
    steps.append(text["step_retry"])
    steps.append(text["step_fallback"])

    workflow = "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, start=1)
    )

    auxiliaries: List[str] = []
    if has_estimate:
        auxiliaries.append("estimate_hand_probabilities")
    if has_equity:
        auxiliaries.append("estimate_equity")
    if has_random:
        auxiliaries.append("get_random_number")
    if has_call_ev:
        auxiliaries.append("call_ev")
    if has_table_talk:
        auxiliaries.append(TABLE_TALK_TOOL_NAME)
    if auxiliaries:
        tool_style = text["tool_style_aux"].format(tools=_join(auxiliaries, locale))
    else:
        tool_style = text["tool_style_action_only"]

    table_talk_section = ""
    if has_table_talk:
        table_talk_section = (
            build_table_talk_section(
                max_chars=table_talk_max_chars,
                max_per_turn=table_talk_max_per_turn,
                locale=locale,
            )
            + "\n\n"
        )
    extra = extra_instructions.strip()
    extra_section = f"{extra}\n\n" if extra else ""
    return text["system_template"].format(
        workflow=workflow,
        tool_style=tool_style,
        table_talk_section=table_talk_section,
        extra_section=extra_section,
    )


SYSTEM_PROMPT = build_system_prompt()
SYSTEM_PROMPT_EN = build_system_prompt(locale="en-US")


def _localize_action(item: dict, locale: str) -> dict:
    action = dict(item)
    action_type = str(action.get("type") or "")
    if action_type:
        action["label"] = action_label(action_type, locale)
    return action


def build_turn_message(
    view: AgentView,
    require_table_talk: bool = False,
    locale: str = "zh-CN",
) -> str:
    locale = _locale(locale)
    text = prompt_text(locale)
    state = view.to_prompt_dict()
    state["actions_this_hand"] = [
        _localize_action(item, locale) for item in state["actions_this_hand"][-10:]
    ]
    if state.get("opponent_last_action"):
        state["opponent_last_action"] = _localize_action(
            state["opponent_last_action"], locale
        )
    state["recent_hands"] = state["recent_hands"][-3:]
    position = tr(locale, "position.button" if view.is_button else "position.big_blind")
    community = " ".join(view.community) if view.community else text["none"]
    lines = [
        text["turn_header"].format(
            hand=view.hand_number,
            street=_street_name(view.street, locale),
        ),
        text["turn_position"].format(seat=view.seat, position=position),
        text["turn_stack"].format(
            pot=view.pot,
            my_chips=view.my_chips,
            opponent_chips=view.opponent_chips,
        ),
        text["turn_cards"].format(
            hole=" ".join(view.hole_cards),
            community=community,
        ),
        text["turn_action"].format(
            to_call=view.to_call,
            actions=", ".join(view.legal_actions),
        ),
        text["turn_state_intro"],
        f"```json\n{json.dumps(state, ensure_ascii=False, indent=2)}\n```",
    ]
    if view.table_talk:
        lines.extend(
            [
                text["turn_talk_header"],
                text["turn_talk_warning"],
            ]
        )
        for item in view.table_talk:
            hand_id = int(item.get("hand_id") or 0)
            street = _street_name(str(item.get("street") or ""), locale)
            name = str(item.get("name") or text["opponent"])
            talk = str(item.get("text") or "")
            lines.append(
                text["turn_talk_item"].format(
                    hand=hand_id,
                    street=street,
                    name=name,
                    text=talk,
                )
            )
    if require_table_talk:
        lines.append(text["turn_talk_required"])
    lines.append(text["turn_closing"])
    return "\n".join(lines)


def build_compact_turn_message(
    view: AgentView,
    require_table_talk: bool = False,
    locale: str = "zh-CN",
) -> str:
    """Build a smaller turn update for context experiments.

    The full state already contains the same values as the human-readable
    header lines, and it repeats the last few hands and actions.  This version
    keeps one canonical state block, only the most recent actions, and no
    match note or recent-hand list.
    """
    locale = _locale(locale)
    text = prompt_text(locale)
    state = view.to_prompt_dict()
    recent_actions = [
        {
            "seat": item.get("seat"),
            "type": item.get("type"),
            "amount": item.get("amount"),
        }
        for item in state["actions_this_hand"][-6:]
    ]
    compact_state = {
        "hand_id": state["hand_id"],
        "street": state["street"],
        "position": state["position"],
        "hole_cards": state["hole_cards"],
        "community_cards": state["community_cards"],
        "pot": state["pot"],
        "your_chips": state["your_chips"],
        "opponent_chips": state["opponent_chips"],
        "your_street_bet": state["your_street_bet"],
        "opponent_street_bet": state["opponent_street_bet"],
        "current_bet": state["current_bet"],
        "to_call": state["to_call"],
        "call_amount": state["call_amount"],
        "min_raise_to": state["min_raise_to"],
        "max_raise_to": state["max_raise_to"],
        "legal_actions": state["legal_actions"],
        "blinds": state["blinds"],
        "recent_actions": recent_actions,
    }
    lines = [
        text["turn_header"].format(
            hand=view.hand_number,
            street=_street_name(view.street, locale),
        ),
        f"```json\n{json.dumps(compact_state, ensure_ascii=False, separators=(',', ':'))}\n```",
    ]
    if view.table_talk:
        lines.extend([text["turn_talk_header"], text["turn_talk_warning"]])
        for item in view.table_talk:
            lines.append(
                text["turn_talk_item"].format(
                    hand=int(item.get("hand_id") or 0),
                    street=_street_name(str(item.get("street") or ""), locale),
                    name=str(item.get("name") or text["opponent"]),
                    text=str(item.get("text") or ""),
                )
            )
    if require_table_talk:
        lines.append(text["turn_talk_required"])
    lines.append(text["turn_closing_compact"])
    return "\n".join(lines)


def build_nudge_message(require_table_talk: bool = False, locale: str = "zh-CN") -> str:
    locale = _locale(locale)
    text = prompt_text(locale)
    prefix = text["nudge_prefix"]
    separator = "" if locale == "zh-CN" else " "
    if require_table_talk:
        prefix += separator + text["nudge_talk"]
    return f"{prefix}{separator}{text['nudge_action']}"


def build_hand_result_message(
    result: Optional[dict],
    seat: int,
    name: str,
    locale: str = "zh-CN",
    final: bool = False,
) -> str:
    locale = _locale(locale)
    text = prompt_text(locale)
    if not result:
        return text["result_empty"]
    winner_seats = result.get("winner_seats") or []
    you_won = seat in winner_seats
    players = {item.get("seat"): item for item in result.get("players", [])}
    mine = players.get(seat, {})
    pot = result.get("pot", 0)
    lines = [
        text["result_header"],
        text["result_won"].format(pot=pot)
        if you_won
        else text["result_lost"].format(pot=pot),
        text["result_net"].format(
            net=int(mine.get("net", 0) or 0),
            chips=mine.get("chips_after", 0),
        ),
    ]
    board = result.get("board") or []
    if board:
        lines.append(text["result_board"].format(board=" ".join(board)))
    showdown = result.get("showdown") or []
    for item in showdown:
        best = " ".join(item.get("best_five") or [])
        category = hand_category_name(
            str(item.get("category") or ""),
            locale,
            str(item.get("category_name") or ""),
        )
        lines.append(
            text["result_showdown"].format(
                seat=item.get("seat"),
                category=category,
                best=best,
            ).strip()
        )
    if result.get("reason") == "fold":
        lines.append(text["result_fold"])
    if not final:
        if result.get("match_over"):
            lines.append(text["result_match_over"])
        else:
            lines.append(text["result_next_hand"])
    return "\n".join(lines)


def build_hand_summary_message(
    result: Optional[dict],
    seat: int,
    locale: str = "zh-CN",
) -> str:
    """Build a compact one-message summary for a completed hand."""
    locale = _locale(locale)
    text = prompt_text(locale)
    if not result:
        return text["result_empty"]
    winner_seats = result.get("winner_seats") or []
    won = seat in winner_seats
    players = {item.get("seat"): item for item in result.get("players", [])}
    mine = players.get(seat, {})
    net = int(mine.get("net", 0) or 0)
    chips = mine.get("chips_after", 0)
    board = " ".join(result.get("board") or []) or text["none"]
    outcome = text["result_summary_won"] if won else text["result_summary_lost"]
    if result.get("reason") == "fold":
        outcome += text["result_summary_fold"]
    showdown = ""
    entries = result.get("showdown") or []
    if entries:
        categories = {
            item.get("seat"): hand_category_name(
                str(item.get("category") or ""),
                locale,
                str(item.get("category_name") or ""),
            )
            for item in entries
        }
        mine_category = categories.get(seat)
        other_category = categories.get(1 - seat)
        if mine_category and other_category:
            showdown = text["result_summary_showdown"].format(
                mine=mine_category,
                theirs=other_category,
            )
    return text["result_summary"].format(
        hand=result.get("hand_id", 0),
        outcome=outcome,
        pot=int(result.get("pot", 0) or 0),
        net=net,
        chips=chips,
        board=board,
        showdown=showdown,
    )


def build_final_table_talk_message(
    result: Optional[dict],
    seat: int,
    name: str,
    locale: str = "zh-CN",
) -> str:
    """Prompt one optional post-match table-talk message."""
    locale = _locale(locale)
    text = prompt_text(locale)
    base = build_hand_result_message(result, seat, name, locale, final=True)
    return (
        f"{base}\n\n"
        f"{text['final_header']}\n"
        f"{text['final_instructions']}"
    )


def build_final_talk_system_prompt(name: str, locale: str = "zh-CN") -> str:
    """Build a short system prompt for the post-match table-talk call."""
    locale = _locale(locale)
    return prompt_text(locale)["final_system"].format(name=name)


def build_final_talk_summary_from_stats(
    stats: MatchStats,
    locale: str = "zh-CN",
) -> str:
    """Summarize the match from aggregate stats for the fresh context.

    Args:
        stats: Aggregated results for one seat.
        locale: Agent locale.

    Returns:
        Localized summary sentence used before the final-talk prompt.
    """
    locale = _locale(locale)
    text = prompt_text(locale)
    if stats.net > 0:
        result_label = text["final_result_profit"]
    elif stats.net < 0:
        result_label = text["final_result_loss"]
    else:
        result_label = text["final_result_even"]
    return text["final_summary"].format(
        hands=stats.hands,
        wins=stats.wins,
        losses=stats.losses,
        net=stats.net,
        result_label=result_label,
        chips=stats.chips,
        opponent=stats.opponent_chips,
    )


def build_final_talk_summary(
    hand_results: Sequence[dict],
    seat: int,
    locale: str = "zh-CN",
) -> str:
    """Summarize raw hand results for the fresh post-match context."""
    stats = MatchStats.from_results(hand_results, seat)
    return build_final_talk_summary_from_stats(stats, locale)


def build_final_talk_fresh_message(
    result: Optional[dict],
    seat: int,
    hand_results: Sequence[dict],
    locale: str = "zh-CN",
    *,
    stats: Optional[MatchStats] = None,
) -> str:
    """Build a compact post-match user message with a match summary.

    Args:
        result: Final hand result.
        seat: Seat whose perspective is being summarized.
        hand_results: Raw history used when ``stats`` is omitted.
        locale: Agent locale.
        stats: Aggregate match stats; preferred over re-scanning history.
    """
    summary = (
        build_final_talk_summary_from_stats(stats, locale)
        if stats is not None
        else build_final_talk_summary(hand_results, seat, locale)
    )
    return f"{summary}\n\n{build_final_table_talk_message(result, seat, '', locale)}"


def _street_name(street: str, locale: str = "zh-CN") -> str:
    key = f"street.{street}"
    value = tr(locale, key)
    return street if value == key else value


__all__ = [
    "AGENT_TOOL_NAMES",
    "ALL_TOOL_NAMES",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_EN",
    "TABLE_TALK_TOOL_NAME",
    "build_compact_turn_message",
    "build_final_table_talk_message",
    "build_final_talk_fresh_message",
    "build_final_talk_summary",
    "build_final_talk_summary_from_stats",
    "build_final_talk_system_prompt",
    "build_hand_result_message",
    "build_hand_summary_message",
    "build_nudge_message",
    "build_system_prompt",
    "build_table_talk_section",
    "build_turn_message",
]
