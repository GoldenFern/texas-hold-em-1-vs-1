"""Small continuous-context preview for DeepSeek.

This script is intentionally narrower than the full evaluation harness: it
plays a short continuous match against the Project DQN bot, records every model
request's token usage and cache fields, and writes a compact report.

Examples
--------
Legacy context::

    python scripts/preview_context.py --policy legacy --hands 10 --seed 2026

Hand-summary compression::

    python scripts/preview_context.py --policy compact --hands 10 --seed 2026

Compression plus reduced state/tool payloads::

    python scripts/preview_context.py --policy compact_slim --hands 10 --seed 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from scripts.eval_llm_vs_bot import DEFAULT_BOT_MODEL, RetryingClient
from src.agents.base import AgentView
from src.agents.project_bot import ProjectBot
from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.analysis.history import HistoryStats, build_public_hand_record
from src.engine.actions import Action, ActionType
from src.engine.heads_up import HeadsUpGame, Street
from src.i18n import tr
from src.llm.agent import LLMAgent
from src.llm.client import LLMClient
from src.llm.prompt import GAME_TOOL_NAMES, build_system_prompt
from src.llm.providers import ProviderRegistry
from src.llm.retry import RetryPolicy
from src.llm.tools import select_tools

AUX_TOOL_NAMES = tuple(name for name in GAME_TOOL_NAMES if name != "apply_action")
MAX_ACTIONS_PER_HAND = 500


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview context policies in a short match.")
    parser.add_argument(
        "--policy",
        default="compact_slim",
        choices=("legacy", "full", "compact", "compact_slim", "checkpoint"),
    )
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--bot-model", type=Path, default=DEFAULT_BOT_MODEL)
    parser.add_argument("--high-water", type=int, default=30000)
    parser.add_argument("--low-water", type=int, default=12000)
    parser.add_argument("--keep-hands", type=int, default=2)
    parser.add_argument("--keep-tokens", type=int, default=20000)
    parser.add_argument("--buffer", type=int, default=0)
    parser.add_argument("--summary-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--summary-retries",
        default="0.85,0.7,0.5,0.35",
        help="Comma-separated shrink steps for checkpoint summary retries.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=0,
        help="Hard prompt budget; 0 uses the model context window minus output reserve.",
    )
    parser.add_argument("--stream-usage", action="store_true")
    parser.add_argument(
        "--cache-scope",
        default="",
        help="DeepSeek user_id used for KV-cache isolation; defaults to a unique run id.",
    )
    parser.add_argument("--price-cache-input", type=float, default=0.003)
    parser.add_argument("--price-miss-input", type=float, default=0.15)
    parser.add_argument("--price-output", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.hands <= 0:
        parser.error("--hands must be positive")
    if args.keep_hands <= 0:
        parser.error("--keep-hands must be positive")
    if args.high_water <= 0 or args.low_water <= 0:
        parser.error("context watermarks must be positive")
    if args.low_water > args.high_water:
        parser.error("--low-water cannot exceed --high-water")
    return args


def build_view(game: HeadsUpGame, seat: int, history: Sequence[dict]) -> AgentView:
    player = game.players[seat]
    opponent = game.players[1 - seat]
    constraints = game.action_constraints(seat)
    return AgentView(
        seat=seat,
        name=player.name,
        hand_id=game.hand_id,
        street=game.street.value,
        hole_cards=[card.code for card in player.hole_cards],
        community=[card.code for card in game.community],
        pot=game.pot,
        my_chips=player.chips,
        opponent_chips=opponent.chips,
        my_street_bet=player.street_bet,
        opponent_street_bet=opponent.street_bet,
        my_total_bet=player.total_bet,
        opponent_total_bet=opponent.total_bet,
        current_bet=game.current_bet,
        to_call=max(0, constraints.get("to_call", 0)),
        call_amount=int(constraints.get("call_amount", 0)),
        min_raise_to=int(constraints.get("min_raise_to", game.big_blind)),
        max_raise_to=int(constraints.get("max_raise_to", player.chips)),
        legal_actions=[action.value for action in game.legal_actions(seat)],
        action_history=list(game.actions[-12:]),
        opponent_last_action=opponent.last_action,
        button=game.button_index,
        small_blind=game.small_blind,
        big_blind=game.big_blind,
        hand_number=game.hand_id,
        match_note=tr(
            "zh-CN",
            "agent.match_note",
            hand=game.hand_id,
            total=len(history),
            small=game.small_blind,
            big=game.big_blind,
            chips=game.starting_chips,
            mine=player.chips,
            opponent=opponent.chips,
        ),
        history_summary=[history_item(item, seat) for item in history[-3:]],
        table_talk=[],
    )


def history_item(item: dict, seat: int) -> dict:
    players = {entry.get("seat"): entry for entry in item.get("players", [])}
    mine = players.get(seat, {})
    return {
        "hand_id": item.get("hand_id"),
        "you_won": seat in (item.get("winner_seats") or []),
        "reason": item.get("reason"),
        "pot": item.get("pot", 0),
        "board": item.get("board", []),
        "your_net": mine.get("net", 0),
        "your_chips_after": mine.get("chips_after", 0),
    }


def safe_action(game: HeadsUpGame, seat: int) -> Action:
    legal = {action.value for action in game.legal_actions(seat)}
    if "check" in legal:
        return Action(seat, ActionType.CHECK, source="system", auto=True)
    if "call" in legal:
        return Action(seat, ActionType.CALL, source="system", auto=True)
    return Action(seat, ActionType.FOLD, source="system", auto=True)


def compact_action(action: dict) -> dict:
    return {
        key: action.get(key)
        for key in ("hand_id", "street", "seat", "type", "amount", "source", "auto")
    }


def request_cost(
    usage: dict,
    price_cache_input: float,
    price_miss_input: float,
    price_output: float,
) -> float:
    return (
        int(usage.get("cached_input_tokens", 0)) * price_cache_input / 1_000_000.0
        + int(usage.get("cache_miss_input_tokens", 0)) * price_miss_input / 1_000_000.0
        + int(usage.get("output_tokens", 0)) * price_output / 1_000_000.0
    )


def play_session(args: argparse.Namespace) -> dict:
    runtime = json.loads(
        (PROJECT_ROOT / "config" / "game.json").read_text(encoding="utf-8")
    )["llm_runtime"]
    runtime.update(
        {
            "context_policy": args.policy,
            "context_high_water_tokens": args.high_water,
            "context_low_water_tokens": args.low_water,
            "context_keep_recent_hands": args.keep_hands,
            "context_keep_recent_tokens": args.keep_tokens,
            "context_buffer_tokens": args.buffer,
            "context_summary_max_tokens": args.summary_max_tokens,
            "context_summary_retries": [
                float(item)
                for item in str(args.summary_retries).split(",")
                if item.strip()
            ],
            "max_context_tokens_estimate": args.max_context,
            "max_retries": int(args.llm_retries),
            "retry_base_delay_ms": float(args.retry_backoff) * 1000.0,
            "retry_max_delay_ms": 10000.0,
        }
    )

    registry = ProviderRegistry()
    spec = registry.resolve(args.provider, args.model)
    if args.stream_usage:
        spec = replace(spec, request={**spec.request, "stream_usage": True})
    params = spec.normalize_params(
        {
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
        }
    )

    game = HeadsUpGame(
        starting_chips=args.chips,
        small_blind=args.sb,
        big_blind=args.bb,
        auto_rebuy=True,
        rebuy_amount=args.chips,
        seed=args.seed,
        locale="zh-CN",
    )
    game.configure_players(
        {"name": "DeepSeek", "kind": "llm"},
        {"name": "Project DQN", "kind": "bot"},
    )
    bot = ProjectBot(
        name="Project DQN",
        model_path=str(args.bot_model),
        algorithm="dqn",
        locale="zh-CN",
    )
    cache_scope = args.cache_scope.strip() or (
        f"ctx-preview-{args.policy}-{args.seed}-{int(time.time() * 1000)}"
    )
    args.cache_scope = cache_scope
    client = RetryingClient(
        LLMClient(
            registry,
            connect_timeout=args.connect_timeout,
            read_timeout=args.timeout,
            locale="zh-CN",
            user_id=cache_scope,
            retry_policy=RetryPolicy(
                max_retries=args.llm_retries,
                base_delay=args.retry_backoff,
                max_delay=10.0,
            ),
        ),
        attempts=1,
    )
    history_stats = HistoryStats()

    def history_provider(
        seat: int,
        *,
        scope: str = "both",
        last_hands: int = 20,
        focus: str = "all",
        checkpoint: bool = False,
    ) -> dict:
        match_info = {
            "small_blind": game.small_blind,
            "big_blind": game.big_blind,
            "your_chips": game.players[seat].chips,
            "opponent_chips": game.players[1 - seat].chips,
        }
        if checkpoint:
            return history_stats.checkpoint_snapshot(
                [],
                seat,
                hero_identity="llm:deepseek:deepseek-flash:DeepSeek",
                opponent_identity="bot:project_dqn:Project DQN",
                last_hands=5,
                match_info=match_info,
            )
        return history_stats.report(
            [],
            seat,
            scope=scope,
            last_hands=last_hands,
            focus=focus,
            hero_identity="llm:deepseek:deepseek-flash:DeepSeek",
            opponent_identity="bot:project_dqn:Project DQN",
            match_info=match_info,
        )

    agent = LLMAgent(
        seat=0,
        name="DeepSeek",
        spec=spec,
        params=params,
        client=client,
        analyzer=HandDistributionAnalyzer(),
        validate_action=lambda action: game.preview(action, locale="zh-CN"),
        runtime=runtime,
        tools=select_tools(GAME_TOOL_NAMES, locale="zh-CN"),
        system_prompt=build_system_prompt(GAME_TOOL_NAMES, locale="zh-CN"),
        history_provider=history_provider,
        locale="zh-CN",
    )

    events: List[dict] = []
    history: List[dict] = []
    hands: List[dict] = []

    def emit(event: dict) -> None:
        payload = dict(event)
        payload["_ts"] = time.time()
        events.append(payload)

    for hand_index in range(args.hands):
        try:
            game.start_hand()
        except RuntimeError as exc:
            print(f"stop before hand {hand_index + 1}: {exc}")
            break
        started = time.perf_counter()
        event_start = len(events)
        actions = 0
        fallbacks = 0
        while game.street != Street.FINISHED:
            seat = game.actor
            if seat is None:
                raise RuntimeError("actor is None before the hand finished")
            actions += 1
            if actions > MAX_ACTIONS_PER_HAND:
                raise RuntimeError("hand exceeded action limit")
            view = build_view(game, seat, history)
            actor = agent if seat == 0 else bot
            action = actor.decide(view, emit)
            result = game.apply(action)
            if not result.ok:
                fallbacks += 1
                result = game.apply(safe_action(game, seat))
                if not result.ok:
                    raise RuntimeError(f"fallback action rejected: {result.error}")

        result = game.result
        if result is None:
            raise RuntimeError("hand finished without a result")
        record = build_public_hand_record(game, result, {})
        history.append(record)
        history_stats.add(record)
        agent.on_hand_finished(record)
        hand_events = events[event_start:]
        usage = {
            key: 0
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "cache_miss_input_tokens",
                "output_tokens",
                "reasoning_tokens",
            )
        }
        estimated = 0
        summary_calls = 0
        for event in hand_events:
            if event.get("type") not in ("message_end", "context_summary_end"):
                continue
            if not event.get("usage"):
                continue
            for key in usage:
                usage[key] += int(event["usage"].get(key, 0))
            if event.get("type") == "message_end":
                estimated += int(bool(event.get("estimated_usage")))
            else:
                summary_calls += int(event.get("attempts", 0) or 0)
        players = {item["seat"]: item for item in result.get("players", [])}
        net = int(players.get(0, {}).get("net", 0))
        turns = sum(1 for event in hand_events if event.get("type") == "turn_start")
        api_calls = (
            sum(1 for event in hand_events if event.get("type") == "llm_request")
            + summary_calls
        )
        auto_turns = sum(
            1
            for event in hand_events
            if event.get("type") == "turn_end" and event.get("auto")
        )
        tool_calls = sum(
            1
            for event in hand_events
            if event.get("type") == "tool_result"
            and event.get("name") in AUX_TOOL_NAMES
        )
        compactions = sum(
            1 for event in hand_events if event.get("type") == "context_compacted"
        )
        trims = sum(
            1 for event in hand_events if event.get("type") == "context_trimmed"
        )
        requests = [event for event in hand_events if event.get("type") == "llm_request"]
        context_start = int(requests[0].get("estimated_context_tokens", 0)) if requests else 0
        context_end = int(requests[-1].get("estimated_context_tokens", 0)) if requests else 0
        duration = time.perf_counter() - started
        cost = request_cost(
            usage,
            args.price_cache_input,
            args.price_miss_input,
            args.price_output,
        )
        row = {
            "policy": args.policy,
            "hand_index": hand_index,
            "hand_id": game.hand_id,
            "seed": args.seed,
            "button": game.button_index,
            "hole_llm": " ".join(card.code for card in game.players[0].hole_cards),
            "board": " ".join(result.get("board") or []),
            "reason": result.get("reason", ""),
            "winner_seats": " ".join(str(seat) for seat in result.get("winner_seats") or []),
            "pot": int(result.get("pot", 0)),
            "net_llm": net,
            "turns": turns,
            "api_calls": api_calls,
            "summary_calls": summary_calls,
            "aux_tool_calls": tool_calls,
            "auto_turns": auto_turns,
            "fallbacks": fallbacks,
            "compactions": compactions,
            "trims": trims,
            "context_start": context_start,
            "context_end": context_end,
            "messages_end": len(agent.messages),
            "rebuy_llm": game.players[0].rebuy_count,
            "rebuy_bot": game.players[1].rebuy_count,
            "duration_sec": round(duration, 3),
            "estimated_usage_messages": estimated,
            **{key: int(value) for key, value in usage.items()},
            "cache_hit_rate": round(
                usage["cached_input_tokens"] / usage["input_tokens"], 4
            )
            if usage["input_tokens"]
            else 0.0,
            "cost_usd": round(cost, 6),
            "actions": [compact_action(item) for item in game.actions],
            "_event_start": event_start,
            "_event_end": len(events),
        }
        hands.append(row)
        if not args.quiet:
            print(
                f"[{args.policy}] hand {hand_index + 1}/{args.hands} "
                f"id={game.hand_id} net={net:+d} pot={row['pot']} "
                f"turns={turns} calls={api_calls} "
                f"in={usage['input_tokens']} cache={usage['cached_input_tokens']} "
                f"ctx={context_end} comp={compactions} "
                f"({duration:.1f}s)"
            )

    return {
        "args": args,
        "events": events,
        "hands": hands,
        "agent": agent,
        "game": game,
        "usage": dict(agent.total_usage),
        "messages": list(agent.messages),
        "hand_results": list(history),
        "final_result": history[-1] if history else None,
        "system_prompt": agent.system_prompt,
        "cache_scope": cache_scope,
    }


def request_rows(session: dict) -> List[dict]:
    args: argparse.Namespace = session["args"]
    events: List[dict] = session["events"]
    turn_hand: Dict[str, int] = {}
    request_meta: Dict[str, dict] = {}
    for event in events:
        if event.get("type") == "turn_start":
            turn_hand[str(event.get("turn_id"))] = int(event.get("hand_id") or 0)
        elif event.get("type") == "llm_request":
            request_meta[str(event.get("message_id"))] = event
    rows: List[dict] = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        usage = event.get("usage") or {}
        if not usage:
            continue
        meta = request_meta.get(str(event.get("message_id")), {})
        input_tokens = int(usage.get("input_tokens", 0))
        cached = int(usage.get("cached_input_tokens", 0))
        rows.append(
            {
                "policy": args.policy,
                "seed": args.seed,
                "hand_id": turn_hand.get(str(event.get("turn_id")), 0),
                "turn_id": event.get("turn_id", ""),
                "message_id": event.get("message_id", ""),
                "step": int(meta.get("step", 0)),
                "message_count": int(meta.get("message_count", 0)),
                "context_estimate": int(meta.get("estimated_context_tokens", 0)),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached,
                "cache_miss_input_tokens": int(
                    usage.get("cache_miss_input_tokens", max(0, input_tokens - cached))
                ),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "reasoning_tokens": int(usage.get("reasoning_tokens", 0)),
                "cache_hit_rate": round(cached / input_tokens, 4) if input_tokens else 0.0,
                "estimated_usage": bool(event.get("estimated_usage")),
                "error": bool(event.get("error")),
                "latency_ms": float(event.get("latency_ms", 0.0) or 0.0),
                "cost_usd": round(
                    request_cost(
                        usage,
                        args.price_cache_input,
                        args.price_miss_input,
                        args.price_output,
                    ),
                    6,
                ),
            }
        )
    return rows


def summarize(session: dict, requests: Sequence[dict]) -> dict:
    args: argparse.Namespace = session["args"]
    hands = session["hands"]
    nets = [int(row["net_llm"]) for row in hands]
    bb_values = [net / float(args.bb) for net in nets]
    mean_bb = statistics.fmean(bb_values) if bb_values else 0.0
    sd_bb = statistics.stdev(bb_values) if len(bb_values) > 1 else 0.0
    ci = 1.96 * sd_bb / math.sqrt(len(bb_values)) if bb_values else 0.0
    total_input = sum(int(row["input_tokens"]) for row in hands)
    total_cached = sum(int(row["cached_input_tokens"]) for row in hands)
    total_miss = sum(int(row["cache_miss_input_tokens"]) for row in hands)
    total_output = sum(int(row["output_tokens"]) for row in hands)
    total_reasoning = sum(int(row["reasoning_tokens"]) for row in hands)
    total_cost = sum(float(row["cost_usd"]) for row in hands)
    reasons: Dict[str, int] = {}
    for row in hands:
        reasons[str(row["reason"])] = reasons.get(str(row["reason"]), 0) + 1

    def hit_rate(rows: Sequence[dict]) -> float:
        tokens = sum(int(row["input_tokens"]) for row in rows)
        cached = sum(int(row["cached_input_tokens"]) for row in rows)
        return round(cached / tokens, 4) if tokens else 0.0

    return {
        "policy": args.policy,
        "hands": len(hands),
        "seed": args.seed,
        "provider": args.provider,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "cache_scope": args.cache_scope,
        "context_limit": (args.max_context or "auto")
        if args.policy == "legacy"
        else args.high_water,
        "high_water": args.high_water,
        "low_water": args.low_water,
        "keep_hands": args.keep_hands,
        "net_llm": sum(nets),
        "bb_per_100": round(mean_bb * 100.0, 2),
        "bb_per_100_ci": round(ci * 100.0, 2),
        "total_input_tokens": total_input,
        "total_cached_input_tokens": total_cached,
        "total_cache_miss_input_tokens": total_miss,
        "total_output_tokens": total_output,
        "total_reasoning_tokens": total_reasoning,
        "cache_hit_rate": round(total_cached / total_input, 4) if total_input else 0.0,
        "first_request_cache_hit_rate": hit_rate(requests[:1]),
        "steady_state_cache_hit_rate": hit_rate(requests[1:]),
        "input_tokens_per_hand": round(total_input / len(hands), 1) if hands else 0.0,
        "cache_miss_tokens_per_hand": round(total_miss / len(hands), 1) if hands else 0.0,
        "output_tokens_per_hand": round(total_output / len(hands), 1) if hands else 0.0,
        "cost_usd": round(total_cost, 6),
        "api_calls": sum(int(row["api_calls"]) for row in hands),
        "summary_calls": sum(int(row.get("summary_calls", 0)) for row in hands),
        "turns": sum(int(row["turns"]) for row in hands),
        "aux_tool_calls": sum(int(row["aux_tool_calls"]) for row in hands),
        "auto_turns": sum(int(row["auto_turns"]) for row in hands),
        "fallbacks": sum(int(row["fallbacks"]) for row in hands),
        "compactions": sum(int(row["compactions"]) for row in hands),
        "trims": sum(int(row.get("trims", 0)) for row in hands),
        "estimated_usage_requests": sum(1 for row in requests if row["estimated_usage"]),
        "api_errors": sum(1 for row in requests if row["error"]),
        "reasons": reasons,
    }


def write_outputs(session: dict, output: Path) -> dict:
    args: argparse.Namespace = session["args"]
    output.mkdir(parents=True, exist_ok=True)
    requests = request_rows(session)
    summary = summarize(session, requests)
    hands = session["hands"]

    hand_fields = [
        "policy", "hand_index", "hand_id", "seed", "button", "hole_llm", "board",
        "reason", "winner_seats", "pot", "net_llm", "turns", "api_calls",
        "summary_calls", "aux_tool_calls", "auto_turns", "fallbacks", "compactions", "trims", "context_start",
        "context_end", "messages_end", "rebuy_llm", "rebuy_bot", "duration_sec",
        "estimated_usage_messages", "input_tokens", "cached_input_tokens",
        "cache_miss_input_tokens", "output_tokens", "reasoning_tokens",
        "cache_hit_rate", "cost_usd",
    ]
    with (output / "hands.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hand_fields)
        writer.writeheader()
        for row in hands:
            writer.writerow({key: row.get(key) for key in hand_fields})

    request_fields = [
        "policy", "seed", "hand_id", "turn_id", "message_id", "step",
        "message_count", "context_estimate", "input_tokens", "cached_input_tokens",
        "cache_miss_input_tokens", "output_tokens", "reasoning_tokens",
        "cache_hit_rate", "estimated_usage", "error", "latency_ms", "cost_usd",
    ]
    with (output / "requests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=request_fields)
        writer.writeheader()
        for row in requests:
            writer.writerow({key: row.get(key) for key in request_fields})

    with (output / "hands.jsonl").open("w", encoding="utf-8") as handle:
        for row in hands:
            payload = dict(row)
            start = int(row.get("_event_start", 0))
            end = int(row.get("_event_end", start))
            payload["events"] = [
                event
                for event in session["events"][start:end]
                if event.get("type")
                not in ("reasoning_delta", "content_delta", "tool_call_start", "tool_call_args_delta")
            ]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "messages.json").write_text(
        json.dumps(
            {
                "policy": args.policy,
                "provider": args.provider,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "cache_scope": session["cache_scope"],
                "system_prompt": session["system_prompt"],
                "messages": session["messages"],
                "hand_results": session["hand_results"],
                "final_result": session["final_result"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Context preview · {args.policy}",
        "",
        f"- model: `{args.provider}/{args.model}`, reasoning `{args.reasoning_effort}`",
        f"- hands: {summary['hands']}, seed: {args.seed}",
        f"- cache scope: `{summary['cache_scope']}`",
        f"- policy: `{args.policy}`, limit {summary['context_limit']}, low {args.low_water}, keep {args.keep_hands}",
        f"- chips: {args.chips}, blinds {args.sb}/{args.bb}",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| net chips | {summary['net_llm']:+d} |",
        f"| bb/100 | {summary['bb_per_100']:+.1f} ± {summary['bb_per_100_ci']:.1f} |",
        f"| input tokens | {summary['total_input_tokens']} |",
        f"| cached input | {summary['total_cached_input_tokens']} |",
        f"| cache miss input | {summary['total_cache_miss_input_tokens']} |",
        f"| cache hit rate | {summary['cache_hit_rate'] * 100:.1f}% |",
        f"| first request hit | {summary['first_request_cache_hit_rate'] * 100:.1f}% |",
        f"| steady state hit | {summary['steady_state_cache_hit_rate'] * 100:.1f}% |",
        f"| output tokens | {summary['total_output_tokens']} |",
        f"| reasoning tokens | {summary['total_reasoning_tokens']} |",
        f"| cost USD | ${summary['cost_usd']:.4f} |",
        f"| API calls | {summary['api_calls']} |",
        f"| summary calls | {summary['summary_calls']} |",
        f"| turns | {summary['turns']} |",
        f"| auxiliary tools | {summary['aux_tool_calls']} |",
        f"| auto turns | {summary['auto_turns']} |",
        f"| fallbacks | {summary['fallbacks']} |",
        f"| compactions | {summary['compactions']} |",
        f"| legacy trims | {summary['trims']} |",
        f"| estimated usage requests | {summary['estimated_usage_requests']} |",
        f"| API errors | {summary['api_errors']} |",
        "",
        "## Hands",
        "",
        "| hand | id | net | pot | result | turns | calls | input | cached | hit | ctx end | comp | sec |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hands:
        lines.append(
            f"| {row['hand_index'] + 1} | {row['hand_id']} | {row['net_llm']:+d} | "
            f"{row['pot']} | {row['reason']} | {row['turns']} | {row['api_calls']} | "
            f"{row['input_tokens']} | {row['cached_input_tokens']} | "
            f"{row['cache_hit_rate'] * 100:.0f}% | {row['context_end']} | "
            f"{row['compactions']} | {row['duration_sec']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT_ROOT / "tmp" / "context_preview" / f"{stamp}_{args.policy}_seed{args.seed}"
    args.output = args.output.resolve()
    session = play_session(args)
    summary = write_outputs(session, args.output)
    print(
        f"\nsummary: {args.output / 'summary.json'}\n"
        f"report:  {args.output / 'report.md'}\n"
        f"hands={summary['hands']} net={summary['net_llm']:+d} "
        f"bb/100={summary['bb_per_100']:+.1f} "
        f"cache={summary['cache_hit_rate'] * 100:.1f}% "
        f"cost=${summary['cost_usd']:.4f} "
        f"errors={summary['api_errors']} estimated={summary['estimated_usage_requests']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
