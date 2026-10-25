"""Compare full-history and fresh-context post-match table talk.

The script loads a `messages.json` produced by ``scripts/preview_context.py``
and calls ``LLMAgent.final_table_talk`` with the same final result under two
context modes:

* ``full``: append the final prompt to the whole match history.
* ``fresh``: use a short system prompt plus a compact match summary.

Each mode gets its own ``user_id`` cache scope.  The first run is reported
separately from later runs, because the first call may populate the cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from scripts.eval_llm_vs_bot import RetryingClient
from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.llm.agent import LLMAgent
from src.llm.client import LLMClient
from src.llm.context import MatchStats
from src.llm.prompt import AGENT_TOOL_NAMES
from src.llm.providers import ModelSpec, ProviderRegistry
from src.llm.retry import RetryPolicy
from src.llm.tools import select_tools


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare final table-talk contexts.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--mode", default="both", choices=("both", "full", "fresh"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--full-cache-scope", default="")
    parser.add_argument("--fresh-cache-scope", default="")
    parser.add_argument("--price-cache-input", type=float, default=0.003)
    parser.add_argument("--price-miss-input", type=float, default=0.15)
    parser.add_argument("--price-output", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    return args


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


def run_once(
    history: dict,
    mode: str,
    args: argparse.Namespace,
    spec: ModelSpec,
    params: dict,
    registry: ProviderRegistry,
    cache_scope: str,
    run_index: int,
) -> dict:
    messages = [copy.deepcopy(item) for item in history.get("messages", [])]
    system_prompt = str(history.get("system_prompt") or "")
    if not system_prompt and messages:
        system_prompt = str(messages[0].get("content") or "")
    final_result = history.get("final_result")
    hand_results = list(history.get("hand_results") or [])
    if final_result and not hand_results:
        hand_results = [final_result]

    runtime = {
        "context_policy": "full",
        "context_high_water_tokens": 900000,
        "context_low_water_tokens": 300000,
        "context_keep_recent_hands": 2,
        "final_talk_context": mode,
        "table_talk_max_chars": 160,
        "table_talk_max_per_turn": 1,
        "max_context_tokens_estimate": 900000,
        "max_steps_per_turn": 12,
        "max_consecutive_failures": 3,
        "request_timeout_seconds": args.timeout,
    }
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
    agent = LLMAgent(
        seat=0,
        name="DeepSeek",
        spec=spec,
        params=params,
        client=client,
        analyzer=HandDistributionAnalyzer(),
        validate_action=lambda action: None,
        runtime=runtime,
        tools=select_tools(AGENT_TOOL_NAMES, locale="zh-CN"),
        system_prompt=system_prompt,
        submit_table_talk=lambda text: {"ok": True, "text": text},
        locale="zh-CN",
    )
    agent.match_stats = MatchStats.from_results(hand_results, 0)
    agent.messages = messages if mode == "full" else messages[:1]

    events: List[dict] = []
    submitted: List[str] = []

    def emit(event: dict) -> None:
        payload = dict(event)
        payload["_ts"] = time.time()
        events.append(payload)

    def submit(text: str) -> dict:
        submitted.append(text)
        return {"ok": True, "text": text}

    started = time.perf_counter()
    agent.final_table_talk(final_result, submit, emit)
    duration = time.perf_counter() - started

    usage = dict(agent.total_usage)
    start_event = next(
        (event for event in events if event.get("type") == "final_talk_start"),
        {},
    )
    input_tokens = usage["input_tokens"]
    cached = usage["cached_input_tokens"]
    return {
        "mode": mode,
        "run": run_index,
        "cache_scope": cache_scope,
        "success": bool(submitted),
        "message": submitted[-1] if submitted else "",
        "message_length": len(submitted[-1]) if submitted else 0,
        "history_messages": len(messages),
        "history_tokens_estimate": int(start_event.get("estimated_tokens", 0) or 0),
        "context_messages": int(start_event.get("message_count", 0) or 0),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_miss_input_tokens": int(
            usage.get("cache_miss_input_tokens", max(0, input_tokens - cached))
        ),
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "cache_hit_rate": round(cached / input_tokens, 4) if input_tokens else 0.0,
        "cost_usd": round(
            request_cost(
                usage,
                args.price_cache_input,
                args.price_miss_input,
                args.price_output,
            ),
            6,
        ),
        "duration_sec": round(duration, 3),
        "events": [
            event
            for event in events
            if event.get("type")
            not in ("reasoning_delta", "content_delta", "tool_call_start", "tool_call_args_delta")
        ],
    }


def summarize(rows: Sequence[dict]) -> dict:
    modes = sorted({row["mode"] for row in rows})
    summary = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        successes = [row for row in mode_rows if row["success"]]
        first = mode_rows[0] if mode_rows else {}
        rest = mode_rows[1:]
        total_input = sum(row["input_tokens"] for row in successes)
        total_cached = sum(row["cached_input_tokens"] for row in successes)
        total_miss = sum(row["cache_miss_input_tokens"] for row in successes)
        total_output = sum(row["output_tokens"] for row in successes)
        total_cost = sum(row["cost_usd"] for row in successes)
        summary[mode] = {
            "runs": len(mode_rows),
            "successes": len(successes),
            "first_input_tokens": first.get("input_tokens", 0),
            "first_cache_hit_rate": first.get("cache_hit_rate", 0.0),
            "steady_cache_hit_rate": round(
                sum(row["cached_input_tokens"] for row in rest)
                / max(1, sum(row["input_tokens"] for row in rest)),
                4,
            ),
            "total_input_tokens": total_input,
            "total_cached_input_tokens": total_cached,
            "total_cache_miss_input_tokens": total_miss,
            "total_output_tokens": total_output,
            "cache_hit_rate": round(total_cached / total_input, 4) if total_input else 0.0,
            "cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / len(successes), 6) if successes else 0.0,
            "avg_input_tokens": round(total_input / len(successes), 1) if successes else 0.0,
            "avg_output_tokens": round(total_output / len(successes), 1) if successes else 0.0,
            "avg_message_length": round(
                sum(row["message_length"] for row in successes) / len(successes), 1
            )
            if successes
            else 0.0,
            "history_tokens_estimate": first.get("history_tokens_estimate", 0),
            "context_messages": first.get("context_messages", 0),
        }
    return summary


def write_outputs(rows: List[dict], summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode", "run", "cache_scope", "success", "message_length",
        "history_messages", "history_tokens_estimate", "context_messages",
        "input_tokens", "cached_input_tokens", "cache_miss_input_tokens",
        "output_tokens", "reasoning_tokens", "cache_hit_rate", "cost_usd",
        "duration_sec", "message",
    ]
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    with (output / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Final table talk context preview",
        "",
        "| mode | runs | success | first input | first hit | steady hit | avg input | avg output | avg cost | avg chars | history estimate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary.items():
        lines.append(
            f"| {mode} | {item['runs']} | {item['successes']} | "
            f"{item['first_input_tokens']} | {item['first_cache_hit_rate'] * 100:.1f}% | "
            f"{item['steady_cache_hit_rate'] * 100:.1f}% | "
            f"{item['avg_input_tokens']:.0f} | {item['avg_output_tokens']:.0f} | "
            f"${item['avg_cost_usd']:.6f} | {item['avg_message_length']:.0f} | "
            f"{item['history_tokens_estimate']} |"
        )
    lines.append("")
    lines.append("## Messages")
    lines.append("")
    for row in rows:
        if not row["success"]:
            continue
        lines.append(f"- `{row['mode']}` run {row['run']}: {row['message']}")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    history = json.loads(args.history.read_text(encoding="utf-8"))
    if not history.get("messages") or not history.get("final_result"):
        print("history file needs messages and final_result")
        return 1
    registry = ProviderRegistry()
    spec = registry.resolve(args.provider, args.model)
    params = spec.normalize_params(
        {
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
        }
    )
    original_scope = str(history.get("cache_scope") or "final-talk")
    full_scope = args.full_cache_scope.strip() or original_scope
    fresh_scope = args.fresh_cache_scope.strip() or f"{original_scope}-fresh"

    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT_ROOT / "tmp" / "final_talk_preview" / stamp
    args.output = args.output.resolve()

    modes = ["full", "fresh"] if args.mode == "both" else [args.mode]
    rows: List[dict] = []
    for mode in modes:
        scope = full_scope if mode == "full" else fresh_scope
        for repeat in range(args.repeats):
            row = run_once(
                history,
                mode,
                args,
                spec,
                params,
                registry,
                scope,
                repeat,
            )
            rows.append(row)
            print(
                f"[{mode}] run {repeat + 1}/{args.repeats} "
                f"success={row['success']} input={row['input_tokens']} "
                f"hit={row['cache_hit_rate'] * 100:.1f}% "
                f"ctx_msgs={row['context_messages']} "
                f"chars={row['message_length']} "
                f"cost=${row['cost_usd']:.6f}"
            )
    summary = summarize(rows)
    write_outputs(rows, summary, args.output)
    print(f"\nsummary: {args.output / 'summary.json'}")
    print(f"report:  {args.output / 'report.md'}")
    for mode, item in summary.items():
        print(
            f"{mode}: runs={item['runs']} success={item['successes']} "
            f"avg_input={item['avg_input_tokens']:.0f} "
            f"hit={item['cache_hit_rate'] * 100:.1f}% "
            f"avg_cost=${item['avg_cost_usd']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
