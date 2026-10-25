"""Run the scripted-opponent and tool-use experiments in budgeted batches.

The runner is intentionally small.  It calls ``eval_llm_vs_bot.py`` once per
condition, keeps a cumulative RMB estimate, and skips conditions that already
have a clean summary.  Experiment B is finally summarised by
``eval_llm_tool_matrix.py --report-only``.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_llm_vs_bot.py"
MATRIX_SCRIPT = PROJECT_ROOT / "scripts" / "eval_llm_tool_matrix.py"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    opponent_kind: str
    tools: str
    recommend_tools: bool


EXPERIMENT_A = (
    Condition("a_all_in", "只会全下", "always_all_in", "none", False),
    Condition("a_raise", "只会最小加注", "always_raise", "none", False),
    Condition("a_call_check", "只会跟注/过牌", "call_check", "none", False),
)

EXPERIMENT_B = (
    Condition("none", "空白", "random", "none", False),
    Condition(
        "random",
        "随机数",
        "random",
        "get_random_number",
        True,
    ),
    Condition(
        "distribution",
        "牌型分布",
        "random",
        "estimate_hand_probabilities",
        True,
    ),
    Condition("equity", "胜率", "random", "estimate_equity", True),
    Condition("ev", "跟注 EV", "random", "call_ev", True),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run budgeted GPT-6 Luna experiments.")
    parser.add_argument("--provider", default="4router")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument(
        "--provider-config",
        type=Path,
        default=None,
        help="Optional alternate providers.json for experiment-only relays.",
    )
    parser.add_argument("--llm-name", default="GPT-6 Luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--experiment", choices=("all", "a", "b"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument(
        "--context-policy",
        default=None,
        choices=("legacy", "full", "compact", "compact_slim", "checkpoint"),
    )
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--pilot-hands", type=int, default=20)
    parser.add_argument("--full-hands", type=int, default=400)
    parser.add_argument("--condition-attempts", type=int, default=3)
    parser.add_argument("--retry-pause", type=float, default=300.0)
    parser.add_argument("--relay-host", default="4router.net")
    parser.add_argument("--relay-port", type=int, default=443)
    parser.add_argument("--relay-wait", type=float, default=1800.0)
    parser.add_argument("--budget-cny", type=float, default=22.0)
    parser.add_argument("--price-input", type=float, default=0.045)
    parser.add_argument(
        "--price-cache-input",
        type=float,
        default=None,
        help="Cached input price; defaults to --price-input when omitted.",
    )
    parser.add_argument("--price-output", type=float, default=0.225)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deal-offset", type=int, default=0)
    parser.add_argument("--agent-locale", default="zh-CN")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("pilot", "full", "all"), default="all")
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.llm_retries <= 0:
        parser.error("--llm-retries must be positive")
    if args.retry_backoff < 0:
        parser.error("--retry-backoff cannot be negative")
    if args.timeout <= 0 or args.connect_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.max_context_tokens is not None and args.max_context_tokens < 0:
        parser.error("--max-context-tokens cannot be negative")
    if args.condition_attempts <= 0:
        parser.error("--condition-attempts must be positive")
    if args.retry_pause < 0:
        parser.error("--retry-pause cannot be negative")
    if args.pilot_hands <= 0 or args.full_hands <= 0:
        parser.error("hand counts must be positive")
    if args.full_hands % 2:
        parser.error("--full-hands must be even so every deal can be mirrored")
    if args.deal_offset < 0:
        parser.error("--deal-offset cannot be negative")
    if args.budget_cny <= 0:
        parser.error("--budget-cny must be positive")
    if args.price_input < 0 or args.price_output < 0:
        parser.error("token prices cannot be negative")
    if args.price_cache_input is not None and args.price_cache_input < 0:
        parser.error("--price-cache-input cannot be negative")
    args.output = args.output or (
        PROJECT_ROOT
        / "tmp"
        / "llm_experiments"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    if args.provider_config is not None:
        config = Path(args.provider_config)
        if not config.is_absolute():
            config = (PROJECT_ROOT / config).resolve()
        args.provider_config = config
    return args


def load_summary(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summary_clean(summary: Optional[dict], hands: int) -> bool:
    if not summary:
        return False
    return (
        int(summary.get("hands", 0)) >= hands
        and int(summary.get("fatal_hands", 0)) == 0
        and int(summary.get("hands_with_api_error", 0)) == 0
        and int(summary.get("api_error_auto", 0)) == 0
    )


def relay_reachable(args: argparse.Namespace) -> bool:
    try:
        with socket.create_connection(
            (args.relay_host, args.relay_port), timeout=8
        ):
            return True
    except OSError:
        return False


def wait_for_relay(args: argparse.Namespace) -> None:
    deadline = time.time() + max(0.0, args.relay_wait)
    while time.time() < deadline:
        if relay_reachable(args):
            return
        print(
            f"relay {args.relay_host}:{args.relay_port} unreachable; "
            "waiting 60s before retry",
            flush=True,
        )
        time.sleep(60)
    raise RuntimeError(f"relay {args.relay_host}:{args.relay_port} is still unreachable")


def build_command(
    args: argparse.Namespace,
    condition: Condition,
    hands: int,
    output: Path,
    remaining_budget: float,
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(EVAL_SCRIPT),
        "--provider",
        args.provider,
        "--model",
        args.model,
        *(
            ["--provider-config", str(args.provider_config)]
            if args.provider_config is not None
            else []
        ),
        "--llm-name",
        args.llm_name,
        "--opponent-kind",
        condition.opponent_kind,
        "--tools",
        condition.tools,
        "--hands",
        str(hands),
        "--workers",
        str(args.workers),
        "--chips",
        str(args.chips),
        "--sb",
        str(args.sb),
        "--bb",
        str(args.bb),
        "--seed",
        str(args.seed),
        "--deal-offset",
        str(args.deal_offset),
        *(
            ["--context-policy", str(args.context_policy)]
            if args.context_policy is not None
            else []
        ),
        *(
            ["--max-context-tokens", str(args.max_context_tokens)]
            if args.max_context_tokens is not None
            else []
        ),
        "--agent-locale",
        args.agent_locale,
        "--reasoning-effort",
        args.reasoning_effort,
        "--group",
        condition.key,
        "--output",
        str(output),
        "--batch-size",
        "50",
        "--budget-cny",
        f"{max(0.05, remaining_budget):.4f}",
        "--price-input",
        str(args.price_input),
        *(
            ["--price-cache-input", str(args.price_cache_input)]
            if args.price_cache_input is not None
            else []
        ),
        "--price-output",
        str(args.price_output),
        "--retries",
        "1",
        "--retry-api-errors",
        "--llm-retries",
        str(args.llm_retries),
        "--retry-backoff",
        str(args.retry_backoff),
        "--timeout",
        str(args.timeout),
        "--connect-timeout",
        str(args.connect_timeout),
    ]
    if condition.recommend_tools:
        command.append("--recommend-tools")
    return command


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_condition(
    args: argparse.Namespace,
    condition: Condition,
    hands: int,
    root: Path,
    spent: float,
) -> tuple[Optional[dict], float, bool]:
    output = root / condition.key
    summary_path = output / "summary.json"
    prior = load_summary(summary_path)
    if args.resume and summary_clean(prior, hands):
        cost = float((prior or {}).get("est_cost_cny", 0.0))
        print(
            f"[skip] {condition.key}: clean summary already has "
            f"{prior.get('hands', 0)} hands, cost Y{cost:.4f}",
            flush=True,
        )
        return prior, spent + cost, False

    remaining = max(0.05, args.budget_cny - spent)
    command = build_command(args, condition, hands, output, remaining)
    print(
        f"\n[{condition.key}] {condition.label}: {hands} hands, "
        f"opponent={condition.opponent_kind}, tools={condition.tools}, "
        f"remaining=Y{remaining:.4f}",
        flush=True,
    )
    print("$ " + subprocess.list2cmdline(command), flush=True)
    started = time.perf_counter()
    exit_code = subprocess.call(command, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - started
    summary = load_summary(summary_path)
    cost = float((summary or {}).get("est_cost_cny", 0.0))
    stopped = bool((summary or {}).get("stopped_for_budget"))
    print(
        f"[done] {condition.key}: exit={exit_code}, elapsed={elapsed:.1f}s, "
        f"hands={(summary or {}).get('hands', 0)}, cost=Y{cost:.4f}, "
        f"stopped_for_budget={stopped}",
        flush=True,
    )
    if summary is None:
        raise RuntimeError(f"missing summary for {condition.key}")
    return summary, spent + cost, stopped


def ensure_condition(
    args: argparse.Namespace,
    condition: Condition,
    hands: int,
    root: Path,
    spent: float,
) -> tuple[dict, float, bool]:
    """Run a condition, retrying the whole condition after relay instability."""
    last_summary: Optional[dict] = None
    last_spent = spent
    last_stopped = False
    for attempt in range(1, args.condition_attempts + 1):
        wait_for_relay(args)
        summary, last_spent, last_stopped = run_condition(
            args, condition, hands, root, last_spent
        )
        last_summary = summary
        if last_stopped or summary_clean(summary, hands):
            return summary or {}, last_spent, last_stopped
        print(
            f"[retry] {condition.key}: attempt {attempt}/{args.condition_attempts} "
            "left API errors; re-running the condition",
            flush=True,
        )
        if attempt < args.condition_attempts:
            time.sleep(max(0.0, args.retry_pause))
    raise RuntimeError(
        f"{condition.key} did not produce a clean summary after "
        f"{args.condition_attempts} attempts"
    )


def budget_payload(
    args: argparse.Namespace,
    spent: float,
    rows: Sequence[dict],
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "budget_cny": args.budget_cny,
        "spent_cny": round(spent, 6),
        "remaining_cny": round(max(0.0, args.budget_cny - spent), 6),
        "price_input_per_m": args.price_input,
        "price_cache_input_per_m": (
            args.price_input
            if args.price_cache_input is None
            else args.price_cache_input
        ),
        "price_output_per_m": args.price_output,
        "seed": args.seed,
        "deal_offset": args.deal_offset,
        "context_policy": args.context_policy,
        "max_context_tokens_estimate": args.max_context_tokens,
        "conditions": list(rows),
    }


def markdown_table(title: str, conditions: Sequence[Condition], summaries: Dict[str, dict]) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| 条件 | 手数 | 每手均值 | bb/100 | 95% CI | 胜率 | 工具/回合 | 费用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        summary = summaries.get(condition.key) or {}
        lines.append(
            "| {label} | {hands} | {mean:+.2f} | {bb100:+.1f} | ±{ci:.1f} | "
            "{win:.1f}% | {tool:.3f} | ¥{cost:.4f} |".format(
                label=condition.label,
                hands=summary.get("hands", 0),
                mean=float(summary.get("mean_net", 0.0)),
                bb100=float(summary.get("mean_bb100", 0.0)),
                ci=float(summary.get("paired_ci95_bb100", 0.0)),
                win=float(summary.get("win_pct", 0.0)),
                tool=float(summary.get("aux_tool_calls_per_turn", 0.0)),
                cost=float(summary.get("est_cost_cny", 0.0)),
            )
        )
    return lines


def write_top_report(
    path: Path,
    args: argparse.Namespace,
    a_summaries: Dict[str, dict],
    b_summaries: Dict[str, dict],
    spent: float,
    full_hands: int,
) -> None:
    lines = [
        "# GPT-6 Luna 实验汇总",
        "",
        f"- 生成时间: {datetime.now(timezone.utc).isoformat()}",
        f"- 模型: `{args.provider}/{args.model}`，推理强度 `{args.reasoning_effort}`",
        f"- 规则: {args.chips} 筹码，盲注 {args.sb}/{args.bb}，每手独立重置",
        f"- 并发: {args.workers}",
        f"- 每组手数: {full_hands}（{full_hands // 2} 副牌 × 2 个座位）",
        f"- 累计估算费用: ¥{spent:.4f} / 预算 ¥{args.budget_cny:.2f}",
        f"- 单价: 输入 ¥{args.price_input:.4f}/M，缓存输入 ¥{args.price_input if args.price_cache_input is None else args.price_cache_input:.4f}/M，输出 ¥{args.price_output:.4f}/M",
        "",
    ]
    if args.experiment in ("all", "a"):
        lines.extend(markdown_table("实验 A：固定策略对手", EXPERIMENT_A, a_summaries))
        lines.extend([""])
    if args.experiment in ("all", "b"):
        lines.extend(
            markdown_table("实验 B：工具对照（对随机 Agent）", EXPERIMENT_B, b_summaries)
        )
        lines.extend([""])
    lines.extend(
        [
            "## 说明",
            "",
            "- 每手都是独立牌局，筹码不跨手累积；GPT 不能跨手观察对手风格。",
            "- 实验 A 的 GPT 只有 `apply_action`；实验 B 每组只多一个辅助工具。",
            "- 实验 B 的提示词对本组工具做强烈推荐，但不强制。",
            "- 费用按本次运行配置的输入/输出单价估算，不含缓存。",
            "- 400 手一组的置信区间仍然较宽，结果只适合作方向性比较。",
        ]
    )
    if args.experiment in ("all", "b"):
        lines.append("- 实验 B 的详细配对比较见 `full/b_tools/combined_report.md`。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"output: {root}", flush=True)
    print(
        f"provider={args.provider}, model={args.model}, workers={args.workers}, "
        f"pilot={args.pilot_hands}, full={args.full_hands}, budget=Y{args.budget_cny:.2f}",
        flush=True,
    )

    spent = 0.0
    ledger: List[dict] = []
    budget_exhausted = False
    run_a = args.experiment in ("all", "a")
    run_b = args.experiment in ("all", "b")
    selected_conditions = (
        (EXPERIMENT_A if run_a else ()) + (EXPERIMENT_B if run_b else ())
    )

    if args.stage in ("pilot", "all") and not args.skip_pilot:
        pilot_rows: Dict[str, dict] = {}
        for condition in selected_conditions:
            summary, spent, stopped = ensure_condition(
                args, condition, args.pilot_hands, root / "pilot", spent
            )
            pilot_rows[condition.key] = summary or {}
            ledger.append({"stage": "pilot", "key": condition.key, "spent": spent})
            write_json(root / "budget.json", budget_payload(args, spent, ledger))
            if stopped:
                print("budget stopped during pilot", flush=True)
                return 0
        pilot_hands = args.pilot_hands * len(pilot_rows)
        pilot_cost = sum(
            float(summary.get("est_cost_cny", 0.0)) for summary in pilot_rows.values()
        )
        projected = (
            pilot_cost / pilot_hands * args.full_hands * len(pilot_rows)
            if pilot_hands
            else 0.0
        )
        print(
            f"\npilot: {pilot_hands} hands, cost Y{pilot_cost:.4f}, "
            f"projected full cost Y{projected:.2f}",
            flush=True,
        )
        remaining = args.budget_cny - spent
        if projected > remaining and projected > 0:
            scaled = int(remaining * 0.9 / (pilot_cost / pilot_hands))
            scaled = max(50, scaled - scaled % 2)
            print(
                f"warning: projected full run exceeds budget; "
                f"reducing full hands from {args.full_hands} to {scaled}",
                flush=True,
            )
            args.full_hands = scaled

    if args.stage == "pilot":
        write_json(root / "budget.json", budget_payload(args, spent, ledger))
        return 0

    a_summaries: Dict[str, dict] = {}
    if run_a:
        for condition in EXPERIMENT_A:
            summary, spent, stopped = ensure_condition(
                args, condition, args.full_hands, root / "full" / "a_bots", spent
            )
            a_summaries[condition.key] = summary or {}
            ledger.append({"stage": "full_a", "key": condition.key, "spent": spent})
            write_json(root / "budget.json", budget_payload(args, spent, ledger))
            if stopped or spent >= args.budget_cny:
                budget_exhausted = True
                print("budget stopped during experiment A", flush=True)
                break

    b_root = root / "full" / "b_tools"
    b_summaries: Dict[str, dict] = {}
    if run_b and not budget_exhausted:
        for condition in EXPERIMENT_B:
            summary, spent, stopped = ensure_condition(
                args, condition, args.full_hands, b_root, spent
            )
            b_summaries[condition.key] = summary or {}
            ledger.append({"stage": "full_b", "key": condition.key, "spent": spent})
            write_json(root / "budget.json", budget_payload(args, spent, ledger))
            if stopped or spent >= args.budget_cny:
                print("budget stopped during experiment B", flush=True)
                break

    if run_b and all(
        summary_clean(b_summaries.get(c.key), args.full_hands) for c in EXPERIMENT_B
    ):
        report_cmd = [
            sys.executable,
            "-u",
            str(MATRIX_SCRIPT),
            "--report-only",
            "--provider",
            args.provider,
            "--model",
            args.model,
            *(
                ["--provider-config", str(args.provider_config)]
                if args.provider_config is not None
                else []
            ),
            "--llm-name",
            args.llm_name,
            "--opponent-kind",
            "random",
            "--hands",
            str(args.full_hands),
            "--workers",
            str(args.workers),
            "--chips",
            str(args.chips),
            "--sb",
            str(args.sb),
            "--bb",
            str(args.bb),
            "--seed",
            str(args.seed),
            "--agent-locale",
            args.agent_locale,
            "--output",
            str(b_root),
        ]
        print("\nbuilding combined experiment B report", flush=True)
        subprocess.call(report_cmd, cwd=str(PROJECT_ROOT))

    write_top_report(
        root / "combined_report.md",
        args,
        a_summaries,
        b_summaries,
        spent,
        args.full_hands,
    )
    write_json(root / "budget.json", budget_payload(args, spent, ledger))
    print(
        f"\nfinished: spent Y{spent:.4f} / Y{args.budget_cny:.2f}; "
        f"report={root / 'combined_report.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
