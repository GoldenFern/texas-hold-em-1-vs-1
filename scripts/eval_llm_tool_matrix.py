"""Run a five-group LLM tool ablation against the project's strongest bot.

Groups differ only in the auxiliary tools exposed to the LLM and the matching
strong-but-optional recommendation in the system prompt.  Every group uses the
same seeds, the same number of hands, the same bot checkpoint, and the same
game rules.  Hands are independent 100 BB heads-up games with fresh chips, and
each deck is normally replayed with the LLM on both seats.

Examples
--------
Dry run::

    python scripts/eval_llm_tool_matrix.py --dry-run

Smoke test (two hands per group, two workers)::

    python scripts/eval_llm_tool_matrix.py --hands 2 --workers 2 \
        --output tmp/llm_tool_matrix/smoke

Full experiment::

    python scripts/eval_llm_tool_matrix.py --hands 300 --workers 10

Rebuild the combined report from existing group outputs::

    python scripts/eval_llm_tool_matrix.py --report-only \
        --output tmp/llm_tool_matrix/<timestamp>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.llm.providers import ProviderRegistry

EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_llm_vs_bot.py"
DEFAULT_BOT_MODEL = PROJECT_ROOT / "models" / "rlcard" / "nlh_dqn_project.pth"
DEFAULT_MODEL_NAME = "LongCat 2.5 Preview"

OPPONENT_KINDS = (
    "project",
    "always_all_in",
    "always_raise",
    "call_check",
    "random",
)
OPPONENT_LABELS = {
    "project": "Project DQN",
    "always_all_in": "All-in Bot",
    "always_raise": "Min-Raise Bot",
    "call_check": "Call/Check Bot",
    "random": "Random Bot",
}

GROUP_ORDER = ("none", "random", "distribution", "equity", "ev")


@dataclass(frozen=True)
class GroupSpec:
    """One tool-ablation group."""

    key: str
    label: str
    tools: tuple[str, ...]
    note: str


GROUPS: Dict[str, GroupSpec] = {
    "none": GroupSpec(
        key="none",
        label="无工具",
        tools=("apply_action",),
        note="只提供动作提交，没有任何辅助分析工具。",
    ),
    "random": GroupSpec(
        key="random",
        label="随机数",
        tools=("get_random_number", "apply_action"),
        note="只提供随机数工具，用于混合策略或打破僵局。",
    ),
    "distribution": GroupSpec(
        key="distribution",
        label="牌型分布",
        tools=("estimate_hand_probabilities", "apply_action"),
        note="只提供最终牌型分布估算，不直接提供胜率。",
    ),
    "equity": GroupSpec(
        key="equity",
        label="胜率",
        tools=("estimate_equity", "apply_action"),
        note="只提供摊牌 equity 估算。",
    ),
    "ev": GroupSpec(
        key="ev",
        label="跟注 EV",
        tools=("call_ev", "apply_action"),
        note="只提供静态跟注 EV；省略 equity 时会自动估算胜率。",
    ),
}


class Reporter:
    """Write progress to both stdout and the run log."""

    def __init__(self, path: Path, mode: str = "w") -> None:
        self.path = path
        self.handle = path.open(mode, encoding="utf-8")

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def raw(self, text: str) -> None:
        print(text, end="", flush=True)
        self.handle.write(text)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a five-group LLM tool ablation against the project bot."
    )
    parser.add_argument("--hands", type=int, default=300, help="Total hands per group.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent hands.")
    parser.add_argument("--provider", default="longcat")
    parser.add_argument("--model", default="LongCat-2.5-Preview")
    parser.add_argument(
        "--provider-config",
        type=Path,
        default=None,
        help="Optional alternate providers.json for experiment-only relays.",
    )
    parser.add_argument("--llm-name", default=None)
    parser.add_argument("--bot-model", type=Path, default=DEFAULT_BOT_MODEL)
    parser.add_argument("--bot-algorithm", default="dqn", choices=("dqn", "nfsp"))
    parser.add_argument(
        "--opponent-kind",
        default="project",
        choices=OPPONENT_KINDS,
        help="Opponent used for every group.",
    )
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--agent-locale",
        default="zh-CN",
        choices=("zh-CN", "en-US"),
        help="Language for LLM prompts, tool schemas and tool results.",
    )
    parser.add_argument("--groups", default="all", help="'all' or comma-separated group keys.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-api-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)

    args = parser.parse_args(argv)
    if args.hands <= 0:
        parser.error("--hands must be positive")
    if args.hands % 2:
        parser.error("--hands must be even so every deal can be mirrored")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.chips <= 0:
        parser.error("--chips must be positive")
    if args.sb <= 0 or args.bb < args.sb:
        parser.error("blinds must satisfy 0 < sb <= bb")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.timeout <= 0 or args.connect_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.report_only and args.output is None:
        parser.error("--report-only requires --output")
    if args.output is not None and args.report_only and not args.output.exists():
        parser.error(f"--output does not exist: {args.output}")
    args.workers = min(args.workers, args.hands)
    args.group_specs = select_groups(args.groups, parser)
    return args


def select_groups(
    raw: str, parser: argparse.ArgumentParser
) -> tuple[GroupSpec, ...]:
    text = str(raw or "").strip().lower()
    if text in ("", "all"):
        return tuple(GROUPS[key] for key in GROUP_ORDER)
    keys = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [key for key in keys if key not in GROUPS]
    if unknown:
        parser.error(
            f"unknown groups: {', '.join(unknown)}; choose from {', '.join(GROUP_ORDER)}"
        )
    return tuple(GROUPS[key] for key in GROUP_ORDER if key in set(keys))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:
        return "", False


def resolve_bot_model(raw: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def build_eval_command(
    args: argparse.Namespace,
    group: GroupSpec,
    output_dir: Path,
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
        "--group",
        group.key,
        "--tools",
        ",".join(group.tools),
        "--recommend-tools",
        "--hands",
        str(args.hands),
        "--workers",
        str(args.workers),
        "--bot-model",
        str(args.bot_model),
        "--bot-algorithm",
        args.bot_algorithm,
        "--opponent-kind",
        args.opponent_kind,
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
        str(output_dir),
        "--retries",
        str(args.retries),
        "--timeout",
        str(args.timeout),
        "--connect-timeout",
        str(args.connect_timeout),
    ]
    if args.retry_api_errors:
        command.append("--retry-api-errors")
    if args.llm_name:
        command.extend(["--llm-name", args.llm_name])
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    if args.top_p is not None:
        command.extend(["--top-p", str(args.top_p)])
    if args.max_tokens is not None:
        command.extend(["--max-tokens", str(args.max_tokens)])
    if args.reasoning_effort:
        command.extend(["--reasoning-effort", args.reasoning_effort])
    if args.thinking is True:
        command.append("--thinking")
    elif args.thinking is False:
        command.append("--no-thinking")
    return command


def summary_complete(summary_path: Path, args: argparse.Namespace, group: GroupSpec) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        int(summary.get("hands", 0)) >= int(args.hands)
        and int(summary.get("fatal_hands", 0)) == 0
        and int(summary.get("hands_with_api_error", 0)) == 0
        and int(summary.get("api_error_auto", 0)) == 0
        and int(summary.get("seed", 0)) == int(args.seed)
        and int(summary.get("big_blind", 0)) == int(args.bb)
        and str(summary.get("agent_locale") or "zh-CN") == str(args.agent_locale)
        and str(summary.get("opponent_kind") or "project") == str(args.opponent_kind)
        and list(summary.get("tools") or []) == list(group.tools)
        and bool(summary.get("recommend_tools"))
    )


def run_process(
    command: Sequence[str],
    log_path: Path,
    reporter: Reporter,
) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    display = subprocess.list2cmdline(list(command))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + display + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                reporter.raw(line)
                handle.write(line)
                handle.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def load_summary(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_hands(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not row.get("fatal_error"):
                rows.append(row)
    return rows


def hand_nets(rows: Sequence[dict]) -> Dict[tuple[int, int], int]:
    """Map ``(deal_index, assignment)`` to the LLM's net chips."""
    result: Dict[tuple[int, int], int] = {}
    for row in rows:
        try:
            key = (int(row["deal_index"]), int(row["assignment"]))
            result[key] = int(row["net_llm"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def compare_group(
    group_rows: Sequence[dict],
    baseline_rows: Sequence[dict],
    big_blind: int,
) -> dict:
    group_nets = hand_nets(group_rows)
    baseline_nets = hand_nets(baseline_rows)
    keys = sorted(set(group_nets).intersection(baseline_nets))
    differences = [group_nets[key] - baseline_nets[key] for key in keys]
    n = len(differences)
    if not n:
        return {
            "pairs": 0,
            "delta_net_per_hand": 0.0,
            "delta_bb100": 0.0,
            "ci95_bb100": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    mean = statistics.fmean(differences)
    sd = statistics.pstdev(differences) if n > 1 else 0.0
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "pairs": n,
        "delta_net_per_hand": round(mean, 3),
        "delta_bb100": round(mean / big_blind * 100.0, 3),
        "ci95_net": round(ci95, 3),
        "ci95_bb100": round(ci95 / big_blind * 100.0, 3),
        "wins": sum(1 for value in differences if value > 0),
        "losses": sum(1 for value in differences if value < 0),
        "ties": sum(1 for value in differences if value == 0),
    }


def build_combined(
    args: argparse.Namespace,
    summaries: Dict[str, dict],
    rows_by_group: Dict[str, List[dict]],
) -> dict:
    comparisons: Dict[str, dict] = {}
    baseline_rows = rows_by_group.get("none") or []
    if baseline_rows:
        for group in args.group_specs:
            if group.key == "none":
                continue
            rows = rows_by_group.get(group.key) or []
            if rows:
                comparisons[group.key] = compare_group(rows, baseline_rows, args.bb)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "provider_config": (
            str(args.provider_config) if args.provider_config is not None else ""
        ),
        "hands_per_group": args.hands,
        "workers": args.workers,
        "seed": args.seed,
        "agent_locale": args.agent_locale,
        "chips": args.chips,
        "small_blind": args.sb,
        "big_blind": args.bb,
        "opponent_kind": args.opponent_kind,
        "opponent_name": OPPONENT_LABELS.get(args.opponent_kind, args.opponent_kind),
        "bot_model": str(args.bot_model) if args.opponent_kind == "project" else "",
        "bot_algorithm": (
            args.bot_algorithm if args.opponent_kind == "project" else "scripted"
        ),
        "groups": {
            group.key: {
                "label": group.label,
                "tools": list(group.tools),
                "note": group.note,
                "summary": summaries.get(group.key),
            }
            for group in args.group_specs
        },
        "comparisons_vs_none": comparisons,
    }


def fmt(value: Any, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if signed:
        return f"{number:+.{digits}f}"
    return f"{number:.{digits}f}"


def write_combined_report(
    path: Path,
    combined: dict,
    args: argparse.Namespace,
    summaries: Dict[str, dict],
) -> None:
    opponent_name = OPPONENT_LABELS.get(args.opponent_kind, args.opponent_kind)
    if args.opponent_kind == "project":
        opponent_line = f"- Bot: `{args.bot_algorithm}` / `{args.bot_model}`"
    else:
        opponent_line = f"- Bot: `{args.opponent_kind}`（内置脚本策略）"
    lines = [
        f"# {args.model} 工具对照实验：{len(args.group_specs)} 组 vs {opponent_name}",
        "",
        f"- 生成时间: {combined['generated_at']}",
        f"- 模型: `{args.provider}/{args.model}`",
        f"- Agent 语言: `{args.agent_locale}`",
        opponent_line,
        f"- 每组手数: {args.hands}（{args.hands // 2} 副牌 × 2 个座位）",
        f"- 并发: {args.workers}；{len(args.group_specs)} 组顺序执行",
        f"- 规则: {args.chips} 筹码，盲注 {args.sb}/{args.bb}，每手重置筹码",
        f"- 随机种子: {args.seed}；各组使用同一批牌",
        "- 工具推荐: 有辅助工具的组在提示词里强烈推荐使用本组工具，但运行时不强制，也不会因未调用而报错。",
        "",
        "## 总表",
        "",
        "| 组 | 工具 | 有效手数 | 失败 | 总净筹码 | 每手均值 | bb/100 | 同牌换座 95% CI | 胜率 | 摊牌率 | 全下率 | 工具/回合 | 工具回合 | API 调用 | 耗时 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in args.group_specs:
        summary = summaries.get(group.key) or {}
        lines.append(
            "| {label} | {tools} | {hands} | {fatal} | {net} | {mean} | {bb100} | ±{ci} | {win}% | {showdown}% | {allin}% | {per_turn} | {tool_turns}% | {api} | {elapsed} |".format(
                label=group.label,
                tools=", ".join(group.tools),
                hands=summary.get("hands", 0),
                fatal=summary.get("fatal_hands", 0),
                net=fmt(summary.get("total_net"), 0, signed=True),
                mean=fmt(summary.get("mean_net"), 2, signed=True),
                bb100=fmt(summary.get("mean_bb100"), 1, signed=True),
                ci=fmt(summary.get("paired_ci95_bb100"), 1),
                win=fmt(summary.get("win_pct"), 1),
                showdown=fmt(summary.get("showdown_pct"), 1),
                allin=fmt(summary.get("allin_hand_pct"), 1),
                per_turn=fmt(summary.get("aux_tool_calls_per_turn"), 3),
                tool_turns=fmt(summary.get("tool_turns_pct"), 1),
                api=summary.get("api_calls", 0),
                elapsed=f"{float(summary.get('elapsed_sec', 0.0)) / 3600.0:.2f}h",
            )
        )

    lines.extend(
        [
            "",
            "## 资源消耗",
            "",
            "| 组 | API 调用 | 输入 token | 输出 token | 推理 token | 估算费用 | 耗时 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in args.group_specs:
        summary = summaries.get(group.key) or {}
        lines.append(
            "| {label} | {api} | {input} | {output} | {reasoning} | ¥{cost:.4f} | {elapsed} |".format(
                label=group.label,
                api=summary.get("api_calls", 0),
                input=summary.get("input_tokens", 0),
                output=summary.get("output_tokens", 0),
                reasoning=summary.get("reasoning_tokens", 0),
                cost=float(summary.get("est_cost_cny", 0.0)),
                elapsed=f"{float(summary.get('elapsed_sec', 0.0)) / 3600.0:.2f}h",
            )
        )

    lines.extend(
        [
            "",
            "## 相对无工具组（同牌同座位配对）",
            "",
            "| 组 | 配对手数 | Δ 每手筹码 | Δ bb/100 | 95% CI | 赢/输/平 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in args.group_specs:
        if group.key == "none":
            continue
        comp = combined.get("comparisons_vs_none", {}).get(group.key) or {}
        lines.append(
            "| {label} | {pairs} | {delta} | {bb100} | ±{ci} | {wins}/{losses}/{ties} |".format(
                label=group.label,
                pairs=comp.get("pairs", 0),
                delta=fmt(comp.get("delta_net_per_hand"), 2, signed=True),
                bb100=fmt(comp.get("delta_bb100"), 1, signed=True),
                ci=fmt(comp.get("ci95_bb100"), 1),
                wins=comp.get("wins", 0),
                losses=comp.get("losses", 0),
                ties=comp.get("ties", 0),
            )
        )

    lines.extend(
        [
            "",
            "## 工具使用",
            "",
            "| 组 | 辅助调用 | 每回合 | 有工具回合 | 工具错误 | 牌型分布 | 胜率 | 随机数 | 跟注 EV | EV 不适用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in args.group_specs:
        summary = summaries.get(group.key) or {}
        lines.append(
            "| {label} | {calls} | {per_turn} | {tool_turns}% | {errors} | {dist} | {equity} | {random} | {ev} | {ev_na} |".format(
                label=group.label,
                calls=summary.get("aux_tool_calls", 0),
                per_turn=fmt(summary.get("aux_tool_calls_per_turn"), 3),
                tool_turns=fmt(summary.get("tool_turns_pct"), 1),
                errors=summary.get("aux_tool_errors", 0),
                dist=summary.get("distribution_calls", 0),
                equity=summary.get("equity_calls", 0),
                random=summary.get("random_calls", 0),
                ev=summary.get("call_ev_calls", 0),
                ev_na=summary.get("call_ev_not_applicable", 0),
            )
        )

    lines.extend(
        [
            "",
            "## 数据质量",
            "",
            "| 组 | 失败手数 | API 错误手数 | 自动行动回合 | 非法动作自动 | API 错误自动 | 重试手数 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in args.group_specs:
        summary = summaries.get(group.key) or {}
        lines.append(
            "| {label} | {fatal} | {api_hands} | {auto} | {invalid} | {api_auto} | {retried} |".format(
                label=group.label,
                fatal=summary.get("fatal_hands", 0),
                api_hands=summary.get("hands_with_api_error", 0),
                auto=summary.get("auto_turns", 0),
                invalid=summary.get("invalid_action_auto", 0),
                api_auto=summary.get("api_error_auto", 0),
                retried=summary.get("hands_with_retry", 0),
            )
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 每手都是独立牌局，筹码不跨手累积；各组使用同一批牌和同一 Bot。",
            "- 总表的 95% CI 使用同牌换座配对；逐手 CI 和行动分布在各组 `report.md`。",
            "- `call_ev` 在省略 `equity` 时会自动估算胜率，因此 EV 组也间接获得了胜率信息。",
            "- `get_random_number` 本身不提供牌局信息，主要用于混合策略或打破僵局。",
            f"- {args.hands} 手/组的 95% 置信区间仍然较宽，组间差异需要结合区间和工具行为一起看，不能只按总净筹码排名。",
            "- `hands.csv` / `hands.jsonl` 保留每手动作与压缩事件，便于复核具体工具调用。",
            "- 如果某一组失败手数不为零，总表会单独标出；不要把它和其他组直接比较。",
        ]
    )
    for group in args.group_specs:
        summary = summaries.get(group.key) or {}
        api_error_hands = int(summary.get("hands_with_api_error", 0))
        api_error_auto = int(summary.get("api_error_auto", 0))
        if api_error_hands or api_error_auto:
            lines.append(
                f"- 警告：{group.label} 有 {api_error_hands} 手出现 API 错误，"
                f"{api_error_auto} 个回合由错误降级为自动行动；这组结果可能被污染，"
                "不应直接当作模型真实策略。"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.bot_model = resolve_bot_model(args.bot_model)
    if args.provider_config is not None:
        provider_config = Path(args.provider_config)
        if not provider_config.is_absolute():
            provider_config = (PROJECT_ROOT / provider_config).resolve()
        args.provider_config = provider_config

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or (PROJECT_ROOT / "tmp" / "llm_tool_matrix" / timestamp)
    output = output.resolve()
    if args.report_only:
        if not output.is_dir():
            print(f"error: output directory not found: {output}", file=sys.stderr)
            return 2
    else:
        output.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if (args.resume or args.report_only) else "w"
    reporter = Reporter(output / "run.log", mode=log_mode)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    problem_groups: List[str] = []

    try:
        reporter.line(f"output: {output}")
        reporter.line(
            "groups: "
            + ", ".join(f"{group.key}({group.label})" for group in args.group_specs)
        )
        reporter.line(
            f"table: {args.hands} hands/group, {args.workers} workers, "
            f"{args.chips} chips, blinds {args.sb}/{args.bb}, seed {args.seed}, "
            f"agent_locale={args.agent_locale}"
        )

        if not args.report_only:
            registry = (
                ProviderRegistry(args.provider_config)
                if args.provider_config is not None
                else ProviderRegistry()
            )
            spec = registry.resolve(args.provider, args.model)
            if not args.dry_run and not ProviderRegistry.key_present(spec):
                reporter.line(
                    f"error: missing environment variable {spec.api_key_env} in .env"
                )
                return 2
            if args.opponent_kind == "project" and not args.bot_model.is_file():
                reporter.line(f"error: bot model not found: {args.bot_model}")
                return 2
            if not args.llm_name:
                args.llm_name = spec.name
            commands = {
                group.key: build_eval_command(args, group, output / group.key)
                for group in args.group_specs
            }
            commit, dirty = git_state()
            manifest = {
                "created_at": started_at.isoformat(),
                "provider": args.provider,
                "model": args.model,
                "provider_config": (
                    str(args.provider_config)
                    if args.provider_config is not None
                    else ""
                ),
                "model_name": args.llm_name,
                "opponent_kind": args.opponent_kind,
                "opponent_name": OPPONENT_LABELS.get(
                    args.opponent_kind, args.opponent_kind
                ),
                "bot_model": (
                    str(args.bot_model) if args.opponent_kind == "project" else ""
                ),
                "bot_sha256": (
                    sha256_file(args.bot_model)
                    if args.opponent_kind == "project"
                    else ""
                ),
                "bot_algorithm": (
                    args.bot_algorithm
                    if args.opponent_kind == "project"
                    else "scripted"
                ),
                "hands_per_group": args.hands,
                "workers": args.workers,
                "seed": args.seed,
                "chips": args.chips,
                "small_blind": args.sb,
                "big_blind": args.bb,
                "git_commit": commit,
                "git_dirty": dirty,
                "groups": {
                    group.key: {
                        "label": group.label,
                        "tools": list(group.tools),
                        "note": group.note,
                        "output": str(output / group.key),
                    }
                    for group in args.group_specs
                },
                "commands": commands,
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reporter.line(
                f"model: {args.llm_name} ({spec.provider_name}/{spec.id})"
            )
            if args.opponent_kind == "project":
                reporter.line(
                    f"bot: {args.bot_model.name} sha256={manifest['bot_sha256'][:12]}"
                )
            else:
                reporter.line(
                    f"opponent: {args.opponent_kind} "
                    f"({OPPONENT_LABELS.get(args.opponent_kind, args.opponent_kind)})"
                )

            if args.dry_run:
                for group in args.group_specs:
                    reporter.line("")
                    reporter.line(f"[{group.key}] {group.note}")
                    reporter.line(
                        "$ " + subprocess.list2cmdline(commands[group.key])
                    )
                reporter.line("")
                reporter.line("dry run only; no hands were played")
                return 0

            for index, group in enumerate(args.group_specs, start=1):
                group_dir = output / group.key
                group_dir.mkdir(parents=True, exist_ok=True)
                summary_path = group_dir / "summary.json"
                if args.resume and summary_complete(summary_path, args, group):
                    reporter.line(
                        f"\n[{index}/{len(args.group_specs)}] skip {group.key}: "
                        "completed summary found"
                    )
                    continue
                reporter.line(
                    f"\n[{index}/{len(args.group_specs)}] {group.key} / {group.label}: "
                    f"tools={','.join(group.tools)}"
                )
                reporter.line(
                    f"$ {subprocess.list2cmdline(commands[group.key])}"
                )
                exit_code = run_process(commands[group.key], group_dir / "run.log", reporter)
                reporter.line(f"[{group.key}] exit code {exit_code}")
                if exit_code not in (0, 1):
                    problem_groups.append(group.key)
                if not summary_path.is_file():
                    problem_groups.append(group.key)
                    reporter.line(
                        f"[{group.key}] warning: summary.json is missing after the run"
                    )

        summaries: Dict[str, dict] = {}
        rows_by_group: Dict[str, List[dict]] = {}
        for group in args.group_specs:
            group_dir = output / group.key
            summary = load_summary(group_dir / "summary.json")
            if summary is None:
                reporter.line(f"[{group.key}] missing summary; skipped in the report")
                if group.key not in problem_groups:
                    problem_groups.append(group.key)
                continue
            summaries[group.key] = summary
            rows_by_group[group.key] = load_hands(group_dir / "hands.jsonl")
            if (
                int(summary.get("fatal_hands", 0)) > 0
                or int(summary.get("hands_with_api_error", 0)) > 0
                or int(summary.get("api_error_auto", 0)) > 0
            ) and group.key not in problem_groups:
                problem_groups.append(group.key)
            has_auxiliary = any(name != "apply_action" for name in group.tools)
            if has_auxiliary and int(summary.get("aux_tool_calls", 0)) == 0:
                reporter.line(
                    f"[{group.key}] warning: the model never called its auxiliary tool"
                )

        if not summaries:
            reporter.line("error: no group summaries found; nothing to aggregate")
            return 2

        combined = build_combined(args, summaries, rows_by_group)
        elapsed = time.perf_counter() - started
        combined["elapsed_sec"] = round(elapsed, 3)
        (output / "combined_summary.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_combined_report(output / "combined_report.md", combined, args, summaries)

        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                manifest = {}
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest["elapsed_sec"] = round(elapsed, 3)
            manifest["available_groups"] = sorted(summaries)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        reporter.line("")
        reporter.line(f"combined report: {output / 'combined_report.md'}")
        for group in args.group_specs:
            summary = summaries.get(group.key)
            if not summary:
                continue
            reporter.line(
                f"{group.label:6s} {summary.get('total_net', 0):+6d} chips, "
                f"{summary.get('mean_bb100', 0.0):+8.1f} bb/100, "
                f"win {summary.get('win_pct', 0.0):5.1f}%, "
                f"tools {summary.get('aux_tool_calls', 0):4d}, "
                f"elapsed {float(summary.get('elapsed_sec', 0.0)) / 3600.0:.2f}h"
            )
        reporter.line(f"total elapsed: {elapsed / 3600.0:.2f}h")
        if problem_groups:
            reporter.line(
                "groups with missing summaries or failed hands: "
                + ", ".join(sorted(set(problem_groups)))
            )
            return 1
        return 0
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
