"""Merge compatible LLM evaluation runs and rebuild paired summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_llm_tool_matrix import GROUP_ORDER, compare_group, hand_nets
from scripts.eval_llm_vs_bot import summarize, write_csv, write_jsonl

GROUP_LABELS = {
    "none": "空白",
    "random": "随机数",
    "distribution": "牌型分布",
    "equity": "胜率",
    "ev": "跟注 EV",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and re-summarize LLM runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--provider", default="dxh")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--provider-config", default="")
    parser.add_argument("--llm-name", default="GPT-6 Luna (DXH)")
    parser.add_argument("--price-input", type=float, default=0.0198)
    parser.add_argument("--price-cache-input", type=float, default=None)
    parser.add_argument("--price-output", type=float, default=0.099)
    parser.add_argument("--expected-hands", type=int, default=0)
    args = parser.parse_args(argv)
    if args.price_input < 0 or args.price_output < 0:
        parser.error("token prices cannot be negative")
    if args.price_cache_input is not None and args.price_cache_input < 0:
        parser.error("--price-cache-input cannot be negative")
    args.runs = [path.resolve() for path in args.runs]
    args.output = args.output.resolve()
    return args


def load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
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


def merged_rows(runs: Sequence[Path], group: str) -> List[dict]:
    by_key: Dict[tuple[int, int], dict] = {}
    for run in runs:
        for row in load_rows(run / group / "hands.jsonl"):
            try:
                key = (int(row["deal_index"]), int(row["assignment"]))
            except (KeyError, TypeError, ValueError):
                continue
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def paired_p_value(
    rows: Sequence[dict],
    baseline_rows: Sequence[dict],
) -> float:
    nets = hand_nets(rows)
    baseline = hand_nets(baseline_rows)
    keys = sorted(set(nets) & set(baseline))
    differences = [nets[key] - baseline[key] for key in keys]
    if len(differences) < 2:
        return 1.0
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences)
    if sd == 0:
        return 0.0 if mean else 1.0
    t = mean / (sd / math.sqrt(len(differences)))
    return math.erfc(abs(t) / math.sqrt(2.0))


def fmt(value: object, digits: int = 1, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_args = SimpleNamespace(bb=args.bb, agent_locale="zh-CN")

    rows_by_group: Dict[str, List[dict]] = {}
    summaries: Dict[str, dict] = {}
    for group in GROUP_ORDER:
        rows = merged_rows(args.runs, group)
        if not rows:
            continue
        rows_by_group[group] = rows
        summary = summarize(rows, summary_args)
        summary.update(
            {
                "group": group,
                "provider": args.provider,
                "model": args.model,
                "provider_config": args.provider_config,
                "llm_name": args.llm_name,
                "source_runs": [str(path) for path in args.runs],
                "price_input_per_m": args.price_input,
                "price_cache_input_per_m": (
                    args.price_input
                    if args.price_cache_input is None
                    else args.price_cache_input
                ),
                "price_output_per_m": args.price_output,
            }
        )
        summaries[group] = summary
        write_jsonl(args.output / group / "hands.jsonl", rows)
        write_csv(args.output / group / "hands.csv", rows)
        (args.output / group / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if args.expected_hands and int(summary.get("hands", 0)) < args.expected_hands:
            print(
                f"warning: {group} has {summary.get('hands', 0)} hands, "
                f"expected at least {args.expected_hands}"
            )

    baseline = rows_by_group.get("none") or []
    comparisons: Dict[str, dict] = {}
    if baseline:
        for group in GROUP_ORDER:
            if group == "none" or group not in rows_by_group:
                continue
            comp = compare_group(rows_by_group[group], baseline, args.bb)
            comp["p_value"] = round(
                paired_p_value(rows_by_group[group], baseline), 4
            )
            comparisons[group] = comp

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "provider_config": args.provider_config,
        "llm_name": args.llm_name,
        "source_runs": [str(path) for path in args.runs],
        "price_input_per_m": args.price_input,
        "price_cache_input_per_m": (
            args.price_input
            if args.price_cache_input is None
            else args.price_cache_input
        ),
        "price_output_per_m": args.price_output,
        "hands_per_group": {
            group: summaries[group].get("hands", 0)
            for group in summaries
        },
        "total_cost_cny": round(
            sum(float(item.get("est_cost_cny", 0.0)) for item in summaries.values()),
            6,
        ),
        "groups": summaries,
        "comparisons_vs_none": comparisons,
    }
    (args.output / "combined_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# DXH 工具对照汇总（合并多次运行）",
        "",
        f"- 生成时间: {combined['generated_at']}",
        f"- 模型: `{args.provider}/{args.model}`",
        f"- 合并运行: {len(args.runs)} 次",
        f"- 每组手数: {', '.join(f'{GROUP_LABELS.get(g, g)} {summaries[g].get('hands', 0)}' for g in summaries)}",
        f"- 累计估算费用: ¥{combined['total_cost_cny']:.4f}",
        f"- 单价: 输入 ¥{args.price_input:.4f}/M，缓存输入 ¥{args.price_input if args.price_cache_input is None else args.price_cache_input:.4f}/M，输出 ¥{args.price_output:.4f}/M",
        "",
        "## 总表",
        "",
        "| 组 | 手数 | bb/100 | 95% CI | 胜率 | 摊牌率 | 工具/回合 | 费用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        item = summaries.get(group)
        if not item:
            continue
        lines.append(
            "| {label} | {hands} | {bb100} | ±{ci} | {win}% | {showdown}% | {tool} | ¥{cost:.4f} |".format(
                label=GROUP_LABELS.get(group, group),
                hands=item.get("hands", 0),
                bb100=fmt(item.get("mean_bb100"), 1, signed=True),
                ci=fmt(item.get("paired_ci95_bb100"), 1),
                win=fmt(item.get("win_pct"), 1),
                showdown=fmt(item.get("showdown_pct"), 1),
                tool=fmt(item.get("aux_tool_calls_per_turn"), 3),
                cost=float(item.get("est_cost_cny", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## 相对空白组（同牌同座位配对）",
            "",
            "| 组 | 配对手数 | Δ bb/100 | 95% CI | p | 胜/负/平 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in GROUP_ORDER:
        if group == "none" or group not in comparisons:
            continue
        comp = comparisons[group]
        lines.append(
            "| {label} | {pairs} | {delta} | ±{ci} | {p} | {wins}/{losses}/{ties} |".format(
                label=GROUP_LABELS.get(group, group),
                pairs=comp.get("pairs", 0),
                delta=fmt(comp.get("delta_bb100"), 1, signed=True),
                ci=fmt(comp.get("ci95_bb100"), 1),
                p=fmt(comp.get("p_value"), 4),
                wins=comp.get("wins", 0),
                losses=comp.get("losses", 0),
                ties=comp.get("ties", 0),
            )
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 合并按 `(deal_index, assignment)` 去重，新增运行使用不同 deal offset。",
            "- 随机 Agent 在牌局分叉后动作会不同，配对差异仍带有噪声。",
            "- 1000 手一组的置信区间依旧很宽，不能把小差异当成稳定结论。",
            "- 费用按输入/输出 token 和本次配置的单价估算，不含缓存。",
        ]
    )
    (args.output / "combined_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"merged output: {args.output}")
    print(
        "hands: "
        + ", ".join(
            f"{group}={summaries[group].get('hands', 0)}" for group in summaries
        )
    )
    print(f"cost: Y{combined['total_cost_cny']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
