"""Plot the 1000-hand DXH tool-ablation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUPS = ("none", "random", "distribution", "equity", "ev")
LABELS = {
    "none": "Blank\n(no tools)",
    "random": "Random number\n(placebo)",
    "distribution": "Hand distribution",
    "equity": "Equity",
    "ev": "Call EV",
}
COLORS = {
    "none": "#6B7280",
    "random": "#94A3B8",
    "distribution": "#D99A45",
    "equity": "#2F9E8F",
    "ev": "#1F6F5C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT
        / "tmp"
        / "llm_experiments"
        / "20261024_dxh_compact1000"
        / "combined_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "images" / "dxh_tool_ablation_1000.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = data["groups"]
    comparisons = data["comparisons_vs_none"]
    hands = data["hands_per_group"]["none"]

    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.2),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    fig.patch.set_facecolor("#F7F5EF")

    # Left: absolute bb/100 with 95% CI.
    y = list(range(len(GROUPS)))[::-1]
    for index, group in zip(y, GROUPS):
        item = groups[group]
        value = float(item["mean_bb100"])
        ci = float(item["paired_ci95_bb100"])
        ax_left.barh(
            index,
            value,
            color=COLORS[group],
            height=0.62,
            edgecolor="white",
            linewidth=0.8,
        )
        ax_left.errorbar(
            value,
            index,
            xerr=ci,
            fmt="none",
            ecolor="#263238",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
        )
        ax_left.text(
            value + ci + 14,
            index,
            f"{value:+.1f}  ±{ci:.1f}",
            va="center",
            fontsize=9.5,
            color="#263238",
        )
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([LABELS[g] for g in GROUPS], fontsize=10)
    ax_left.axvline(0, color="#263238", linewidth=0.9)
    ax_left.set_xlim(-120, 900)
    ax_left.set_xlabel("bb/100 (higher is better; error bars: 95% CI)", fontsize=10)
    ax_left.set_title("Absolute performance", fontsize=12, fontweight="bold", loc="left")
    ax_left.grid(axis="x", color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax_left.set_axisbelow(True)
    ax_left.spines[["top", "right"]].set_visible(False)

    # Right: paired difference vs blank.
    pair_groups = ("random", "distribution", "equity", "ev")
    y2 = list(range(len(pair_groups)))[::-1]
    for index, group in zip(y2, pair_groups):
        comp = comparisons[group]
        value = float(comp["delta_bb100"])
        ci = float(comp["ci95_bb100"])
        p_value = float(comp["p_value"])
        significant = p_value < 0.05 and value > 0
        color = "#1F6F5C" if significant else "#A8B0B8"
        if value < 0:
            color = "#C26B5A"
        ax_right.barh(
            index,
            value,
            color=color,
            height=0.6,
            edgecolor="white",
            linewidth=0.8,
        )
        ax_right.errorbar(
            value,
            index,
            xerr=ci,
            fmt="none",
            ecolor="#263238",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
        )
        label = f"{value:+.1f}  ±{ci:.1f}\np={p_value:.4f}"
        ax_right.text(
            value + ci + 12,
            index,
            label,
            va="center",
            fontsize=9,
            color="#263238",
        )
    ax_right.set_yticks(y2)
    ax_right.set_yticklabels([LABELS[g] for g in pair_groups], fontsize=10)
    ax_right.axvline(0, color="#263238", linewidth=1.0)
    ax_right.set_xlim(-180, 900)
    ax_right.set_xlabel("Paired difference vs blank (bb/100, 95% CI)", fontsize=10)
    ax_right.set_title("Gain over blank", fontsize=12, fontweight="bold", loc="left")
    ax_right.grid(axis="x", color="#D8D3C7", linewidth=0.7, alpha=0.8)
    ax_right.set_axisbelow(True)
    ax_right.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "GPT-6 Luna vs Random Agent: Tool Ablation (1,000 Hands per Group)",
        fontsize=16,
        fontweight="bold",
        x=0.045,
        ha="left",
        y=0.965,
        va="top",
    )

    fig.subplots_adjust(left=0.095, right=0.975, top=0.855, bottom=0.14, wspace=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
