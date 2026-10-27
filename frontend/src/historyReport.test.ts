import { describe, expect, it } from "vitest";

import {
  buildHistoryGroups,
  completedHistory,
  formatHistoryPercent,
  historyBarWidth,
  historyConfidenceClass,
  historyGroupLabel,
  historyPositionLabel,
  historyScopeLabel,
  recentHistoryHands,
} from "./historyReport";

const CURRENT = {
  hands: 12,
  blinds: { small: 5, big: 10 },
  stacks: { you: 1200, opponent: 800 },
  you: {
    preflop: {
      vpip: { value: 0.5, n: 12, confidence: "medium" },
      pfr: { value: 0.25, n: 12, confidence: "medium" },
    },
    showdown: {
      wtsd: { value: 0.5, n: 6, confidence: "low" },
    },
  },
  opponent: {
    preflop: {
      vpip: { value: 0.75, n: 12, confidence: "medium" },
      pfr: { value: 0.0833, n: 12, confidence: "medium" },
    },
    showdown: {
      wtsd: { value: 0.3333, n: 6, confidence: "low" },
    },
  },
  recent_hands: [
    { hand_id: 11, you_won: true, your_net: 120 },
    { hand_id: 12, you_won: false, your_net: -60 },
  ],
};

describe("historyReport", () => {
  it("builds only the groups present in the report", () => {
    const groups = buildHistoryGroups(CURRENT);
    expect(groups.map((group) => group.key)).toEqual(["preflop", "showdown"]);
    expect(groups[0].label).toBe("翻前");
    expect(groups[0].metrics.map((metric) => metric.label)).toEqual(["VPIP", "PFR"]);
    expect(groups[1].label).toBe("摊牌");
    expect(groups[1].metrics.map((metric) => metric.label)).toEqual(["WTSD"]);
  });

  it("ignores removed result and table-talk blocks", () => {
    const groups = buildHistoryGroups({
      ...CURRENT,
      you: {
        ...CURRENT.you,
        result: { net: 200, bb_per_100: 33.33, avg_pot: 90.5, all_ins: 2 },
        table_talk: { messages: 11, per_hand: 3.67 },
      },
      opponent: {
        ...CURRENT.opponent,
        result: { net: -200, bb_per_100: -33.33, avg_pot: 90.5, all_ins: 2 },
        table_talk: { messages: 13, per_hand: 4.33 },
      },
    });
    expect(groups.map((group) => group.key)).toEqual(["preflop", "showdown"]);
  });

  it("keeps a single focus group", () => {
    const groups = buildHistoryGroups({
      you: { showdown: { wtsd: { value: 0.4, n: 10, confidence: "medium" } } },
      opponent: { showdown: { wtsd: { value: 0.2, n: 10, confidence: "medium" } } },
    });
    expect(groups.map((group) => group.key)).toEqual(["showdown"]);
    expect(formatHistoryPercent(groups[0].metrics[0].opponent)).toBe("20.0%");
  });

  it("drops metrics neither side has an opportunity for", () => {
    const groups = buildHistoryGroups({
      you: {
        preflop: {
          vpip: { value: 1, n: 3, confidence: "low" },
          three_bet: { value: null, n: 0, confidence: "none" },
        },
        showdown: {
          wtsd: { value: null, n: 0, confidence: "none" },
          w_sd: { value: null, n: 0, confidence: "none" },
        },
      },
      opponent: {
        preflop: {
          vpip: { value: 0, n: 3, confidence: "low" },
          three_bet: { value: null, n: 0, confidence: "none" },
        },
      },
    });
    expect(groups.map((group) => group.key)).toEqual(["preflop"]);
    expect(groups[0].metrics.map((metric) => metric.key)).toEqual(["vpip"]);
  });

  it("formats percentages and empty cells", () => {
    expect(formatHistoryPercent({ value: 0.6667, n: 3, confidence: "low" })).toBe("66.7%");
    expect(formatHistoryPercent({ value: 0, n: 3, confidence: "low" })).toBe("0.0%");
    expect(formatHistoryPercent({ value: null, n: 0, confidence: "none" })).toBe("—");
  });

  it("maps bar widths and confidence labels", () => {
    expect(historyBarWidth({ value: 0.6667, n: 3, confidence: "low" })).toBe("66.67%");
    expect(historyBarWidth({ value: 0, n: 3, confidence: "low" })).toBe("0%");
    expect(historyBarWidth({ value: 1.4, n: 3, confidence: "low" })).toBe("100%");
    expect(historyBarWidth({ value: null, n: 0, confidence: "none" })).toBe("0%");
    expect(historyConfidenceClass({ value: 0.5, n: 3, confidence: "low" })).toBe("low");
    expect(historyConfidenceClass({ value: 0.5, n: 20, confidence: "medium" })).toBe("medium");
    expect(historyConfidenceClass({ value: 0.5, n: 40, confidence: "high" })).toBe("high");
    expect(historyConfidenceClass({ value: 0.5, n: null, confidence: "none" })).toBe("none");
  });

  it("labels scopes and groups for the focus chip", () => {
    expect(historyScopeLabel("current_match")).toBe("当前对局");
    expect(historyScopeLabel("completed_matches")).toBe("已完成对局");
    expect(historyScopeLabel("both")).toBe("当前 + 历史");
    expect(historyScopeLabel("unknown")).toBe("当前 + 历史");
    expect(historyGroupLabel("postflop")).toBe("翻后");
    expect(historyGroupLabel("unknown")).toBe("unknown");
    expect(historyPositionLabel("all")).toBe("全部");
    expect(historyPositionLabel("button")).toBe("按钮");
    expect(historyPositionLabel("bb")).toBe("大盲");
  });

  it("summarizes recent hands and completed matches", () => {
    expect(recentHistoryHands(CURRENT)).toEqual([
      { handId: 11, won: true, net: 120 },
      { handId: 12, won: false, net: -60 },
    ]);

    const completed = completedHistory({
      completed_matches: {
        count: 3,
        vs_current_opponent: { matches: 2, hands: 30 },
        overall: { matches: 3, hands: 50 },
      },
    });
    expect(completed).toEqual({ vsCurrent: true, matches: 2, hands: 30 });

    expect(
      completedHistory({
        completed_matches: {
          count: 0,
          vs_current_opponent: { matches: 0, hands: 0 },
          overall: { matches: 0, hands: 0 },
        },
      }),
    ).toBeNull();
  });
});
