import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";

const storeMock = vi.hoisted(() => {
  const card = (code: string) => {
    const suit = code[1];
    const symbols: Record<string, string> = { c: "♣", d: "♦", h: "♥", s: "♠" };
    return {
      code,
      rank: code[0],
      suit,
      symbol: symbols[suit] || suit,
      red: suit === "h" || suit === "d",
      label: code,
    };
  };
  return {
    game: {
      snapshot: {
        has_match: true,
        hand_id: 1,
        street: "flop",
        community: [card("Ah"), card("Kd"), card("7c")],
        players: [
          { seat: 0, name: "Hero", hole_cards: [card("As"), card("Ad")], kind: "human" },
        ],
        match: {
          seats: [{ seat: 0, name: "Hero", kind: "human" }],
        },
      },
      selectedAnalysisSeat: 0,
      godView: false,
      analysisLoading: false,
      analysisError: "",
      analysis: {
        0: {
          hero_seat: 0,
          method: "monte_carlo",
          iterations: 30000,
          elapsed_ms: 12.5,
          total: 30000,
          hand_id: 1,
          street: "flop",
          known_cards: ["As", "Ad", "Ah", "Kd", "7c"],
          needed_board_cards: 2,
          current_best: { category: "three_of_a_kind", category_name: "三条" },
          distribution: [
            { key: "royal_flush", name: "皇家同花顺", probability: 0, count: 0, color: "#fff" },
            { key: "three_of_a_kind", name: "三条", probability: 0.46, count: 13800, color: "#54d6a8" },
            { key: "high_card", name: "高牌", probability: 0.2, count: 6000, color: "#94a49b" },
          ],
        },
      },
    },
  };
});

vi.mock("../store", async () => {
  const { ref } = await import("vue");
  return {
    game: storeMock.game,
    humanSeat: ref({ seat: 0, name: "Hero", kind: "human" }),
    refreshAnalysis: vi.fn(),
    selectAnalysisSeat: vi.fn(),
  };
});

vi.mock("../boardReveal", async () => {
  const { ref } = await import("vue");
  return { boardLocked: ref(false) };
});

import AnalysisPanel from "./AnalysisPanel.vue";

describe("AnalysisPanel collapsed summary", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("starts collapsed and shows the most likely category", async () => {
    const wrapper = mount(AnalysisPanel);
    await nextTick();

    expect(wrapper.find(".chart").exists()).toBe(false);
    const summary = wrapper.find(".analysis-toggle").text();
    expect(summary).toContain("最可能");
    expect(summary).toContain("三条");
    expect(summary).not.toContain("皇家同花顺");
    expect(summary).not.toContain("当前");
    expect(wrapper.find(".analysis-toggle .pcard").exists()).toBe(false);

    await wrapper.find(".analysis-toggle").trigger("click");
    await nextTick();

    expect(wrapper.find(".chart").exists()).toBe(true);
    expect(wrapper.findAll(".chart-row")).toHaveLength(3);
    expect(wrapper.find(".analysis-foot").exists()).toBe(false);
    expect(window.localStorage.getItem("hulab.analysis.expanded")).toBe("1");
    wrapper.unmount();
  });

  it("restores the expanded state from localStorage", async () => {
    window.localStorage.setItem("hulab.analysis.expanded", "1");
    const wrapper = mount(AnalysisPanel);
    await nextTick();

    expect(wrapper.find(".chart").exists()).toBe(true);
    wrapper.unmount();
  });
});
