import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";

const storeMock = vi.hoisted(() => {
  function turn(id: string, handId: number) {
    return {
      id,
      header: {
        handId,
        handNumber: handId,
        street: "preflop",
        pot: 15,
        myChips: 995,
        opponentChips: 990,
        toCall: 5,
        legalActions: ["fold", "call", "raise"],
        position: "button",
      },
      messages: [],
      notes: [],
      stateText: "",
      action: null,
      auto: false,
      status: "done",
      usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: 0 },
      latencyMs: 10,
      steps: 0,
      contextTokens: 0,
      startedAt: 1,
      endedAt: 11,
    };
  }
  return {
    game: {
      selectedSeat: 0,
      pinned: false,
      consoleMode: "stream",
      matchId: "match-1",
      connected: true,
      actionFeed: [] as unknown[],
      tableTalk: [] as unknown[],
      lastResult: null,
      snapshot: {
        has_match: true,
        hand_id: 0,
        players: [
          { seat: 0, name: "Agent A", kind: "llm" },
          { seat: 1, name: "Agent B", kind: "llm" },
        ],
        match: {
          table_talk: true,
          seats: [
            { seat: 0, name: "Agent A", kind: "llm", model_name: "Model A", icon: "", agent: {} },
            { seat: 1, name: "Agent B", kind: "llm", model_name: "Model B", icon: "", agent: {} },
          ],
        },
      },
      turns: {
        0: [turn("t1", 1), turn("t2", 2)],
      } as Record<number, unknown[]>,
    },
  };
});

vi.mock("../store", async () => {
  const { ref } = await import("vue");
  return {
    game: storeMock.game,
    agentSeats: ref([
      { seat: 0, name: "Agent A", kind: "llm", model_name: "Model A", icon: "", agent: {} },
    ]),
    humanSeat: ref(null),
    followLive: vi.fn(),
    selectSeat: vi.fn(),
    setConsoleMode: (mode: string) => { storeMock.game.consoleMode = mode; },
    openSetup: vi.fn(),
  };
});

import AgentConsole from "./AgentConsole.vue";

describe("AgentConsole stream", () => {
  beforeEach(() => {
    storeMock.game.snapshot.hand_id = 0;
    storeMock.game.consoleMode = "stream";
    storeMock.game.tableTalk = [];
  });

  it("shows the table-talk tab and renders the conversation", async () => {
    storeMock.game.consoleMode = "talk";
    storeMock.game.tableTalk = [
      {
        id: "talk-1",
        seq: 1,
        hand_id: 1,
        street: "flop",
        seat: 0,
        name: "Agent A",
        text: "我这次不会再让你便宜看牌。",
      },
    ];
    const wrapper = mount(AgentConsole);
    await nextTick();

    expect(wrapper.find(".talk-tab").exists()).toBe(true);
    expect(wrapper.find(".talk-note").exists()).toBe(false);
    expect(wrapper.text()).toContain("我这次不会再让你便宜看牌。");
    wrapper.unmount();
  });

  it("renders a visual divider at the start of every hand", async () => {
    const wrapper = mount(AgentConsole);
    await nextTick();
    const dividers = wrapper.findAll(".hand-divider");
    expect(dividers).toHaveLength(2);
    expect(dividers[0].text()).toContain("HAND #1");
    expect(dividers[1].text()).toContain("HAND #2");
    wrapper.unmount();
  });

  it("keeps turn headers free of duplicated table stats and legal actions", async () => {
    const wrapper = mount(AgentConsole);
    await nextTick();

    expect(wrapper.find(".turn-stats").exists()).toBe(false);
    expect(wrapper.find(".legal-line").exists()).toBe(false);
    expect(wrapper.find(".position-chip").text()).toContain("BTN");
    wrapper.unmount();
  });

  it("opens the next hand before the agent has acted", async () => {
    storeMock.game.snapshot.hand_id = 3;
    const wrapper = mount(AgentConsole);
    await nextTick();
    const labels = wrapper.findAll(".hand-divider").map((divider) => divider.text());
    expect(labels).toEqual(["HAND #1", "HAND #2", "HAND #3"]);
    wrapper.unmount();
  });
});
