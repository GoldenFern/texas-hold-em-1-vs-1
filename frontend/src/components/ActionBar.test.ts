import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";

const storeMock = vi.hoisted(() => ({
  humanAction: vi.fn(),
  nextHand: vi.fn(),
  openSetup: vi.fn(),
  humanSeat: { value: { seat: 0, kind: "human" } },
  game: {
    snapshot: null as any,
    busy: false,
    setupOpen: false,
    lastResult: null,
  },
}));

vi.mock("../store", () => ({
  game: storeMock.game,
  humanAction: storeMock.humanAction,
  humanSeat: storeMock.humanSeat,
  nextHand: storeMock.nextHand,
  openSetup: storeMock.openSetup,
}));

import ActionBar from "./ActionBar.vue";

function snapshot(overrides: Record<string, any> = {}) {
  return {
    has_match: true,
    hand_id: 1,
    street: "flop",
    pot: 30,
    current_bet: 10,
    min_raise_to: 20,
    max_raise_to: 900,
    to_call: 0,
    call_amount: 0,
    acting_seat: 0,
    acting_kind: "human",
    legal_actions: ["fold", "check", "bet", "all_in"],
    players: [
      { seat: 0, kind: "human", chips: 990, street_bet: 10, assessment: null },
      { seat: 1, kind: "bot", chips: 980, street_bet: 10, assessment: null },
    ],
    match: { match_over: false, god_view: false, can_next_hand: false, seats: [] },
    ...overrides,
  };
}

beforeEach(() => {
  storeMock.game.snapshot = snapshot();
  storeMock.game.busy = false;
  storeMock.game.setupOpen = false;
  storeMock.game.lastResult = null;
  storeMock.humanAction.mockClear();
});

afterEach(() => {
  storeMock.humanAction.mockClear();
});

describe("ActionBar keyboard shortcuts", () => {
  it("does not fold, call, or go all-in with a modifier key", async () => {
    const wrapper = mount(ActionBar);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "a", ctrlKey: true }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "f", metaKey: true }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "c", altKey: true }));
    await nextTick();
    expect(storeMock.humanAction).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("still supports the plain all-in shortcut", async () => {
    const wrapper = mount(ActionBar);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "a" }));
    await nextTick();
    expect(storeMock.humanAction).toHaveBeenCalledWith("all_in", 0);
    wrapper.unmount();
  });
});

describe("ActionBar raise type", () => {
  it("submits a bet rather than a raise when the engine offers bet", async () => {
    const wrapper = mount(ActionBar);
    const button = wrapper
      .findAll("button")
      .find((item) => item.text().includes("下注到"));
    expect(button).toBeTruthy();
    await button!.trigger("click");
    expect(storeMock.humanAction).toHaveBeenCalledWith("bet", 20);
    wrapper.unmount();
  });

  it("keeps the typed raise amount visible until blur clamps it", async () => {
    const wrapper = mount(ActionBar);
    const input = wrapper.find('input[type="number"]');
    expect(input.exists()).toBe(true);
    await input.setValue("5");
    expect((input.element as HTMLInputElement).value).toBe("5");
    await input.trigger("blur");
    expect((input.element as HTMLInputElement).value).toBe("20");
    wrapper.unmount();
  });

  it("shows a call amount only for the human's own outstanding bet", () => {
    storeMock.game.snapshot = snapshot({
      acting_kind: "bot",
      acting_seat: 1,
      current_bet: 100,
      legal_actions: [],
    });
    const wrapper = mount(ActionBar);
    expect(wrapper.text()).not.toContain("需跟注");
    wrapper.unmount();
  });
});

describe("ActionBar aborted match", () => {
  it("shows the aborted state and hides the action controls", () => {
    storeMock.game.snapshot = snapshot({
      match: {
        match_over: true,
        aborted: true,
        god_view: false,
        can_next_hand: false,
        seats: [],
      },
    });
    const wrapper = mount(ActionBar);
    expect(wrapper.text()).toContain("对局已中止");
    expect(wrapper.find(".controls").exists()).toBe(false);
    expect(wrapper.text()).toContain("新比赛");
    wrapper.unmount();
  });
});
