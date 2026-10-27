import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ToolCard from "./ToolCard.vue";
import type { ToolView } from "../types";

function talkTool(overrides: Partial<ToolView> = {}): ToolView {
  return {
    index: 0,
    callId: "call-1",
    name: "send_table_message",
    argsText: '{"text":"我这次不会再让你便宜看牌。"}',
    parsed: { text: "我这次不会再让你便宜看牌。" },
    status: "done",
    result: {
      ok: true,
      entry: { text: "我这次不会再让你便宜看牌。" },
      remaining_turn: 0,
    },
    summary: "已喊话",
    durationMs: 2,
    ...overrides,
  };
}

describe("ToolCard research history", () => {
  function historyTool(overrides: Partial<ToolView> = {}): ToolView {
    return {
      index: 0,
      callId: "call-history",
      name: "research_history",
      argsText: '{"scope":"both"}',
      parsed: { scope: "both" },
      status: "done",
      result: {
        ok: true,
        scope: "both",
        current_match: {
          hands: 3,
          blinds: { small: 5, big: 10 },
          stacks: { you: 1000, opponent: 1000 },
          you: {
            preflop: {
              vpip: { value: 1, n: 3, confidence: "low" },
              pfr: { value: 0.3333, n: 3, confidence: "low" },
            },
          },
          opponent: {
            preflop: {
              vpip: { value: 0.6667, n: 3, confidence: "low" },
              pfr: { value: 0, n: 3, confidence: "low" },
            },
          },
          recent_hands: [
            { hand_id: 12, you_won: true, your_net: 120 },
            { hand_id: 13, you_won: false, your_net: -60 },
          ],
        },
        completed_matches: {
          count: 1,
          vs_current_opponent: { matches: 1, hands: 10 },
          overall: { matches: 1, hands: 10 },
        },
      },
      summary: "3 手 · 对手 VPIP 66.7%（n=3）",
      durationMs: 1,
      ...overrides,
    };
  }

  it("renders a comparison grid instead of only raw JSON", () => {
    const wrapper = mount(ToolCard, { props: { tool: historyTool() } });
    expect(wrapper.text()).toContain("历史统计");
    expect(wrapper.text()).toContain("当前 + 历史");
    expect(wrapper.text()).toContain("盲注 5/10");
    expect(wrapper.text()).toContain("VPIP");
    expect(wrapper.text()).toContain("66.7%");
    expect(wrapper.text()).toContain("n=3");
    expect(wrapper.text()).toContain("最近手牌");
    expect(wrapper.text()).toContain("#12");
    expect(wrapper.text()).toContain("已完成对局 · 对当前对手 1 场 / 10 手");
    expect(wrapper.findAll(".history-row").length).toBeGreaterThan(2);
    expect(wrapper.findAll(".history-group").map((group) => group.text())).toEqual(["翻前"]);
    wrapper.unmount();
  });

  it("shows only the requested metric group", () => {
    const tool = historyTool({
      argsText: '{"scope":"current_match","focus":"preflop"}',
      parsed: { scope: "current_match", focus: "preflop" },
      result: {
        ok: true,
        scope: "current_match",
        current_match: {
          hands: 2,
          you: { preflop: { vpip: { value: 0.5, n: 2, confidence: "low" } } },
          opponent: { preflop: { vpip: { value: 0, n: 2, confidence: "low" } } },
        },
        completed_matches: { count: 0 },
      },
    });
    const wrapper = mount(ToolCard, { props: { tool } });
    expect(wrapper.text()).toContain("只看翻前");
    expect(wrapper.text()).toContain("当前对局");
    expect(wrapper.text()).toContain("50.0%");
    expect(wrapper.text()).not.toContain("最近手牌");
    wrapper.unmount();
  });

  it("renders the error reason", () => {
    const wrapper = mount(ToolCard, {
      props: {
        tool: historyTool({
          status: "error",
          result: { ok: false, error: "history provider failed" },
          summary: "history provider failed",
        }),
      },
    });
    expect(wrapper.text()).toContain("历史统计");
    expect(wrapper.text()).toContain("history provider failed");
    expect(wrapper.find(".history-grid").exists()).toBe(false);
    wrapper.unmount();
  });

  it("switches between aggregate and position splits", async () => {
    const wrapper = mount(ToolCard, {
      props: {
        tool: historyTool({
          result: {
            ok: true,
            scope: "current_match",
            current_match: {
              hands: 4,
              you: { preflop: { vpip: { value: 0.5, n: 4, confidence: "low" } } },
              opponent: { preflop: { vpip: { value: 0.25, n: 4, confidence: "low" } } },
              positions: {
                you: {
                  button: { preflop: { vpip: { value: 1, n: 2, confidence: "low" } } },
                  bb: { preflop: { vpip: { value: 0, n: 2, confidence: "low" } } },
                },
                opponent: {
                  button: { preflop: { vpip: { value: 0, n: 2, confidence: "low" } } },
                  bb: { preflop: { vpip: { value: 0.5, n: 2, confidence: "low" } } },
                },
              },
            },
            completed_matches: { count: 0 },
          },
        }),
      },
    });
    expect(wrapper.text()).toContain("50.0%");
    const buttons = wrapper.findAll(".history-position");
    expect(buttons.map((button) => button.text())).toEqual(["全部", "按钮", "大盲"]);
    await buttons[2].trigger("click");
    expect(wrapper.findAll(".history-value").map((value) => value.text())).toEqual(["0.0%", "50.0%"]);
    wrapper.unmount();
  });

  it("summarizes completed-only scopes in the header", () => {
    const wrapper = mount(ToolCard, {
      props: {
        tool: historyTool({
          result: {
            ok: true,
            scope: "completed_matches",
            current_match: { hands: 0, you: {}, opponent: {} },
            completed_matches: {
              count: 2,
              vs_current_opponent: { matches: 2, hands: 30 },
              overall: { matches: 3, hands: 45 },
            },
          },
        }),
      },
    });
    expect(wrapper.find(".tool-detail").text()).toBe("已完成 2 场 · 30 手");
    expect(wrapper.text()).toContain("已完成对局 · 对当前对手 2 场 / 30 手");
    wrapper.unmount();
  });
});

describe("ToolCard action states", () => {
  function actionTool(overrides: Partial<ToolView> = {}): ToolView {
    return {
      index: 0,
      callId: "call-action",
      name: "apply_action",
      argsText: '{"action":"bet","amount":30}',
      parsed: { action: "bet", amount: 30 },
      status: "running",
      result: null,
      summary: "",
      durationMs: null,
      ...overrides,
    };
  }

  it("does not show a rejection while the action is still running", () => {
    const wrapper = mount(ToolCard, { props: { tool: actionTool() } });
    expect(wrapper.text()).toContain("正在提交动作");
    expect(wrapper.text()).not.toContain("动作被拒绝");
    expect(wrapper.text()).not.toContain("正在执行执行动作");
    wrapper.unmount();
  });

  it("shows only the accepted result after completion", () => {
    const wrapper = mount(ToolCard, {
      props: {
        tool: actionTool({
          status: "done",
          result: { ok: true, accepted: true, action: "bet", amount: 30 },
        }),
      },
    });
    expect(wrapper.text()).toContain("动作已接受");
    expect(wrapper.text()).not.toContain("动作被拒绝");
    wrapper.unmount();
  });
});

describe("ToolCard table talk", () => {
  it("renders an accepted message with remaining quotas", () => {
    const wrapper = mount(ToolCard, { props: { tool: talkTool() } });
    expect(wrapper.text()).toContain("桌边喊话");
    expect(wrapper.text()).toContain("我这次不会再让你便宜看牌。");
    expect(wrapper.text()).toContain("本回合喊话完成");
    wrapper.unmount();
  });

  it("renders inline table talk inside the action card", () => {
    const wrapper = mount(ToolCard, {
      props: {
        tool: {
          index: 0,
          callId: "call-2",
          name: "apply_action",
          argsText: '{"action":"call","table_talk":"我先放个烟雾弹。"}',
          parsed: { action: "call", table_talk: "我先放个烟雾弹。" },
          status: "done",
          result: {
            ok: true,
            accepted: true,
            action: "call",
            table_talk: { text: "我先放个烟雾弹。" },
          },
          summary: "",
          durationMs: 1,
        },
      },
    });
    expect(wrapper.text()).toContain("我先放个烟雾弹。");
    wrapper.unmount();
  });

  it("renders a rejected message reason", () => {
    const wrapper = mount(ToolCard, {
      props: {
        tool: talkTool({
          status: "error",
          result: { ok: false, error: "喊话内容不能为空" },
          summary: "喊话内容不能为空",
        }),
      },
    });
    expect(wrapper.text()).toContain("桌边喊话");
    expect(wrapper.text()).toContain("喊话内容不能为空");
    expect(wrapper.text()).not.toContain("对手下一次决策时看到");
    wrapper.unmount();
  });
});
