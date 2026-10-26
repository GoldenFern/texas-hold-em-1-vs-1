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
