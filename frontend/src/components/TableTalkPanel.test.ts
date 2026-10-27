import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import TableTalkPanel from "./TableTalkPanel.vue";

const seats = [
  { seat: 0, name: "Agent A", kind: "llm" as const, is_human: false, provider_id: "deepseek", model_id: "a", model_name: "Model A", icon: "", table_talk_style_name: "温文尔雅", table_talk_style_intensity: 0, agent: {} },
  { seat: 1, name: "Agent B", kind: "llm" as const, is_human: false, provider_id: "zhipu", model_id: "b", model_name: "Model B", icon: "", table_talk_style_name: "贴吧老哥", table_talk_style_intensity: 2, agent: {} },
];

describe("TableTalkPanel", () => {
  it("groups messages by hand", () => {
    const wrapper = mount(TableTalkPanel, {
      props: {
        seats,
        entries: [
          { id: "a", seq: 1, hand_id: 1, street: "preflop", seat: 0, name: "Agent A", text: "第一句" },
          { id: "b", seq: 2, hand_id: 1, street: "flop", seat: 1, name: "Agent B", text: "第二句" },
          { id: "c", seq: 3, hand_id: 2, street: "finished", seat: 0, name: "Agent A", text: "下一手", kind: "final" },
        ],
      },
    });

    expect(wrapper.find(".talk-note").exists()).toBe(false);
    expect(wrapper.findAll(".talk-group")).toHaveLength(2);
    expect(wrapper.text()).toContain("第一句");
    expect(wrapper.text()).toContain("第二句");
    expect(wrapper.text()).toContain("下一手");
    expect(wrapper.text()).toContain("翻牌");
    expect(wrapper.text()).toContain("赛后");
    expect(wrapper.findAll(".talk-style")).toHaveLength(3);
    expect(wrapper.text()).toContain("温文尔雅");
    expect(wrapper.text()).toContain("贴吧老哥");
    expect(wrapper.findAll(".talk-style.aggressive")).toHaveLength(1);
    wrapper.unmount();
  });

  it("shows an empty state before anyone talks", () => {
    const wrapper = mount(TableTalkPanel, { props: { seats, entries: [] } });
    expect(wrapper.text()).toContain("还没有人喊话");
    wrapper.unmount();
  });
});
