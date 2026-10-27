import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";

const storeMock = vi.hoisted(() => {
  const loadedModel = {
    provider_id: "deepseek",
    provider_name: "DeepSeek",
    provider_name_en: "DeepSeek",
    icon: "deepseek.svg",
    accent: "#5B7CFA",
    base_url: "https://api.deepseek.com",
    api_key_env: "DEEPSEEK_API_KEY",
    key_present: true,
    docs_url: "https://api-docs.deepseek.com",
    id: "deepseek-flash",
    name: "DeepSeek V4.1 Flash",
    tagline: "1M 上下文 · 原生工具调用",
    tagline_en: "1M context · native tool calls",
    context_window: 1048576,
    max_output: 393216,
    defaults: {
      temperature: 1,
      top_p: 1,
      max_tokens: 32768,
      thinking: true,
      reasoning_effort: "low",
    },
    controls: {
      temperature: { show: true, min: 0, max: 2, step: 0.1 },
      top_p: { show: true, min: 0, max: 1, step: 0.05 },
      max_tokens: { show: true, min: 1024, max: 393216, step: 1024 },
      thinking: { mode: "toggle" },
      reasoning_effort: { show: true, options: ["low", "high", "max"] },
    },
  };

  const defaultSeat = (kind: string) => {
    if (kind === "human") return { kind: "human", name: "You" };
    if (kind === "bot") {
      return {
        kind: "bot",
        name: "RLCard",
        bot: { agent_type: "random", model_path: null, delay_seconds: 3 },
      };
    }
    return {
      kind: "llm",
      name: loadedModel.name,
      provider_id: loadedModel.provider_id,
      model_id: loadedModel.id,
      params: { ...loadedModel.defaults },
      table_talk_style: "gentle",
    };
  };

  return {
    model: loadedModel,
    defaultSeat,
    game: {
      setupOpen: true,
      busy: false,
      starting: false,
      aborting: false,
      snapshot: null,
      testResults: {},
      boot: {
        providers: [
          {
            id: "deepseek",
            name: "DeepSeek",
            name_en: "DeepSeek",
            icon: "deepseek.svg",
            accent: "#5B7CFA",
            models: [loadedModel],
          },
        ],
        rlcard: { available: true, models: [], agents: [], model_dir: "models" },
        table_talk_styles: [
          {
            id: "gentle",
            name: "温文尔雅",
            name_en: "The Gentleman",
            description: "礼貌、克制。",
            description_en: "Polite.",
            sample: "承让。",
            sample_en: "Well played.",
            intensity: 0,
            default: true,
          },
        ],
        game: {
          defaults: { table_talk_style: "gentle" },
          limits: {
            starting_chips: { min: 100, max: 100000 },
            rebuy_amount: { min: 100, max: 100000 },
            max_hands: { max: 1000 },
            blind_increase_every_hands: { min: 0, max: 1000 },
            blind_increase_factor: { min: 1, max: 10 },
          },
        },
      },
    },
  };
});

vi.mock("../store", () => ({
  game: storeMock.game,
  buildDefaultSetup: () => {
    const seats = [storeMock.defaultSeat("llm"), storeMock.defaultSeat("human")];
    return {
      seats,
      rules: {
        starting_chips: 1000,
        small_blind: 5,
        big_blind: 10,
        auto_rebuy: false,
        rebuy_amount: 1000,
        max_hands: 0,
        blind_increase_every_hands: 0,
        blind_increase_factor: 2,
        seed: null,
        god_view: false,
      },
      table_talk: true,
      table_talk_style: "gentle",
      agent_locale: "auto",
    };
  },
  closeSetup: vi.fn(),
  defaultGodView: (seats: Array<{ kind: string }>) =>
    !seats.some((seat) => seat.kind === "human"),
  defaultSeat: storeMock.defaultSeat,
  defaultTableTalkStyle: () => "gentle",
  modelFor: (providerId: string, modelId: string) =>
    providerId === "deepseek" && modelId === "deepseek-flash" ? storeMock.model : null,
  providerOf: () => storeMock.game.boot.providers[0],
  pushToast: vi.fn(),
  startMatch: vi.fn(),
  testProvider: vi.fn(),
  abortMatch: vi.fn(),
}));

import SetupModal from "./SetupModal.vue";

function mountModal() {
  return mount(SetupModal, {
    global: {
      stubs: {
        AppIcon: true,
        LanguageSwitch: true,
      },
    },
  });
}

async function setSeatKind(wrapper: ReturnType<typeof mountModal>, index: number, label: string) {
  const seat = wrapper.findAll(".seat-config")[index];
  const button = seat.findAll(".kind-switch button").find((item) => item.text() === label);
  await button!.trigger("click");
  await nextTick();
}

describe("SetupModal layout and defaults", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("keeps model parameters out of the main view", async () => {
    const wrapper = mountModal();
    await nextTick();

    expect(wrapper.find(".advanced-toggle").exists()).toBe(true);
    expect(wrapper.find(".advanced-body").exists()).toBe(false);
    expect(wrapper.find(".param-grid").exists()).toBe(false);
    expect(wrapper.text()).toContain("DeepSeek V4.1 Flash");
    expect(wrapper.text()).toContain("起始筹码");
    wrapper.unmount();
  });

  it("turns god view on when no human remains", async () => {
    const wrapper = mountModal();
    await nextTick();
    await wrapper.find(".advanced-toggle").trigger("click");
    await nextTick();
    expect(wrapper.find(".god-view-toggle .toggle").classes()).not.toContain("on");

    await setSeatKind(wrapper, 1, "LLM");
    expect(wrapper.find(".god-view-toggle .toggle").classes()).toContain("on");
    wrapper.unmount();
  });

  it("keeps a manual god-view choice when the lineup changes", async () => {
    const wrapper = mountModal();
    await nextTick();
    await wrapper.find(".advanced-toggle").trigger("click");
    await nextTick();

    await wrapper.find(".god-view-toggle .toggle").trigger("click");
    await nextTick();
    expect(wrapper.find(".god-view-toggle .toggle").classes()).toContain("on");

    await setSeatKind(wrapper, 1, "LLM");
    expect(wrapper.find(".god-view-toggle .toggle").classes()).toContain("on");

    await setSeatKind(wrapper, 0, "人类");
    expect(wrapper.find(".god-view-toggle .toggle").classes()).toContain("on");
    wrapper.unmount();
  });
});
