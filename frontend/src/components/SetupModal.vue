<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from "vue";
import AppIcon from "./AppIcon.vue";
import LanguageSwitch from "./LanguageSwitch.vue";
import { iconUrl } from "../api";
import { agentNameFor, providerDisplayName } from "../format";
import { languageName, locale, localizedText, t } from "../i18n";
import {
  buildDefaultSetup,
  closeSetup,
  defaultSeat,
  game,
  modelFor,
  providerOf,
  pushToast,
  startMatch,
  testProvider,
} from "../store";
import type { MatchSetupPayload, ProviderInfo, ProviderModel, SeatSetup } from "../types";

interface SetupDraft extends MatchSetupPayload {}

const FALLBACK_RULES: SetupDraft["rules"] = {
  starting_chips: 1000,
  small_blind: 5,
  big_blind: 10,
  auto_rebuy: false,
  rebuy_amount: 1000,
  max_hands: 0,
  blind_increase_every_hands: 0,
  blind_increase_factor: 2,
  seed: null,
  god_view: true,
};

const BOT_DELAY_DEFAULT = 3;
const BOT_DELAY_MIN = 0;
const BOT_DELAY_MAX = 10;

const setup = reactive<SetupDraft>({
  seats: [
    { kind: "llm", name: "LLM Agent" },
    { kind: "human", name: t("common.you") },
  ],
  rules: { ...FALLBACK_RULES },
  table_talk: true,
  agent_locale: "auto",
});

const boot = computed(() => game.boot);
const providers = computed<ProviderInfo[]>(() => boot.value?.providers || []);
const rlcard = computed(() => boot.value?.rlcard || null);
const limits = computed(() => boot.value?.game?.limits || {});
const busy = computed(() => game.busy);
const canTableTalk = computed(() => setup.seats.every((item) => item.kind === "llm"));
const tableTalkTouched = ref(false);
const modalRef = ref<HTMLElement | null>(null);

watch(canTableTalk, (enabled) => {
  if (!enabled) {
    setup.table_talk = false;
    return;
  }
  if (!tableTalkTouched.value) setup.table_talk = true;
});

function toggleTableTalk(): void {
  if (!canTableTalk.value) return;
  tableTalkTouched.value = true;
  setup.table_talk = !setup.table_talk;
}

function deepCopy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function normalizeBotDelay(value: unknown): number {
  const numeric = Number(value ?? BOT_DELAY_DEFAULT);
  if (!Number.isFinite(numeric)) return BOT_DELAY_DEFAULT;
  const clamped = Math.min(BOT_DELAY_MAX, Math.max(BOT_DELAY_MIN, numeric));
  return Math.round(clamped * 2) / 2;
}

function resetSetup(): void {
  const source = buildDefaultSetup();
  const copy = deepCopy(source);
  setup.seats = copy.seats;
  setup.rules = { ...copy.rules };
  setup.agent_locale = copy.agent_locale || "auto";
  const savedTalk = Boolean(copy.table_talk ?? true);
  setup.table_talk = savedTalk && setup.seats.every((item) => item.kind === "llm");
  tableTalkTouched.value = !savedTalk;
  if (!setup.rules.rebuy_amount) setup.rules.rebuy_amount = setup.rules.starting_chips;
  for (const config of setup.seats) {
    if (config.kind !== "bot") continue;
    config.bot = {
      agent_type: config.bot?.agent_type || "random",
      model_path: config.bot?.model_path || null,
      delay_seconds: normalizeBotDelay(config.bot?.delay_seconds),
    };
  }
}

watch(
  () => game.setupOpen,
  (open) => {
    if (open) {
      resetSetup();
      void nextTick(() => modalRef.value?.focus());
    }
  },
  { immediate: true },
);

function onGlobalKeydown(event: KeyboardEvent): void {
  if (!game.setupOpen) return;
  if (event.key === "Escape") {
    event.preventDefault();
    close();
    return;
  }
  if (event.key !== "Tab" || !modalRef.value) return;
  const focusable = Array.from(
    modalRef.value.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => element.getClientRects().length > 0);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const active = document.activeElement as HTMLElement | null;
  if (event.shiftKey && (active === first || active === modalRef.value)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
}

onMounted(() => window.addEventListener("keydown", onGlobalKeydown));
onUnmounted(() => window.removeEventListener("keydown", onGlobalKeydown));

function seat(index: number): SeatSetup {
  return setup.seats[index];
}

function otherSeat(index: number): SeatSetup {
  return setup.seats[index === 0 ? 1 : 0];
}

function setKind(index: number, kind: SeatSetup["kind"]): void {
  const current = seat(index);
  if (current.kind === kind) return;
  if (kind === "human" && otherSeat(index).kind === "human") {
    pushToast("warn", t("setup.toast.twoHumans"), t("setup.toast.twoHumansDetail"));
    return;
  }
  if (kind === "human") {
    setup.seats[index] = { kind: "human", name: t("common.you") };
  } else if (kind === "bot") {
    setup.seats[index] = {
      kind: "bot",
      name: "RLCard",
      bot: { agent_type: "random", model_path: null, delay_seconds: BOT_DELAY_DEFAULT },
    };
  } else {
    const model = modelFor(current.provider_id || "", current.model_id || "")
      || providers.value.flatMap((provider) => provider.models).find((item) => item.key_present)
      || providers.value.flatMap((provider) => provider.models)[0]
      || null;
    setup.seats[index] = defaultSeat("llm", model);
  }
}

function currentModel(index: number): ProviderModel | null {
  const config = seat(index);
  if (config.kind !== "llm" || !config.provider_id || !config.model_id) return null;
  return modelFor(config.provider_id, config.model_id);
}

function selectProvider(index: number, provider: ProviderInfo): void {
  const current = seat(index);
  if (current.kind === "llm" && current.provider_id === provider.id) return;
  const model = provider.models[0];
  if (!model) return;
  setup.seats[index] = {
    ...seat(index),
    kind: "llm",
    name: agentNameFor(providerDisplayName(provider.name, provider.name_en), model.name),
    provider_id: provider.id,
    model_id: model.id,
    params: { ...model.defaults },
  } as SeatSetup;
}

function selectModel(index: number, model: ProviderModel): void {
  setup.seats[index] = {
    ...seat(index),
    kind: "llm",
    name: agentNameFor(
      providerDisplayName(model.provider_name, model.provider_name_en),
      model.name,
    ),
    provider_id: model.provider_id,
    model_id: model.id,
    params: { ...model.defaults },
  } as SeatSetup;
}

function providerName(provider: ProviderInfo): string {
  return providerDisplayName(provider.name, provider.name_en);
}

function modelProviderName(model: ProviderModel): string {
  return providerDisplayName(model.provider_name, model.provider_name_en);
}

function modelTagline(model: ProviderModel): string {
  return localizedText(model.tagline, model.tagline_en);
}

function controlHint(control?: { hint?: string; hint_en?: string }): string {
  return localizedText(control?.hint, control?.hint_en);
}

function controlLabel(
  control: { label?: string; label_en?: string } | undefined,
  fallbackKey: "setup.thinking",
): string {
  return localizedText(control?.label, control?.label_en) || t(fallbackKey);
}

function botDescription(agent: { description: string; description_en?: string }): string {
  return localizedText(agent.description, agent.description_en);
}

function setAgentLocale(value: "auto" | "zh-CN" | "en-US"): void {
  setup.agent_locale = value;
}

const resolvedAgentLanguage = computed(() => {
  const target = setup.agent_locale === "auto" ? locale.value : setup.agent_locale;
  return languageName(target);
});

function paramsOf(index: number): Record<string, any> {
  const config = seat(index);
  if (!config.params) config.params = {};
  return config.params as Record<string, any>;
}

function setParam(index: number, key: string, value: unknown): void {
  paramsOf(index)[key] = value;
  const model = currentModel(index);
  if (model) delete game.testResults[testKey(index, model)];
}

function setBotType(index: number, agentType: string): void {
  const current = seat(index).bot || { agent_type: "random", model_path: null };
  const preferred: Record<string, string> = {
    dqn: "nlh_dqn.pth",
    nfsp: "nlh_nfsp.pth",
    dmc: "nlh_dmc.pth",
    project_dqn: "nlh_dqn_project.pth",
    project_nfsp: "nlh_nfsp_project.pth",
  };
  const candidates = botModels(agentType);
  const preferredName = preferred[agentType];
  const chosen =
    candidates.find((model) => model.name === preferredName) ||
    candidates[0] ||
    null;
  setup.seats[index] = {
    ...seat(index),
    kind: "bot",
    name: agentType.startsWith("project_") ? "Project Bot" : "RLCard",
    bot: {
      agent_type: agentType,
      model_path: agentType === "random" ? null : (chosen?.path || current.model_path || null),
      delay_seconds: normalizeBotDelay(current.delay_seconds),
    },
  } as SeatSetup;
}

function botModels(agentType: string) {
  const models = rlcard.value?.models || [];
  const project = agentType.startsWith("project_");
  return models.filter((model) => model.name.includes("_project") === project);
}

function setBotModel(index: number, path: string): void {
  const bot = seat(index).bot || { agent_type: "random", model_path: null };
  setup.seats[index] = {
    ...seat(index),
    kind: "bot",
    bot: { ...bot, model_path: path || null },
  } as SeatSetup;
}

function botDelay(index: number): number {
  return normalizeBotDelay(seat(index).bot?.delay_seconds);
}

function setBotDelay(index: number, event: Event): void {
  const bot = seat(index).bot || { agent_type: "random", model_path: null };
  setup.seats[index] = {
    ...seat(index),
    kind: "bot",
    bot: { ...bot, delay_seconds: normalizeBotDelay((event.target as HTMLInputElement).value) },
  } as SeatSetup;
}

function setRule(key: keyof SetupDraft["rules"], value: unknown): void {
  (setup.rules as Record<string, unknown>)[key] = value;
}

function numericRule(key: keyof SetupDraft["rules"]): number {
  const value = Number(setup.rules[key]);
  return Number.isFinite(value) ? value : 0;
}

function setNumericRule(key: keyof SetupDraft["rules"], event: Event): void {
  const value = Number((event.target as HTMLInputElement).value);
  setRule(key, Number.isFinite(value) ? Math.round(value) : 0);
}

function setSeed(event: Event): void {
  const raw = (event.target as HTMLInputElement).value.trim();
  if (raw === "") {
    setRule("seed", null);
    return;
  }
  const value = Number(raw);
  setRule("seed", Number.isFinite(value) ? Math.round(value) : null);
}

function setBlinds(small: number, big: number): void {
  setRule("small_blind", small);
  setRule("big_blind", big);
}

const blindIncreaseEnabled = computed(
  () => numericRule("blind_increase_every_hands") > 0,
);

function toggleBlindIncrease(): void {
  if (blindIncreaseEnabled.value) {
    setRule("blind_increase_every_hands", 0);
    return;
  }
  setRule("blind_increase_every_hands", 10);
  setRule("blind_increase_factor", 2);
}

function setBlindFactor(event: Event): void {
  const value = Number((event.target as HTMLSelectElement).value);
  setRule("blind_increase_factor", Number.isFinite(value) ? value : 2);
}

function limitError(key: string, label: string): string {
  const limit = limits.value[key];
  if (!limit) return "";
  const value = numericRule(key as keyof SetupDraft["rules"]);
  if (value < limit.min || value > limit.max) {
    return t("setup.validation.range", { label, min: limit.min, max: limit.max });
  }
  return "";
}

const validation = computed(() => {
  if (setup.seats.filter((item) => item.kind === "human").length >= 2) {
    return t("setup.validation.twoHumans");
  }
  for (const config of setup.seats) {
    if (config.kind === "llm") {
      const model = modelFor(config.provider_id || "", config.model_id || "");
      if (!model) return t("setup.validation.selectModel");
      if (!model.key_present) return t("setup.validation.missingKey", { env: model.api_key_env });
    }
    if (config.kind === "bot") {
      const agentType = config.bot?.agent_type || "random";
      if (agentType !== "random" && !config.bot?.model_path) {
        return t("setup.validation.botModel", {
          framework: agentType.startsWith("project_") ? "Project" : "RLCard",
          type: agentType.toUpperCase(),
        });
      }
      if (!agentType.startsWith("project_") && rlcard.value && !rlcard.value.available) {
        return t("setup.validation.rlcardMissing");
      }
    }
  }
  for (const [key, label] of [
    ["starting_chips", t("setup.startingChips")],
    ["small_blind", t("setup.smallBlind")],
    ["big_blind", t("setup.bigBlind")],
    ["rebuy_amount", t("setup.rebuyAmount")],
    ["max_hands", t("setup.maxHands")],
    ["blind_increase_every_hands", t("setup.blindIncreaseEvery")],
  ] as const) {
    const error = limitError(key, label);
    if (error) return error;
  }
  if (numericRule("starting_chips") < numericRule("big_blind")) {
    return t("setup.validation.startingChipsBelowBB");
  }
  if (numericRule("small_blind") <= 0 || numericRule("big_blind") <= 0) {
    return t("setup.validation.blindsPositive");
  }
  if (numericRule("small_blind") >= numericRule("big_blind")) {
    return t("setup.validation.bigBlindGreater");
  }
  if (numericRule("max_hands") < 0) return t("setup.validation.maxHandsNegative");
  if (blindIncreaseEnabled.value) {
    const factor = numericRule("blind_increase_factor");
    if (factor < 1 || factor > 10) return t("setup.validation.blindFactor");
  }
  return "";
});

const canStart = computed(() => !validation.value && !busy.value);

function applyPreset(kind: "human_deepseek" | "deepseek_glm" | "human_bot"): void {
  const allModels = providers.value.flatMap((provider) => provider.models);
  const deepseek = allModels.find((model) => model.provider_id === "deepseek" && model.key_present) || allModels.find((model) => model.key_present) || null;
  const glm = allModels.find((model) => model.provider_id === "zhipu" && model.key_present) || allModels.find((model) => model.key_present) || deepseek;
  if (kind === "human_deepseek") {
    setup.seats = [defaultSeat("llm", deepseek), defaultSeat("human")];
  } else if (kind === "deepseek_glm") {
    setup.seats = [defaultSeat("llm", deepseek), defaultSeat("llm", glm)];
    tableTalkTouched.value = false;
    setup.table_talk = true;
  } else {
    setup.seats = [defaultSeat("bot"), defaultSeat("human")];
  }
}

function testKey(index: number, model: ProviderModel): string {
  return `${index}:${model.provider_id}:${model.id}`;
}

function testConnection(index: number): void {
  const model = currentModel(index);
  if (!model) return;
  void testProvider(model, paramsOf(index), testKey(index, model));
}

function testResult(index: number) {
  const model = currentModel(index);
  if (!model) return null;
  return game.testResults[testKey(index, model)] || null;
}

function isTesting(index: number): boolean {
  return testResult(index)?.status === "testing";
}

function compactNumber(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value % 1_000_000 === 0 ? 0 : 1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}K`;
  return String(value);
}

function rangePct(value: number, min: number, max: number): string {
  const span = max - min;
  if (span <= 0) return "100%";
  return `${Math.max(0, Math.min(100, ((value - min) / span) * 100))}%`;
}

function modelQuirk(model: ProviderModel): string {
  if (model.provider_id === "minimax") return t("setup.quirk.minimax");
  if (model.provider_id === "zhipu") return t("setup.quirk.zhipu");
  if (model.provider_id === "kimi") return t("setup.quirk.kimi");
  if (model.provider_id === "qwen") return t("setup.quirk.qwen");
  if (model.provider_id === "deepseek") return t("setup.quirk.deepseek");
  if (model.provider_id === "longcat") return t("setup.quirk.longcat");
  if (model.provider_id === "mimo") return t("setup.quirk.mimo");
  if (model.provider_id === "4router") return t("setup.quirk.4router");
  return t("setup.quirk.generic");
}

async function submit(): Promise<void> {
  if (!canStart.value) return;
  const payload = deepCopy(setup);
  payload.table_talk = Boolean(payload.table_talk && canTableTalk.value);
  await startMatch(payload);
}

function close(): void {
  if (!busy.value) closeSetup();
}
</script>

<template>
  <div v-if="game.setupOpen" class="modal-backdrop" @click.self="close" @keydown.esc="close">
    <div ref="modalRef" class="modal setup-modal" role="dialog" aria-modal="true" :aria-label="t('setup.aria')" tabindex="-1">
      <header class="modal-header">
        <div>
          <div class="eyebrow">MATCH SETUP</div>
          <h2 class="modal-title">{{ t("setup.title") }}</h2>
          <p class="modal-subtitle">{{ t("setup.subtitle") }}</p>
        </div>
        <div class="modal-head-tools">
          <LanguageSwitch />
          <button class="btn btn-ghost btn-icon" :aria-label="t('common.close')" @click="close">
            <AppIcon name="x" :size="18" />
          </button>
        </div>
      </header>

      <div class="modal-body setup-body">
        <div class="preset-row">
          <span class="preset-label">{{ t("setup.quickStart") }}</span>
          <button class="preset" @click="applyPreset('human_deepseek')">
            <AppIcon name="user" :size="12" /> {{ t("setup.presetHumanDeepseek") }}
          </button>
          <button class="preset" @click="applyPreset('deepseek_glm')">
            <AppIcon name="sparkles" :size="12" /> {{ t("setup.presetLlmDuel") }}
          </button>
          <button class="preset" @click="applyPreset('human_bot')">
            <AppIcon name="bot" :size="12" /> {{ t("setup.presetHumanBot") }}
          </button>
          <span class="preset-hint">{{ t("setup.presetHint") }}</span>
        </div>

        <section class="setup-section">
          <div class="section-head">
            <div>
              <span class="section-kicker">01</span>
              <h3>{{ t("setup.players") }}</h3>
            </div>
            <span class="section-note">{{ t("setup.twoHumansNote") }}</span>
          </div>

          <div class="seat-grid">
            <article v-for="(config, index) in setup.seats" :key="index" class="seat-config" :class="{ human: config.kind === 'human' }">
              <header class="seat-config-head">
                <div class="seat-label">
                  <span class="seat-tag">{{ t("setup.seatLabel", { seat: index === 0 ? "A" : "B" }) }}</span>
                  <span class="seat-position">{{ index === 0 ? t("setup.topSeat") : t("setup.bottomSeat") }}</span>
                </div>
                <div class="segmented kind-switch">
                  <button
                    :class="{ active: config.kind === 'human' }"
                    :disabled="config.kind !== 'human' && otherSeat(index).kind === 'human'"
                    :title="otherSeat(index).kind === 'human' && config.kind !== 'human' ? t('setup.twoHumansNote') : ''"
                    @click="setKind(index, 'human')"
                  >
                    {{ t("setup.human") }}
                  </button>
                  <button :class="{ active: config.kind === 'llm' }" @click="setKind(index, 'llm')">LLM</button>
                  <button :class="{ active: config.kind === 'bot' }" @click="setKind(index, 'bot')">RLCard</button>
                </div>
              </header>

              <!-- Human -->
              <div v-if="config.kind === 'human'" class="seat-body human-body">
                <div class="human-orb"><AppIcon name="user" :size="24" /></div>
                <h4>{{ t("setup.humanTitle") }}</h4>
                <p>{{ t("setup.humanBody") }}</p>
                <div class="human-features">
                  <span class="chip"><AppIcon name="eye" :size="10" /> {{ t("setup.humanFeatureHole") }}</span>
                  <span class="chip"><AppIcon name="clock" :size="10" /> {{ t("setup.humanFeatureTime") }}</span>
                  <span class="chip"><AppIcon name="cards" :size="10" /> {{ t("setup.humanFeatureGod") }}</span>
                </div>
                <blockquote class="human-quote">{{ t("setup.humanQuote") }}</blockquote>
              </div>

              <!-- LLM -->
              <div v-else-if="config.kind === 'llm'" class="seat-body">
                <div class="provider-grid">
                  <button
                    v-for="provider in providers"
                    :key="provider.id"
                    class="provider-button"
                    :class="{ active: config.provider_id === provider.id, missing: !provider.models.some((model) => model.key_present) }"
                    :style="{ '--provider-accent': provider.accent }"
                    @click="selectProvider(index, provider)"
                  >
                    <span class="provider-icon logo-tile">
                      <img :src="iconUrl(provider.icon)" :alt="providerName(provider)" />
                    </span>
                    <span class="provider-name">{{ providerName(provider) }}</span>
                    <span v-if="provider.models.some((model) => model.key_present)" class="key-dot ok"></span>
                    <span v-else class="key-dot missing"></span>
                  </button>
                </div>

                <template v-if="currentModel(index)">
                  <div class="model-card" :style="{ '--provider-accent': currentModel(index)!.accent }">
                    <div class="model-head">
                      <span class="model-icon logo-tile"><img :src="iconUrl(currentModel(index)!.icon)" :alt="modelProviderName(currentModel(index)!)" /></span>
                      <div class="model-title">
                        <strong>{{ currentModel(index)!.name }}</strong>
                        <span>{{ modelTagline(currentModel(index)!) }}</span>
                      </div>
                      <div class="model-specs">
                        <span class="chip mono">CTX {{ compactNumber(currentModel(index)!.context_window) }}</span>
                        <span class="chip mono">OUT {{ compactNumber(currentModel(index)!.max_output) }}</span>
                      </div>
                    </div>

                    <div class="key-row">
                      <span v-if="currentModel(index)!.key_present" class="key-state ok">
                        <span class="dot live"></span>
                        {{ t("setup.keyConfigured", { env: currentModel(index)!.api_key_env }) }}
                      </span>
                      <span v-else class="key-state missing">
                        <AppIcon name="alert" :size="11" />
                        {{ t("setup.keyMissing", { env: currentModel(index)!.api_key_env }) }}
                      </span>
                      <a class="docs-link" :href="currentModel(index)!.docs_url" target="_blank" rel="noopener">
                        {{ t("setup.docs") }} <AppIcon name="external" :size="10" />
                      </a>
                    </div>

                    <div class="param-grid">
                      <label v-if="currentModel(index)!.controls.temperature?.show" class="param">
                        <div class="param-head">
                          <span>{{ t("setup.temperature") }} <span class="param-en">temperature</span></span>
                          <strong class="mono">{{ Number(paramsOf(index).temperature ?? 0).toFixed(2) }}</strong>
                        </div>
                        <input
                          class="range"
                          type="range"
                          :min="currentModel(index)!.controls.temperature!.min ?? 0"
                          :max="currentModel(index)!.controls.temperature!.max ?? 2"
                          :step="currentModel(index)!.controls.temperature!.step ?? 0.1"
                          :value="paramsOf(index).temperature"
                          :style="{ '--pct': rangePct(Number(paramsOf(index).temperature ?? 0), currentModel(index)!.controls.temperature!.min ?? 0, currentModel(index)!.controls.temperature!.max ?? 1) }"
                          @input="setParam(index, 'temperature', Number(($event.target as HTMLInputElement).value))"
                        />
                        <span class="param-hint">{{ controlHint(currentModel(index)!.controls.temperature) }}</span>
                      </label>

                      <label v-if="currentModel(index)!.controls.top_p?.show" class="param">
                        <div class="param-head">
                          <span>{{ t("setup.topP") }} <span class="param-en">top_p</span></span>
                          <strong class="mono">{{ Number(paramsOf(index).top_p ?? 0).toFixed(2) }}</strong>
                        </div>
                        <input
                          class="range"
                          type="range"
                          :min="currentModel(index)!.controls.top_p!.min ?? 0"
                          :max="currentModel(index)!.controls.top_p!.max ?? 1"
                          :step="currentModel(index)!.controls.top_p!.step ?? 0.05"
                          :value="paramsOf(index).top_p"
                          :style="{ '--pct': rangePct(Number(paramsOf(index).top_p ?? 0), currentModel(index)!.controls.top_p!.min ?? 0, currentModel(index)!.controls.top_p!.max ?? 1) }"
                          @input="setParam(index, 'top_p', Number(($event.target as HTMLInputElement).value))"
                        />
                        <span class="param-hint">{{ controlHint(currentModel(index)!.controls.top_p) }}</span>
                      </label>

                      <label v-if="currentModel(index)!.controls.max_tokens?.show" class="param span-2">
                        <div class="param-head">
                          <span>{{ t("setup.maxOutput") }} <span class="param-en">max_tokens</span></span>
                          <strong class="mono">{{ Number(paramsOf(index).max_tokens ?? 0).toLocaleString("en-US") }}</strong>
                        </div>
                        <input
                          class="range"
                          type="range"
                          :min="currentModel(index)!.controls.max_tokens!.min ?? 1024"
                          :max="currentModel(index)!.controls.max_tokens!.max ?? 131072"
                          :step="currentModel(index)!.controls.max_tokens!.step ?? 1024"
                          :value="paramsOf(index).max_tokens"
                          :style="{ '--pct': rangePct(Number(paramsOf(index).max_tokens ?? 0), currentModel(index)!.controls.max_tokens!.min ?? 1024, currentModel(index)!.controls.max_tokens!.max ?? 8192) }"
                          @input="setParam(index, 'max_tokens', Number(($event.target as HTMLInputElement).value))"
                        />
                        <span class="param-hint">{{ controlHint(currentModel(index)!.controls.max_tokens) }}</span>
                      </label>
                    </div>

                    <div class="mode-row" v-if="currentModel(index)!.controls.thinking">
                      <div class="mode-label">
                        <span>{{ controlLabel(currentModel(index)!.controls.thinking, "setup.thinking") }}</span>
                        <span class="param-hint">{{ controlHint(currentModel(index)!.controls.thinking) }}</span>
                      </div>
                      <button
                        v-if="currentModel(index)!.controls.thinking!.mode === 'toggle'"
                        class="toggle"
                        :class="{ on: Boolean(paramsOf(index).thinking) }"
                        @click="setParam(index, 'thinking', !paramsOf(index).thinking)"
                      ></button>
                      <span v-else-if="currentModel(index)!.controls.thinking!.mode === 'always'" class="chip gold">
                        {{ t("setup.thinkingAlways") }}
                      </span>
                      <span v-else class="chip">{{ t("setup.thinkingUnsupported") }}</span>
                    </div>

                    <div
                      v-if="currentModel(index)!.controls.reasoning_effort?.show && (currentModel(index)!.controls.reasoning_effort!.options || []).length"
                      class="mode-row"
                    >
                      <div class="mode-label">
                        <span>{{ t("setup.reasoningEffort") }} <span class="param-en">reasoning_effort</span></span>
                        <span class="param-hint">{{ controlHint(currentModel(index)!.controls.reasoning_effort) }}</span>
                      </div>
                      <div class="segmented effort-switch">
                        <button
                          v-for="option in currentModel(index)!.controls.reasoning_effort!.options || []"
                          :key="option"
                          :class="{ active: paramsOf(index).reasoning_effort === option }"
                          @click="setParam(index, 'reasoning_effort', option)"
                        >
                          {{ option }}
                        </button>
                      </div>
                    </div>

                    <div class="quirk-row">
                      <AppIcon name="zap" :size="11" />
                      <span>{{ modelQuirk(currentModel(index)!) }}</span>
                    </div>

                    <div class="test-row">
                      <button class="btn btn-sm" :disabled="!currentModel(index)!.key_present || isTesting(index)" @click="testConnection(index)">
                        <span v-if="isTesting(index)" class="spinner"></span>
                        <AppIcon v-else name="play" :size="12" />
                        {{ t("setup.testConnection") }}
                      </button>
                      <div v-if="testResult(index)" class="test-result" :class="testResult(index)!.status">
                        <span v-if="testResult(index)!.status === 'testing'" class="spinner"></span>
                        <AppIcon v-else-if="testResult(index)!.status === 'ok'" name="check" :size="12" />
                        <AppIcon v-else name="alert" :size="12" />
                        <span class="test-text">{{ testResult(index)!.text }}</span>
                        <span v-if="testResult(index)!.latency" class="mono">{{ testResult(index)!.latency }}ms</span>
                      </div>
                    </div>
                  </div>
                </template>
                <div v-else class="seat-placeholder">{{ t("setup.selectProviderModel") }}</div>
              </div>

              <!-- Bot -->
              <div v-else class="seat-body">
                <div class="bot-grid">
                  <button
                    v-for="agent in rlcard?.agents || []"
                    :key="agent.id"
                    class="bot-option"
                    :class="{ active: config.bot?.agent_type === agent.id }"
                    @click="setBotType(index, agent.id)"
                  >
                    <span class="bot-icon"><AppIcon :name="agent.id === 'random' ? 'zap' : 'brain'" :size="16" /></span>
                    <span class="bot-text">
                      <strong>{{ agent.name }}</strong>
                      <small>{{ botDescription(agent) }}</small>
                    </span>
                  </button>
                </div>

                <label class="field">
                  <div class="param-head">
                    <span>{{ t("setup.botDelay") }} <span class="param-en">delay</span></span>
                    <strong class="mono">{{ botDelay(index).toFixed(1) }}s</strong>
                  </div>
                  <input
                    class="range"
                    type="range"
                    :min="BOT_DELAY_MIN"
                    :max="BOT_DELAY_MAX"
                    step="0.5"
                    :value="botDelay(index)"
                    :style="{ '--pct': rangePct(botDelay(index), BOT_DELAY_MIN, BOT_DELAY_MAX) }"
                    @input="setBotDelay(index, $event)"
                  />
                  <span class="field-hint">{{ t("setup.botDelayHint") }}</span>
                </label>

                <label v-if="config.bot?.agent_type !== 'random'" class="field">
                  <span class="field-label">{{ t("setup.modelFile") }}</span>
                  <select class="input" :value="config.bot?.model_path || ''" @change="setBotModel(index, ($event.target as HTMLSelectElement).value)">
                    <option value="">{{ t("setup.selectModel", { dir: rlcard?.model_dir || "" }) }}</option>
                    <option v-for="model in botModels(config.bot?.agent_type || 'random')" :key="model.path" :value="model.path">
                      {{ model.name }} · {{ model.size_kb }} KB
                    </option>
                  </select>
                  <span class="field-hint" v-if="!(rlcard?.models || []).length">
                    {{ t("setup.noModels") }}
                  </span>
                </label>

                <div class="bot-note">
                  <AppIcon name="bot" :size="13" />
                  <span>{{ t("setup.botNoReasoning") }}</span>
                </div>

                <div v-if="rlcard && !rlcard.available" class="inline-warning">
                  <AppIcon name="alert" :size="12" />
                  {{ t("setup.rlcardMissing") }}
                </div>
              </div>
            </article>
          </div>
        </section>

        <section class="setup-section">
          <div class="section-head">
            <div>
              <span class="section-kicker">02</span>
              <h3>{{ t("setup.rules") }}</h3>
            </div>
            <span class="section-note">{{ t("setup.godViewNote") }}</span>
          </div>

          <div class="rules-grid">
            <label class="field">
              <span class="field-label">{{ t("setup.startingChips") }}</span>
              <input class="input" type="number" :min="limits.starting_chips?.min || 100" :max="limits.starting_chips?.max || 100000" :value="numericRule('starting_chips')" @input="setNumericRule('starting_chips', $event)" />
            </label>
            <label class="field">
              <span class="field-label">{{ t("setup.blinds") }}</span>
              <div class="blind-row">
                <input class="input" type="number" :min="1" :value="numericRule('small_blind')" @input="setNumericRule('small_blind', $event)" />
                <span class="blind-sep">/</span>
                <input class="input" type="number" :min="2" :value="numericRule('big_blind')" @input="setNumericRule('big_blind', $event)" />
              </div>
            </label>
            <label class="field">
              <span class="field-label">{{ t("setup.maxHands") }} <span class="field-hint-inline">{{ t("setup.unlimited") }}</span></span>
              <input class="input" type="number" min="0" :max="limits.max_hands?.max || 1000" :value="numericRule('max_hands')" @input="setNumericRule('max_hands', $event)" />
            </label>
            <label class="field">
              <span class="field-label">{{ t("setup.seed") }} <span class="field-hint-inline">{{ t("setup.seedHint") }}</span></span>
              <input class="input" type="number" :placeholder="t('setup.seedPlaceholder')" :value="setup.rules.seed ?? ''" @input="setSeed" />
            </label>
          </div>

          <div class="rules-row blind-rules">
            <div class="rule-toggle">
              <div class="rule-text">
                <strong>{{ t("setup.blindIncrease") }}</strong>
                <span>{{ t("setup.blindIncreaseHint") }}</span>
              </div>
              <button
                class="toggle"
                :class="{ on: blindIncreaseEnabled }"
                @click="toggleBlindIncrease"
              ></button>
            </div>
            <label v-if="blindIncreaseEnabled" class="field blind-every-field">
              <span class="field-label">{{ t("setup.blindIncreaseEvery") }}</span>
              <input
                class="input"
                type="number"
                min="1"
                :max="limits.blind_increase_every_hands?.max || 1000"
                :value="numericRule('blind_increase_every_hands')"
                @input="setNumericRule('blind_increase_every_hands', $event)"
              />
            </label>
            <label v-if="blindIncreaseEnabled" class="field blind-factor-field">
              <span class="field-label">{{ t("setup.blindIncreaseFactor") }}</span>
              <select class="input" :value="numericRule('blind_increase_factor')" @change="setBlindFactor">
                <option :value="1.5">1.5×</option>
                <option :value="2">2×</option>
                <option :value="3">3×</option>
              </select>
            </label>
          </div>

          <div class="rules-row">
            <div class="rule-toggle">
              <div class="rule-text">
                <strong>{{ t("setup.autoRebuy") }}</strong>
                <span>{{ t("setup.autoRebuyHint") }}</span>
              </div>
              <button class="toggle" :class="{ on: setup.rules.auto_rebuy }" @click="setRule('auto_rebuy', !setup.rules.auto_rebuy)"></button>
            </div>
            <label v-if="setup.rules.auto_rebuy" class="field rebuy-field">
              <span class="field-label">{{ t("setup.rebuyAmount") }}</span>
              <input class="input" type="number" :min="limits.rebuy_amount?.min || 100" :max="limits.rebuy_amount?.max || 100000" :value="numericRule('rebuy_amount')" @input="setNumericRule('rebuy_amount', $event)" />
            </label>
            <div class="rule-toggle">
              <div class="rule-text">
                <strong>{{ t("setup.godView") }}</strong>
                <span>{{ t("setup.godViewHint") }}</span>
              </div>
              <button class="toggle" :class="{ on: setup.rules.god_view }" @click="setRule('god_view', !setup.rules.god_view)"></button>
            </div>
            <div class="rule-toggle" :class="{ disabled: !canTableTalk }">
              <div class="rule-text">
                <strong>{{ t("setup.tableTalk") }}</strong>
                <span>
                  {{ canTableTalk ? t("setup.tableTalkHint") : t("setup.tableTalkUnavailable") }}
                </span>
              </div>
              <button
                class="toggle"
                :class="{ on: setup.table_talk }"
                :disabled="!canTableTalk"
                @click="toggleTableTalk"
              ></button>
            </div>
          </div>

          <div v-if="setup.seats.some((item) => item.kind === 'llm')" class="rule-toggle agent-language">
            <div class="rule-text">
              <strong>{{ t("setup.agentLanguage") }}</strong>
              <span>{{ t("setup.agentLanguageHint") }}</span>
              <span class="field-hint-inline">{{ t("setup.agentLanguageResolved", { language: resolvedAgentLanguage }) }}</span>
            </div>
            <div class="segmented effort-switch">
              <button :class="{ active: setup.agent_locale === 'auto' }" @click="setAgentLocale('auto')">
                {{ t("setup.agentLanguageAuto") }}
              </button>
              <button :class="{ active: setup.agent_locale === 'zh-CN' }" @click="setAgentLocale('zh-CN')">
                {{ t("setup.agentLanguageZh") }}
              </button>
              <button :class="{ active: setup.agent_locale === 'en-US' }" @click="setAgentLocale('en-US')">
                {{ t("setup.agentLanguageEn") }}
              </button>
            </div>
          </div>

          <div class="blind-presets">
            <span class="faint">{{ t("setup.quickBlinds") }}</span>
            <button class="chip" @click="setBlinds(5, 10)">5 / 10</button>
            <button class="chip" @click="setBlinds(10, 20)">10 / 20</button>
            <button class="chip" @click="setBlinds(25, 50)">25 / 50</button>
          </div>
        </section>
      </div>

      <footer class="modal-footer">
        <div class="footer-note">
          <span v-if="validation" class="validation error">
            <AppIcon name="alert" :size="12" />
            {{ validation }}
          </span>
          <span v-else class="validation ok">
            <AppIcon name="shield" :size="12" />
            {{ t("setup.valid") }}
          </span>
          <span class="footer-hint">
            <template v-if="game.snapshot?.has_match">{{ t("setup.willEndMatch") }}</template>
            {{ t("setup.shortcuts") }}
          </span>
        </div>
        <div class="footer-actions">
          <button class="btn" :disabled="busy" @click="close">{{ t("common.cancel") }}</button>
          <button class="btn btn-primary start-btn" :disabled="!canStart" @click="submit">
            <span v-if="busy" class="spinner"></span>
            <AppIcon v-else name="play" :size="14" />
            {{ busy ? t("setup.starting") : t("setup.start") }}
          </button>
        </div>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.setup-modal {
  width: min(1120px, 100%);
  max-height: min(960px, calc(100vh - 28px));
}

.modal-head-tools {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: none;
}

.agent-language .rule-text {
  flex: 1;
}

@media (max-width: 720px) {
  .agent-language {
    align-items: flex-start;
    flex-direction: column;
  }
}

.setup-body { gap: 16px; }

.preset-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.022);
}

.preset-label {
  font-size: calc(9px * var(--type-scale));
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--faint);
  margin-right: 4px;
}

.preset {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 9px;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-soft);
  font-size: calc(10px * var(--type-scale));
  font-weight: 600;
}

.preset:hover { border-color: rgba(232, 195, 122, 0.3); color: var(--gold); }

.preset-hint { margin-left: auto; font-size: calc(9px * var(--type-scale)); color: var(--faint); }

.setup-section { display: flex; flex-direction: column; gap: 10px; }

.section-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}

.section-head > div { display: flex; align-items: baseline; gap: 8px; }

.section-kicker {
  font-family: var(--font-mono);
  font-size: calc(9px * var(--type-scale));
  color: var(--gold);
  border: 1px solid rgba(232, 195, 122, 0.25);
  border-radius: 4px;
  padding: 1px 5px;
}

.section-head h3 { margin: 0; font-size: calc(14px * var(--type-scale)); font-weight: 750; letter-spacing: -0.01em; }
.section-note { font-size: calc(9.5px * var(--type-scale)); color: var(--faint); }

.seat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.seat-config {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.018);
  overflow: hidden;
}

.seat-config.human { border-color: rgba(84, 214, 168, 0.16); }

.seat-config-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.022);
}

.seat-label { display: flex; align-items: center; gap: 6px; }

.seat-tag {
  font-family: var(--font-mono);
  font-size: calc(8px * var(--type-scale));
  font-weight: 800;
  letter-spacing: 0.12em;
  color: var(--gold);
  border: 1px solid rgba(232, 195, 122, 0.28);
  border-radius: 4px;
  padding: 1px 5px;
}

.seat-position { font-size: calc(10px * var(--type-scale)); color: var(--faint); }

.kind-switch button { padding: 4px 8px; font-size: calc(10px * var(--type-scale)); }

.seat-body {
  padding: 9px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 200px;
}

.human-body {
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 18px 22px;
  min-height: 260px;
}

.human-orb {
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  border: 1px solid rgba(84, 214, 168, 0.24);
  background: radial-gradient(circle at 35% 28%, rgba(84, 214, 168, 0.18), transparent 60%);
  color: var(--emerald);
  margin-bottom: 2px;
}

.human-body h4 { margin: 0; font-size: calc(14px * var(--type-scale)); font-weight: 750; }
.human-body p { margin: 0; max-width: 280px; font-size: calc(10.5px * var(--type-scale)); line-height: 1.65; color: var(--muted); }
.human-features { display: flex; gap: 5px; flex-wrap: wrap; justify-content: center; margin-top: 3px; }

.human-quote {
  margin: 10px 0 0;
  padding: 9px 12px;
  max-width: 300px;
  border-left: 2px solid rgba(232, 195, 122, 0.3);
  background: linear-gradient(90deg, rgba(232, 195, 122, 0.045), transparent);
  color: var(--faint);
  font-size: calc(9.5px * var(--type-scale));
  line-height: 1.7;
  text-align: left;
  border-radius: 0 8px 8px 0;
}

.provider-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px;
}

.provider-button {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 6px 4px;
  border-radius: 9px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.022);
  transition: all 0.15s ease;
  min-width: 0;
}

.provider-button:hover { border-color: var(--line-strong); background: rgba(255, 255, 255, 0.045); }

.provider-button.active {
  border-color: color-mix(in srgb, var(--provider-accent) 60%, transparent);
  background: color-mix(in srgb, var(--provider-accent) 10%, transparent);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--provider-accent) 20%, transparent) inset;
}

.provider-button.missing { opacity: 0.62; }

.logo-tile {
  display: grid;
  place-items: center;
  width: 27px;
  height: 27px;
  border-radius: 7px;
  background: #f2ede1;
  flex: none;
  overflow: hidden;
}

.logo-tile img { width: 19px; height: 19px; object-fit: contain; }
.provider-name { font-size: calc(9px * var(--type-scale)); font-weight: 650; color: var(--text-soft); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }

.key-dot {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
}

.key-dot.ok { background: var(--emerald); box-shadow: 0 0 6px rgba(84, 214, 168, 0.5); }
.key-dot.missing { background: var(--red); }

.model-card {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 8px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: rgba(4, 12, 9, 0.35);
}

.model-head { display: flex; align-items: center; gap: 8px; }
.model-icon { width: 30px; height: 30px; }
.model-icon img { width: 21px; height: 21px; }
.model-title { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.model-title strong { font-size: calc(11.5px * var(--type-scale)); font-weight: 750; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-title span { font-size: calc(8.5px * var(--type-scale)); color: var(--faint); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-specs { margin-left: auto; display: flex; gap: 3px; flex-wrap: wrap; justify-content: flex-end; }
.model-specs .chip { font-size: calc(7px * var(--type-scale)); padding: 1px 4px; }

.key-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

.key-state {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
}

.key-state.ok { color: var(--emerald); }
.key-state.missing { color: var(--red); }

.docs-link { margin-left: auto; display: inline-flex; align-items: center; gap: 3px; font-size: calc(9px * var(--type-scale)); color: var(--faint); }
.docs-link:hover { color: var(--gold); }

.param-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px 12px;
}

.param { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.param.span-2 { grid-column: span 2; }

.param-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: calc(9.5px * var(--type-scale));
  color: var(--muted);
}

.param-head strong { font-size: calc(9.5px * var(--type-scale)); color: var(--gold); }
.param-en { font-family: var(--font-mono); font-size: calc(7.5px * var(--type-scale)); color: var(--faint); margin-left: 3px; }
.param-hint { font-size: calc(8px * var(--type-scale)); line-height: 1.45; color: var(--faint); }

.mode-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 6px 0;
  border-top: 1px solid var(--line);
}

.mode-label { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.mode-label > span:first-child { font-size: calc(9.5px * var(--type-scale)); font-weight: 650; color: var(--muted); }
.effort-switch { flex: none; }
.effort-switch button { font-size: calc(8.5px * var(--type-scale)); padding: 3px 7px; }

.quirk-row {
  display: flex;
  align-items: flex-start;
  gap: 5px;
  font-size: calc(8.5px * var(--type-scale));
  line-height: 1.5;
  color: var(--faint);
  padding-top: 2px;
}

.test-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding-top: 2px; }
.test-result { display: inline-flex; align-items: center; gap: 4px; font-size: calc(8.5px * var(--type-scale)); min-width: 0; }
.test-result.ok { color: var(--emerald); }
.test-result.error { color: var(--red); }
.test-result.testing { color: var(--muted); }
.test-text { max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.bot-grid { display: flex; flex-direction: column; gap: 5px; }

.bot-option {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 7px 8px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.02);
  text-align: left;
}

.bot-option:hover { border-color: var(--line-strong); }
.bot-option.active { border-color: rgba(121, 169, 255, 0.4); background: var(--blue-dim); }
.bot-icon { color: var(--blue); display: flex; margin-top: 1px; }
.bot-text { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.bot-text strong { font-size: calc(10.5px * var(--type-scale)); font-weight: 700; }
.bot-text small { font-size: calc(8.5px * var(--type-scale)); line-height: 1.45; color: var(--faint); }

.bot-note {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: calc(8.5px * var(--type-scale));
  line-height: 1.5;
  color: var(--faint);
  padding: 7px 8px;
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px dashed var(--line);
}

.inline-warning {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: calc(9px * var(--type-scale));
  color: var(--red);
}

.seat-placeholder { font-size: calc(10px * var(--type-scale)); color: var(--faint); padding: 14px; text-align: center; }

.rules-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.blind-row { display: flex; align-items: center; gap: 5px; }
.blind-sep { color: var(--faint); font-size: calc(11px * var(--type-scale)); }
.field-hint-inline { font-size: calc(8px * var(--type-scale)); color: var(--faint); font-weight: 400; }

.rules-row {
  display: flex;
  align-items: center;
  gap: 18px;
  flex-wrap: wrap;
  padding: 9px 10px;
  border-radius: 9px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.018);
}

.rule-toggle { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 220px; }
.rule-toggle.disabled { opacity: 0.55; }
.rule-toggle .toggle:disabled { cursor: not-allowed; }
.rule-text { display: flex; flex-direction: column; gap: 1px; }
.rule-text strong { font-size: calc(10.5px * var(--type-scale)); font-weight: 700; }
.rule-text span { font-size: calc(8.5px * var(--type-scale)); color: var(--faint); line-height: 1.45; }
.rebuy-field { width: 130px; flex: none; }
.blind-rules { margin-top: 10px; }
.blind-every-field { width: 132px; flex: none; }
.blind-factor-field { width: 108px; flex: none; }

.blind-presets { display: flex; align-items: center; gap: 5px; font-size: calc(9px * var(--type-scale)); }
.blind-presets .chip { cursor: pointer; }
.blind-presets .chip:hover { color: var(--gold); border-color: rgba(232, 195, 122, 0.3); }

.footer-note { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.validation { display: inline-flex; align-items: center; gap: 5px; font-size: calc(10px * var(--type-scale)); }
.validation.ok { color: var(--emerald); }
.validation.error { color: var(--red); }
.footer-hint { font-size: calc(8.5px * var(--type-scale)); color: var(--faint); }
.footer-actions { display: flex; align-items: center; gap: 8px; flex: none; }
.start-btn { min-width: 128px; }

@media (max-width: 980px) {
  .seat-grid { grid-template-columns: minmax(0, 1fr); }
  .rules-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 640px) {
  .provider-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .rules-grid { grid-template-columns: minmax(0, 1fr); }
  .param-grid { grid-template-columns: minmax(0, 1fr); }
  .param.span-2 { grid-column: span 1; }
  .preset-hint { display: none; }
  .footer-note { display: none; }
  .footer-actions { width: 100%; justify-content: flex-end; }
}
</style>
