<script setup lang="ts">
import { computed, ref } from "vue";
import AppIcon from "./AppIcon.vue";
import { actionLabel, chips, compactPercent, duration, signedChips } from "../format";
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
} from "../historyReport";
import { handCategoryLabel, handCategoryShort, toolLabel } from "../i18n/labels";
import { t, type MessageKey } from "../i18n";
import type { ToolView } from "../types";

const props = defineProps<{ tool: ToolView }>();

const isEstimate = computed(() => props.tool.name === "estimate_hand_probabilities");
const isEquity = computed(() => props.tool.name === "estimate_equity");
const isAction = computed(() => props.tool.name === "apply_action");
const isRandom = computed(() => props.tool.name === "get_random_number");
const isCallEv = computed(() => props.tool.name === "call_ev");
const isTalk = computed(() => props.tool.name === "send_table_message");
const isHistory = computed(() => props.tool.name === "research_history");
const parsed = computed<Record<string, unknown>>(() => {
  if (props.tool.parsed) return props.tool.parsed;
  try {
    const value = JSON.parse(props.tool.argsText || "{}");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
});
const result = computed<Record<string, any>>(() => props.tool.result || {});
const ok = computed(() => props.tool.status === "done" && result.value.ok !== false);
const ignored = computed(() => Boolean(props.tool.ignored));
const historyCurrent = computed<Record<string, any>>(() => {
  const current = result.value.current_match;
  return current && typeof current === "object" ? (current as Record<string, any>) : {};
});
const historyHands = computed(() => Number(historyCurrent.value.hands || 0));
const historyBlinds = computed<{ small?: number; big?: number } | null>(() => {
  const value = historyCurrent.value.blinds;
  return value && typeof value === "object" ? (value as { small?: number; big?: number }) : null;
});
const historyStacks = computed<{ you?: number; opponent?: number } | null>(() => {
  const value = historyCurrent.value.stacks;
  return value && typeof value === "object" ? (value as { you?: number; opponent?: number }) : null;
});
const historyPositions = computed<Record<string, any> | null>(() => {
  const value = historyCurrent.value.positions;
  return value && typeof value === "object" ? (value as Record<string, any>) : null;
});
const historyPosition = ref<"all" | "button" | "bb">("all");
const historyPositionOptions = ["all", "button", "bb"] as const;
const historySource = computed<Record<string, any>>(() => {
  if (historyPosition.value === "all" || !historyPositions.value) return historyCurrent.value;
  const you = historyPositions.value.you?.[historyPosition.value];
  const opponent = historyPositions.value.opponent?.[historyPosition.value];
  return {
    ...historyCurrent.value,
    you: you && typeof you === "object" ? you : {},
    opponent: opponent && typeof opponent === "object" ? opponent : {},
  };
});
const historyGroups = computed(() => buildHistoryGroups(historySource.value));
const historyRecent = computed(() => recentHistoryHands(historyCurrent.value));
const historyCompleted = computed(() => completedHistory(result.value));
const historyScope = computed(() => historyScopeLabel(String(result.value.scope || "both")));
const historyFocus = computed(() => {
  const focus = String(parsed.value.focus || "all").trim().toLowerCase();
  return focus && focus !== "all" ? historyGroupLabel(focus) : "";
});
const randomValue = computed<number | null>(() => {
  const raw = result.value.value;
  const value = Number(raw);
  return raw !== undefined && raw !== null && Number.isFinite(value) ? value : null;
});
const randomPercent = computed(() => `${Math.min(100, Math.max(0, (randomValue.value ?? 0) * 100))}%`);
const distribution = computed<any[]>(() => {
  const value = result.value.distribution;
  return Array.isArray(value) ? value : [];
});
const currentBest = computed(() => result.value.current_best || null);
const method = computed(() => result.value.method || "");
const samples = computed(() => Number(result.value.total || result.value.samples || 0));
const elapsed = computed(() => Number(result.value.elapsed_ms || 0));
const bestKey = computed(() => currentBest.value?.category || "");
const equityRate = computed(() => Number(result.value.equity || 0));
const winRate = computed(() => Number(result.value.win || 0));
const tieRate = computed(() => Number(result.value.tie || 0));
const loseRate = computed(() => Number(result.value.lose || 0));
const stdError = computed(() => Number(result.value.std_error || 0));
const evCall = computed(() => Number(result.value.ev_call || 0));
const breakeven = computed(() => Number(result.value.breakeven_equity || 0));
const callAmount = computed(() => Number(result.value.call_amount || 0));
const effectivePot = computed(() => Number(result.value.effective_pot || 0));
const equitySource = computed(() => String(result.value.equity_source || "internal"));
const equityMethod = computed(() => String(result.value.equity_method || result.value.method || ""));
const resultStreet = computed(() => {
  const street = String(result.value.street || "");
  if (!street) return "";
  return t(`tool.street.${street}` as MessageKey);
});
const action = computed(() => String(parsed.value.action || result.value.action || ""));
const amount = computed(() => Number(parsed.value.amount || result.value.amount || 0));
const errorText = computed(() => String(result.value.error || props.tool.summary || ""));
const talkText = computed(() => String(
  parsed.value.text
  || result.value.entry?.text
  || result.value.text
  || "",
));
const inlineTalk = computed(() => String(
  result.value.table_talk?.text
  || result.value.table_talk
  || "",
));
const talkAccepted = computed(() => props.tool.status !== "error" && result.value.ok !== false);
const constraint = computed(() => {
  if (!isAction.value || !result.value || result.value.ok !== false) return null;
  if (result.value.legal_actions === undefined && result.value.to_call === undefined) return null;
  return result.value as Record<string, unknown>;
});
const legalActions = computed<string[]>(() => {
  const value = constraint.value?.legal_actions;
  return Array.isArray(value) ? (value as string[]) : [];
});
const statusLabel = computed(() => {
  if (ignored.value) return t("tool.status.ignored");
  if (props.tool.status === "running") return t("tool.status.running");
  if (props.tool.status === "error") return t("tool.status.error");
  return t("tool.status.done");
});
const variant = computed(() => {
  if (isEstimate.value) return "estimate";
  if (isEquity.value) return "equity";
  if (isAction.value) return "action";
  if (isRandom.value) return "random";
  if (isCallEv.value) return "call-ev";
  if (isTalk.value) return "talk";
  if (isHistory.value) return "history";
  return "generic";
});
const icon = computed(() =>
  isHistory.value
    ? "history"
    : isEstimate.value || isEquity.value
      ? "chart"
      : isAction.value
        ? "zap"
        : isRandom.value
          ? "dice"
          : isCallEv.value
            ? "coins"
            : isTalk.value
              ? "message"
              : "wrench",
);
const title = computed(() =>
  isHistory.value
    ? t("tool.title.history")
    : isEstimate.value
      ? t("tool.title.estimate")
      : isEquity.value
        ? t("tool.title.equity")
        : isAction.value
          ? t("tool.title.action")
          : isRandom.value
            ? t("tool.title.random")
            : isCallEv.value
              ? t("tool.title.callEv")
              : isTalk.value
                ? t("tool.title.talk")
                : props.tool.name || t("tool.title.generic"),
);
const detail = computed(() => {
  if (isHistory.value) {
    if (props.tool.status === "running") return t("tool.detail.calculating");
    const vpip = historyCurrent.value.opponent?.preflop?.vpip;
    if (vpip && Number.isFinite(Number(vpip.value))) {
      return t("tool.detail.history", {
        hands: historyHands.value,
        value: percent(Number(vpip.value)),
        n: Number(vpip.n || 0),
      });
    }
    const net = historyCurrent.value.you?.result?.net;
    if (typeof net === "number" && Number.isFinite(net)) {
      return t("tool.detail.historyNet", {
        hands: historyHands.value,
        value: signedChips(net),
      });
    }
    if (historyCompleted.value) {
      return t("tool.detail.historyCompleted", {
        matches: historyCompleted.value.matches,
        hands: historyCompleted.value.hands,
      });
    }
    return t("tool.detail.historyEmpty", { hands: historyHands.value });
  }
  if (isEstimate.value || isEquity.value) {
    if (method.value === "exact") {
      return t("tool.detail.exact", { count: samples.value.toLocaleString() });
    }
    if (method.value === "monte_carlo") {
      return t("tool.detail.monteCarlo", { count: samples.value.toLocaleString() });
    }
    return t("tool.detail.calculating");
  }
  if (isAction.value) {
    const label = actionLabel(action.value) || action.value || t("tool.title.generic");
    return amount.value
      ? t("tool.detail.action", { action: label, amount: chips(amount.value) })
      : label;
  }
  if (isRandom.value) {
    return randomValue.value !== null ? randomValue.value.toFixed(4) : t("tool.detail.random");
  }
  if (isCallEv.value) {
    if (result.value.applicable === false) return t("tool.detail.callEvNa");
    return t("tool.detail.callEv", { value: `${evCall.value > 0 ? "+" : ""}${evCall.value.toFixed(1)}` });
  }
  if (isTalk.value) {
    if (props.tool.status === "running") return t("tool.detail.sending");
    return talkAccepted.value ? t("tool.detail.sent") : t("tool.detail.rejected");
  }
  return props.tool.name || "";
});

const bestLabel = computed(() => handCategoryLabel(bestKey.value, currentBest.value?.category_name || ""));
const methodLabel = computed(() => {
  if (method.value === "exact") return t("tool.method.exact");
  if (method.value === "monte_carlo") return t("tool.method.monteCarlo");
  return t("tool.method.calculating");
});
const argsPretty = computed(() => {
  if (props.tool.parsed) return JSON.stringify(props.tool.parsed, null, 2);
  if (props.tool.argsText) {
    try {
      return JSON.stringify(JSON.parse(props.tool.argsText), null, 2);
    } catch {
      return props.tool.argsText;
    }
  }
  return "{}";
});
const resultPretty = computed(() => {
  try {
    return JSON.stringify(result.value, null, 2);
  } catch {
    return String(result.value);
  }
});
const copied = ref<string>("");

async function copy(text: string, key: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    copied.value = key;
    window.setTimeout(() => {
      if (copied.value === key) copied.value = "";
    }, 1400);
  } catch {
    copied.value = "";
  }
}

function barWidth(item: any): string {
  const value = Number(item?.probability || 0);
  return `${Math.max(value > 0 ? 2 : 0, Math.min(100, value * 100))}%`;
}

function pctWidth(value: number): string {
  if (!Number.isFinite(value)) return "0%";
  return `${Math.max(0, Math.min(100, value * 100))}%`;
}

function percent(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}
</script>

<template>
  <details class="tool-card" :class="[variant, tool.status]" open>
    <summary class="tool-head">
      <span class="tool-icon"><AppIcon :name="icon" :size="13" /></span>
      <span class="tool-title">{{ title }}</span>
      <span class="tool-detail">{{ detail }}</span>
      <span class="tool-status" :class="tool.status">
        <span v-if="tool.status === 'running'" class="spinner"></span>
        <AppIcon v-else-if="tool.status === 'error'" name="alert" :size="12" />
        <AppIcon v-else-if="ignored" name="x" :size="12" />
        <AppIcon v-else name="check" :size="12" />
        <span>{{ statusLabel }}</span>
        <span v-if="tool.durationMs" class="mono duration">{{ duration(tool.durationMs) }}</span>
      </span>
      <AppIcon name="chevronDown" :size="12" class="tool-chevron" />
    </summary>

    <div class="tool-body">
      <!-- Estimate result -->
      <div v-if="isEstimate" class="estimate-block">
        <div v-if="tool.status === 'error'" class="generic-error">
          <AppIcon name="alert" :size="13" />
          {{ errorText || t("tool.error.estimate") }}
        </div>
        <div class="estimate-meta">
          <span class="chip gold">{{ methodLabel }}</span>
          <span v-if="samples" class="mono">
            {{ samples.toLocaleString() }} {{ method === "exact" ? t("tool.unit.combos") : t("tool.unit.samples") }}
          </span>
          <span v-if="elapsed" class="mono faint">{{ elapsed.toFixed(1) }}ms</span>
          <span v-if="currentBest" class="best-inline">
            {{ t("tool.best") }} <strong>{{ bestLabel }}</strong>
          </span>
          <span v-else class="faint">{{ t("tool.preflopNoHand") }}</span>
        </div>

        <div v-if="distribution.length" class="dist">
          <div
            v-for="item in distribution"
            :key="item.key"
            class="dist-row"
            :class="{ best: bestKey === item.key }"
          >
            <span class="dist-name">{{ handCategoryShort(item.key, item.short || item.name) }}</span>
            <div class="dist-track">
              <div
                class="dist-fill"
                :style="{ width: barWidth(item), background: item.color }"
              ></div>
            </div>
            <span class="dist-pct mono">{{ compactPercent(item.probability) }}</span>
          </div>
        </div>
        <div v-else-if="tool.status === 'running'" class="dist-skeleton">
          <div v-for="n in 5" :key="n" class="skeleton-row">
            <span class="skeleton-name"></span>
            <span class="skeleton-bar" :style="{ width: `${70 - n * 8}%` }"></span>
          </div>
        </div>
        <div v-else class="empty-result">{{ t("tool.emptyDistribution") }}</div>
      </div>

      <!-- Equity result -->
      <div v-else-if="isEquity" class="equity-block">
        <div v-if="tool.status === 'error'" class="generic-error">
          <AppIcon name="alert" :size="13" />
          {{ errorText || t("tool.error.equity") }}
        </div>
        <template v-else>
          <div class="equity-readout">
            <div class="equity-main">
              <span class="equity-value mono">{{ percent(equityRate) }}</span>
              <span class="equity-caption">{{ t("tool.equityCaption") }}</span>
            </div>
            <div class="equity-meta">
              <span class="chip gold">{{ methodLabel }}</span>
              <span v-if="samples" class="mono faint">
                {{ samples.toLocaleString() }} {{ method === "exact" ? t("tool.unit.combos") : t("tool.unit.samples") }}
              </span>
              <span v-if="result.cached" class="chip">{{ t("tool.cached") }}</span>
            </div>
          </div>
          <div class="equity-bar">
            <span class="equity-seg win" :style="{ width: pctWidth(winRate) }"></span>
            <span class="equity-seg tie" :style="{ width: pctWidth(tieRate) }"></span>
            <span class="equity-seg lose" :style="{ width: pctWidth(loseRate) }"></span>
          </div>
          <div class="equity-legend">
            <span><i class="dot win"></i>{{ t("tool.win", { value: percent(winRate) }) }}</span>
            <span><i class="dot tie"></i>{{ t("tool.tie", { value: percent(tieRate) }) }}</span>
            <span><i class="dot lose"></i>{{ t("tool.lose", { value: percent(loseRate) }) }}</span>
            <span class="faint">{{ t("tool.opponentRandom", { street: resultStreet }) }}</span>
          </div>
          <div v-if="stdError > 0" class="equity-note">
            {{ t("tool.samplingError", { value: percent(stdError * 1.96) }) }}
          </div>
        </template>
      </div>

      <!-- Call EV result -->
      <div v-else-if="isCallEv" class="callev-block">
        <div v-if="tool.status === 'error'" class="generic-error">
          <AppIcon name="alert" :size="13" />
          {{ errorText || t("tool.error.callEv") }}
        </div>
        <div v-else-if="result.applicable === false" class="callev-na">
          <AppIcon name="alert" :size="13" />
          <span>{{ result.reason || t("tool.detail.callEvNa") }}</span>
        </div>
        <template v-else>
          <div class="callev-readout">
            <span class="callev-value mono" :class="{ positive: evCall > 0, negative: evCall < 0 }">
              {{ evCall > 0 ? "+" : "" }}{{ evCall.toFixed(1) }}
            </span>
            <span class="callev-caption">{{ t("tool.title.callEv") }}</span>
            <span class="chip" :class="evCall > 0 ? 'gold' : evCall < 0 ? 'red' : ''">
              {{ evCall > 0 ? t("tool.positiveEv") : evCall < 0 ? t("tool.negativeEv") : t("tool.breakeven") }}
            </span>
            <span class="chip">{{ equitySource === "model" ? t("tool.fromModel") : t("tool.internalEstimate") }}</span>
            <span v-if="resultStreet" class="chip">{{ resultStreet }}</span>
          </div>
          <div class="callev-grid">
            <span>{{ t("tool.pot") }} <strong class="mono">{{ chips(Number(result.pot || 0)) }}</strong></span>
            <span>{{ t("tool.call") }} <strong class="mono">{{ chips(Number(callAmount)) }}</strong></span>
            <span>{{ t("tool.effectivePot") }} <strong class="mono">{{ chips(Number(effectivePot)) }}</strong></span>
            <span>{{ t("tool.breakevenEquity") }} <strong class="mono">{{ percent(breakeven) }}</strong></span>
            <span>{{ t("tool.yourEquity") }} <strong class="mono">{{ percent(equityRate) }}</strong></span>
            <span>{{ t("tool.potOdds") }} <strong class="mono">{{ result.pot_to_call_ratio ? `${Number(result.pot_to_call_ratio).toFixed(2)} : 1` : t("common.none") }}</strong></span>
          </div>
          <div class="callev-compare">
            <div class="compare-track">
              <span class="compare-fill" :style="{ width: pctWidth(equityRate) }"></span>
              <span class="compare-marker" :style="{ left: pctWidth(breakeven) }"></span>
            </div>
            <div class="compare-scale mono">
              <span>0</span>
              <span>{{ t("tool.breakevenShort", { value: percent(breakeven) }) }}</span>
              <span>100%</span>
            </div>
          </div>
          <div v-if="equityMethod && equityMethod !== 'model'" class="callev-note mono">
            {{ equityMethod === "exact" ? t("tool.exact") : t("tool.sampled") }} {{ Number(result.equity_samples || 0).toLocaleString() }}
          </div>
          <div class="callev-note">{{ result.note || t("tool.staticNote") }}</div>
        </template>
      </div>

      <!-- Action result -->
      <div v-else-if="isAction" class="action-block">
        <div v-if="tool.status === 'running'" class="running-line">
          <span class="spinner"></span>
          <span>{{ t("tool.submitting") }}</span>
        </div>
        <div v-else-if="ok && result.ok !== false" class="action-ok">
          <span class="action-pill" :class="action">{{ actionLabel(action) || action }}</span>
          <span v-if="amount" class="mono amount">→ {{ chips(amount) }}</span>
          <span class="ok-text"><AppIcon name="check" :size="12" /> {{ t("tool.actionAccepted") }}</span>
        </div>
        <div v-else class="action-fail">
          <div class="fail-line">
            <AppIcon name="alert" :size="13" />
            <span>{{ errorText || t("tool.error.action") }}</span>
          </div>
          <div v-if="constraint" class="constraint-grid">
            <span v-if="constraint.to_call !== undefined" class="chip">{{ t("tool.toCall", { amount: chips(Number(constraint.to_call)) }) }}</span>
            <span v-if="constraint.min_raise_to !== undefined" class="chip">{{ t("tool.minRaise", { amount: chips(Number(constraint.min_raise_to)) }) }}</span>
            <span v-if="constraint.max_raise_to !== undefined" class="chip">{{ t("tool.max", { amount: chips(Number(constraint.max_raise_to)) }) }}</span>
            <span v-if="constraint.pot !== undefined" class="chip">{{ t("tool.pot") }} <strong class="mono">{{ chips(Number(constraint.pot)) }}</strong></span>
          </div>
          <div v-if="legalActions.length" class="legal-row">
            <span class="faint">{{ t("tool.legalActions") }}</span>
            <span v-for="item in legalActions" :key="item" class="chip">{{ actionLabel(item) }}</span>
          </div>
        </div>
        <div v-if="inlineTalk" class="inline-talk">
          <div class="talk-quote">{{ t("tool.talkQuote", { text: inlineTalk }) }}</div>
        </div>
      </div>

      <!-- Random number result -->
      <div v-else-if="isRandom" class="random-block">
        <div v-if="tool.status === 'error'" class="generic-error">
          <AppIcon name="alert" :size="13" />
          {{ errorText || t("tool.error.random") }}
        </div>
        <template v-else>
          <div class="random-readout">
            <span class="random-value mono">{{ randomValue !== null ? randomValue.toFixed(4) : t("common.none") }}</span>
            <span class="random-unit">{{ t("tool.randomUniform") }}</span>
          </div>
          <template v-if="randomValue !== null">
            <div class="random-track">
              <span class="random-marker" :style="{ left: randomPercent }"></span>
            </div>
            <div class="random-scale mono">
              <span>0</span>
              <span>0.5</span>
              <span>1</span>
            </div>
          </template>
        </template>
      </div>

      <!-- Table talk result -->
      <div v-else-if="isTalk" class="talk-block">
        <div v-if="!talkAccepted" class="generic-error">
          <AppIcon name="alert" :size="13" />
          {{ errorText || t("tool.error.talk") }}
        </div>
        <template v-else>
          <div class="talk-quote">{{ t("tool.talkQuote", { text: talkText || t("tool.emptyMessage") }) }}</div>
          <div class="talk-meta-row">
            <span v-if="tool.status === 'running'" class="chip">{{ t("tool.detail.sending") }}</span>
            <template v-else>
              <span class="chip gold">{{ t("tool.talkDone") }}</span>
              <span class="chip">{{ t("tool.talkSeen") }}</span>
            </template>
          </div>
        </template>
      </div>

      <!-- Research history result -->
      <div v-else-if="isHistory" class="history-block">
        <div v-if="tool.status === 'error'" class="generic-error">
          <AppIcon name="alert" :size="13" />
          {{ errorText || t("tool.error.generic") }}
        </div>
        <div v-else-if="tool.status === 'running'" class="dist-skeleton">
          <div v-for="n in 4" :key="n" class="skeleton-row">
            <span class="skeleton-name"></span>
            <span class="skeleton-bar" :style="{ width: `${70 - n * 8}%` }"></span>
          </div>
        </div>
        <template v-else>
          <div class="history-meta">
            <span class="chip gold">{{ historyScope }}</span>
            <span class="mono">{{ t("tool.history.hands", { count: historyHands }) }}</span>
            <span v-if="historyBlinds" class="mono faint">
              {{ t("tool.history.blinds", { sb: chips(historyBlinds.small), bb: chips(historyBlinds.big) }) }}
            </span>
            <span v-if="historyStacks" class="mono faint">
              {{ t("tool.history.stacks", { you: chips(historyStacks.you), opponent: chips(historyStacks.opponent) }) }}
            </span>
            <span v-if="historyFocus" class="chip">{{ t("tool.history.focus", { group: historyFocus }) }}</span>
          </div>

          <div v-if="historyPositions && historyHands" class="history-positions">
            <button
              v-for="position in historyPositionOptions"
              :key="position"
              type="button"
              class="history-position"
              :class="{ active: historyPosition === position }"
              @click="historyPosition = position"
            >{{ historyPositionLabel(position) }}</button>
          </div>

          <div v-if="historyGroups.length" class="history-grid">
            <div class="history-row history-head-row">
              <span></span>
              <span>{{ t("tool.history.you") }}</span>
              <span>{{ t("tool.history.opponent") }}</span>
            </div>
            <template v-for="group in historyGroups" :key="group.key">
              <div class="history-row history-group-row">
                <span class="history-group">{{ group.label }}</span>
              </div>
              <div v-for="metric in group.metrics" :key="metric.key" class="history-row">
                <span class="history-name" :title="metric.label">{{ metric.label }}</span>
                <span class="history-cell" :class="{ empty: metric.you.value === null }">
                  <span class="history-track">
                    <span class="history-fill you" :style="{ width: historyBarWidth(metric.you) }"></span>
                  </span>
                  <span class="history-value mono">{{ formatHistoryPercent(metric.you) }}</span>
                  <span
                    v-if="metric.you.n"
                    class="history-n mono"
                    :class="historyConfidenceClass(metric.you)"
                  >n={{ metric.you.n }}</span>
                </span>
                <span class="history-cell" :class="{ empty: metric.opponent.value === null }">
                  <span class="history-track">
                    <span class="history-fill opponent" :style="{ width: historyBarWidth(metric.opponent) }"></span>
                  </span>
                  <span class="history-value mono">{{ formatHistoryPercent(metric.opponent) }}</span>
                  <span
                    v-if="metric.opponent.n"
                    class="history-n mono"
                    :class="historyConfidenceClass(metric.opponent)"
                  >n={{ metric.opponent.n }}</span>
                </span>
              </div>
            </template>
          </div>
          <div v-else class="empty-result">{{ t("tool.history.noData") }}</div>

          <div v-if="historyRecent.length" class="history-recent">
            <span class="history-recent-label">{{ t("tool.history.recent") }}</span>
            <span
              v-for="hand in historyRecent"
              :key="hand.handId"
              class="history-hand"
              :class="{ won: hand.won }"
            >
              <span class="mono">#{{ hand.handId }}</span>
              <span class="mono">{{ signedChips(hand.net) }}</span>
            </span>
          </div>

          <div v-if="historyCompleted" class="history-completed">
            <span>
              {{
                historyCompleted.vsCurrent
                  ? t("tool.history.completedVs", {
                      matches: historyCompleted.matches,
                      hands: historyCompleted.hands,
                    })
                  : t("tool.history.completedOverall", {
                      matches: historyCompleted.matches,
                      hands: historyCompleted.hands,
                    })
              }}
            </span>
          </div>

          <div class="history-note">{{ t("tool.history.note") }}</div>
        </template>
      </div>

      <!-- Generic result -->
      <div v-else-if="tool.status === 'error'" class="generic-error">
        <AppIcon name="alert" :size="13" />
        {{ errorText || t("tool.error.generic") }}
      </div>

      <div v-if="tool.status === 'running' && !isAction && !isTalk" class="running-line">
        <span class="spinner"></span>
        <span>{{ t("tool.executing", { name: toolLabel(tool.name) }) }}</span>
      </div>

      <details class="tool-section">
        <summary>
          <span>{{ t("tool.args") }}</span>
          <button class="copy-btn" type="button" @click.stop.prevent="copy(argsPretty, 'args')">
            {{ copied === "args" ? t("common.copied") : t("common.copy") }}
          </button>
        </summary>
        <pre class="tool-pre">{{ argsPretty }}</pre>
      </details>

      <details class="tool-section" v-if="tool.result">
        <summary>
          <span>{{ t("tool.rawResult") }}</span>
          <button class="copy-btn" type="button" @click.stop.prevent="copy(resultPretty, 'result')">
            {{ copied === "result" ? t("common.copied") : t("common.copy") }}
          </button>
        </summary>
        <pre class="tool-pre result-pre">{{ resultPretty }}</pre>
      </details>
    </div>
  </details>
</template>

<style scoped>
.tool-card {
  margin: 8px 0;
  border: 1px solid var(--line);
  border-left: 2px solid var(--line-strong);
  border-radius: 0 10px 10px 0;
  background: rgba(255, 255, 255, 0.022);
  overflow: hidden;
  transition: border-color 0.18s ease, background 0.18s ease;
}

.tool-card.estimate { border-left-color: rgba(232, 195, 122, 0.55); }
.tool-card.equity { border-left-color: rgba(232, 195, 122, 0.55); }
.tool-card.action { border-left-color: rgba(84, 214, 168, 0.5); }
.tool-card.random { border-left-color: rgba(121, 169, 255, 0.55); }
.tool-card.call-ev { border-left-color: rgba(121, 169, 255, 0.55); }
.tool-card.talk { border-left-color: rgba(232, 195, 122, 0.7); }
.tool-card.history { border-left-color: rgba(232, 195, 122, 0.45); }
.tool-card.action.error { border-left-color: rgba(226, 112, 95, 0.6); }
.tool-card.running { border-left-color: rgba(121, 169, 255, 0.55); }
.tool-card:hover { background: rgba(255, 255, 255, 0.035); }

.tool-head {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 7px 10px;
  cursor: pointer;
  list-style: none;
  user-select: none;
}

.tool-head::-webkit-details-marker { display: none; }

.tool-icon {
  display: flex;
  color: var(--muted);
  flex: none;
}

.tool-card.estimate .tool-icon,
.tool-card.equity .tool-icon { color: var(--gold); }
.tool-card.action .tool-icon { color: var(--emerald); }
.tool-card.random .tool-icon,
.tool-card.call-ev .tool-icon { color: var(--blue); }
.tool-card.talk .tool-icon { color: var(--gold); }
.tool-card.history .tool-icon { color: var(--gold); }
.tool-card.action.error .tool-icon { color: var(--red); }

.tool-title {
  font-size: calc(11px * var(--type-scale));
  font-weight: 750;
  color: var(--text-soft);
  white-space: nowrap;
}

.tool-detail {
  font-family: var(--font-mono);
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.tool-status {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: calc(9px * var(--type-scale));
  color: var(--muted);
  white-space: nowrap;
  flex: none;
}

.tool-status.done { color: var(--emerald); }
.tool-status.error { color: var(--red); }
.tool-status.running { color: var(--blue); }

.duration { color: var(--faint); font-size: calc(8.5px * var(--type-scale)); }
.tool-chevron { color: var(--faint); flex: none; }

.tool-body {
  padding: 0 10px 9px;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.estimate-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
  padding-top: 2px;
}

.estimate-meta .chip { padding: 2px 6px; font-size: calc(8.5px * var(--type-scale)); }
.best-inline { color: var(--muted); font-size: calc(9.5px * var(--type-scale)); }
.best-inline strong { color: var(--gold); }

.dist {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 6px 0 2px;
}

.dist-row {
  display: flex;
  align-items: center;
  gap: 7px;
  min-height: 14px;
}

.dist-name {
  width: 48px;
  flex: none;
  font-size: calc(9.5px * var(--type-scale));
  color: var(--muted);
  white-space: nowrap;
}

.dist-row.best .dist-name { color: var(--gold); font-weight: 700; }

.dist-track {
  flex: 1;
  height: 4px;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.055);
  overflow: hidden;
}

.dist-fill {
  height: 100%;
  border-radius: 99px;
  opacity: 0.82;
  transition: width 0.45s cubic-bezier(0.2, 0.8, 0.3, 1);
}

.dist-row.best .dist-fill { opacity: 1; }

.dist-pct {
  width: 42px;
  flex: none;
  text-align: right;
  font-size: calc(9px * var(--type-scale));
  color: var(--muted);
}

.dist-row.best .dist-pct { color: var(--gold); }

.dist-skeleton { display: flex; flex-direction: column; gap: 6px; padding: 6px 0; }

.skeleton-row { display: flex; align-items: center; gap: 8px; }
.skeleton-name { width: 40px; height: 6px; border-radius: 99px; background: rgba(255, 255, 255, 0.06); }
.skeleton-bar {
  height: 5px;
  border-radius: 99px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.04), rgba(232, 195, 122, 0.14), rgba(255, 255, 255, 0.04));
  background-size: 200% 100%;
  animation: shimmer 1.4s linear infinite;
}

@keyframes shimmer { to { background-position: -200% 0; } }

.empty-result { font-size: calc(10px * var(--type-scale)); color: var(--faint); }

.random-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 4px 0 1px;
}

.random-readout {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.random-value {
  font-size: calc(20px * var(--type-scale));
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.02em;
  line-height: 1.1;
}

.random-unit {
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
}

.random-track {
  position: relative;
  height: 5px;
  margin: 2px 4px 0;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.06);
}

.random-marker {
  position: absolute;
  top: 50%;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--blue);
  box-shadow: 0 0 8px rgba(121, 169, 255, 0.6);
  transform: translate(-50%, -50%);
  transition: left 0.4s cubic-bezier(0.2, 0.8, 0.3, 1);
}

.random-scale {
  display: flex;
  justify-content: space-between;
  padding: 0 1px;
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
}

.equity-block,
.callev-block {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 4px 0 1px;
}

.equity-readout,
.callev-readout {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}

.equity-value {
  font-size: calc(22px * var(--type-scale));
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.01em;
  line-height: 1.05;
}

.equity-caption,
.callev-caption {
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
}

.equity-meta {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: calc(8.5px * var(--type-scale));
}

.equity-meta .chip {
  padding: 2px 6px;
  font-size: calc(8px * var(--type-scale));
}

.equity-bar {
  display: flex;
  height: 6px;
  border-radius: 99px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.055);
}

.equity-seg {
  height: 100%;
  transition: width 0.45s cubic-bezier(0.2, 0.8, 0.3, 1);
}

.equity-seg.win { background: var(--emerald); }
.equity-seg.tie { background: var(--gold); }
.equity-seg.lose { background: rgba(226, 112, 95, 0.75); }

.equity-legend {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 9px;
  font-size: calc(9px * var(--type-scale));
  color: var(--muted);
}

.equity-legend .dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  margin-right: 3px;
  vertical-align: middle;
}

.dot.win { background: var(--emerald); }
.dot.tie { background: var(--gold); }
.dot.lose { background: rgba(226, 112, 95, 0.75); }

.equity-note,
.callev-note {
  font-size: calc(8.5px * var(--type-scale));
  color: var(--faint);
  line-height: 1.45;
}

.callev-value {
  font-size: calc(22px * var(--type-scale));
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.01em;
  line-height: 1.05;
}

.callev-value.positive { color: var(--emerald); }
.callev-value.negative { color: var(--red); }

.callev-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
}

.callev-grid strong {
  color: var(--text-soft);
  font-weight: 600;
}

.callev-compare {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding-top: 1px;
}

.compare-track {
  position: relative;
  height: 5px;
  margin: 2px 4px 0;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.06);
}

.compare-fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 99px;
  background: rgba(84, 214, 168, 0.55);
}

.compare-marker {
  position: absolute;
  top: 50%;
  width: 3px;
  height: 11px;
  border-radius: 2px;
  background: var(--gold);
  box-shadow: 0 0 6px rgba(232, 195, 122, 0.55);
  transform: translate(-50%, -50%);
}

.compare-scale {
  display: flex;
  justify-content: space-between;
  padding: 0 1px;
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
}

.callev-na {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--gold);
  line-height: 1.5;
}

.action-ok {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 5px 0 1px;
}

.action-pill {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: calc(10px * var(--type-scale));
  font-weight: 800;
  letter-spacing: 0.04em;
  background: rgba(84, 214, 168, 0.12);
  border: 1px solid rgba(84, 214, 168, 0.3);
  color: var(--emerald);
}

.action-pill.fold { color: var(--red); border-color: rgba(226, 112, 95, 0.3); background: var(--red-dim); }
.action-pill.raise, .action-pill.bet { color: var(--blue); border-color: rgba(121, 169, 255, 0.3); background: var(--blue-dim); }
.action-pill.all_in { color: var(--purple); border-color: rgba(196, 140, 255, 0.35); background: var(--purple-dim); }
.action-pill.check { color: var(--muted); border-color: var(--line); background: rgba(255, 255, 255, 0.04); }

.amount { font-size: calc(10px * var(--type-scale)); color: var(--text-soft); }

.ok-text {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: calc(9.5px * var(--type-scale));
  color: var(--emerald);
  margin-left: auto;
}

.action-fail {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 5px 0 1px;
}

.fail-line {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: calc(10.5px * var(--type-scale));
  color: #ffb3a8;
  line-height: 1.5;
}

.constraint-grid { display: flex; flex-wrap: wrap; gap: 4px; }
.constraint-grid .chip { font-size: calc(8.5px * var(--type-scale)); padding: 2px 6px; }
.constraint-grid strong { color: var(--text-soft); }

.legal-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
  font-size: calc(9px * var(--type-scale));
}

.legal-row .chip { font-size: calc(8.5px * var(--type-scale)); padding: 2px 6px; }

.talk-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px 10px 9px;
}

.talk-quote {
  padding: 8px 10px;
  border-radius: 3px 10px 10px 10px;
  border: 1px solid rgba(232, 195, 122, 0.22);
  background: rgba(232, 195, 122, 0.055);
  color: var(--text-soft);
  font-size: calc(11.5px * var(--type-scale));
  line-height: 1.65;
  overflow-wrap: anywhere;
}

.inline-talk {
  margin-top: 8px;
}

.talk-meta-row {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
}

.generic-error {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--red);
}

.running-line {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: calc(9.5px * var(--type-scale));
  color: var(--blue);
}

.tool-section {
  border-top: 1px solid var(--line);
  padding-top: 5px;
}

.tool-section summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
  cursor: pointer;
  list-style: none;
}

.tool-section summary::-webkit-details-marker { display: none; }

.copy-btn {
  padding: 1px 6px;
  border-radius: 4px;
  border: 1px solid var(--line);
  color: var(--faint);
  font-size: calc(8.5px * var(--type-scale));
}

.copy-btn:hover { color: var(--text); border-color: var(--line-strong); }

.tool-pre {
  margin: 6px 0 0;
  padding: 7px 8px;
  max-height: 220px;
  overflow: auto;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: rgba(3, 9, 7, 0.55);
  font-family: var(--font-mono);
  font-size: calc(9.5px * var(--type-scale));
  line-height: 1.55;
  color: #a9b8af;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.result-pre { max-height: 260px; color: #93a49a; }

.history-block {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 4px 0 1px;
}

.history-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
}

.history-meta .chip {
  padding: 2px 6px;
  font-size: calc(8px * var(--type-scale));
}

.history-positions {
  display: inline-flex;
  align-self: flex-start;
  gap: 3px;
  padding: 2px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.03);
}

.history-position {
  padding: 1px 7px;
  border-radius: 4px;
  background: transparent;
  color: var(--faint);
  font-size: calc(8.5px * var(--type-scale));
}

.history-position:hover { color: var(--text-soft); }

.history-position.active {
  background: rgba(232, 195, 122, 0.12);
  color: var(--gold);
}

.history-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-top: 2px;
}

.history-row {
  display: grid;
  grid-template-columns: minmax(52px, 68px) minmax(0, 1fr) minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  min-height: 15px;
}

.history-head-row {
  padding-bottom: 1px;
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.history-head-row span:nth-child(2),
.history-head-row span:nth-child(3) { padding-left: 2px; }

.history-group-row { margin-top: 4px; }

.history-group {
  grid-column: 1 / -1;
  font-size: calc(8px * var(--type-scale));
  font-weight: 700;
  color: var(--gold);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.78;
}

.history-name {
  overflow: hidden;
  font-size: calc(9px * var(--type-scale));
  color: var(--muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-cell {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
}

.history-cell.empty .history-value { color: var(--faint); }

.history-track {
  flex: 1;
  min-width: 16px;
  height: 4px;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.055);
  overflow: hidden;
}

.history-fill {
  display: block;
  height: 100%;
  border-radius: 99px;
  opacity: 0.85;
  transition: width 0.4s cubic-bezier(0.2, 0.8, 0.3, 1);
}

.history-fill.you { background: var(--emerald); }
.history-fill.opponent { background: var(--gold); }

.history-value {
  flex: none;
  min-width: 34px;
  font-size: calc(9px * var(--type-scale));
  color: var(--text-soft);
  text-align: right;
}

.history-n {
  flex: none;
  font-size: calc(7.5px * var(--type-scale));
  color: var(--faint);
}

.history-n.low { color: var(--gold); }
.history-n.medium { color: var(--blue); }
.history-n.high { color: var(--emerald); opacity: 0.75; }

.history-recent {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
  padding-top: 3px;
}

.history-recent-label {
  margin-right: 2px;
  font-size: calc(8.5px * var(--type-scale));
  color: var(--faint);
}

.history-hand {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 5px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.03);
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
}

.history-hand.won {
  border-color: rgba(84, 214, 168, 0.25);
  background: rgba(84, 214, 168, 0.06);
  color: var(--emerald);
}

.history-completed {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px 10px;
  font-size: calc(8.5px * var(--type-scale));
  color: var(--faint);
}

.history-note {
  font-size: calc(8px * var(--type-scale));
  line-height: 1.45;
  color: var(--faint);
  opacity: 0.85;
}
</style>
