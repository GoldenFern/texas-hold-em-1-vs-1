<script setup lang="ts">
import { computed, ref } from "vue";
import AppIcon from "./AppIcon.vue";
import { compactPercent, seatDisplayName } from "../format";
import { handCategoryLabel } from "../i18n/labels";
import { t } from "../i18n";
import { boardLocked } from "../boardReveal";
import {
  game,
  humanSeat,
  refreshAnalysis,
  selectAnalysisSeat,
} from "../store";

const STORAGE_KEY = "hulab.analysis.expanded";

function readExpanded(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

const expanded = ref(readExpanded());

const snapshot = computed(() => game.snapshot);
const match = computed(() => snapshot.value?.match || null);
const seats = computed(() => match.value?.seats || []);
const analysis = computed(() => {
  const seat = game.selectedAnalysisSeat;
  return seat === null ? null : game.analysis[seat] || null;
});
const selectedSeat = computed(() => game.selectedAnalysisSeat);
const selectedInfo = computed(() => seats.value.find((seat) => seat.seat === selectedSeat.value) || null);
const selectedPlayer = computed(() => snapshot.value?.players?.find((player) => player.seat === selectedSeat.value) || null);
const visibleSeats = computed(() => {
  if (game.godView) return seats.value;
  const human = humanSeat.value;
  return human ? [human] : [];
});
const canAnalyze = computed(() => Boolean(selectedPlayer.value?.hole_cards?.length === 2 && snapshot.value?.has_match));
const distribution = computed(() => analysis.value?.distribution || []);
const bestKey = computed(() => analysis.value?.current_best?.category || "");
const topCategory = computed(() => {
  const list = distribution.value;
  if (!list.length) return null;
  return list.reduce((best, item) => (item.probability > best.probability ? item : best));
});
const stale = computed(() => {
  if (!analysis.value || !snapshot.value) return false;
  return analysis.value.hand_id !== snapshot.value.hand_id || analysis.value.street !== snapshot.value.street;
});
const maxProbability = computed(() => {
  const values = distribution.value.map((item) => item.probability);
  return Math.max(0.0001, ...values);
});
const summaryName = computed(() => {
  const player = selectedPlayer.value;
  if (!player) return t("common.none");
  return seatDisplayName(player.name, player.kind, selectedInfo.value?.model_name) || player.name;
});

const topCategoryName = computed(() => {
  const item = topCategory.value;
  return item ? handCategoryLabel(item.key, item.name) : "";
});

function barWidth(value: number): string {
  if (value <= 0) return "0%";
  // Scale relative to the largest category so small probabilities remain visible.
  const relative = value / maxProbability.value;
  return `${Math.max(2, relative * 100)}%`;
}

function pickSeat(seat: number): void {
  selectAnalysisSeat(seat);
}

function reload(): void {
  void refreshAnalysis(true);
}

function toggleExpanded(): void {
  expanded.value = !expanded.value;
  try {
    window.localStorage.setItem(STORAGE_KEY, expanded.value ? "1" : "0");
  } catch {
    // Storage can be unavailable in private mode; the panel still works.
  }
}
</script>

<template>
  <section class="analysis-panel surface" :class="{ collapsed: !expanded }">
    <header class="analysis-head">
      <button
        class="analysis-toggle"
        type="button"
        :title="t('analysis.toggleTitle')"
        :aria-expanded="expanded"
        @click="toggleExpanded"
      >
        <span class="toggle-icon"><AppIcon name="chart" :size="14" /></span>
        <span class="toggle-title">{{ t("analysis.title") }}</span>

        <template v-if="canAnalyze">
          <span class="summary-name">{{ summaryName }}</span>
          <span v-if="topCategory" class="summary-stat summary-top">
            <span class="summary-key">{{ t("analysis.mostLikely") }}</span>
            {{ topCategoryName }} {{ compactPercent(topCategory.probability) }}
          </span>
        </template>
        <span v-else class="summary-muted">{{ match ? t("analysis.waitingHole") : t("analysis.showAfterStart") }}</span>

        <span v-if="boardLocked" class="summary-state"><span class="spinner"></span>{{ t("analysis.dealing") }}</span>
        <span v-else-if="stale" class="summary-state"><span class="spinner"></span>{{ t("analysis.recalculating") }}</span>
        <span v-else-if="game.analysisLoading" class="summary-state"><span class="spinner"></span>{{ t("analysis.calculating") }}</span>

        <AppIcon name="chevronDown" :size="14" class="toggle-chevron" />
      </button>

      <div v-if="expanded" class="head-controls">
        <div v-if="visibleSeats.length > 1" class="segmented seat-switch">
          <button
            v-for="seat in visibleSeats"
            :key="seat.seat"
            :class="{ active: selectedSeat === seat.seat }"
            @click="pickSeat(seat.seat)"
          >
            {{ seatDisplayName(seat.name, seat.kind, seat.model_name) || seat.name }}
          </button>
        </div>
        <button class="btn btn-ghost btn-sm" :disabled="game.analysisLoading || !canAnalyze" @click="reload">
          <AppIcon name="refresh" :size="13" :class="{ spin: game.analysisLoading }" />
          {{ t("analysis.recalculate") }}
        </button>
      </div>
    </header>

    <div v-if="expanded" class="analysis-body">
      <div v-if="!match || !canAnalyze" class="analysis-empty">
        <AppIcon name="chart" :size="18" />
        <span>
          {{ !match ? t("analysis.emptyNoMatch") : game.godView ? t("analysis.emptyWaitingDeal") : t("analysis.emptyGodView") }}
        </span>
      </div>

      <div v-else-if="game.analysisLoading && !analysis" class="analysis-loading">
        <span class="spinner"></span>
        {{ t("analysis.calculatingDistribution") }}
      </div>

      <div v-else-if="game.analysisError" class="analysis-error">
        <AppIcon name="alert" :size="14" />
        <span>{{ game.analysisError }}</span>
        <button class="btn btn-sm" @click="reload">{{ t("analysis.retry") }}</button>
      </div>

      <div v-else-if="distribution.length" class="chart" :class="{ stale, locked: boardLocked }">
        <div
          v-for="(item, index) in distribution"
          :key="item.key"
          class="chart-row"
          :class="{ best: bestKey === item.key, zero: item.probability === 0 }"
          :style="{ '--delay': `${index * 28}ms`, '--bar-color': item.color }"
        >
          <span class="chart-name">{{ handCategoryLabel(item.key, item.name) }}</span>
          <div class="chart-track">
            <div class="chart-fill" :style="{ width: barWidth(item.probability) }"></div>
          </div>
          <span class="chart-pct mono">{{ compactPercent(item.probability) }}</span>
        </div>
      </div>

      <div v-else class="analysis-loading">
        <span class="spinner"></span>
        {{ t("analysis.waitingData") }}
      </div>
    </div>
  </section>
</template>

<style scoped>
.analysis-panel {
  padding: 14px 16px 13px;
}

.analysis-panel.collapsed {
  padding: 10px 14px;
}

.analysis-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.analysis-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  flex: 1;
  padding: 0;
  text-align: left;
  color: var(--muted);
  transition: color 0.16s ease;
}

.analysis-toggle:hover { color: var(--text-soft); }

.toggle-icon {
  display: flex;
  flex: none;
  color: var(--faint);
}

.toggle-title {
  flex: none;
  font-size: calc(12.5px * var(--type-scale));
  font-weight: 750;
  color: var(--text-soft);
  letter-spacing: -0.01em;
}

.summary-name {
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: calc(11px * var(--type-scale));
  color: var(--muted);
}

.summary-stat {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--text-soft);
  white-space: nowrap;
}

.summary-key {
  font-size: calc(9px * var(--type-scale));
  color: var(--faint);
  letter-spacing: 0.04em;
}

.summary-top { color: var(--muted); }

.summary-muted {
  font-size: calc(10.5px * var(--type-scale));
  color: var(--faint);
}

.summary-state {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-left: auto;
  font-size: calc(9px * var(--type-scale));
  color: var(--gold);
  white-space: nowrap;
}

.summary-state .spinner { width: 10px; height: 10px; }

.toggle-chevron {
  flex: none;
  margin-left: auto;
  color: var(--faint);
  transition: transform 0.18s ease;
}

.analysis-panel:not(.collapsed) .toggle-chevron { transform: rotate(180deg); }

.head-controls {
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
}

.seat-switch button { font-size: calc(10px * var(--type-scale)); padding: 4px 8px; }
.seat-switch { max-width: 100%; overflow-x: auto; }

.analysis-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 12px;
}

.chart {
  display: flex;
  flex-direction: column;
  gap: 2px;
  transition: opacity 0.2s ease;
}

.chart.stale,
.chart.locked { opacity: 0.55; }

.chart-row {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 22px;
  padding: 1px 2px;
  border-radius: 5px;
  animation: row-in 0.35s ease-out both;
  animation-delay: var(--delay);
}

@keyframes row-in {
  from { opacity: 0; transform: translateX(-4px); }
  to { opacity: 1; transform: none; }
}

.chart-row:hover { background: rgba(255, 255, 255, 0.025); }
.chart-row.zero { opacity: 0.38; }

.chart-name {
  flex: 0 0 8.8em;
  min-width: 0;
  overflow: hidden;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--muted);
  white-space: nowrap;
  text-overflow: ellipsis;
}

.chart-row.best .chart-name {
  color: var(--gold);
  font-weight: 750;
}

.chart-track {
  position: relative;
  flex: 1;
  height: 8px;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.045);
  overflow: hidden;
}

.chart-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, color-mix(in srgb, var(--bar-color) 65%, transparent), var(--bar-color));
  opacity: 0.72;
  transition: width 0.5s cubic-bezier(0.2, 0.8, 0.3, 1);
}

.chart-row.best .chart-fill {
  opacity: 1;
  box-shadow: 0 0 14px color-mix(in srgb, var(--bar-color) 40%, transparent);
}

.chart-pct {
  width: 52px;
  flex: none;
  text-align: right;
  font-size: calc(10px * var(--type-scale));
  font-weight: 700;
  color: var(--text-soft);
}

.chart-row.best .chart-pct { color: var(--gold); }

.analysis-empty {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 2px 2px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--muted);
}

.analysis-loading {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 12px 4px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--muted);
}

.analysis-error {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(226, 112, 95, 0.28);
  background: var(--red-dim);
  color: #ffb3a8;
  font-size: calc(10.5px * var(--type-scale));
}

.analysis-error button { margin-left: auto; }

.spin { animation: spin 0.9s linear infinite; }

@media (max-width: 720px) {
  .analysis-panel { padding: 12px 12px 11px; }
  .analysis-panel.collapsed { padding: 9px 12px; }
  .summary-name { display: none; }
  .chart-name { flex-basis: 8.8em; font-size: calc(10px * var(--type-scale)); }
  .chart-pct { width: 46px; }
}
</style>
