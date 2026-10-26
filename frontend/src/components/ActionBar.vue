<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import AppIcon from "./AppIcon.vue";
import { chips, percent, seatDisplayName, streetLabel } from "../format";
import { t } from "../i18n";
import { board, boardLocked, streetForCommunity } from "../boardReveal";
import {
  game,
  humanAction,
  humanSeat,
  nextHand,
  openSetup,
} from "../store";

const raiseTo = ref(0);
const raiseDraft = ref("0");
const raiseInput = ref<HTMLInputElement | null>(null);
const quickFraction = ref<string>("");

const snapshot = computed(() => (
  (board.busy || board.holeHold) && board.baseSnapshot ? board.baseSnapshot : game.snapshot
));
const match = computed(() => snapshot.value?.match || null);
const legal = computed(() => snapshot.value?.legal_actions || []);
const displayStreet = computed(() => {
  if (board.busy) return streetForCommunity(board.community.length);
  if (board.holeHold) return "preflop";
  return snapshot.value?.street || "waiting";
});
const canAct = computed(
  () =>
    Boolean(snapshot.value?.has_match) &&
    snapshot.value?.acting_kind === "human" &&
    snapshot.value?.street !== "finished" &&
    !match.value?.match_over &&
    !match.value?.aborted &&
    !game.busy &&
    !boardLocked.value,
);
const isRaise = computed(() => legal.value.includes("bet") || legal.value.includes("raise"));
const canCheck = computed(() => legal.value.includes("check"));
const canCall = computed(() => legal.value.includes("call"));
const canFold = computed(() => legal.value.includes("fold"));
const canAllIn = computed(() => legal.value.includes("all_in") && isRaise.value);
const pot = computed(() => Number(snapshot.value?.pot || 0));
const currentBet = computed(() => Number(snapshot.value?.current_bet || 0));
const minRaise = computed(() => Number(snapshot.value?.min_raise_to || 0));
const maxRaise = computed(() => Number(snapshot.value?.max_raise_to || 0));
const humanPlayer = computed(() => {
  if (!humanSeat.value) return null;
  return snapshot.value?.players?.find((player) => player.seat === humanSeat.value?.seat) || null;
});
const toCall = computed(() => {
  const player = humanPlayer.value;
  const owed = currentBet.value - Number(player?.street_bet || 0);
  return Math.max(0, owed);
});
const callAmount = computed(() => Math.min(toCall.value, Number(humanPlayer.value?.chips || 0)));
const potOdds = computed(() => {
  if (toCall.value <= 0) return 0;
  return toCall.value / (pot.value + toCall.value);
});
const actingName = computed(() => {
  const seat = snapshot.value?.acting_seat;
  const player = snapshot.value?.players?.find((item) => item.seat === seat);
  return player
    ? seatDisplayName(player.name, player.kind) || player.name
    : t("common.opponent");
});
const winnerNames = computed(() => {
  const result = game.lastResult;
  if (!result) return "";
  return result.winner_seats
    .map((seat) => {
      const player = snapshot.value?.players?.find((item) => item.seat === seat);
      return player
        ? seatDisplayName(player.name, player.kind) || player.name
        : t("common.seat", { seat });
    })
    .join(" · ");
});
const phaseText = computed(() => {
  if (board.holeHold) return t("table.dealing");
  if (board.busy) {
    return board.community.length
      ? t("table.revealing", { street: streetLabel(displayStreet.value) })
      : t("table.dealing");
  }
  if (game.busy) return t("table.submitting");
  if (!match.value) return t("topbar.noMatch");
  if (match.value.aborted) return t("table.matchAborted");
  if (match.value.match_over) return t("table.matchOver");
  if (snapshot.value?.street === "finished") return t("table.handOver");
  if (snapshot.value?.acting_kind === "human") return t("table.yourTurn");
  if (snapshot.value?.acting_kind === "llm") {
    return t("table.thinking", { name: actingName.value });
  }
  if (snapshot.value?.acting_kind === "bot") {
    return t("table.deciding", { name: actingName.value });
  }
  return t("table.waitingShort");
});

function clampRaise(value: number): number {
  const min = minRaise.value || 0;
  const max = maxRaise.value || min;
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, Math.round(value)));
}

function targetFor(fraction: number): number {
  const potAfterCall = pot.value + toCall.value;
  return clampRaise(currentBet.value + potAfterCall * fraction);
}

function setFraction(fraction: number, label: string): void {
  quickFraction.value = label;
  raiseTo.value = targetFor(fraction);
}

function setRaise(value: number): void {
  quickFraction.value = "";
  raiseTo.value = clampRaise(value);
}

function onRaiseInput(event: Event): void {
  const input = event.target as HTMLInputElement;
  raiseDraft.value = input.value;
  quickFraction.value = "";
  const value = Number(input.value);
  if (Number.isFinite(value)) raiseTo.value = clampRaise(value);
}

function onRaiseBlur(): void {
  const clamped = clampRaise(Number(raiseDraft.value));
  raiseTo.value = clamped;
  raiseDraft.value = String(clamped);
}

const sliderPct = computed(() => {
  const span = maxRaise.value - minRaise.value;
  if (span <= 0) return 100;
  return Math.max(0, Math.min(100, ((raiseTo.value - minRaise.value) / span) * 100));
});

watch(raiseTo, (value) => {
  raiseDraft.value = String(value);
});

watch(
  () => [
    snapshot.value?.hand_id,
    snapshot.value?.street,
    snapshot.value?.acting_seat,
    currentBet.value,
    minRaise.value,
    maxRaise.value,
  ],
  () => {
    if (!isRaise.value) return;
    raiseTo.value = clampRaise(minRaise.value);
    raiseDraft.value = String(raiseTo.value);
    quickFraction.value = "";
  },
  { immediate: true },
);

async function act(action: string, amount = 0): Promise<void> {
  if (game.busy || boardLocked.value) return;
  await humanAction(action, amount);
}

function raiseAction(): void {
  if (!isRaise.value) return;
  const type = legal.value.includes("bet") ? "bet" : "raise";
  void act(type, clampRaise(raiseTo.value));
}

function onKeydown(event: KeyboardEvent): void {
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
  if (game.setupOpen) return;
  const target = event.target as HTMLElement | null;
  if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
  if (!canAct.value) return;
  const key = event.key.toLowerCase();
  if (key === "f" && canFold.value) {
    event.preventDefault();
    void act("fold");
  } else if (key === "c" && (canCheck.value || canCall.value)) {
    event.preventDefault();
    void act(canCheck.value ? "check" : "call");
  } else if (key === "a" && canAllIn.value) {
    event.preventDefault();
    void act("all_in");
  } else if (key === "r" && isRaise.value) {
    event.preventDefault();
    raiseInput.value?.focus();
    raiseInput.value?.select();
  }
}

onMounted(() => window.addEventListener("keydown", onKeydown));
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <div class="action-bar" :class="{ 'your-turn': canAct }">
    <div class="status-side">
      <div class="status-line">
        <span v-if="canAct" class="dot live"></span>
        <span v-else-if="snapshot?.street === 'finished' || match?.match_over || match?.aborted" class="dot"></span>
        <span v-else class="spinner"></span>
        <span class="status-text">{{ phaseText }}</span>
      </div>
      <div v-if="canAct && toCall > 0" class="status-sub">
        <span class="pot-odds">{{ t("table.potOdds") }} <strong class="mono">{{ percent(potOdds, 0) }}</strong></span>
      </div>
    </div>

    <div v-if="match && displayStreet === 'finished'" class="end-actions">
      <div class="winner-line">
        <AppIcon name="sparkles" :size="14" />
        <span>{{ t("table.winnerPrefix", { names: winnerNames || t("table.handOver") }) }} <strong class="mono gold">{{ chips(game.lastResult?.pot || 0) }}</strong></span>
      </div>
      <button v-if="match.can_next_hand" class="btn btn-primary" :disabled="game.busy || boardLocked" @click="nextHand">
        {{ t("table.nextHand") }}
        <AppIcon name="arrowRight" :size="15" />
      </button>
      <button class="btn btn-ghost" @click="openSetup">{{ t("table.newMatch") }}</button>
    </div>

    <div v-else-if="canAct" class="controls">
      <div class="primary-actions">
        <button v-if="canFold" class="btn btn-danger action-btn" @click="act('fold')" :title="t('table.shortcut', { key: 'F' })">
          <AppIcon name="x" :size="15" />
          {{ t("action.fold") }}
          <kbd>F</kbd>
        </button>
        <button
          v-if="canCheck || canCall"
          class="btn btn-emerald action-btn main-action"
          @click="act(canCheck ? 'check' : 'call')"
          :title="t('table.shortcut', { key: 'C' })"
        >
          <AppIcon :name="canCheck ? 'check' : 'arrowRight'" :size="15" />
          {{ canCheck ? t("action.check") : t("action.callAmount", { amount: chips(callAmount) }) }}
          <kbd>C</kbd>
        </button>
        <button v-if="isRaise" class="btn btn-primary action-btn main-action" @click="raiseAction" :title="t('table.shortcut', { key: 'R' })">
          <AppIcon name="zap" :size="15" />
          {{ t(legal.includes("bet") ? "action.betTo" : "action.raiseTo", { amount: chips(raiseTo) }) }}
          <kbd>R</kbd>
        </button>
        <button v-if="canAllIn" class="btn allin-btn" @click="act('all_in')" :title="t('table.shortcut', { key: 'A' })">
          <AppIcon name="flame" :size="14" />
          {{ t("action.allInShort") }}
          <kbd>A</kbd>
        </button>
      </div>

      <div v-if="isRaise" class="raise-panel">
        <div class="quick-row">
          <button class="quick" :class="{ active: quickFraction === 'half' }" @click="setFraction(0.5, 'half')">{{ t("table.halfPot") }}</button>
          <button class="quick" :class="{ active: quickFraction === 'three' }" @click="setFraction(0.75, 'three')">{{ t("table.threeQuarterPot") }}</button>
          <button class="quick" :class="{ active: quickFraction === 'pot' }" @click="setFraction(1, 'pot')">{{ t("table.onePot") }}</button>
        </div>
        <div class="slider-row">
          <input
            class="range"
            type="range"
            :min="minRaise"
            :max="maxRaise"
            step="1"
            :value="raiseTo"
            :style="{ '--pct': `${sliderPct}%` }"
            @input="setRaise(Number(($event.target as HTMLInputElement).value))"
          />
          <div class="amount-input">
            <span class="amount-prefix">{{ t("table.to") }}</span>
            <input
              ref="raiseInput"
              class="input"
              type="number"
              :min="minRaise"
              :max="maxRaise"
              :value="raiseDraft"
              @input="onRaiseInput"
              @blur="onRaiseBlur"
            />
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="match?.match_over || match?.aborted" class="waiting-side">
      <span class="match-end">{{ match?.aborted ? t("table.matchAborted") : t("table.matchEnded") }}</span>
      <button class="btn btn-primary" @click="openSetup">{{ t("table.newMatch") }}</button>
    </div>
  </div>
</template>

<style scoped>
.action-bar {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 13px 16px;
  border-top: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(10, 22, 17, 0.65), rgba(6, 14, 11, 0.8));
  min-height: 88px;
}

.action-bar.your-turn {
  background:
    radial-gradient(500px 100px at 18% 0%, rgba(84, 214, 168, 0.08), transparent 70%),
    linear-gradient(180deg, rgba(10, 22, 17, 0.65), rgba(6, 14, 11, 0.8));
}

.status-side {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 190px;
  max-width: 300px;
  flex: none;
}

.status-line {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}

.status-text {
  font-size: calc(12px * var(--type-scale));
  font-weight: 750;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.status-sub {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 10px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--muted);
}

.status-sub strong { color: var(--text-soft); font-weight: 700; }
.pot-odds { color: var(--faint); }
.pot-odds strong { color: var(--gold); }

.controls {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
  min-width: 0;
}

.primary-actions {
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
  flex: none;
}

.action-btn {
  height: 38px;
  padding: 0 13px;
}

.main-action { min-width: 108px; }

kbd {
  font-family: var(--font-mono);
  font-size: calc(8px * var(--type-scale));
  padding: 1px 4px;
  border-radius: 3px;
  border: 1px solid currentColor;
  opacity: 0.42;
  margin-left: 2px;
}

.allin-btn {
  height: 38px;
  padding: 0 10px;
  border-radius: 10px;
  border: 1px solid rgba(196, 140, 255, 0.32);
  background: var(--purple-dim);
  color: var(--purple);
  font-size: calc(10.5px * var(--type-scale));
  font-weight: 800;
  letter-spacing: 0.04em;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.allin-btn:hover { background: rgba(196, 140, 255, 0.2); }

.raise-panel {
  flex: 1;
  min-width: 220px;
  max-width: 520px;
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding-left: 14px;
  border-left: 1px solid var(--line);
}

.quick-row {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
}

.quick {
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
  color: var(--muted);
  font-size: calc(9.5px * var(--type-scale));
  font-weight: 700;
  transition: all 0.15s ease;
}

.quick:hover { color: var(--text); border-color: var(--line-strong); }
.quick.active { color: var(--gold); border-color: rgba(232, 195, 122, 0.32); background: var(--gold-dim); }

.slider-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.slider-row .range { flex: 1; }

.amount-input {
  display: flex;
  align-items: center;
  gap: 5px;
  flex: none;
}

.amount-prefix { font-size: calc(9.5px * var(--type-scale)); color: var(--faint); }

.amount-input .input {
  width: 84px;
  height: 30px;
  padding: 4px 8px;
  font-size: calc(12px * var(--type-scale));
  text-align: right;
}

.waiting-side {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-left: auto;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.match-end { font-size: calc(12px * var(--type-scale)); font-weight: 700; color: var(--gold); }

.end-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.winner-line {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-right: 6px;
  font-size: calc(11.5px * var(--type-scale));
  color: var(--text-soft);
}

.winner-line strong { color: var(--gold); }

@media (max-width: 1520px) {
  .controls { flex-wrap: wrap; }
  .raise-panel {
    flex: 1 1 100%;
    max-width: none;
    border-left: 0;
    border-top: 1px solid var(--line);
    padding-left: 0;
    padding-top: 8px;
  }
}

@media (max-width: 1180px) {
  .action-bar { flex-wrap: wrap; gap: 12px; }
  .status-side { max-width: none; min-width: 0; flex: 1 1 100%; }
  .controls { flex: 1 1 100%; }
  .raise-panel { max-width: none; }
}

@media (max-width: 720px) {
  .action-bar { padding: 10px 12px; min-height: 0; }
  .controls { flex-direction: column; align-items: stretch; gap: 9px; }
  .primary-actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .primary-actions .action-btn { flex: none; }
  .raise-panel { border-left: 0; border-top: 1px solid var(--line); padding-left: 0; padding-top: 8px; }
  .end-actions { margin-left: 0; justify-content: flex-start; }
  .waiting-side { margin-left: 0; justify-content: flex-start; }
}
</style>
