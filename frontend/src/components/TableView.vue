<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, watch } from "vue";
import ActionBar from "./ActionBar.vue";
import AnimatedNumber from "./AnimatedNumber.vue";
import DeckStrip from "./DeckStrip.vue";
import FlipCard from "./FlipCard.vue";
import PlayingCard from "./PlayingCard.vue";
import SeatPod from "./SeatPod.vue";
import AppIcon from "./AppIcon.vue";
import { lastActionForSeat } from "../actionMeta";
import { streetLabel } from "../format";
import { t } from "../i18n";
import { board, prefersReducedMotion, streetForCommunity } from "../boardReveal";
import { game, openSetup, openTableTalk } from "../store";
import type { ActionLog, CardInfo, PlayerState, TableTalkEntry } from "../types";

const snapshot = computed(() => (
  (board.busy || board.holeHold) && board.baseSnapshot ? board.baseSnapshot : game.snapshot
));
const blindText = computed(() => {
  const small = snapshot.value?.small_blind;
  const big = snapshot.value?.big_blind;
  if (!small || !big) return "";
  return `${small} / ${big}`;
});
const match = computed(() => snapshot.value?.match || null);
const players = computed(() => snapshot.value?.players || []);
const seats = computed(() => match.value?.seats || []);

const humanSeatIndex = computed(() => seats.value.find((seat) => seat.kind === "human")?.seat ?? null);
const topSeat = computed(() => {
  if (humanSeatIndex.value === null) return 0;
  return humanSeatIndex.value === 0 ? 1 : 0;
});
const bottomSeat = computed(() => (topSeat.value === 0 ? 1 : 0));

const actionStreet = computed(() => snapshot.value?.street || "preflop");
const handFinished = computed(
  () => actionStreet.value === "finished" || actionStreet.value === "showdown",
);
const topLastAction = computed<ActionLog | null>(() => lastActionForSeat(
  snapshot.value?.actions || [],
  Number(snapshot.value?.hand_id || 0),
  actionStreet.value,
  topSeat.value,
  handFinished.value,
));
const bottomLastAction = computed<ActionLog | null>(() => lastActionForSeat(
  snapshot.value?.actions || [],
  Number(snapshot.value?.hand_id || 0),
  actionStreet.value,
  bottomSeat.value,
  handFinished.value,
));

function playerAt(seat: number): PlayerState | null {
  return players.value.find((player) => player.seat === seat) || null;
}

function openTalkFromTable(): void {
  openTableTalk();
  if (window.innerWidth > 1180) return;
  requestAnimationFrame(() => {
    document.querySelector<HTMLElement>(".agent-console")?.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "start",
    });
  });
}

function seatInfoAt(seat: number) {
  return seats.value.find((item) => item.seat === seat) || null;
}

function showCardsAt(seat: number): boolean {
  const player = playerAt(seat);
  return Boolean(player && !player.hole_hidden);
}

const community = computed<CardInfo[]>(() => board.community);
const boardSlots = computed(() => Array.from({ length: 5 }, (_, index) => community.value[index] || null));
const pot = computed(() => Number(snapshot.value?.pot || 0));
const street = computed(() => {
  if (board.busy) return streetForCommunity(community.value.length);
  if (board.holeHold) return "preflop";
  return snapshot.value?.street || "waiting";
});
const rawResult = computed(() => (
  game.lastResult?.hand_id === game.snapshot?.hand_id ? game.lastResult : null
));
const result = computed(() => ((board.busy || board.holeHold) ? null : rawResult.value));
const winnerSeats = computed(() => result.value?.winner_seats || []);
const deckRemaining = computed(() => snapshot.value?.deck_remaining || []);
const burned = computed(() => snapshot.value?.burned || []);
const deckCount = computed(() => snapshot.value?.deck_remaining_count || 0);

const phases = computed(() => [
  { key: "preflop", label: streetLabel("preflop"), short: "PRE" },
  { key: "flop", label: streetLabel("flop"), short: "FLOP" },
  { key: "turn", label: streetLabel("turn"), short: "TURN" },
  { key: "river", label: streetLabel("river"), short: "RIVER" },
]);

const streetIndex = computed(() => {
  const order = ["preflop", "flop", "turn", "river", "showdown", "finished"];
  return order.indexOf(street.value);
});

function phaseState(index: number): "done" | "active" | "todo" {
  if (street.value === "finished" || street.value === "showdown") return "done";
  if (index < streetIndex.value) return "done";
  if (index === streetIndex.value) return "active";
  return "todo";
}

const boardCaption = computed(() => {
  if ((board.holeHold || board.busy) && community.value.length === 0) return t("table.dealing");
  return "";
});

const streetText = computed(() => {
  if (board.holeHold) return t("table.dealing");
  if (board.busy) {
    return community.value.length
      ? t("table.revealing", { street: streetLabel(street.value) })
      : t("table.dealing");
  }
  if (match.value?.match_over) return t("table.matchOver");
  return streetLabel(street.value);
});

const feltInner = ref<HTMLElement | null>(null);
const deckOrigin = ref<HTMLElement | null>(null);
const potEl = ref<HTMLElement | null>(null);
const flyChip = ref<HTMLElement | null>(null);
const dealing = ref(false);
const potShockKey = ref(0);
const talkTicker = ref<TableTalkEntry | null>(null);
let talkTimer = 0;
let lastDealNonce = board.dealNonce;
let dealTimer = 0;
let dealToken = 0;
let dealAnimations: Animation[] = [];
let chipAnimations: Animation[] = [];
let chipNodes: HTMLElement[] = [];
let fxHandId = -1;
let fxActionCount = 0;

function centerOf(rect: DOMRect): { x: number; y: number } {
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
}

function clearDealAnimations(): void {
  for (const animation of dealAnimations) {
    try {
      animation.cancel();
    } catch {
      // The element may already be gone; cancelling is best effort.
    }
  }
  dealAnimations = [];
}

async function playHoleDeal(): Promise<void> {
  const token = ++dealToken;
  if (!game.snapshot?.match || prefersReducedMotion()) return;
  await nextTick();
  if (token !== dealToken) return;
  const root = feltInner.value;
  if (!root) return;
  const targets = Array.from(root.querySelectorAll<HTMLElement>("[data-deal-card]"));
  if (!targets.length) return;
  const deckRect = deckOrigin.value?.getBoundingClientRect();
  const origin = deckRect && deckRect.width > 0
    ? centerOf(deckRect)
    : centerOf(root.getBoundingClientRect());
  clearDealAnimations();
  dealing.value = true;
  window.clearTimeout(dealTimer);
  dealTimer = window.setTimeout(() => {
    dealing.value = false;
  }, 1100);
  for (const element of targets) {
    if (typeof element.animate !== "function") continue;
    const rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height) continue;
    const target = centerOf(rect);
    const dx = origin.x - target.x;
    const dy = origin.y - target.y;
    const order = Number(element.dataset.dealOrder || 0);
    const animation = element.animate(
      [
        {
          transform: `translate(${dx}px, ${dy}px) rotate(${-8 + order * 3}deg) scale(0.82)`,
          opacity: 0,
        },
        { opacity: 1, offset: 0.12 },
        { transform: "translate(0, 0) rotate(0deg) scale(1)", opacity: 1 },
      ],
      {
        duration: 400,
        delay: order * 95,
        easing: "cubic-bezier(0.16, 0.9, 0.3, 1)",
        fill: "both",
      },
    );
    dealAnimations.push(animation);
  }
}

watch(() => board.dealNonce, (nonce) => {
  if (nonce === lastDealNonce) return;
  lastDealNonce = nonce;
  void playHoleDeal();
});

function movesChips(action: ActionLog): boolean {
  return ["call", "bet", "raise", "all_in"].includes(action.type)
    && Number(action.amount || 0) > 0;
}

function clearChipFx(): void {
  for (const animation of chipAnimations) {
    try {
      animation.cancel();
    } catch {
      // The chip may already be gone; cancelling is best effort.
    }
  }
  chipAnimations = [];
  for (const chip of chipNodes) chip.remove();
  chipNodes = [];
}

async function playChipToss(action: ActionLog): Promise<void> {
  if (prefersReducedMotion() || board.holeHold) return;
  await nextTick();
  const root = feltInner.value;
  const target = potEl.value;
  const template = flyChip.value;
  if (!root || !target || !template) return;
  const pod = root.querySelector<HTMLElement>(`[data-seat-pod="${action.seat}"]`);
  if (!pod) return;
  const rootRect = root.getBoundingClientRect();
  const from = centerOf(pod.getBoundingClientRect());
  const to = centerOf(target.getBoundingClientRect());
  if (!from.x || !from.y || !to.x || !to.y) return;
  const chip = template.cloneNode(true) as HTMLElement;
  chip.style.display = "block";
  chip.style.opacity = "1";
  chip.style.left = `${to.x - rootRect.left}px`;
  chip.style.top = `${to.y - rootRect.top}px`;
  if (action.seat === 1) chip.classList.add("seat-b");
  root.appendChild(chip);
  chipNodes.push(chip);
  if (typeof chip.animate !== "function") {
    chip.remove();
    chipNodes = chipNodes.filter((node) => node !== chip);
    return;
  }
  const dx = from.x - to.x;
  const dy = from.y - to.y;
  const animation = chip.animate(
    [
      {
        transform: `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(0.7)`,
        opacity: 0,
      },
      { opacity: 1, offset: 0.18 },
      {
        transform: `translate(calc(-50% + ${dx * 0.45}px), calc(-50% + ${dy * 0.45}px - 14px)) scale(1)`,
        offset: 0.55,
      },
      { transform: "translate(-50%, -50%) scale(0.85)", opacity: 0.9 },
    ],
    {
      duration: 620,
      easing: "cubic-bezier(0.2, 0.8, 0.25, 1)",
      fill: "both",
    },
  );
  chipAnimations.push(animation);
  animation.onfinish = () => {
    chip.remove();
    chipNodes = chipNodes.filter((node) => node !== chip);
    chipAnimations = chipAnimations.filter((item) => item !== animation);
  };
}

watch(
  () => {
    const snap = snapshot.value;
    return [Number(snap?.hand_id || 0), (snap?.actions || []).length] as const;
  },
  ([handId, count]) => {
    const actions = snapshot.value?.actions || [];
    if (fxHandId !== handId) {
      clearChipFx();
      fxHandId = handId;
      fxActionCount = count;
      return;
    }
    if (count < fxActionCount) {
      fxActionCount = count;
      return;
    }
    if (count > fxActionCount) {
      for (let index = fxActionCount; index < actions.length; index += 1) {
        const action = actions[index];
        if (!action || !movesChips(action)) continue;
        potShockKey.value += 1;
        if (index === actions.length - 1) void playChipToss(action);
      }
      fxActionCount = count;
    }
  },
  { immediate: true },
);

watch(
  () => [game.tableTalk.length, board.busy, board.holeHold] as const,
  ([length, busy, hold]) => {
    if (!length) {
      talkTicker.value = null;
      return;
    }
    if (busy || hold) return;
    const latest = game.tableTalk[length - 1];
    if (!latest || talkTicker.value?.id === latest.id) return;
    talkTicker.value = latest;
    window.clearTimeout(talkTimer);
    talkTimer = window.setTimeout(() => {
      if (talkTicker.value?.id === latest.id) talkTicker.value = null;
    }, 12000);
  },
);

onUnmounted(() => {
  dealToken += 1;
  window.clearTimeout(dealTimer);
  window.clearTimeout(talkTimer);
  clearDealAnimations();
  clearChipFx();
});
</script>

<template>
  <section class="table-panel surface">
    <div class="table-toolbar">
      <div class="phase-track" :aria-label="streetText">
        <span class="street-now">{{ streetText }}</span>
        <div class="phase-dots" aria-hidden="true">
          <span
            v-for="(phase, index) in phases"
            :key="phase.key"
            class="phase-dot"
            :class="phaseState(index)"
            :title="phase.label"
          ></span>
        </div>
      </div>
      <span v-if="blindText" class="blind-level" :title="t('table.blinds')">
        {{ t("table.blinds") }} <strong>{{ blindText }}</strong>
      </span>
      <button
        v-if="talkTicker && !board.busy && !board.holeHold"
        class="talk-ticker"
        type="button"
        :title="`${talkTicker.name}: ${talkTicker.text}`"
        @click="openTalkFromTable"
      >
        <AppIcon name="message" :size="12" />
        <span class="talk-ticker-name">{{ talkTicker.name }}</span>
        <span class="talk-ticker-text">{{ talkTicker.text }}</span>
      </button>
    </div>

    <div class="felt-wrap">
      <div class="felt">
        <div class="felt-ring"></div>
        <div ref="feltInner" class="felt-inner">
          <div class="felt-watermark">♠</div>
          <span ref="flyChip" class="action-fly-chip" aria-hidden="true"></span>

          <template v-if="match">
            <div
              ref="deckOrigin"
              class="deal-deck"
              :class="{ active: dealing }"
              aria-hidden="true"
            >
              <PlayingCard class="deck-layer deck-layer-1" :face-down="true" size="mini" />
              <PlayingCard class="deck-layer deck-layer-2" :face-down="true" size="mini" />
              <PlayingCard class="deck-layer deck-layer-3" :face-down="true" size="mini" />
            </div>

            <div v-if="board.banner" :key="board.banner.key" class="street-banner">
              {{ board.banner.label }}
            </div>

            <SeatPod
              :player="playerAt(topSeat)"
              :seat-info="seatInfoAt(topSeat)"
              position="top"
              :active="snapshot?.acting_seat === topSeat && street !== 'finished' && !board.busy && !board.holeHold"
              :god-view="game.godView"
              :result="result"
              :show-cards="showCardsAt(topSeat)"
              :deal-nonce="board.dealNonce"
              :hole-hold="board.holeHold"
              :hold-assessment="board.busy"
              :last-action="topLastAction"
            />
            <SeatPod
              :player="playerAt(bottomSeat)"
              :seat-info="seatInfoAt(bottomSeat)"
              position="bottom"
              :active="snapshot?.acting_seat === bottomSeat && street !== 'finished' && !board.busy && !board.holeHold"
              :god-view="game.godView"
              :result="result"
              :show-cards="showCardsAt(bottomSeat)"
              :deal-nonce="board.dealNonce"
              :hole-hold="board.holeHold"
              :hold-assessment="board.busy"
              :last-action="bottomLastAction"
            />

            <div class="board-block">
              <div class="board-row">
                <div class="board">
                  <FlipCard
                    v-for="(card, index) in boardSlots"
                    :key="card?.code || `slot-${index}`"
                    :card="card"
                    :face-down="Boolean(card)"
                    :index="index"
                    :size="'community'"
                    :auto-reveal="true"
                    :flip-delay="50"
                    :instant="board.hydrated"
                    :dim="street === 'finished' && winnerSeats.length === 0"
                  />
                </div>
                <div ref="potEl" class="pot-display" :class="{ empty: pot <= 0 }">
                  <div class="pot-label"><span>{{ t("table.pot") }}</span></div>
                  <div class="pot-value">
                    <AnimatedNumber :key="`pot-${snapshot?.hand_id || 0}-${board.dealNonce}`" :value="pot" />
                  </div>
                  <span v-if="potShockKey" :key="potShockKey" class="pot-shock" aria-hidden="true"></span>
                </div>
              </div>
              <div v-if="boardCaption" class="board-caption">
                <span class="caption-line"></span>
                <span>{{ boardCaption }}</span>
              </div>
            </div>
          </template>

          <div v-if="!match" class="empty-overlay">
            <div class="empty-mark">♠</div>
            <p>{{ t("table.emptyTitle") }}</p>
            <span>{{ t("table.emptyBody") }}</span>
            <button class="btn btn-primary empty-cta" @click="openSetup">
              <AppIcon name="settings" :size="14" />
              {{ t("table.configure") }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <DeckStrip
      v-if="match && game.godView"
      :remaining="deckRemaining"
      :burned="burned"
      :count="deckCount"
      :dealing="board.holeHold"
    />

    <ActionBar v-if="match" />
  </section>
</template>

<style scoped>
.table-panel {
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.table-toolbar {
  display: flex;
  align-items: center;
  padding: 9px 16px;
  border-bottom: 1px solid var(--line);
  background: rgba(5, 13, 10, 0.35);
}

.phase-track {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.blind-level {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
  font-size: calc(9px * var(--type-scale));
  white-space: nowrap;
}

.blind-level strong {
  color: var(--gold);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.talk-ticker {
  margin-left: 8px;
  min-width: 0;
  max-width: min(58%, 560px);
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 9px;
  border-radius: 999px;
  border: 1px solid rgba(232, 195, 122, 0.24);
  background: rgba(232, 195, 122, 0.06);
  color: var(--muted);
  font-size: calc(9.5px * var(--type-scale));
  text-align: left;
  transition: border-color 0.16s ease, background 0.16s ease;
}

.talk-ticker:hover {
  border-color: rgba(232, 195, 122, 0.42);
  background: rgba(232, 195, 122, 0.1);
}

.talk-ticker .app-icon { color: var(--gold); }

.talk-ticker-name {
  flex: none;
  max-width: 96px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--gold);
  font-weight: 700;
}

.talk-ticker-text {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.street-now {
  font-size: calc(10.5px * var(--type-scale));
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--gold);
  white-space: nowrap;
}

.phase-dots {
  display: flex;
  align-items: center;
  gap: 4px;
}

.phase-dot {
  width: 16px;
  height: 3px;
  border-radius: 99px;
  background: var(--line-strong);
  transition: width 0.2s ease, background 0.2s ease;
}

.phase-dot.done { background: rgba(84, 214, 168, 0.42); }

.phase-dot.active {
  width: 26px;
  background: var(--gold);
  box-shadow: 0 0 10px rgba(232, 195, 122, 0.32);
}

.felt-wrap {
  padding: 14px 14px 8px;
}

.felt {
  position: relative;
  width: 100%;
  aspect-ratio: 1.62 / 1;
  min-height: 430px;
  max-height: 610px;
  border-radius: 46% / 50%;
  padding: 11px;
  background:
    linear-gradient(160deg, rgba(232, 195, 122, 0.16), rgba(232, 195, 122, 0.02) 38%, rgba(0, 0, 0, 0.3)),
    linear-gradient(180deg, #1b2b23, #0c1712);
  box-shadow:
    0 34px 70px rgba(0, 0, 0, 0.46),
    0 0 0 1px rgba(232, 195, 122, 0.14) inset,
    0 2px 0 rgba(255, 255, 255, 0.05) inset;
}

.felt-ring {
  position: absolute;
  inset: 7px;
  border-radius: 46% / 50%;
  border: 1px solid rgba(232, 195, 122, 0.16);
  pointer-events: none;
}

.felt-inner {
  position: relative;
  width: 100%;
  height: 100%;
  border-radius: 46% / 50%;
  overflow: hidden;
  background:
    radial-gradient(ellipse at 50% 42%, rgba(38, 112, 84, 0.72) 0%, rgba(20, 72, 54, 0.7) 38%, rgba(10, 40, 30, 0.92) 70%, #071d15 100%),
    radial-gradient(ellipse at 50% 100%, rgba(0, 0, 0, 0.32), transparent 60%);
  box-shadow:
    inset 0 0 80px rgba(0, 0, 0, 0.42),
    inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}

.felt-inner::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.3;
  background-image:
    repeating-radial-gradient(ellipse at 50% 50%, transparent 0, transparent 18px, rgba(255, 255, 255, 0.018) 19px, transparent 20px);
  pointer-events: none;
}

.felt-inner::after {
  content: "";
  position: absolute;
  left: 8%;
  right: 8%;
  top: 50%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(232, 195, 122, 0.1), transparent);
  pointer-events: none;
}

.felt-watermark {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -52%);
  font-size: clamp(calc(150px * var(--type-scale)), 26vw, calc(280px * var(--type-scale)));
  line-height: 1;
  color: rgba(255, 255, 255, 0.025);
  pointer-events: none;
  user-select: none;
}

.deal-deck {
  position: absolute;
  right: 4.5%;
  top: 31%;
  z-index: 3;
  width: 30px;
  height: 42px;
  transform: rotate(-7deg);
  pointer-events: none;
  opacity: 0.68;
  transition: opacity 0.2s ease, filter 0.2s ease;
}

.deal-deck.active {
  opacity: 1;
  filter: drop-shadow(0 0 12px rgba(232, 195, 122, 0.24));
}

.deck-layer {
  position: absolute;
  inset: 0;
}

.deck-layer-1 {
  transform: translate(-3px, -3px) rotate(-4deg);
  opacity: 0.5;
}

.deck-layer-2 {
  transform: translate(-1.5px, -1.5px) rotate(-2deg);
  opacity: 0.74;
}

.street-banner {
  position: absolute;
  left: 50%;
  top: 50%;
  z-index: 5;
  transform: translate(-50%, -50%);
  font-family: var(--font-mono);
  font-size: clamp(calc(34px * var(--type-scale)), 6vw, calc(72px * var(--type-scale)));
  font-weight: 800;
  letter-spacing: 0.3em;
  color: rgba(232, 195, 122, 0.12);
  white-space: nowrap;
  pointer-events: none;
  animation: street-banner 1s ease-out both;
}

@keyframes street-banner {
  0% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(1.12);
  }
  18% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(1);
  }
}

.board-block {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 6;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 7px;
}

.board-row {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.pot-display {
  position: absolute;
  left: calc(100% + 16px);
  top: 50%;
  transform: translateY(-50%);
  z-index: 5;
  display: flex;
  align-items: baseline;
  gap: 7px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(232, 195, 122, 0.2);
  background: rgba(5, 14, 10, 0.58);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(10px);
  white-space: nowrap;
}

.pot-display.empty { opacity: 0.38; }

.pot-label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-family: var(--font-mono);
  font-size: calc(8px * var(--type-scale));
  font-weight: 700;
  letter-spacing: 0.18em;
  color: var(--gold);
  opacity: 0.85;
}

.pot-value {
  font-family: var(--font-mono);
  font-size: clamp(calc(17px * var(--type-scale)), 1.7vw, calc(22px * var(--type-scale)));
  font-weight: 700;
  line-height: 1.1;
  color: var(--gold-bright);
  letter-spacing: -0.03em;
  text-shadow: 0 2px 14px rgba(232, 195, 122, 0.2);
}

.pot-shock {
  position: absolute;
  inset: -4px;
  border-radius: 999px;
  border: 1px solid rgba(232, 195, 122, 0.6);
  pointer-events: none;
  animation: pot-shock 0.62s ease-out both;
}

@keyframes pot-shock {
  from { opacity: 0.85; transform: scale(0.94); }
  to { opacity: 0; transform: scale(1.16); }
}

.action-fly-chip {
  position: absolute;
  left: 0;
  top: 0;
  width: 13px;
  height: 13px;
  display: none;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.52);
  background: linear-gradient(145deg, #f5d89d, #b98a3f);
  box-shadow: 0 0 10px rgba(232, 195, 122, 0.45), 0 2px 5px rgba(0, 0, 0, 0.36);
  pointer-events: none;
  z-index: 9;
}

.action-fly-chip.seat-b {
  background: linear-gradient(145deg, #b9d1ff, #4f7ed6);
  box-shadow: 0 0 10px rgba(121, 169, 255, 0.45), 0 2px 5px rgba(0, 0, 0, 0.36);
}

.board {
  display: flex;
  gap: 7px;
}

.board-caption {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: calc(9px * var(--type-scale));
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(232, 226, 210, 0.42);
  white-space: nowrap;
}

.caption-line {
  width: 20px;
  height: 1px;
  background: rgba(232, 195, 122, 0.28);
}

.caption-pot { color: rgba(232, 195, 122, 0.5); }

.empty-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  background:
    radial-gradient(ellipse at 50% 42%, rgba(16, 44, 33, 0.3), transparent 62%),
    rgba(4, 12, 9, 0.35);
  backdrop-filter: blur(2px);
  z-index: 10;
}

.empty-cta { margin-top: 10px; }

.empty-mark {
  font-size: calc(38px * var(--type-scale));
  color: rgba(232, 195, 122, 0.4);
  line-height: 1;
}

.empty-overlay p {
  margin: 6px 0 0;
  font-size: calc(15px * var(--type-scale));
  font-weight: 750;
  letter-spacing: -0.01em;
}

.empty-overlay span {
  font-size: calc(11px * var(--type-scale));
  color: var(--muted);
}

@media (max-width: 1180px) {
  .felt { min-height: 460px; }
}

@media (max-width: 720px) {
  .table-toolbar { padding: 8px 10px; }
  .talk-ticker { display: none; }
  .phase-track { gap: 9px; }
  .phase-dot { width: 12px; }
  .phase-dot.active { width: 22px; }
  .felt-wrap { padding: 8px 8px 4px; }
  .felt { height: 430px; min-height: 430px; aspect-ratio: auto; border-radius: 42% / 46%; }
  .felt-inner { height: 100%; border-radius: 42% / 46%; }
  .board { gap: 4px; }
  .deal-deck { display: none; }
  .street-banner { top: 50%; letter-spacing: 0.2em; }
  .pot-display { left: calc(100% + 10px); padding: 4px 9px; gap: 5px; }
  .pot-value { font-size: calc(16px * var(--type-scale)); }
}

@media (max-width: 560px) {
  .board-row { flex-direction: column; gap: 6px; }
  .pot-display {
    position: static;
    transform: none;
    padding: 4px 10px;
  }
}
</style>
