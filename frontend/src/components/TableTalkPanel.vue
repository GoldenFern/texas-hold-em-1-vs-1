<script setup lang="ts">
import { computed } from "vue";
import AppIcon from "./AppIcon.vue";
import { iconUrl } from "../api";
import { seatDisplayName, streetLabel } from "../format";
import { t } from "../i18n";
import type { SeatInfo, TableTalkEntry } from "../types";

const props = defineProps<{
  entries: TableTalkEntry[];
  seats: SeatInfo[];
}>();

const groups = computed(() => {
  const map = new Map<number, TableTalkEntry[]>();
  for (const entry of props.entries) {
    const key = Number(entry.hand_id || 0);
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(entry);
  }
  return [...map.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([handId, entries]) => ({
      handId,
      entries: [...entries].sort((a, b) => Number(a.seq || 0) - Number(b.seq || 0)),
    }));
});

function seatInfo(seat: number): SeatInfo | null {
  return props.seats.find((item) => item.seat === seat) || null;
}

function seatName(entry: TableTalkEntry): string {
  const info = seatInfo(entry.seat);
  const raw = info?.name || entry.name || t("common.seat", { seat: entry.seat });
  return seatDisplayName(raw, info?.kind, info?.model_name) || raw;
}

function seatIcon(entry: TableTalkEntry): string {
  return seatInfo(entry.seat)?.icon || "";
}

function seatStyle(entry: TableTalkEntry): string {
  return seatInfo(entry.seat)?.table_talk_style_name || "";
}

function seatStyleAggressive(entry: TableTalkEntry): boolean {
  return Number(seatInfo(entry.seat)?.table_talk_style_intensity || 0) >= 2;
}
</script>

<template>
  <div class="talk-panel">
    <div class="talk-note">
      <AppIcon name="shield" :size="12" />
      <span>{{ t("talk.note") }}</span>
    </div>

    <div v-if="!entries.length" class="talk-empty">
      <div class="talk-empty-orb"><AppIcon name="message" :size="22" /></div>
      <strong>{{ t("talk.emptyTitle") }}</strong>
      <p>{{ t("talk.emptyBody") }}</p>
    </div>

    <section v-for="group in groups" :key="group.handId" class="talk-group">
      <div class="talk-hand">
        <span class="mono">HAND #{{ group.handId }}</span>
      </div>
      <article
        v-for="entry in group.entries"
        :key="entry.id"
        class="talk-entry"
        :class="entry.seat === 1 ? 'right' : 'left'"
      >
        <div class="talk-avatar" :class="{ provider: Boolean(seatIcon(entry)) }">
          <img v-if="seatIcon(entry)" :src="iconUrl(seatIcon(entry))" :alt="seatName(entry)" />
          <AppIcon v-else name="sparkles" :size="13" />
        </div>
        <div class="talk-content">
          <div class="talk-meta">
            <strong>{{ seatName(entry) }}</strong>
            <span
              v-if="seatStyle(entry)"
              class="talk-style"
              :class="{ aggressive: seatStyleAggressive(entry) }"
            >
              {{ seatStyle(entry) }}
            </span>
            <span v-if="entry.kind === 'final'" class="talk-final">{{ t("talk.final") }}</span>
            <span v-else class="mono">{{ streetLabel(entry.street) }}</span>
          </div>
          <div class="talk-bubble">{{ entry.text }}</div>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.talk-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px 2px 10px;
}

.talk-note {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 7px 9px;
  border-radius: 8px;
  border: 1px solid rgba(232, 195, 122, 0.2);
  background: rgba(232, 195, 122, 0.05);
  color: var(--muted);
  font-size: calc(9px * var(--type-scale));
  line-height: 1.55;
}

.talk-note .app-icon {
  color: var(--gold);
  margin-top: 1px;
}

.talk-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 7px;
  padding: 54px 24px;
  text-align: center;
}

.talk-empty-orb {
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  border: 1px solid rgba(232, 195, 122, 0.22);
  background: rgba(232, 195, 122, 0.06);
  color: var(--gold);
  margin-bottom: 3px;
}

.talk-empty strong {
  font-size: calc(12px * var(--type-scale));
  color: var(--text-soft);
}

.talk-empty p {
  margin: 0;
  max-width: 290px;
  color: var(--faint);
  font-size: calc(10px * var(--type-scale));
  line-height: 1.6;
}

.talk-group {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.talk-hand {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--gold);
  font-size: calc(8px * var(--type-scale));
  font-weight: 700;
  letter-spacing: 0.1em;
}

.talk-hand::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(232, 195, 122, 0.2), transparent);
}

.talk-entry {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  max-width: 94%;
}

.talk-entry.right {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.talk-avatar {
  width: 25px;
  height: 25px;
  flex: none;
  display: grid;
  place-items: center;
  border-radius: 7px;
  border: 1px solid var(--line-strong);
  background: rgba(255, 255, 255, 0.045);
  color: var(--text-soft);
  margin-top: 1px;
}

.talk-avatar.provider {
  background: #f1ecdf;
  padding: 3px;
}

.talk-avatar img {
  width: 17px;
  height: 17px;
  object-fit: contain;
}

.talk-content {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.talk-entry.right .talk-content {
  align-items: flex-end;
}

.talk-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: calc(8.5px * var(--type-scale));
  color: var(--faint);
}

.talk-meta strong {
  font-size: calc(9.5px * var(--type-scale));
  color: var(--text-soft);
  font-weight: 700;
}

.talk-final {
  padding: 0 5px;
  border-radius: 999px;
  border: 1px solid rgba(232, 195, 122, 0.34);
  background: rgba(232, 195, 122, 0.1);
  color: var(--gold);
  font-size: calc(7.5px * var(--type-scale));
  font-weight: 700;
  letter-spacing: 0.06em;
}

.talk-style {
  padding: 0 5px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
  color: var(--muted);
  font-size: calc(7.5px * var(--type-scale));
  white-space: nowrap;
}

.talk-style.aggressive {
  border-color: rgba(224, 138, 94, 0.35);
  color: var(--amber);
}

.talk-bubble {
  padding: 7px 10px;
  border-radius: 3px 10px 10px 10px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-soft);
  font-size: calc(11px * var(--type-scale));
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.talk-entry.right .talk-bubble {
  border-radius: 10px 3px 10px 10px;
  border-color: rgba(232, 195, 122, 0.2);
  background: rgba(232, 195, 122, 0.055);
}
</style>
