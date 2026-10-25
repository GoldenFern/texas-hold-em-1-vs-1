<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import AppIcon from "./AppIcon.vue";
import ReasoningBlock from "./ReasoningBlock.vue";
import RetryNotice from "./RetryNotice.vue";
import ToolCard from "./ToolCard.vue";
import { actionLabel, chips, duration, secondsSince, streetLabel, tokenText } from "../format";
import { t } from "../i18n";
import { turnPendingLabel } from "../turnPhase";
import type { TurnView } from "../types";

const props = defineProps<{ turn: TurnView; seatName: string; live: boolean; turnNumber: number }>();

const action = computed(() => (props.turn.action || null) as Record<string, unknown> | null);
const actionType = computed(() => String(action.value?.type || ""));
const actionAmount = computed(() => Number(action.value?.amount || 0));
const totalTokens = computed(() => {
  const usage = props.turn.usage || { input_tokens: 0, output_tokens: 0 };
  return Number(usage.input_tokens || 0) + Number(usage.output_tokens || 0);
});
const now = ref(Date.now());
let timer: number | null = null;

function syncTimer(): void {
  const running = props.turn.status === "running" && !props.turn.endedAt;
  if (running && timer === null) {
    timer = window.setInterval(() => {
      now.value = Date.now();
    }, 1000);
  } else if (!running && timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
}

watch(
  () => [props.turn.status, props.turn.endedAt],
  syncTimer,
  { immediate: true },
);
onUnmounted(() => {
  if (timer !== null) window.clearInterval(timer);
});

const elapsed = computed(() => {
  if (props.turn.latencyMs) return duration(props.turn.latencyMs);
  if (!props.turn.startedAt) return "";
  if (props.turn.endedAt) return duration(props.turn.endedAt - props.turn.startedAt);
  return secondsSince(props.turn.startedAt, now.value);
});
const statusText = computed(() => {
  if (props.turn.status === "running") {
    if (props.turn.retry) return t("console.statusRetrying");
    return props.live ? t("console.statusThinking") : t("console.statusRunning");
  }
  if (props.turn.auto) return t("console.statusAuto");
  return t("console.statusDone");
});
const pendingText = computed(() => turnPendingLabel(props.turn));
const streetClass = computed(() => props.turn.header.street || "preflop");
</script>

<template>
  <article class="turn" :class="[turn.status, { live, auto: turn.auto }]">
    <div class="turn-rail">
      <span class="rail-node" :class="streetClass"></span>
    </div>

    <div class="turn-content">
      <header class="turn-head">
        <div class="turn-head-main">
          <span class="turn-index mono">TURN {{ String(turnNumber || 1).padStart(2, "0") }}</span>
          <span class="street-pill" :class="streetClass">{{ streetLabel(turn.header.street) }}</span>
          <span v-if="turn.header.position" class="position-chip">
            {{ turn.header.position === "button" ? t("turn.positionBtn") : t("turn.positionBb") }}
          </span>
          <span class="faint mono hand-no">#{{ turn.header.handId }}</span>
          <span v-if="turn.header.receivedTalk" class="talk-received">
            <AppIcon name="message" :size="10" />
            {{ t("turn.receivedTalk", { count: turn.header.receivedTalk }) }}
          </span>
          <span class="turn-status" :class="{ live, auto: turn.auto }">
            <span v-if="turn.status === 'running'" class="dot live"></span>
            <AppIcon v-else-if="turn.auto" name="alert" :size="11" />
            <AppIcon v-else name="check" :size="11" />
            {{ statusText }}
          </span>
          <span class="turn-time mono">{{ elapsed }}</span>
        </div>
      </header>

      <details v-if="turn.stateText" class="state-input">
        <summary>
          <AppIcon name="terminal" :size="11" />
          <span>{{ t("turn.stateInput") }}</span>
          <span class="mono state-size">{{ t("common.chars", { count: turn.stateText.length }) }}</span>
          <span class="state-hint">{{ t("turn.stateHint") }}</span>
          <AppIcon name="chevronDown" :size="11" class="state-chevron" />
        </summary>
        <pre>{{ turn.stateText }}</pre>
      </details>

      <div class="messages">
        <div v-if="!turn.messages.length && turn.status === 'running'" class="thinking-placeholder">
          <span class="spinner"></span>
          <span>{{ t("turn.reading", { name: seatName }) }}</span>
        </div>

        <div v-for="message in turn.messages" :key="message.id" class="message" :class="message.status">
          <ReasoningBlock
            v-if="message.reasoning"
            :text="message.reasoning"
            :streaming="message.status === 'streaming'"
            :collapse="Boolean(message.content)"
          />

          <div v-if="message.content" class="assistant-text">{{ message.content }}</div>
          <div v-else-if="message.status === 'streaming' && !message.reasoning && !message.tools.length" class="content-shimmer">
            <span></span><span></span><span></span>
          </div>

          <div v-if="message.incomplete" class="incomplete-chip">{{ t("turn.incomplete") }}</div>

          <ToolCard v-for="tool in message.tools" :key="`${tool.index}-${tool.callId}`" :tool="tool" />

          <div v-if="message.error" class="message-error">
            <AppIcon name="alert" :size="13" />
            {{ message.error }}
          </div>

          <div v-if="message.status === 'done' && (message.usage || message.latencyMs)" class="message-meta">
            <span v-if="message.step" class="mono">STEP {{ message.step }}</span>
            <span v-if="message.latencyMs" class="mono">{{ duration(message.latencyMs) }}</span>
            <span v-if="message.usage" class="mono">
              {{ tokenText(message.usage.input_tokens) }} in / {{ tokenText(message.usage.output_tokens) }} out
            </span>
            <span v-if="message.finishReason" class="mono faint">{{ message.finishReason }}</span>
          </div>
        </div>
      </div>

      <RetryNotice v-if="turn.retry" :retry="turn.retry" />

      <div v-if="turn.notes.length" class="notes">
        <div v-for="(note, index) in turn.notes" :key="index" class="note" :class="note.level">
          <AppIcon :name="note.level === 'error' ? 'alert' : 'zap'" :size="11" />
          <span>{{ note.text }}</span>
        </div>
      </div>

      <footer class="turn-foot">
        <div v-if="action" class="final-action" :class="actionType">
          <span class="final-label">{{ t("turn.finalAction") }}</span>
          <strong>{{ actionLabel(actionType) }}<template v-if="actionAmount"> {{ chips(actionAmount) }}</template></strong>
          <span v-if="turn.auto" class="auto-chip">AUTO</span>
        </div>
        <div v-else-if="pendingText" class="final-pending">
          <span class="spinner"></span>
          {{ pendingText }}
        </div>
        <div class="turn-usage">
          <span v-if="turn.steps" class="mono">STEP {{ turn.steps }}</span>
          <span v-if="turn.contextTokens" class="mono">~{{ tokenText(turn.contextTokens) }} ctx</span>
          <span v-if="totalTokens" class="mono">{{ tokenText(totalTokens) }} tokens</span>
          <span v-if="turn.latencyMs" class="mono">{{ duration(turn.latencyMs) }}</span>
        </div>
      </footer>
      <div v-if="turn.reason" class="turn-reason">{{ turn.reason }}</div>
    </div>
  </article>
</template>

<style scoped>
.turn {
  position: relative;
  display: flex;
  gap: 10px;
  padding: 4px 0 12px;
  animation: turn-in 0.24s ease-out;
}

@keyframes turn-in {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: none; }
}

.turn-rail {
  width: 16px;
  flex: none;
  display: flex;
  justify-content: center;
  position: relative;
}

.turn-rail::before {
  content: "";
  position: absolute;
  top: 16px;
  bottom: -14px;
  width: 1px;
  background: linear-gradient(180deg, var(--line-strong), var(--line) 70%, transparent);
}

.turn:last-child .turn-rail::before { display: none; }

.rail-node {
  position: relative;
  z-index: 1;
  width: 9px;
  height: 9px;
  margin-top: 9px;
  border-radius: 50%;
  border: 2px solid var(--panel-solid);
  background: var(--faint);
  box-shadow: 0 0 0 1px var(--line-strong);
}

.rail-node.preflop { background: var(--gold); }
.rail-node.flop { background: var(--emerald); }
.rail-node.turn { background: var(--blue); }
.rail-node.river { background: var(--purple); }
.rail-node.finished, .rail-node.showdown { background: var(--muted); }

.turn-content {
  flex: 1;
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.018);
  overflow: hidden;
}

.turn.live .turn-content {
  border-color: rgba(232, 195, 122, 0.22);
  box-shadow: 0 0 0 1px rgba(232, 195, 122, 0.05) inset;
}

.turn-head {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 8px 10px 6px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.028), transparent);
  border-bottom: 1px solid var(--line);
}

.turn-head-main {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  flex-wrap: wrap;
}

.turn-index {
  font-size: calc(9px * var(--type-scale));
  font-weight: 800;
  letter-spacing: 0.14em;
  color: var(--faint);
}

.street-pill {
  display: inline-flex;
  align-items: center;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: calc(8.5px * var(--type-scale));
  font-weight: 800;
  letter-spacing: 0.08em;
  border: 1px solid var(--line);
  color: var(--muted);
}

.street-pill.preflop { color: var(--gold); border-color: rgba(232, 195, 122, 0.28); background: var(--gold-dim); }
.street-pill.flop { color: var(--emerald); border-color: rgba(84, 214, 168, 0.28); background: var(--emerald-dim); }
.street-pill.turn { color: var(--blue); border-color: rgba(121, 169, 255, 0.28); background: var(--blue-dim); }
.street-pill.river { color: var(--purple); border-color: rgba(196, 140, 255, 0.28); background: var(--purple-dim); }

.hand-no { font-size: calc(8.5px * var(--type-scale)); }

.turn-status {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: calc(8.5px * var(--type-scale));
  font-weight: 700;
  color: var(--faint);
  border: 1px solid var(--line);
}

.turn-status.live { color: var(--gold); border-color: rgba(232, 195, 122, 0.25); background: var(--gold-dim); }
.turn-status.auto { color: var(--amber); border-color: rgba(231, 179, 94, 0.28); background: rgba(231, 179, 94, 0.08); }

.turn-time { font-size: calc(8.5px * var(--type-scale)); color: var(--faint); }

.position-chip {
  font-family: var(--font-mono);
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
  letter-spacing: 0.08em;
}

.talk-received {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 5px;
  border-radius: 999px;
  border: 1px solid rgba(232, 195, 122, 0.24);
  background: rgba(232, 195, 122, 0.06);
  color: var(--gold);
  font-size: calc(8px * var(--type-scale));
  font-weight: 700;
  white-space: nowrap;
}

.state-input {
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.012);
}

.state-input > summary {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  cursor: pointer;
  list-style: none;
  font-size: calc(8.5px * var(--type-scale));
  color: var(--faint);
  user-select: none;
}

.state-input > summary::-webkit-details-marker { display: none; }
.state-input[open] > summary { color: var(--muted); }
.state-size { color: var(--faint); }
.state-hint { margin-left: auto; color: var(--faint); font-size: calc(8px * var(--type-scale)); }
.state-chevron { flex: none; transition: transform 0.15s ease; }
.state-input[open] .state-chevron { transform: rotate(180deg); }

.state-input pre {
  margin: 0;
  padding: 0 10px 8px;
  max-height: 280px;
  overflow: auto;
  font-family: var(--font-mono);
  font-size: calc(8.5px * var(--type-scale));
  line-height: 1.55;
  color: #8fa097;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.messages {
  padding: 7px 10px 4px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.message { min-width: 0; }

.assistant-text {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: calc(12.5px * var(--type-scale));
  line-height: 1.68;
  color: var(--text-soft);
  padding: 3px 0 6px;
}

.content-shimmer {
  display: flex;
  gap: 5px;
  padding: 9px 0 10px;
}

.content-shimmer span {
  width: 34px;
  height: 5px;
  border-radius: 99px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.04), rgba(232, 195, 122, 0.14), rgba(255, 255, 255, 0.04));
  background-size: 200% 100%;
  animation: shimmer 1.4s linear infinite;
}

.content-shimmer span:nth-child(2) { width: 58px; animation-delay: 0.1s; }
.content-shimmer span:nth-child(3) { width: 24px; animation-delay: 0.2s; }

@keyframes shimmer { to { background-position: -200% 0; } }

.thinking-placeholder {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px 0 10px;
  font-size: calc(10.5px * var(--type-scale));
  color: var(--faint);
}

.incomplete-chip {
  display: inline-flex;
  align-items: center;
  margin: 2px 0 5px;
  padding: 0 5px;
  border-radius: 999px;
  border: 1px solid rgba(231, 179, 94, 0.3);
  background: rgba(231, 179, 94, 0.08);
  color: var(--amber);
  font-size: calc(7.5px * var(--type-scale));
  font-weight: 700;
  letter-spacing: 0.04em;
}

.message-error {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 6px 8px;
  margin: 4px 0;
  border-radius: 7px;
  border: 1px solid rgba(226, 112, 95, 0.28);
  background: var(--red-dim);
  color: #ffb3a8;
  font-size: calc(10px * var(--type-scale));
  line-height: 1.5;
}

.message-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  padding: 2px 0 4px;
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
  letter-spacing: 0.02em;
}

.notes {
  padding: 0 10px 4px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.note {
  display: flex;
  align-items: flex-start;
  gap: 5px;
  font-size: calc(9.5px * var(--type-scale));
  line-height: 1.5;
  color: var(--faint);
}

.note.warn { color: var(--amber); }
.note.error { color: var(--red); }

.turn-foot {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 6px 10px 7px;
  border-top: 1px solid var(--line);
  background: rgba(4, 11, 8, 0.28);
}

.final-action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: calc(10px * var(--type-scale));
}

.final-label { color: var(--faint); font-size: calc(8.5px * var(--type-scale)); letter-spacing: 0.06em; }
.final-action strong { color: var(--emerald); font-size: calc(10.5px * var(--type-scale)); }
.final-action.fold strong { color: var(--red); }
.final-action.raise strong, .final-action.bet strong { color: var(--blue); }
.final-action.all_in strong { color: var(--purple); }
.final-action.check strong { color: var(--muted); }

.auto-chip {
  font-family: var(--font-mono);
  font-size: calc(7px * var(--type-scale));
  font-weight: 800;
  color: var(--amber);
  border: 1px solid rgba(231, 179, 94, 0.3);
  border-radius: 3px;
  padding: 0 3px;
}

.final-pending {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: calc(9.5px * var(--type-scale));
  color: var(--faint);
}

.turn-usage {
  margin-left: auto;
  display: flex;
  gap: 8px;
  font-size: calc(8px * var(--type-scale));
  color: var(--faint);
}

.turn-reason {
  padding: 0 10px 8px;
  font-size: calc(9px * var(--type-scale));
  color: var(--amber);
}
</style>
