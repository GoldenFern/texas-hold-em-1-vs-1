<script setup lang="ts">
import { computed, onUnmounted, ref } from "vue";
import AppIcon from "./AppIcon.vue";
import { t } from "../i18n";
import type { RetryView } from "../types";

const props = defineProps<{ retry: RetryView }>();

const now = ref(Date.now());
const timer = window.setInterval(() => {
  now.value = Date.now();
}, 1000);
onUnmounted(() => window.clearInterval(timer));

const seconds = computed(() => Math.max(0, Math.round((props.retry.nextAt - now.value) / 1000)));
const info = computed(() =>
  seconds.value > 0
    ? t("turn.retry.waiting", { attempt: props.retry.attempt, seconds: seconds.value })
    : t("turn.retry.now", { attempt: props.retry.attempt }),
);
const scope = computed(() =>
  props.retry.scope === "summary" ? t("turn.retry.summary") : t("turn.retry.request"),
);
const message = computed(() => {
  const text = props.retry.message;
  return text.length > 80 ? `${text.slice(0, 80)}…` : text;
});
</script>

<template>
  <div class="retry-card" :title="retry.message">
    <AppIcon name="alert" :size="13" />
    <div class="retry-body">
      <div v-if="message" class="retry-message">{{ message }}</div>
      <div class="retry-info">
        <span class="retry-scope">{{ scope }}</span>
        <span>{{ info }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.retry-card {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  margin: 4px 0;
  padding: 7px 9px;
  border-radius: 8px;
  border: 1px solid rgba(231, 179, 94, 0.3);
  background: rgba(231, 179, 94, 0.07);
  color: var(--amber);
}

.retry-body {
  min-width: 0;
  flex: 1;
}

.retry-message {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: calc(10px * var(--type-scale));
  line-height: 1.45;
  color: #f0cf9a;
}

.retry-info {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 2px;
  font-size: calc(8.5px * var(--type-scale));
  line-height: 1.45;
  color: var(--faint);
}

.retry-scope {
  font-family: var(--font-mono);
  font-size: calc(7.5px * var(--type-scale));
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--amber);
}
</style>
