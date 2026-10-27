<script setup lang="ts">
import { computed } from "vue";

type IconDef = { paths?: string[]; circles?: Array<[number, number, number]>; lines?: Array<[number, number, number, number]>; rects?: Array<[number, number, number, number, number?]>; polygon?: string };

const ICONS: Record<string, IconDef> = {
  spade: { paths: ["M12 3c2.8 3.1 7.5 5.6 7.5 9.2A4.3 4.3 0 0 1 12 15a4.3 4.3 0 0 1-7.5-2.8C4.5 8.6 9.2 6.1 12 3Z", "M12 14.5c-.4 3-1.8 5-4 6.5h8c-2.2-1.5-3.6-3.5-4-6.5Z"] },
  eye: { paths: ["M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"], circles: [[12, 12, 3]] },
  eyeOff: { paths: ["M3 3l18 18", "M10.6 6.1A10.9 10.9 0 0 1 12 6c6 0 9.5 6 9.5 6a17 17 0 0 1-3.2 3.9", "M6.6 6.6C3.8 8.5 2.5 12 2.5 12S6 18 12 18c1.6 0 3-.4 4.3-1"], circles: [] },
  plus: { paths: ["M12 5v14", "M5 12h14"] },
  settings: { paths: ["M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z", "M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1 1.55V21a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1.1-1.55 1.7 1.7 0 0 0-1.88.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.55-1H3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.6 8.9a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.7 1.7 0 0 0 8.96 4.6 1.7 1.7 0 0 0 10 3.05V3a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87V9a1.7 1.7 0 0 0 1.55 1H21a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.55 1Z"] },
  chevronDown: { paths: ["m6 9 6 6 6-6"] },
  chevronRight: { paths: ["m9 18 6-6-6-6"] },
  check: { paths: ["M20 6 9 17l-5-5"] },
  alert: { paths: ["M10.3 4.2 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z", "M12 9v4", "M12 17h.01"] },
  sparkles: { paths: ["m12 3 1.8 4.9L18.7 9.7l-4.9 1.8L12 16.4l-1.8-4.9L5.3 9.7l4.9-1.8L12 3Z", "M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"] },
  terminal: { paths: ["m4 17 6-6-6-6", "M12 19h8"] },
  cards: { paths: ["M7 3h12a2 2 0 0 1 2 2v12", "M5 7h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z"] },
  chart: { paths: ["M3 3v18h18", "M7 16v-4", "M12 16V8", "M17 16v-6"] },
  bot: { paths: ["M12 8V4", "M8 4h8", "M6 8h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2Z"], circles: [[9, 13, 1], [15, 13, 1]], lines: [[9, 17, 15, 17]] },
  user: { paths: ["M20 21a8 8 0 0 0-16 0"], circles: [[12, 7, 4]] },
  brain: { paths: ["M9.5 4A3.5 3.5 0 0 0 6 7.5v.2A3.5 3.5 0 0 0 4 11v2a3.5 3.5 0 0 0 2 3.2v.3A3.5 3.5 0 0 0 9.5 20H12V4H9.5Z", "M14.5 4A3.5 3.5 0 0 1 18 7.5v.2a3.5 3.5 0 0 1 2 3.3v2a3.5 3.5 0 0 1-2 3.2v.3a3.5 3.5 0 0 1-3.5 3.5H12V4h2.5Z"] },
  wrench: { paths: ["M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.1-2.1 2.5-2.5Z"] },
  play: { paths: ["m8 5 11 7-11 7V5Z"] },
  refresh: { paths: ["M21 12a9 9 0 1 1-2.6-6.4", "M21 3v6h-6"] },
  x: { paths: ["M18 6 6 18", "M6 6l12 12"] },
  external: { paths: ["M15 3h6v6", "M10 14 21 3", "M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"] },
  clock: { circles: [[12, 12, 9]], paths: ["M12 7v5l3 2"] },
  coins: { circles: [[8, 8, 5], [16, 16, 5]], paths: [] },
  history: { paths: ["M3 12a9 9 0 1 0 2.7-6.4", "M3 4v5h5", "M12 8v4l3 2"] },
  zap: { paths: ["M13 2 4 14h7l-1 8 9-12h-7l1-8Z"] },
  dice: {
    rects: [[3, 3, 18, 18, 4]],
    paths: ["M8 8h.01", "M16 8h.01", "M12 12h.01", "M8 16h.01", "M16 16h.01"],
  },
  list: { paths: ["M8 6h13", "M8 12h13", "M8 18h13", "M3 6h.01", "M3 12h.01", "M3 18h.01"] },
  message: { paths: ["M21 15a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"] },
  copy: { paths: ["M8 8h11a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2V8Z", "M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"] },
  arrowRight: { paths: ["M5 12h14", "m13 6 6 6-6 6"] },
  shield: { paths: ["M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"] },
  flame: { paths: ["M12 22c4 0 7-2.7 7-6.5 0-3.8-3.5-7-5-10.5-1.5 3.5-3 4.5-3 7 0 1.5-1 2.5-2 2.5-1.2 0-2-1-2-2.5C5 15.5 5 22 12 22Z"] },
};

const props = withDefaults(
  defineProps<{
    name: string;
    size?: number;
    strokeWidth?: number;
    fill?: boolean;
  }>(),
  { size: 16, strokeWidth: 1.8, fill: false },
);

const def = computed(() => ICONS[props.name] || ICONS.wrench);
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    :fill="fill ? 'currentColor' : 'none'"
    :stroke="fill ? 'none' : 'currentColor'"
    :stroke-width="strokeWidth"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    class="app-icon"
  >
    <path v-for="(path, index) in def.paths || []" :key="`p-${index}`" :d="path" />
    <circle v-for="(circle, index) in def.circles || []" :key="`c-${index}`" :cx="circle[0]" :cy="circle[1]" :r="circle[2]" />
    <line v-for="(line, index) in def.lines || []" :key="`l-${index}`" :x1="line[0]" :y1="line[1]" :x2="line[2]" :y2="line[3]" />
    <rect v-for="(rect, index) in def.rects || []" :key="`r-${index}`" :x="rect[0]" :y="rect[1]" :width="rect[2]" :height="rect[3]" :rx="rect[4] ?? 0" />
  </svg>
</template>

<style scoped>
.app-icon { display: block; flex: none; }
</style>
