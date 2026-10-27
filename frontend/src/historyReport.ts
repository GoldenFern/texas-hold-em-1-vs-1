import { percent } from "./format";
import { t, type MessageKey } from "./i18n";

/**
 * Shape the public statistics returned by ``research_history`` into rows that
 * the tool card can render.  The report only carries preflop, postflop and
 * showdown metrics; match results and table-talk counts stay in the game state
 * and are deliberately not part of the statistics payload.
 */

export interface HistoryCell {
  value: number | null;
  n: number | null;
  confidence: string;
}

export interface HistoryMetric {
  key: string;
  label: string;
  you: HistoryCell;
  opponent: HistoryCell;
}

export interface HistoryGroup {
  key: string;
  label: string;
  metrics: HistoryMetric[];
}

export interface HistoryRecentHand {
  handId: number;
  won: boolean;
  net: number;
}

export interface HistoryCompleted {
  vsCurrent: boolean;
  matches: number;
  hands: number;
}

type MetricDef = { key: string; label: MessageKey };

const GROUP_ORDER = ["preflop", "postflop", "showdown"] as const;

const GROUP_LABELS: Record<string, MessageKey> = {
  preflop: "tool.history.group.preflop",
  postflop: "tool.history.group.postflop",
  showdown: "tool.history.group.showdown",
};

const SCOPE_LABELS: Record<string, MessageKey> = {
  current_match: "tool.history.scope.current",
  completed_matches: "tool.history.scope.completed",
  both: "tool.history.scope.both",
};

const POSITION_LABELS: Record<string, MessageKey> = {
  all: "tool.history.position.all",
  button: "tool.history.position.button",
  bb: "tool.history.position.bb",
};

const METRIC_DEFS: Record<string, MetricDef[]> = {
  preflop: [
    { key: "vpip", label: "tool.history.metric.vpip" },
    { key: "pfr", label: "tool.history.metric.pfr" },
    { key: "limp", label: "tool.history.metric.limp" },
    { key: "three_bet", label: "tool.history.metric.threeBet" },
    { key: "fold_to_three_bet", label: "tool.history.metric.foldToThreeBet" },
    { key: "steal", label: "tool.history.metric.steal" },
    { key: "fold_to_steal", label: "tool.history.metric.foldToSteal" },
  ],
  postflop: [
    { key: "c_bet", label: "tool.history.metric.cBet" },
    { key: "fold_to_c_bet", label: "tool.history.metric.foldToCBet" },
    { key: "check_raise", label: "tool.history.metric.checkRaise" },
    { key: "aggression", label: "tool.history.metric.aggression" },
  ],
  showdown: [
    { key: "wtsd", label: "tool.history.metric.wtsd" },
    { key: "w_sd", label: "tool.history.metric.wsd" },
  ],
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalizeCell(raw: unknown): HistoryCell {
  const record = asRecord(raw);
  if (!record) {
    return { value: asNumber(raw), n: null, confidence: "none" };
  }
  return {
    value: asNumber(record.value),
    n: asNumber(record.n),
    confidence: typeof record.confidence === "string" ? record.confidence : "none",
  };
}

/** Build the metric groups present in one ``current_match`` block. */
export function buildHistoryGroups(
  report: Record<string, unknown> | null | undefined,
): HistoryGroup[] {
  if (!report) return [];
  const you = asRecord(report.you);
  const opponent = asRecord(report.opponent);
  if (!you && !opponent) return [];

  const groups: HistoryGroup[] = [];
  for (const groupKey of GROUP_ORDER) {
    const defs = METRIC_DEFS[groupKey];
    if (!defs) continue;
    const youGroup = asRecord(you?.[groupKey]);
    const opponentGroup = asRecord(opponent?.[groupKey]);
    if (!youGroup && !opponentGroup) continue;

    const metrics: HistoryMetric[] = [];
    for (const def of defs) {
      const youRaw = youGroup?.[def.key];
      const opponentRaw = opponentGroup?.[def.key];
      if (youRaw === undefined && opponentRaw === undefined) continue;
      const youCell = normalizeCell(youRaw);
      const opponentCell = normalizeCell(opponentRaw);
      // Hide metrics neither side has had an opportunity for yet; keeping the
      // whole table would bury the three useful rows under twenty dashes.
      if (youCell.value === null && opponentCell.value === null) continue;
      metrics.push({
        key: def.key,
        label: t(def.label),
        you: youCell,
        opponent: opponentCell,
      });
    }
    if (!metrics.length) continue;
    groups.push({ key: groupKey, label: t(GROUP_LABELS[groupKey]), metrics });
  }
  return groups;
}

export function historyGroupLabel(group: string): string {
  const key = GROUP_LABELS[group];
  return key ? t(key) : group;
}

export function historyScopeLabel(scope: string): string {
  const key = SCOPE_LABELS[scope];
  return key ? t(key) : t("tool.history.scope.both");
}

export function historyPositionLabel(position: string): string {
  const key = POSITION_LABELS[position];
  return key ? t(key) : position;
}

export function formatHistoryPercent(cell: HistoryCell): string {
  if (cell.value === null || !Number.isFinite(cell.value)) return "—";
  return percent(cell.value);
}

/** Bar width for one percentage cell. */
export function historyBarWidth(cell: HistoryCell): string {
  if (cell.value === null || !Number.isFinite(cell.value)) return "0%";
  const width = Math.max(0, Math.min(100, cell.value * 100));
  return `${width > 0 ? Math.max(2, width) : 0}%`;
}

export function historyConfidenceClass(cell: HistoryCell): string {
  if (cell.confidence === "low" || cell.confidence === "medium" || cell.confidence === "high") {
    return cell.confidence;
  }
  return "none";
}

/** Latest hands from the asking seat's perspective. */
export function recentHistoryHands(
  report: Record<string, unknown> | null | undefined,
  limit = 8,
): HistoryRecentHand[] {
  const list = Array.isArray(report?.recent_hands) ? (report?.recent_hands as unknown[]) : [];
  const hands: HistoryRecentHand[] = [];
  for (const item of list) {
    const record = asRecord(item);
    const handId = record ? asNumber(record.hand_id) : null;
    if (!record || handId === null) continue;
    hands.push({
      handId: Math.round(handId),
      won: Boolean(record.you_won),
      net: asNumber(record.your_net) ?? 0,
    });
  }
  return hands.slice(-limit);
}

/** Compact line for the completed-match blocks, preferring the current opponent. */
export function completedHistory(
  report: Record<string, unknown> | null | undefined,
): HistoryCompleted | null {
  const completed = asRecord(report?.completed_matches);
  if (!completed) return null;
  const vs = asRecord(completed.vs_current_opponent);
  const overall = asRecord(completed.overall);

  let block: Record<string, unknown> | null = null;
  let vsCurrent = false;
  if (vs && (asNumber(vs.matches) ?? 0) > 0) {
    block = vs;
    vsCurrent = true;
  } else if (overall && (asNumber(overall.matches) ?? 0) > 0) {
    block = overall;
  }
  if (!block) return null;

  const matches = asNumber(block.matches) ?? 0;
  if (matches <= 0) return null;
  return {
    vsCurrent,
    matches,
    hands: asNumber(block.hands) ?? 0,
  };
}
