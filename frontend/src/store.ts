import { computed, reactive, watch } from "vue";
import { io, type Socket } from "socket.io-client";
import { api, ApiError } from "./api";
import {
  board,
  boardLocked,
  ingestSnapshot,
  resetBoardReveal,
} from "./boardReveal";
import { agentNameFor, providerDisplayName } from "./format";
import { locale, t } from "./i18n";
import type {
  ActionEntry,
  AgentEvent,
  AnalysisResult,
  Bootstrap,
  BotDecisionView,
  GameSnapshot,
  HandResult,
  MatchSetupPayload,
  MessageView,
  ProviderModel,
  RetryView,
  SeatSetup,
  TableTalkEntry,
  Toast,
  ToolView,
  TurnView,
} from "./types";

export type ConsoleMode = "stream" | "log" | "talk";

const STORAGE_KEY = "heads-up-lab.setup.v1";

interface StoreState {
  boot: Bootstrap | null;
  snapshot: GameSnapshot | null;
  connected: boolean;
  loading: boolean;
  fatalError: string;
  matchId: string | null;
  turns: Record<number, Array<TurnView | BotDecisionView>>;
  currentTurns: Record<number, TurnView | null>;
  seenEvents: Record<number, Set<number>>;
  lastEventIds: Record<number, number>;
  actionFeed: ActionEntry[];
  actionKeys: Set<string>;
  selectedSeat: number | null;
  pinned: boolean;
  consoleMode: ConsoleMode;
  tableTalk: TableTalkEntry[];
  tableTalkSeen: Set<string>;
  selectedAnalysisSeat: number | null;
  analysis: Record<number, AnalysisResult | null>;
  analysisLoading: boolean;
  analysisError: string;
  analysisKey: string;
  toasts: Toast[];
  setupOpen: boolean;
  lastResult: HandResult | null;
  lastResultHandId: number | null;
  godView: boolean;
  godViewBusy: boolean;
  busy: boolean;
  starting: boolean;
  aborting: boolean;
  testResults: Record<string, { status: "idle" | "testing" | "ok" | "error"; text: string; latency?: number }>;
}

const state = reactive<StoreState>({
  boot: null,
  snapshot: null,
  connected: false,
  loading: true,
  fatalError: "",
  matchId: null,
  turns: {},
  currentTurns: {},
  seenEvents: {},
  lastEventIds: {},
  actionFeed: [],
  actionKeys: new Set<string>(),
  selectedSeat: null,
  pinned: false,
  consoleMode: "stream",
  tableTalk: [],
  tableTalkSeen: new Set<string>(),
  selectedAnalysisSeat: null,
  analysis: {},
  analysisLoading: false,
  analysisError: "",
  analysisKey: "",
  toasts: [],
  setupOpen: false,
  lastResult: null,
  lastResultHandId: null,
  godView: true,
  godViewBusy: false,
  busy: false,
  starting: false,
  aborting: false,
  testResults: {},
});

let socket: Socket | null = null;
let toastSeq = 0;
let eventBuffer: AgentEvent[] | null = null;
let bufferDepth = 0;
let transcriptEpoch = 0;
let analysisSeq = 0;
let lastErrorToastAt = 0;

watch(locale, (next) => {
  if (!state.boot) return;
  void api.setLocale(next).catch(() => undefined);
});

export const game = state;

export const agentSeats = computed(() => {
  const seats = state.snapshot?.match?.seats || [];
  return seats.filter((seat) => seat.kind !== "human");
});

export const humanSeat = computed(() => {
  const seats = state.snapshot?.match?.seats || [];
  return seats.find((seat) => seat.kind === "human") || null;
});

export const selectedSeatInfo = computed(() => {
  if (state.selectedSeat === null) return null;
  return state.snapshot?.match?.seats.find((seat) => seat.seat === state.selectedSeat) || null;
});

export const isHumanTurn = computed(() => {
  const snap = state.snapshot;
  if (!snap?.has_match) return false;
  return snap.acting_kind === "human" && snap.street !== "finished";
});

export const canAct = computed(() => {
  const snap = state.snapshot;
  return Boolean(
    snap?.has_match
      && snap.match
      && !snap.match.match_over
      && isHumanTurn.value
      && !boardLocked.value,
  );
});

export function selectedTurns(): Array<TurnView | BotDecisionView> {
  if (state.selectedSeat === null) return [];
  return state.turns[state.selectedSeat] || [];
}

function toast(level: Toast["level"], title: string, detail = ""): void {
  const id = ++toastSeq;
  state.toasts.push({ id, level, title, detail });
  const ttl = level === "error" ? 9000 : 4500;
  window.setTimeout(() => dismissToast(id), ttl);
}

function dismissToast(id: number): void {
  const index = state.toasts.findIndex((item) => item.id === id);
  if (index >= 0) state.toasts.splice(index, 1);
}

function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

// ---------------------------------------------------------------------
// Bootstrap / setup
// ---------------------------------------------------------------------

function pickDefaultModel(): ProviderModel | null {
  const models = (state.boot?.providers || []).flatMap((provider) => provider.models);
  return models.find((model) => model.key_present && model.provider_id === "deepseek")
    || models.find((model) => model.key_present)
    || models[0]
    || null;
}

export function defaultSeat(kind: SeatSetup["kind"], model?: ProviderModel | null): SeatSetup {
  if (kind === "human") {
    return { kind: "human", name: t("common.you") };
  }
  if (kind === "bot") {
    return { kind: "bot", name: "RLCard", bot: { agent_type: "random", model_path: null, delay_seconds: 3 } };
  }
  const picked = model || pickDefaultModel();
  return {
    kind: "llm",
    name: picked
      ? agentNameFor(providerDisplayName(picked.provider_name, picked.provider_name_en), picked.name)
      : t("table.roleLlm"),
    provider_id: picked?.provider_id || "deepseek",
    model_id: picked?.id || "deepseek-flash",
    params: picked ? { ...picked.defaults } : {},
  };
}

export function buildDefaultSetup(): MatchSetupPayload {
  const defaults = state.boot?.game.defaults || {
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
    table_talk: true,
  };
  const model = pickDefaultModel();
  const saved = loadSavedSetup();
  if (saved) return saved;
  return {
    seats: [defaultSeat("llm", model), defaultSeat("human")],
    rules: {
      starting_chips: defaults.starting_chips,
      small_blind: defaults.small_blind,
      big_blind: defaults.big_blind,
      auto_rebuy: defaults.auto_rebuy,
      rebuy_amount: defaults.rebuy_amount,
      max_hands: defaults.max_hands,
      blind_increase_every_hands: defaults.blind_increase_every_hands ?? 0,
      blind_increase_factor: defaults.blind_increase_factor ?? 2,
      seed: defaults.seed,
      god_view: defaults.god_view,
    },
    table_talk: defaults.table_talk ?? true,
    agent_locale: "auto",
  };
}

function loadSavedSetup(): MatchSetupPayload | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as MatchSetupPayload;
    if (!parsed?.seats || parsed.seats.length !== 2 || !parsed.rules) return null;
    for (const seat of parsed.seats) {
      if (seat.kind === "llm") {
        const exists = (state.boot?.providers || []).some((provider) =>
          provider.models.some((model) => model.provider_id === seat.provider_id && model.id === seat.model_id),
        );
        if (!exists) return null;
      }
    }
    if (typeof parsed.table_talk !== "boolean") parsed.table_talk = true;
    if (!["auto", "zh-CN", "en-US"].includes(String(parsed.agent_locale || ""))) {
      parsed.agent_locale = "auto";
    }
    if (typeof parsed.rules.blind_increase_every_hands !== "number") {
      parsed.rules.blind_increase_every_hands = 0;
    }
    if (typeof parsed.rules.blind_increase_factor !== "number") {
      parsed.rules.blind_increase_factor = 2;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function saveSetup(payload: MatchSetupPayload): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Storage may be unavailable; setup still works for this session.
  }
}

export function modelFor(providerId: string, modelId: string): ProviderModel | null {
  const provider = state.boot?.providers.find((item) => item.id === providerId);
  return provider?.models.find((item) => item.id === modelId) || null;
}

export function providerOf(providerId: string) {
  return state.boot?.providers.find((item) => item.id === providerId) || null;
}

// ---------------------------------------------------------------------
// Boot / socket
// ---------------------------------------------------------------------

export async function boot(): Promise<void> {
  state.loading = true;
  try {
    state.boot = await api.bootstrap();
    const snapshot = await api.state();
    applySnapshot(snapshot);
    if (snapshot.has_match) {
      await refreshTranscripts();
    } else {
      state.setupOpen = true;
    }
    connectSocket();
  } catch (error) {
    state.fatalError = errorText(error);
  } finally {
    state.loading = false;
  }
}

function connectSocket(): void {
  if (socket) return;
  // Polling first gives engine.io an sid before the websocket upgrade. A
  // direct websocket connection ends with an empty WSGI response and makes
  // Werkzeug log a spurious "write() before start_response" on disconnect.
  socket = io({
    path: "/socket.io",
    transports: ["polling", "websocket"],
    query: { locale: locale.value },
  });
  socket.on("connect", () => {
    state.connected = true;
    if (state.snapshot?.has_match) void refreshTranscripts();
  });
  socket.on("disconnect", () => {
    state.connected = false;
  });
  socket.on("connect_error", (error: Error) => {
    state.connected = false;
    console.warn("socket connect error", error?.message || error);
  });
  socket.on("state", (snapshot: GameSnapshot) => {
    applySnapshot(snapshot);
  });
  socket.on("agent_event", (event: AgentEvent) => {
    handleAgentEvent(event);
  });
  socket.on("transcripts", (payload: { match_id: string; seats: Record<string, AgentEvent[]> }) => {
    applyTranscripts(payload);
  });
}

// ---------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------

function applySnapshot(snapshot: GameSnapshot): void {
  const previousSnapshot = state.snapshot;
  const previousMatchId = state.matchId;
  const nextMatchId = snapshot.match?.id || null;
  const sameMatch = Boolean(nextMatchId && nextMatchId === previousMatchId);
  const incomingSeq = Number(snapshot.seq || 0);
  const currentSeq = Number(state.snapshot?.seq || 0);
  // REST responses and socket broadcasts can arrive out of order. Never let
  // an older snapshot overwrite a newer one from the same match.
  if (sameMatch && incomingSeq && currentSeq && incomingSeq < currentSeq) {
    return;
  }
  // A delayed snapshot from a previous match must not pull the UI back.
  const incomingStarted = Number(snapshot.match?.started_at || 0);
  const currentStarted = Number(state.snapshot?.match?.started_at || 0);
  if (
    nextMatchId &&
    previousMatchId &&
    nextMatchId !== previousMatchId &&
    incomingStarted &&
    currentStarted &&
    incomingStarted < currentStarted
  ) {
    return;
  }
  if (nextMatchId !== previousMatchId) {
    resetMatchLocalState();
    state.matchId = nextMatchId;
  }
  ingestSnapshot(snapshot, previousSnapshot, sameMatch);
  state.snapshot = snapshot;
  state.godView = Boolean(snapshot.match?.god_view ?? true);
  mergeTableTalk(snapshot.table_talk || []);

  if (snapshot.match) {
    if (state.selectedSeat === null || !state.pinned) {
      const seats = snapshot.match.seats || [];
      const agent = seats.find((seat) => seat.kind !== "human");
      if (agent) state.selectedSeat = agent.seat;
      else if (seats.length) state.selectedSeat = seats[0].seat;
    }
    if (state.selectedAnalysisSeat === null) {
      const human = snapshot.match.seats?.find((seat) => seat.kind === "human");
      state.selectedAnalysisSeat = human?.seat ?? snapshot.match.seats?.[0]?.seat ?? 0;
    }
    if (snapshot.match.last_result && snapshot.match.last_result.hand_id !== state.lastResultHandId) {
      state.lastResult = snapshot.match.last_result;
      state.lastResultHandId = snapshot.match.last_result.hand_id;
    }
    if (snapshot.match.error && snapshot.match.error !== state.fatalError) {
      state.fatalError = snapshot.match.error;
    }
    if (snapshot.match.aborted) {
      for (const seat of [0, 1]) {
        const turn = state.currentTurns[seat];
        if (turn && turn.status === "running") {
          turn.status = "done";
          turn.endedAt = turn.endedAt ?? Date.now() / 1000;
          turn.reason = t("table.matchAborted");
        }
      }
    }
    state.godView = Boolean(snapshot.match.god_view);
  } else {
    state.selectedSeat = null;
    state.selectedAnalysisSeat = null;
  }

  addActions(snapshot);
  followActingSeat(snapshot);
}

watch(boardLocked, (locked) => {
  if (!locked) return;
  // An analysis response that lands mid-reveal would describe cards the user
  // has not seen yet. Drop it and let App.vue request a fresh one afterwards.
  analysisSeq += 1;
  state.analysisLoading = false;
});

function followActingSeat(snapshot: GameSnapshot): void {
  if (state.pinned) return;
  if (snapshot.acting_seat === null || snapshot.acting_seat === undefined) return;
  const seat = snapshot.match?.seats.find((item) => item.seat === snapshot.acting_seat);
  if (seat && seat.kind !== "human") {
    state.selectedSeat = seat.seat;
  }
}

function addActions(snapshot: GameSnapshot): void {
  const actions = snapshot.actions || [];
  const baseHand = snapshot.hand_id || 0;
  actions.forEach((action, index) => {
    const key = `${action.hand_id ?? baseHand}:${index}`;
    if (state.actionKeys.has(key)) return;
    state.actionKeys.add(key);
    state.actionFeed.push({
      id: key,
      handId: action.hand_id ?? baseHand,
      street: action.street,
      seat: action.seat,
      type: action.type,
      label: action.label,
      amount: action.amount,
      potAfter: action.pot_after,
      auto: action.auto,
      source: action.source,
      at: Date.now(),
    });
  });
  if (state.actionFeed.length > 400) state.actionFeed.splice(0, state.actionFeed.length - 400);
  if (state.actionKeys.size > 2000) {
    state.actionKeys = new Set(state.actionFeed.map((item) => item.id));
  }
}

function mergeTableTalk(entries: TableTalkEntry[]): void {
  let added = false;
  for (const entry of entries) {
    if (!entry?.id || state.tableTalkSeen.has(entry.id)) continue;
    state.tableTalkSeen.add(entry.id);
    state.tableTalk.push({ ...entry });
    added = true;
  }
  if (!added) return;
  state.tableTalk.sort((a, b) => Number(a.seq || 0) - Number(b.seq || 0));
  if (state.tableTalk.length > 200) {
    state.tableTalk.splice(0, state.tableTalk.length - 200);
    state.tableTalkSeen = new Set(state.tableTalk.map((item) => item.id));
  }
}

function tableTalkFromEvent(event: AgentEvent): TableTalkEntry {
  return {
    id: String(event.id || `talk-${event.event_id || Date.now()}`),
    seq: Number(event.seq || 0),
    hand_id: Number(event.hand_id || state.snapshot?.hand_id || 0),
    street: String(event.street || state.snapshot?.street || ""),
    seat: Number(event.seat ?? 0),
    name: String(event.name || ""),
    text: String(event.text || ""),
    turn_id: event.turn_id ? String(event.turn_id) : undefined,
    created_at: Number(event.created_at || 0) || undefined,
    kind: event.kind === "final" ? "final" : "turn",
  };
}

function resetMatchLocalState(): void {
  analysisSeq += 1;
  state.turns = {};
  state.currentTurns = {};
  state.seenEvents = {};
  state.lastEventIds = {};
  state.actionFeed = [];
  state.actionKeys = new Set();
  state.tableTalk = [];
  state.tableTalkSeen = new Set();
  state.consoleMode = "stream";
  state.analysis = {};
  state.analysisKey = "";
  state.analysisError = "";
  state.analysisLoading = false;
  state.selectedAnalysisSeat = null;
  state.lastResult = null;
  state.lastResultHandId = null;
  resetBoardReveal();
  state.fatalError = "";
  state.selectedSeat = null;
  state.pinned = false;
}

// ---------------------------------------------------------------------
// Transcripts
// ---------------------------------------------------------------------

export async function refreshTranscripts(): Promise<void> {
  if (!state.snapshot?.has_match) return;
  const epoch = ++transcriptEpoch;
  beginEventBuffer();
  try {
    const payload = await api.transcripts();
    if (epoch !== transcriptEpoch) return;
    applyTranscripts(payload);
  } catch (error) {
    console.warn("transcript refresh failed", errorText(error));
  } finally {
    endEventBuffer();
  }
}

function beginEventBuffer(): void {
  bufferDepth += 1;
  if (eventBuffer === null) eventBuffer = [];
}

function endEventBuffer(): void {
  bufferDepth = Math.max(0, bufferDepth - 1);
  if (bufferDepth === 0) eventBuffer = null;
}

function applyTranscripts(payload: { match_id: string; seats: Record<string, AgentEvent[]> }): void {
  if (payload.match_id !== state.matchId) return;
  resetTranscripts();
  for (const [seatKey, events] of Object.entries(payload.seats || {})) {
    const seat = Number(seatKey);
    for (const event of events) handleAgentEvent({ ...event, seat }, true);
  }
  if (eventBuffer) {
    for (const event of eventBuffer) handleAgentEvent(event, true);
  }
}

function resetTranscripts(): void {
  state.turns = {};
  state.currentTurns = {};
  state.seenEvents = {};
  state.lastEventIds = {};
}

function handleAgentEvent(event: AgentEvent, replay = false): void {
  if (eventBuffer && !replay) {
    eventBuffer.push(event);
  }
  if (event.match_id && state.matchId && event.match_id !== state.matchId) return;
  const seat = Number(event.seat ?? 0);
  const eventId = Number(event.event_id ?? 0);
  if (eventId) {
    const seen = (state.seenEvents[seat] ||= new Set<number>());
    if (seen.has(eventId)) return;
    seen.add(eventId);
    if (seen.size > 20000) {
      const cutoff = eventId - 20000;
      for (const id of seen) {
        if (id < cutoff) seen.delete(id);
      }
    }
    state.lastEventIds[seat] = Math.max(state.lastEventIds[seat] || 0, eventId);
  }
  if (!state.turns[seat]) state.turns[seat] = [];
  switch (String(event.type)) {
    case "turn_start":
      onTurnStart(seat, event);
      break;
    case "turn_state": {
      const turn = currentTurn(seat);
      if (turn) turn.stateText = String(event.text || "");
      break;
    }
    case "message_start":
      onMessageStart(seat, event);
      break;
    case "reasoning_delta":
    case "content_delta":
      onDelta(seat, event);
      break;
    case "tool_call_start":
      onToolStart(seat, event);
      break;
    case "tool_call_args_delta":
      onToolArgsDelta(seat, event);
      break;
    case "tool_call":
      onToolCall(seat, event);
      break;
    case "tool_result":
      onToolResult(seat, event);
      break;
    case "message_end":
      onMessageEnd(seat, event);
      break;
    case "turn_end":
      onTurnEnd(seat, event);
      break;
    case "llm_request": {
      const turn = currentTurn(seat);
      if (turn) {
        turn.contextTokens = Number(event.estimated_context_tokens || turn.contextTokens || 0);
      }
      break;
    }
    case "retry_scheduled":
      onRetryScheduled(seat, event);
      break;
    case "note":
      addNote(seat, String(event.level || "info"), String(event.text || ""), event.turn_id as string | undefined);
      break;
    case "error":
      onError(seat, event);
      break;
    case "table_talk":
      mergeTableTalk([tableTalkFromEvent(event)]);
      break;
    case "bot_decision":
      onBotDecision(seat, event);
      break;
    case "hand_result":
      onHandResultEvent(event);
      break;
    case "match_over":
      toast("info", t("errors.matchOver"), String(event.text || ""));
      break;
    default:
      break;
  }
}

function currentTurn(seat: number): TurnView | null {
  return state.currentTurns[seat] || null;
}

function onTurnStart(seat: number, event: AgentEvent): void {
  const running = state.currentTurns[seat];
  if (running && running.status === "running") {
    running.status = "done";
    running.retry = null;
    running.endedAt = Date.now();
  }
  const turn: TurnView = {
    id: String(event.turn_id || `turn-${Date.now()}`),
    header: {
      handId: Number(event.hand_id || 0),
      handNumber: Number(event.hand_number || event.hand_id || 0),
      street: String(event.street || "preflop"),
      pot: Number(event.pot || 0),
      myChips: Number(event.my_chips || 0),
      opponentChips: Number(event.opponent_chips || 0),
      toCall: Number(event.to_call || 0),
      legalActions: Array.isArray(event.legal_actions) ? (event.legal_actions as string[]) : [],
      position: String(event.position || ""),
      receivedTalk: Number(event.table_talk_received || 0),
    },
    messages: [],
    notes: [],
    stateText: "",
    action: null,
    auto: false,
    status: "running",
    usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: 0 },
    latencyMs: 0,
    steps: 0,
    contextTokens: 0,
    startedAt: Date.now(),
    endedAt: null,
  };
  state.turns[seat].push(turn);
  if (state.turns[seat].length > 80) {
    state.turns[seat].splice(0, state.turns[seat].length - 80);
  }
  state.currentTurns[seat] = turn;
  if (!state.pinned) state.selectedSeat = seat;
}

function ensureTurn(seat: number, event: AgentEvent): TurnView {
  let turn = currentTurn(seat);
  if (!turn) {
    onTurnStart(seat, { ...event, type: "turn_start" });
    turn = currentTurn(seat)!;
  }
  return turn;
}

function messageById(turn: TurnView, id: string): MessageView | null {
  return turn.messages.find((message) => message.id === id) || null;
}

function onMessageStart(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  turn.retry = null;
  const id = String(event.message_id || `msg-${Date.now()}`);
  if (messageById(turn, id)) return;
  turn.messages.push({
    id,
    content: "",
    reasoning: "",
    tools: [],
    status: "streaming",
    step: Number(event.step || turn.messages.length + 1),
  });
}

function onDelta(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  const id = String(event.message_id || "");
  let message = messageById(turn, id);
  if (!message) {
    onMessageStart(seat, event);
    message = messageById(turn, id);
  }
  if (!message) return;
  turn.retry = null;
  const delta = String(event.delta || "");
  if (event.type === "reasoning_delta") message.reasoning += delta;
  else message.content += delta;
}

function toolByIndex(message: MessageView, index: number): ToolView | null {
  return message.tools.find((tool) => tool.index === index) || null;
}

function onToolStart(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  turn.retry = null;
  const messageId = String(event.message_id || "");
  let message = messageById(turn, messageId);
  if (!message) {
    onMessageStart(seat, event);
    message = messageById(turn, messageId);
  }
  if (!message) return;
  const index = Number(event.index ?? message.tools.length);
  let tool = toolByIndex(message, index);
  if (!tool) {
    tool = {
      index,
      callId: String(event.call_id || ""),
      name: String(event.name || ""),
      argsText: "",
      parsed: null,
      status: "running",
      result: null,
      summary: "",
      durationMs: null,
    };
    message.tools.push(tool);
  } else {
    if (event.call_id) tool.callId = String(event.call_id);
    if (event.name) tool.name = String(event.name);
  }
}

function onToolArgsDelta(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  const message = messageById(turn, String(event.message_id || ""));
  if (!message) return;
  const index = Number(event.index ?? 0);
  let tool = toolByIndex(message, index);
  if (!tool) {
    onToolStart(seat, { ...event, type: "tool_call_start" });
    tool = toolByIndex(message, index);
  }
  if (tool) tool.argsText += String(event.delta || "");
}

function onToolCall(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  const message = messageById(turn, String(event.message_id || ""));
  if (!message) return;
  const index = Number(event.index ?? 0);
  let tool = toolByIndex(message, index);
  if (!tool) {
    onToolStart(seat, { ...event, type: "tool_call_start" });
    tool = toolByIndex(message, index);
  }
  if (!tool) return;
  tool.name = String(event.name || tool.name);
  tool.callId = String(event.call_id || tool.callId);
  tool.argsText = String(event.arguments || tool.argsText || "");
  tool.parsed = (event.parsed as Record<string, unknown> | null) ?? null;
  if (event.parse_error) tool.status = "error";
}

function onToolResult(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  const messageId = String(event.message_id || "");
  const message = messageId ? messageById(turn, messageId) : turn.messages[turn.messages.length - 1] || null;
  if (!message) return;
  const index = Number(event.index ?? 0);
  let tool = toolByIndex(message, index);
  if (!tool) {
    tool = {
      index,
      callId: String(event.call_id || ""),
      name: String(event.name || "tool"),
      argsText: "",
      parsed: null,
      status: "running",
      result: null,
      summary: "",
      durationMs: null,
    };
    message.tools.push(tool);
  }
  tool.name = String(event.name || tool.name);
  tool.callId = String(event.call_id || tool.callId);
  tool.status = event.ok ? "done" : "error";
  tool.result = (event.result as Record<string, unknown> | null) ?? null;
  tool.summary = String(event.summary || "");
  tool.durationMs = Number(event.duration_ms ?? 0);
  tool.ignored = Boolean(event.ignored);
}

function onMessageEnd(seat: number, event: AgentEvent): void {
  const turn = ensureTurn(seat, event);
  const id = String(event.message_id || "");
  let message = messageById(turn, id);
  if (!message) {
    onMessageStart(seat, event);
    message = messageById(turn, id);
  }
  if (!message) return;
  turn.retry = null;
  if (event.content) message.content = String(event.content);
  if (event.reasoning) message.reasoning = String(event.reasoning);
  message.incomplete = Boolean(event.incomplete);
  message.incompleteReason = String(event.incomplete_reason || "");
  if (message.incomplete) {
    for (const tool of message.tools) {
      if (tool.status === "running") {
        tool.status = "error";
        tool.summary = tool.summary || t("turn.toolIncomplete");
      }
    }
  }
  message.status = event.error || message.incomplete ? "error" : "done";
  message.usage = (event.usage as MessageView["usage"]) || message.usage;
  message.latencyMs = Number(event.latency_ms ?? 0);
  message.finishReason = String(event.finish_reason || "");
  if (event.error && !message.incomplete) message.error = t("errors.llmFailed");
}

function onTurnEnd(seat: number, event: AgentEvent): void {
  const turn = currentTurn(seat);
  if (!turn) return;
  turn.action = (event.action as Record<string, unknown> | null) ?? null;
  turn.auto = Boolean(event.auto);
  turn.status = event.auto ? "done" : "done";
  turn.usage = (event.usage as TurnView["usage"]) || turn.usage;
  turn.latencyMs = Number(event.latency_ms ?? 0);
  turn.steps = Number(event.steps ?? turn.steps);
  turn.reason = String(event.reason || "");
  turn.retry = null;
  turn.endedAt = Date.now();
  state.currentTurns[seat] = null;
}

function onRetryScheduled(seat: number, event: AgentEvent): void {
  const turn = currentTurn(seat);
  if (!turn) return;
  const delayMs = Number(event.delay_ms ?? 0);
  const nextAt = Number(event.at ?? Date.now() + delayMs);
  const retry: RetryView = {
    scope: String(event.scope || "turn"),
    attempt: Number(event.attempt ?? 1),
    maxRetries: Number(event.max_retries ?? 0),
    delayMs,
    nextAt,
    message: String(event.text || ""),
  };
  turn.retry = retry;
}

function addNote(seat: number, level: string, text: string, turnId?: string): void {
  if (!text) return;
  let turn = currentTurn(seat);
  if (!turn && turnId) {
    turn = (state.turns[seat] || []).find((item) => item.id === turnId && "messages" in item) as TurnView | undefined || null;
  }
  if (turn) {
    turn.notes.push({ level, text });
  } else if (level === "error" || level === "warn") {
    toast(level === "error" ? "error" : "warn", text);
  }
}

function onError(seat: number, event: AgentEvent): void {
  const text = String(event.text || t("errors.unknown"));
  addNote(seat, "error", text, event.turn_id as string | undefined);
  const messageId = String(event.message_id || "");
  if (messageId) {
    const turn = currentTurn(seat);
    const message = turn ? messageById(turn, messageId) : null;
    if (message) {
      message.status = "error";
      message.error = text;
    }
  }
  const now = Date.now();
  if (now - lastErrorToastAt > 2500) {
    lastErrorToastAt = now;
    toast("error", t("errors.agentCall"), text);
  }
}

function onBotDecision(seat: number, event: AgentEvent): void {
  const turn: BotDecisionView = {
    id: `bot-${event.event_id || Date.now()}`,
    type: "bot",
    agentType: String(event.agent_type || "random"),
    actionId: Number(event.action_id ?? 0),
    legalIds: Array.isArray(event.legal_ids) ? (event.legal_ids as number[]) : [],
    probabilities: (event.probabilities as Record<string, number>) || {},
    action: null,
    mappedAction: (event.mapped as Record<string, unknown> | undefined) || null,
    handId: Number(event.hand_id ?? state.snapshot?.hand_id ?? 0),
    street: String(event.street || state.snapshot?.street || ""),
    startedAt: Date.now(),
    status: "done",
  };
  state.turns[seat].push(turn);
  if (state.turns[seat].length > 80) state.turns[seat].splice(0, state.turns[seat].length - 80);
}

function onHandResultEvent(event: AgentEvent): void {
  const result = event.result as HandResult | undefined;
  if (!result) return;
  if (result.hand_id === state.lastResultHandId) return;
  state.lastResult = result;
  state.lastResultHandId = result.hand_id;
}

// ---------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------

export async function startMatch(payload: MatchSetupPayload): Promise<boolean> {
  state.starting = true;
  state.busy = true;
  beginEventBuffer();
  try {
    const agentLocale = !payload.agent_locale || payload.agent_locale === "auto"
      ? locale.value
      : payload.agent_locale;
    const apiPayload: MatchSetupPayload = {
      ...payload,
      agent_locale: agentLocale,
      seats: [
        payload.seats[0].kind === "human"
          ? { ...payload.seats[0], name: t("common.you") }
          : payload.seats[0],
        payload.seats[1].kind === "human"
          ? { ...payload.seats[1], name: t("common.you") }
          : payload.seats[1],
      ] as [SeatSetup, SeatSetup],
    };
    const response = await api.newGame(apiPayload);
    saveSetup(payload);
    // Close the setup sheet before applying the first snapshot. Some fast
    // bot-vs-bot hands finish before the response returns.
    state.setupOpen = false;
    applySnapshot(response.state);
    await refreshTranscripts();
    if (response.state.match?.aborted) {
      toast("info", t("errors.aborted"), t("errors.abortedDetail"));
      return false;
    }
    for (const seat of response.state.match?.seats || []) {
      if (seat.agent?.warning) {
        toast("warn", t("errors.degradedTitle", { name: seat.name }), seat.agent.warning);
      }
    }
    const actual = response.state.match?.rules || payload.rules;
    toast(
      "success",
      t("errors.startSuccess"),
      t("errors.startDetail", {
        small: actual.small_blind,
        big: actual.big_blind,
        chips: Number(actual.starting_chips).toLocaleString("en-US"),
      }),
    );
    return true;
  } catch (error) {
    const timedOut = error instanceof ApiError && error.kind === "timeout";
    const title = timedOut
      ? t("errors.startTimeout")
      : t("errors.startFailed");
    toast("error", title, errorText(error));
    if (timedOut) {
      // The server may still be building the match after the client gave up.
      // Cancel it so a ghost match does not appear later.
      void abortMatch(true);
    }
    return false;
  } finally {
    endEventBuffer();
    state.starting = false;
    state.busy = false;
  }
}

export async function abortMatch(silent = false): Promise<boolean> {
  if (state.aborting) return false;
  state.aborting = true;
  try {
    const response = await api.abortGame();
    if (response.state) applySnapshot(response.state);
    if (!silent) {
      toast("info", t("errors.abortSuccess"), t("errors.abortDetail"));
    }
    return true;
  } catch (error) {
    if (!silent) {
      toast("error", t("errors.abortFailed"), errorText(error));
    }
    return false;
  } finally {
    state.aborting = false;
  }
}

export async function humanAction(action: string, amount = 0): Promise<void> {
  if (state.busy) return;
  const decisionId = state.snapshot?.decision_id;
  // Without a decision id the view is stale; sending the action would answer
  // whichever decision is current on the server instead of the one on screen.
  if (!decisionId) return;
  state.busy = true;
  try {
    const response = await api.action({ action, amount, decision_id: decisionId });
    if (response.state) applySnapshot(response.state);
  } catch (error) {
    toast("error", t("errors.actionFailed"), errorText(error));
  } finally {
    state.busy = false;
  }
}

export async function nextHand(): Promise<void> {
  if (state.busy || boardLocked.value) return;
  state.busy = true;
  try {
    const response = await api.nextHand();
    applySnapshot(response.state);
  } catch (error) {
    toast("error", t("errors.nextHandFailed"), errorText(error));
  } finally {
    state.busy = false;
  }
}

export async function toggleGodView(): Promise<void> {
  if (state.godViewBusy) return;
  state.godViewBusy = true;
  try {
    const response = await api.setGodView(!state.godView);
    applySnapshot(response.state);
    if (!state.godView) {
      const human = response.state.match?.seats.find((seat) => seat.kind === "human");
      if (human) {
        state.selectedAnalysisSeat = human.seat;
        state.analysisKey = "";
      }
    }
  } catch (error) {
    toast("error", t("errors.godViewFailed"), errorText(error));
  } finally {
    state.godViewBusy = false;
  }
}

export function openSetup(): void {
  state.setupOpen = true;
}

export function closeSetup(): void {
  state.setupOpen = false;
}

export function selectSeat(seat: number, pin = false): void {
  state.selectedSeat = seat;
  if (pin) state.pinned = true;
}

export function setConsoleMode(mode: ConsoleMode): void {
  state.consoleMode = mode;
}

export function openTableTalk(): void {
  if (!state.snapshot?.match?.table_talk && !state.tableTalk.length) return;
  state.consoleMode = "talk";
}

export function followLive(): void {
  state.pinned = false;
  const snap = state.snapshot;
  if (snap?.acting_seat !== null && snap?.acting_seat !== undefined) {
    const seat = snap.match?.seats.find((item) => item.seat === snap.acting_seat);
    if (seat && seat.kind !== "human") state.selectedSeat = seat.seat;
  }
}

export function selectAnalysisSeat(seat: number): void {
  state.selectedAnalysisSeat = seat;
  state.analysisKey = "";
  void refreshAnalysis(true);
}

export async function refreshAnalysis(force = false): Promise<void> {
  const snap = state.snapshot;
  if (!snap?.has_match || boardLocked.value) return;
  const seat = state.selectedAnalysisSeat;
  if (seat === null || seat === undefined) return;
  const player = snap.players?.find((item) => item.seat === seat);
  if (player && player.kind !== "human" && !snap.match?.god_view) {
    analysisSeq += 1;
    state.analysis[seat] = null;
    state.analysisError = t("errors.godViewRequired");
    state.analysisKey = "";
    state.analysisLoading = false;
    return;
  }
  const key = `${snap.hand_id}:${snap.street}:${seat}:${snap.community?.length || 0}`;
  if (!force && key === state.analysisKey) return;
  state.analysisKey = key;
  state.analysisLoading = true;
  state.analysisError = "";
  const requestSeq = ++analysisSeq;
  const requestMatchId = state.matchId;
  try {
    const response = await api.analyze(seat, 30000);
    if (
      requestSeq !== analysisSeq
      || requestMatchId !== state.matchId
      || key !== state.analysisKey
    ) {
      return;
    }
    state.analysis[seat] = response.analysis;
  } catch (error) {
    if (requestSeq === analysisSeq && requestMatchId === state.matchId) {
      state.analysisError = errorText(error);
    }
  } finally {
    if (requestSeq === analysisSeq && requestMatchId === state.matchId) {
      state.analysisLoading = false;
    }
  }
}

export async function testProvider(
  model: ProviderModel,
  params: Record<string, unknown>,
  key = `${model.provider_id}:${model.id}`,
): Promise<void> {
  state.testResults[key] = { status: "testing", text: t("errors.testing") };
  try {
    const response = await api.testLLM({
      provider_id: model.provider_id,
      model_id: model.id,
      params,
    });
    state.testResults[key] = {
      status: "ok",
      text: response.text || t("errors.connected"),
      latency: response.latency_ms,
    };
  } catch (error) {
    state.testResults[key] = { status: "error", text: errorText(error) };
  }
}

export function pushToast(level: Toast["level"], title: string, detail = ""): void {
  toast(level, title, detail);
}

export { dismissToast, errorText };
