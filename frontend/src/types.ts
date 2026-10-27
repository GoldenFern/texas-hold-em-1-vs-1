export type SeatKind = "human" | "llm" | "bot";
export type AgentLocaleChoice = "auto" | "zh-CN" | "en-US";

export interface CardInfo {
  code: string;
  rank: string;
  suit: string;
  symbol: string;
  red: boolean;
  label: string;
}

export interface Assessment {
  category: string;
  category_name: string;
  score: number;
  best_five: string[];
}

export interface ActionLog {
  hand_id: number;
  street: string;
  seat: number;
  type: string;
  label: string;
  amount: number;
  added?: number;
  pot_after: number;
  chips_after: number;
  source: string;
  auto: boolean;
}

export interface PlayerState {
  seat: number;
  name: string;
  kind: SeatKind;
  chips: number;
  status: "active" | "folded" | "all_in" | "out";
  street_bet: number;
  total_bet: number;
  is_button: boolean;
  is_small_blind: boolean;
  is_big_blind: boolean;
  is_acting: boolean;
  hole_cards: CardInfo[];
  hole_hidden: boolean;
  last_action: ActionLog | null;
  revealed: boolean;
  assessment: Assessment | null;
  rebuy_count: number;
}

export interface HandResultPlayer {
  seat: number;
  chips_after: number;
  total_bet: number;
  net: number;
  category?: string;
  category_name?: string;
}

export interface ShowdownItem {
  seat: number;
  category: string;
  category_name: string;
  best_five: string[];
  score: number | null;
}

export interface HandResult {
  hand_id: number;
  reason: "fold" | "showdown";
  winner_seats: number[];
  awarded: Record<string, number>;
  pot: number;
  refunds: Record<string, number>;
  board: string[];
  players: HandResultPlayer[];
  showdown: ShowdownItem[] | null;
  match_over: boolean;
  chips?: number[];
}

export interface Rules {
  starting_chips: number;
  small_blind: number;
  big_blind: number;
  starting_small_blind?: number;
  starting_big_blind?: number;
  blind_level?: number;
  auto_rebuy: boolean;
  rebuy_amount: number;
  max_hands: number;
  blind_increase_every_hands: number;
  blind_increase_factor: number;
  seed: number | null;
}

export interface AgentDescribe {
  kind?: string;
  name?: string;
  provider_id?: string;
  provider_name?: string;
  model?: string;
  model_name?: string;
  params?: Record<string, unknown>;
  turns?: number;
  usage?: { input_tokens: number; output_tokens: number; reasoning_tokens: number };
  last_error?: string;
  table_talk_style?: string;
  table_talk_style_name?: string;
  agent_type?: string;
  model_path?: string | null;
  available?: boolean;
  warning?: string;
}

export interface SeatInfo {
  seat: number;
  name: string;
  kind: SeatKind;
  is_human: boolean;
  provider_id: string;
  provider_name?: string;
  model_id: string;
  model_name: string;
  icon?: string;
  accent?: string;
  table_talk_style?: string;
  table_talk_style_name?: string;
  table_talk_style_intensity?: number;
  agent: AgentDescribe;
}

export interface HistoryItem {
  hand_id: number;
  reason: string;
  winner_seats: number[];
  pot: number;
  board: string[];
  players: HandResultPlayer[];
}

export interface TableTalkEntry {
  id: string;
  seq: number;
  hand_id: number;
  street: string;
  seat: number;
  name: string;
  text: string;
  turn_id?: string;
  created_at?: number;
  kind?: "turn" | "final";
  style?: string;
}

export interface MatchState {
  id: string;
  hands_played: number;
  max_hands: number;
  match_over: boolean;
  aborted?: boolean;
  god_view: boolean;
  table_talk?: boolean;
  table_talk_mode?: "tool" | "inline" | "off";
  table_talk_style?: string;
  agent_locale?: "zh-CN" | "en-US";
  can_next_hand: boolean;
  started_at: number;
  error: string;
  rules: Rules;
  seats: SeatInfo[];
  history: HistoryItem[];
  last_result: HandResult | null;
}

export interface GameSnapshot {
  has_match: boolean;
  match: MatchState | null;
  hand_id?: number;
  street?: string;
  phase?: string;
  pot?: number;
  current_bet?: number;
  min_raise_to?: number;
  max_raise_to?: number;
  to_call?: number;
  call_amount?: number;
  button?: number;
  acting_seat?: number | null;
  acting_kind?: SeatKind | null;
  legal_actions?: string[];
  decision_id?: string | null;
  constraints?: Record<string, number | boolean>;
  community?: CardInfo[];
  burned?: CardInfo[];
  deck_remaining?: CardInfo[];
  deck_remaining_count?: number;
  players?: PlayerState[];
  actions?: ActionLog[];
  result?: HandResult | null;
  match_over?: boolean;
  small_blind?: number;
  big_blind?: number;
  starting_small_blind?: number;
  starting_big_blind?: number;
  blind_level?: number;
  blind_increase_every_hands?: number;
  blind_increase_factor?: number;
  starting_chips?: number;
  transcript_counts?: Record<string, number>;
  table_talk?: TableTalkEntry[];
  server_time?: number;
  seq?: number;
}

export interface ControlSpec {
  show: boolean;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  hint_en?: string;
  mode?: "toggle" | "always" | "off";
  label?: string;
  label_en?: string;
  options?: string[];
}

export interface ModelControls {
  temperature?: ControlSpec;
  top_p?: ControlSpec;
  max_tokens?: ControlSpec;
  thinking?: ControlSpec;
  reasoning_effort?: ControlSpec;
}

export interface ProviderModel {
  provider_id: string;
  provider_name: string;
  provider_name_en?: string;
  icon: string;
  accent: string;
  base_url: string;
  api_key_env: string;
  key_present: boolean;
  docs_url: string;
  id: string;
  name: string;
  tagline: string;
  tagline_en?: string;
  context_window: number;
  max_output: number;
  defaults: {
    temperature: number;
    top_p: number;
    max_tokens: number;
    thinking: boolean;
    reasoning_effort: string;
  };
  controls: ModelControls;
}

export interface ProviderInfo {
  id: string;
  name: string;
  name_en?: string;
  icon: string;
  accent: string;
  base_url: string;
  api_key_env: string;
  docs_url: string;
  models: ProviderModel[];
}

export interface RLCardModel {
  name: string;
  path: string;
  size_kb: number;
}

export interface RLCardAgent {
  id: string;
  name: string;
  description: string;
  description_en?: string;
  needs_model: boolean;
}

export interface RLCardInfo {
  available: boolean;
  agents: RLCardAgent[];
  models: RLCardModel[];
  model_dir: string;
}

export interface GameConfig {
  defaults: Rules & { god_view: boolean; table_talk?: boolean; table_talk_style?: string };
  limits: Record<string, { min: number; max: number }>;
  llm_runtime: Record<string, number | string | boolean | number[]>;
}

export interface TableTalkStyleInfo {
  id: string;
  name: string;
  name_en?: string;
  description: string;
  description_en?: string;
  sample: string;
  sample_en?: string;
  intensity: number;
  default: boolean;
}

export interface Bootstrap {
  app: { name: string; version: string; description: string };
  providers: ProviderInfo[];
  game: GameConfig;
  table_talk_styles?: TableTalkStyleInfo[];
  rlcard: RLCardInfo;
  env: { god_view_default: boolean };
  server_time: number;
}

export interface SeatSetup {
  kind: SeatKind;
  name: string;
  provider_id?: string;
  model_id?: string;
  params?: Record<string, unknown>;
  table_talk_style?: string;
  bot?: { agent_type: string; model_path: string | null; delay_seconds?: number };
}

export interface MatchSetupPayload {
  seats: [SeatSetup, SeatSetup];
  rules: Rules & { god_view: boolean };
  table_talk?: boolean;
  table_talk_style?: string;
  agent_locale?: AgentLocaleChoice;
}

export interface DistributionItem {
  key: string;
  name: string;
  short: string;
  color: string;
  count: number;
  probability: number;
}

export interface AnalysisResult {
  hero_seat: number;
  method: "exact" | "monte_carlo";
  iterations: number;
  elapsed_ms: number;
  total: number;
  distribution: DistributionItem[];
  current_best: Assessment | null;
  known_cards: string[];
  needed_board_cards: number;
  hand_id: number;
  street: string;
  seat: number;
}

export interface AgentEvent {
  type: string;
  match_id?: string;
  seat?: number;
  event_id?: number;
  [key: string]: unknown;
}

export interface ToolView {
  index: number;
  callId: string;
  name: string;
  argsText: string;
  parsed: Record<string, unknown> | null;
  status: "running" | "done" | "error";
  result: Record<string, unknown> | null;
  summary: string;
  durationMs: number | null;
  ignored?: boolean;
}

export interface MessageView {
  id: string;
  content: string;
  reasoning: string;
  tools: ToolView[];
  status: "streaming" | "done" | "error";
  step: number;
  usage?: { input_tokens?: number; output_tokens?: number; reasoning_tokens?: number };
  latencyMs?: number;
  finishReason?: string;
  error?: string;
  incomplete?: boolean;
  incompleteReason?: string;
}

export interface RetryView {
  scope: string;
  attempt: number;
  maxRetries: number;
  delayMs: number;
  nextAt: number;
  message: string;
}

export interface TurnView {
  id: string;
  header: {
    handId: number;
    handNumber: number;
    street: string;
    pot: number;
    myChips: number;
    opponentChips: number;
    toCall: number;
    legalActions: string[];
    position: string;
    receivedTalk?: number;
  };
  messages: MessageView[];
  notes: { level: string; text: string }[];
  stateText: string;
  action: Record<string, unknown> | null;
  auto: boolean;
  status: "running" | "done" | "error";
  usage: { input_tokens: number; output_tokens: number; reasoning_tokens: number };
  latencyMs: number;
  steps: number;
  contextTokens: number;
  startedAt: number;
  endedAt: number | null;
  reason?: string;
  retry?: RetryView | null;
}

export interface BotDecisionView {
  id: string;
  type: "bot";
  agentType: string;
  actionId: number;
  legalIds: number[];
  probabilities: Record<string, number>;
  action: Record<string, unknown> | null;
  mappedAction: Record<string, unknown> | null;
  handId: number;
  street: string;
  startedAt: number;
  status: "running" | "done" | "error";
}

export interface ActionEntry {
  id: string;
  handId: number;
  street: string;
  seat: number;
  type: string;
  label: string;
  amount: number;
  potAfter: number;
  auto: boolean;
  source: string;
  at: number;
}

export interface Toast {
  id: number;
  level: "info" | "success" | "warn" | "error";
  title: string;
  detail?: string;
}
