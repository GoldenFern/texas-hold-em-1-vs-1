import { t, type MessageKey } from "./index";

const STREET_KEYS: Record<string, MessageKey> = {
  waiting: "street.waiting",
  preflop: "street.preflop",
  flop: "street.flop",
  turn: "street.turn",
  river: "street.river",
  showdown: "street.showdown",
  finished: "street.finished",
};

const ACTION_KEYS: Record<string, MessageKey> = {
  fold: "action.fold",
  check: "action.check",
  call: "action.call",
  bet: "action.bet",
  raise: "action.raise",
  all_in: "action.allIn",
};

const STATUS_KEYS: Record<string, MessageKey> = {
  active: "status.active",
  folded: "status.folded",
  all_in: "status.allIn",
  out: "status.out",
};

const POSITION_KEYS: Record<string, MessageKey> = {
  button: "position.button",
  big_blind: "position.bigBlind",
};

const HAND_KEYS: Record<string, MessageKey> = {
  royal_flush: "hand.royalFlush",
  straight_flush: "hand.straightFlush",
  four_of_a_kind: "hand.fourOfAKind",
  full_house: "hand.fullHouse",
  flush: "hand.flush",
  straight: "hand.straight",
  three_of_a_kind: "hand.threeOfAKind",
  two_pair: "hand.twoPair",
  one_pair: "hand.onePair",
  high_card: "hand.highCard",
  incomplete: "hand.incomplete",
};

const HAND_SHORT_KEYS: Record<string, MessageKey> = {
  royal_flush: "handShort.royalFlush",
  straight_flush: "handShort.straightFlush",
  four_of_a_kind: "handShort.fourOfAKind",
  full_house: "handShort.fullHouse",
  flush: "handShort.flush",
  straight: "handShort.straight",
  three_of_a_kind: "handShort.threeOfAKind",
  two_pair: "handShort.twoPair",
  one_pair: "handShort.onePair",
  high_card: "handShort.highCard",
  incomplete: "handShort.incomplete",
};

const TOOL_KEYS: Record<string, MessageKey> = {
  estimate_hand_probabilities: "tool.title.estimate",
  estimate_equity: "tool.title.equity",
  apply_action: "tool.title.action",
  get_random_number: "tool.title.random",
  call_ev: "tool.title.callEv",
  send_table_message: "tool.title.talk",
  research_history: "tool.title.history",
};

export function streetLabel(street?: string): string {
  const key = street ? STREET_KEYS[street] : undefined;
  return key ? t(key) : street || "";
}

export function actionLabel(type?: string): string {
  const key = type ? ACTION_KEYS[type] : undefined;
  return key ? t(key) : type || "";
}

export function statusLabel(status?: string): string {
  const key = status ? STATUS_KEYS[status] : undefined;
  return key ? t(key) : status || "";
}

export function positionLabel(position?: string): string {
  const key = position ? POSITION_KEYS[position] : undefined;
  return key ? t(key) : position || "";
}

export function handCategoryLabel(key?: string, fallback = ""): string {
  const messageKey = key ? HAND_KEYS[key] : undefined;
  return messageKey ? t(messageKey) : fallback || key || "";
}

export function handCategoryShort(key?: string, fallback = ""): string {
  const messageKey = key ? HAND_SHORT_KEYS[key] : undefined;
  return messageKey ? t(messageKey) : fallback || key || "";
}

export function toolLabel(name: string): string {
  const key = name ? TOOL_KEYS[name] : undefined;
  return key ? t(key) : name || t("tool.title.generic");
}
