import { beforeEach, describe, expect, it, vi } from "vitest";

const socketMock = vi.hoisted(() => ({
  handlers: {} as Record<string, (payload: any) => void>,
}));

vi.mock("socket.io-client", () => ({
  io: () => ({
    on: (name: string, handler: (payload: any) => void) => {
      socketMock.handlers[name] = handler;
    },
    disconnect: () => undefined,
  }),
}));

vi.mock("./api", () => {
  class ApiError extends Error {
    status = 0;
    kind = "unknown";
  }
  return {
    ApiError,
    api: {
      bootstrap: vi.fn(),
      state: vi.fn(),
      transcripts: vi.fn(),
      newGame: vi.fn(),
      action: vi.fn(),
      nextHand: vi.fn(),
      setGodView: vi.fn(),
      setLocale: vi.fn(),
      analyze: vi.fn(),
      testLLM: vi.fn(),
    },
  };
});

import { api } from "./api";
import { boot, game, humanAction, refreshAnalysis, refreshTranscripts } from "./store";

const mockedApi = api as unknown as Record<string, ReturnType<typeof vi.fn>>;

function snapshot(overrides: Record<string, any> = {}): any {
  const base = {
    has_match: true,
    seq: 1,
    hand_id: 1,
    street: "preflop",
    phase: "preflop",
    pot: 15,
    current_bet: 10,
    min_raise_to: 20,
    max_raise_to: 1000,
    to_call: 5,
    call_amount: 5,
    button: 0,
    acting_seat: 0,
    acting_kind: "human",
    legal_actions: ["fold", "call", "raise", "all_in"],
    constraints: {},
    community: [],
    players: [
      { seat: 0, kind: "human", chips: 995, street_bet: 5, total_bet: 5, hole_cards: [], assessment: null },
      { seat: 1, kind: "bot", chips: 990, street_bet: 10, total_bet: 10, hole_cards: [], assessment: null },
    ],
    actions: [],
    result: null,
    match_over: false,
    match: {
      id: "m1",
      hands_played: 0,
      max_hands: 0,
      match_over: false,
      god_view: false,
      can_next_hand: false,
      started_at: 1,
      rules: { starting_chips: 1000, small_blind: 5, big_blind: 10, auto_rebuy: false, rebuy_amount: 1000, max_hands: 0, seed: 1 },
      seats: [
        { seat: 0, name: "You", kind: "human", agent: {} },
        { seat: 1, name: "Bot", kind: "bot", agent: {} },
      ],
      history: [],
      last_result: null,
      error: "",
    },
  };
  return { ...base, ...overrides };
}

function turnEvent(eventId: number, turnId: string, handId = 1) {
  return {
    match_id: "m1",
    seat: 0,
    event_id: eventId,
    type: "turn_start",
    turn_id: turnId,
    hand_id: handId,
    hand_number: handId,
    street: "flop",
    pot: 30,
    my_chips: 900,
    opponent_chips: 900,
    to_call: 0,
    legal_actions: ["fold", "check", "bet"],
    position: "button",
  };
}

beforeEach(async () => {
  game.snapshot = null;
  game.matchId = null;
  game.lastResult = null;
  game.lastResultHandId = null;
  game.turns = {};
  game.currentTurns = {};
  game.seenEvents = {};
  game.lastEventIds = {};
  game.tableTalk = [];
  game.tableTalkSeen = new Set();
  game.consoleMode = "stream";
  game.analysis = {};
  game.analysisKey = "";
  game.analysisLoading = false;
  mockedApi.bootstrap.mockResolvedValue({
    app: {},
    providers: [],
    game: {
      defaults: {},
      limits: { starting_chips: { min: 100, max: 100000 } },
      llm_runtime: {},
    },
    rlcard: { available: true, models: [] },
    env: {},
  });
  mockedApi.state.mockResolvedValue(snapshot());
  mockedApi.transcripts.mockResolvedValue({ match_id: "m1", seats: {} });
  await boot();
});

describe("human action decision id", () => {
  it("sends the decision id the state was rendered from", async () => {
    game.busy = false;
    game.snapshot = snapshot({ decision_id: "m1:1:0:0" });
    mockedApi.action.mockClear();
    mockedApi.action.mockResolvedValue({
      ok: true,
      state: snapshot({ seq: 2, decision_id: null, acting_kind: "bot", acting_seat: 1 }),
    });

    await humanAction("call");

    expect(mockedApi.action).toHaveBeenCalledWith({
      action: "call",
      amount: 0,
      decision_id: "m1:1:0:0",
    });
  });

  it("ignores an action when the view has no decision id", async () => {
    game.busy = false;
    game.snapshot = snapshot({ decision_id: null });
    mockedApi.action.mockClear();

    await humanAction("call");

    expect(mockedApi.action).not.toHaveBeenCalled();
  });
});

describe("transcript resync", () => {  it("replays live events that arrive while the snapshot request is in flight", async () => {
    let resolveTranscripts: (value: any) => void = () => undefined;
    mockedApi.transcripts.mockImplementationOnce(
      () => new Promise((resolve) => { resolveTranscripts = resolve; }),
    );

    const pending = refreshTranscripts();
    await Promise.resolve();
    socketMock.handlers.agent_event?.(turnEvent(2, "live-turn"));
    resolveTranscripts({
      match_id: "m1",
      seats: { "0": [turnEvent(1, "snapshot-turn")] },
    });
    await pending;

    const ids = (game.turns[0] || []).map((turn) => turn.id);
    expect(ids).toContain("snapshot-turn");
    expect(ids).toContain("live-turn");
  });
});

describe("aborted transcript replay", () => {
  it("settles running turns and tools after replaying an aborted match", async () => {
    socketMock.handlers.state?.(snapshot({
      seq: 5,
      match: {
        ...snapshot().match,
        aborted: true,
        match_over: true,
      },
    }));
    mockedApi.transcripts.mockResolvedValueOnce({
      match_id: "m1",
      seats: {
        "0": [
          turnEvent(100, "aborted-turn"),
          {
            ...turnEvent(101, "aborted-turn"),
            type: "retry_scheduled",
            scope: "turn",
            attempt: 2,
            max_retries: 10,
            delay_ms: 4000,
            at: Date.now() + 4000,
            text: "retry",
          },
          {
            ...turnEvent(102, "aborted-turn"),
            type: "tool_call_start",
            message_id: "msg-1",
            index: 0,
            call_id: "call-1",
            name: "apply_action",
          },
        ],
      },
    });

    await refreshTranscripts();

    const turn = (game.turns[0] || []).find(
      (item) => item.id === "aborted-turn",
    ) as any;
    expect(turn?.status).toBe("done");
    expect(turn?.retry).toBeNull();
    expect(turn?.messages[0]?.status).toBe("error");
    expect(turn?.messages[0]?.tools[0]?.status).toBe("error");
  });
});

describe("hand result", () => {
  it("keeps the result available for the table without an overlay", () => {
    const boardCodes = ["2c", "7d", "Th", "Jh", "As"];
    const result = {
      hand_id: 1,
      reason: "showdown",
      winner_seats: [0],
      awarded: { "0": 100 },
      pot: 100,
      refunds: {},
      board: boardCodes,
      players: [],
      showdown: [],
      match_over: false,
    };
    const finishedState = snapshot({
      seq: 3,
      street: "finished",
      community: boardCodes.map((code) => ({ code })),
      result,
      match: {
        ...snapshot().match,
        can_next_hand: true,
        last_result: result,
      },
    });
    socketMock.handlers.state?.(finishedState);

    expect(game.lastResult?.hand_id).toBe(1);
    expect(game.snapshot?.street).toBe("finished");
  });
});

function talkEntry(overrides: Record<string, any> = {}) {
  return {
    id: "talk-1",
    seq: 1,
    hand_id: 1,
    street: "flop",
    seat: 0,
    name: "Agent A",
    text: "我这次不会再让你便宜看牌。",
    turn_id: "turn-1",
    created_at: 1,
    ...overrides,
  };
}

describe("table talk", () => {
  it("merges live events and snapshot entries without duplicates", () => {
    const entry = talkEntry();
    socketMock.handlers.agent_event?.({
      match_id: "m1",
      event_id: 50,
      type: "table_talk",
      ...entry,
    });
    expect(game.tableTalk).toHaveLength(1);

    socketMock.handlers.state?.(snapshot({ seq: 2, table_talk: [entry] }));
    expect(game.tableTalk).toHaveLength(1);
    expect(game.tableTalk[0].text).toContain("便宜看牌");
  });

  it("drops entries from a previous match", () => {
    socketMock.handlers.agent_event?.({
      match_id: "m1",
      event_id: 51,
      type: "table_talk",
      ...talkEntry({ id: "old", seq: 2 }),
    });
    expect(game.tableTalk).toHaveLength(1);

    socketMock.handlers.state?.(snapshot({
      seq: 1,
      table_talk: [],
      match: {
        ...snapshot().match,
        id: "m2",
        started_at: 2,
      },
    }));
    expect(game.tableTalk).toHaveLength(0);
  });
});

describe("analysis resync", () => {
  it("ignores an older response that arrives after a newer one", async () => {
    const resolvers: Array<(value: any) => void> = [];
    mockedApi.analyze.mockImplementation(
      () => new Promise((resolve) => { resolvers.push(resolve); }),
    );

    game.snapshot = snapshot({
      street: "flop",
      community: [{ code: "Qh" }, { code: "Jh" }, { code: "2c" }],
      match: { ...snapshot().match, god_view: true },
    });
    game.selectedAnalysisSeat = 0;

    const first = refreshAnalysis(true);
    game.snapshot = snapshot({
      street: "turn",
      community: [{ code: "Qh" }, { code: "Jh" }, { code: "2c" }, { code: "7d" }],
      match: { ...snapshot().match, god_view: true },
    });
    const second = refreshAnalysis(true);

    resolvers[1]({ analysis: { seat: 0, street: "turn", distribution: [] } });
    await second;
    resolvers[0]({ analysis: { seat: 0, street: "flop", distribution: [] } });
    await first;

    expect((game.analysis[0] as any)?.street).toBe("turn");
    expect(game.analysisLoading).toBe(false);
  });
});

describe("retry state", () => {
  it("stores a retry countdown and clears it on message start", () => {
    socketMock.handlers.agent_event?.(turnEvent(60, "retry-turn"));
    socketMock.handlers.agent_event?.({
      ...turnEvent(61, "retry-turn"),
      type: "retry_scheduled",
      scope: "turn",
      attempt: 2,
      max_retries: 10,
      delay_ms: 4000,
      at: Date.now() + 4000,
      text: "retry",
    });

    expect(game.currentTurns[0]?.retry?.attempt).toBe(2);
    expect(game.currentTurns[0]?.retry?.nextAt).toBeGreaterThan(Date.now());

    socketMock.handlers.agent_event?.({
      ...turnEvent(62, "retry-turn"),
      type: "message_start",
      message_id: "msg-2",
      step: 2,
    });
    expect(game.currentTurns[0]?.retry).toBeNull();
  });

  it("marks an incomplete message without treating it as complete", () => {
    socketMock.handlers.agent_event?.(turnEvent(70, "incomplete-turn"));
    socketMock.handlers.agent_event?.({
      ...turnEvent(71, "incomplete-turn"),
      type: "message_start",
      message_id: "msg-1",
      step: 1,
    });
    socketMock.handlers.agent_event?.({
      ...turnEvent(72, "incomplete-turn"),
      type: "tool_call_start",
      message_id: "msg-1",
      index: 0,
      call_id: "call-1",
      name: "apply_action",
    });
    socketMock.handlers.agent_event?.({
      ...turnEvent(73, "incomplete-turn"),
      type: "message_end",
      message_id: "msg-1",
      content: "half",
      incomplete: true,
      incomplete_reason: "network",
      error: true,
    });

    const message = game.currentTurns[0]?.messages[0];
    expect(message?.incomplete).toBe(true);
    expect(message?.incompleteReason).toBe("network");
    expect(message?.status).toBe("error");
    expect(message?.tools[0]?.status).toBe("error");
    expect(message?.tools[0]?.summary).toBe("工具调用未完成");
  });
});
