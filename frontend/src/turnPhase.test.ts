import { describe, expect, it } from "vitest";
import { turnPendingLabel } from "./turnPhase";
import type { MessageView, ToolView, TurnView } from "./types";

function tool(overrides: Partial<ToolView> = {}): ToolView {
  return {
    index: 0,
    callId: "call-1",
    name: "estimate_hand_probabilities",
    argsText: "",
    parsed: null,
    status: "running",
    result: null,
    summary: "",
    durationMs: null,
    ...overrides,
  };
}

function message(overrides: Partial<MessageView> = {}): MessageView {
  return {
    id: "msg-1",
    content: "",
    reasoning: "",
    tools: [],
    status: "streaming",
    step: 1,
    ...overrides,
  };
}

function turn(overrides: Partial<TurnView> = {}): TurnView {
  return {
    id: "turn-1",
    header: {
      handId: 1,
      handNumber: 1,
      street: "preflop",
      pot: 0,
      myChips: 1000,
      opponentChips: 1000,
      toCall: 0,
      legalActions: [],
      position: "button",
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
    startedAt: 0,
    endedAt: null,
    ...overrides,
  };
}

describe("turnPendingLabel", () => {
  it("is empty for finished turns", () => {
    expect(turnPendingLabel(turn({ status: "done" }))).toBe("");
  });

  it("shows thinking before the first message", () => {
    expect(turnPendingLabel(turn())).toBe("模型思考中…");
  });

  it("shows thinking while the model streams", () => {
    expect(turnPendingLabel(turn({ messages: [message()] }))).toBe("模型思考中…");
  });

  it("shows tool preparation while arguments stream", () => {
    expect(turnPendingLabel(turn({ messages: [message({ tools: [tool()] })] }))).toBe(
      "正在准备工具调用…",
    );
  });

  it("shows the executing tool name", () => {
    expect(
      turnPendingLabel(turn({ messages: [message({ status: "done", tools: [tool()] })] })),
    ).toBe("正在执行 牌型概率…");
  });

  it("shows action submission for apply_action", () => {
    expect(
      turnPendingLabel(
        turn({
          messages: [message({ status: "done", tools: [tool({ name: "apply_action" })] })],
        }),
      ),
    ).toBe("正在提交动作…");
  });

  it("waits for the model after a tool result", () => {
    expect(
      turnPendingLabel(
        turn({ messages: [message({ status: "done", tools: [tool({ status: "done" })] })] }),
      ),
    ).toBe("等待模型继续…");
  });

  it("waits for the model after a failed tool", () => {
    expect(
      turnPendingLabel(
        turn({ messages: [message({ status: "done", tools: [tool({ status: "error" })] })] }),
      ),
    ).toBe("等待模型继续…");
  });

  it("shows a retry phase while the request is backing off", () => {
    expect(
      turnPendingLabel(
        turn({
          retry: {
            scope: "turn",
            attempt: 2,
            maxRetries: 10,
            delayMs: 4000,
            nextAt: Date.now() + 4000,
            message: "retry",
          },
        }),
      ),
    ).toBe("第 2 次重试中…");
  });
});
