import { locale, t } from "./i18n";
import type {
  AnalysisResult,
  Bootstrap,
  GameSnapshot,
  MatchSetupPayload,
  AgentEvent,
} from "./types";

export class ApiError extends Error {
  status: number;
  kind?: string;
  constructor(message: string, status = 0, kind?: string) {
    super(message);
    this.status = status;
    this.kind = kind;
  }
}

const DEFAULT_TIMEOUT_MS = 30000;

async function request<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "X-HU-Locale": locale.value,
        ...(options.headers || {}),
      },
    });
    let payload: any = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }
    if (!response.ok) {
      const message = payload?.error || t("errors.http", { status: response.status });
      throw new ApiError(message, response.status, payload?.kind);
    }
    return payload as T;
  } catch (error) {
    if ((error as { name?: string } | null)?.name === "AbortError") {
      throw new ApiError(t("errors.timeout"), 0, "timeout");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  bootstrap(): Promise<Bootstrap> {
    return request<Bootstrap>("/api/bootstrap");
  },
  state(): Promise<GameSnapshot> {
    return request<GameSnapshot>("/api/game/state");
  },
  transcripts(): Promise<{ match_id: string; seats: Record<string, AgentEvent[]> }> {
    return request("/api/game/transcripts");
  },
  newGame(payload: MatchSetupPayload): Promise<{ ok: boolean; state: GameSnapshot }> {
    return request("/api/game/new", { method: "POST", body: JSON.stringify(payload) }, 60000);
  },
  abortGame(): Promise<{ ok: boolean; state: GameSnapshot }> {
    return request("/api/game/abort", { method: "POST", body: "{}" }, 15000);
  },
  action(payload: { action: string; amount?: number; decision_id?: string | null }): Promise<{ ok: boolean; state: GameSnapshot; error?: string }> {
    return request("/api/game/action", { method: "POST", body: JSON.stringify(payload) });
  },
  nextHand(): Promise<{ ok: boolean; state: GameSnapshot }> {
    return request("/api/game/next-hand", { method: "POST", body: "{}" });
  },
  setGodView(enabled: boolean): Promise<{ ok: boolean; state: GameSnapshot }> {
    return request("/api/game/god-view", { method: "POST", body: JSON.stringify({ enabled }) });
  },
  setLocale(locale: string): Promise<{ ok: boolean; locale: string }> {
    return request("/api/preferences", { method: "POST", body: JSON.stringify({ locale }) });
  },
  analyze(seat: number, iterations = 30000): Promise<{ ok: boolean; analysis: AnalysisResult }> {
    return request("/api/game/analyze", { method: "POST", body: JSON.stringify({ seat, iterations }) }, 60000);
  },
  testLLM(payload: {
    provider_id: string;
    model_id: string;
    params: Record<string, unknown>;
  }): Promise<{ ok: boolean; text: string; latency_ms: number; usage: Record<string, number>; error?: string }> {
    return request("/api/llm/test", { method: "POST", body: JSON.stringify(payload) }, 60000);
  },
};

export const iconUrl = (icon: string) => `${import.meta.env.BASE_URL}icons/${icon}`;
