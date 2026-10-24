"""The tool-using LLM poker agent."""

from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid
from typing import Callable, List, Optional, Sequence

from src.agents.base import AgentView, BaseAgent
from src.analysis.equity import (
    DEFAULT_ITERATIONS as DEFAULT_EQUITY_ITERATIONS,
    MIN_ITERATIONS as MIN_EQUITY_ITERATIONS,
    EquityAnalyzer,
    EquityResult,
)
from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.analysis.pot_odds import analyze_call
from src.engine.actions import Action, ActionType, action_label
from src.engine.cards import Card
from src.engine.evaluator import hand_category_name
from src.i18n import normalize_locale, tr

from .client import LLMClient, LLMError, ToolCall, estimate_tokens
from .context import (
    ContextSettings,
    MatchStats,
    compact_history,
    compact_tool_payload,
)
from .prompt import (
    TABLE_TALK_TOOL_NAME,
    build_compact_turn_message,
    build_final_table_talk_message,
    build_final_talk_fresh_message,
    build_final_talk_system_prompt,
    build_hand_result_message,
    build_hand_summary_message,
    build_nudge_message,
    build_system_prompt,
    build_turn_message,
)
from .prompt_texts import prompt_text
from .providers import ModelSpec
from .table_talk import TableTalkLimits
from .tools import (
    TABLE_TALK_TOOL,
    TOOLS,
    TOOL_SCHEMAS,
    build_table_talk_tool,
    build_tools,
    select_tools,
)

_MAX_RECENT_HAND_RESULTS = 200

_ACTION_ALIASES = {
    "x": "check",
    "pass": "check",
    "check": "check",
    "call": "call",
    "fold": "fold",
    "bet": "bet",
    "raise": "raise",
    "allin": "all_in",
    "all-in": "all_in",
    "all_in": "all_in",
    "jam": "all_in",
}


class EquityBudgetError(RuntimeError):
    """Raised when the per-turn equity budget is exhausted."""


class LLMAgent(BaseAgent):
    """An agent that keeps a full conversation and decides through tools."""

    kind = "llm"

    def __init__(
        self,
        seat: int,
        name: str,
        spec: ModelSpec,
        params: dict,
        client: LLMClient,
        analyzer: HandDistributionAnalyzer,
        validate_action: Callable[[Action], Optional[str]],
        runtime: dict,
        tools: Optional[Sequence[dict]] = None,
        system_prompt: Optional[str] = None,
        equity_analyzer: Optional[EquityAnalyzer] = None,
        submit_table_talk: Optional[Callable[[str, str], dict]] = None,
        locale: str = "zh-CN",
    ) -> None:
        self.seat = seat
        self.name = name
        self.spec = spec
        self.params = spec.normalize_params(params)
        self.locale = normalize_locale(locale, "zh-CN") or "zh-CN"
        self._prompt_text = prompt_text(self.locale)
        self._turn_marker = self._prompt_text["turn_marker"]
        self._summary_marker = self._prompt_text["summary_marker"]
        self.context = ContextSettings.from_runtime(runtime)
        self.client = client
        self.analyzer = analyzer
        self.equity_analyzer = equity_analyzer or EquityAnalyzer()
        self.validate_action = validate_action
        self.max_steps = int(runtime.get("max_steps_per_turn", 12))
        self.max_failures = int(runtime.get("max_consecutive_failures", 3))
        self.require_estimate_first = bool(runtime.get("require_estimate_first", False))
        self.require_estimate_each_turn = bool(
            runtime.get("require_estimate_each_turn", False)
        )
        self.skip_estimate_on_complete_board = bool(
            runtime.get("skip_estimate_on_complete_board", False)
        )
        runtime_budget = int(runtime.get("max_context_tokens_estimate", 220000))
        reserved_output = int(self.params.get("max_tokens", 32768))
        context_window = int(getattr(spec, "context_window", 0) or 0)
        if context_window > 0:
            runtime_budget = min(runtime_budget, context_window - reserved_output - 2048)
        self.max_context_tokens = max(8000, runtime_budget)
        self.context = self.context.with_context_window(self.max_context_tokens)
        self.max_estimate_calls = int(runtime.get("max_estimate_calls_per_turn", 2))
        self.max_estimate_iterations = int(runtime.get("max_estimate_iterations", 50000))
        self.max_equity_calls = int(runtime.get("max_equity_calls_per_turn", 2))
        self.max_equity_iterations = int(runtime.get("max_equity_iterations", 50000))
        self.table_talk_limits = TableTalkLimits.from_runtime(runtime)
        self.tools = list(tools) if tools is not None else build_tools(self.locale)
        self.tool_names = {tool["function"]["name"] for tool in self.tools}
        self.table_talk_required = TABLE_TALK_TOOL_NAME in self.tool_names
        if self.require_estimate_each_turn and "estimate_hand_probabilities" not in self.tool_names:
            raise ValueError(self._t("agent.constructor.estimate_required"))
        self.submit_table_talk = submit_table_talk
        if TABLE_TALK_TOOL_NAME in self.tool_names and submit_table_talk is None:
            raise ValueError(self._t("agent.constructor.table_talk_callback"))
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = build_system_prompt(
                tuple(self.tool_names),
                require_estimate_each_turn=self.require_estimate_each_turn,
                table_talk_max_chars=self.table_talk_limits.max_chars,
                table_talk_max_per_turn=self.table_talk_limits.max_per_turn,
                locale=self.locale,
            )
        self.messages: List[dict] = [{"role": "system", "content": self.system_prompt}]
        self.match_stats = MatchStats()
        self.hand_results: List[dict] = []
        self.turn_count = 0
        self._estimated_keys: set[tuple[int, str]] = set()
        self._nudged_keys: set[tuple[int, str]] = set()
        self._estimated_this_turn = False
        self._estimate_calls_this_turn = 0
        self._estimate_iterations_this_turn = 0
        self._equity_calls_this_turn = 0
        self._equity_iterations_this_turn = 0
        self._talk_count_this_turn = 0
        self._tools_tokens = estimate_tokens(json.dumps(self.tools, ensure_ascii=False))
        self.total_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_miss_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        self.last_error = ""
        self._trimmed = False
        self._compactions = 0
        self._cancel_event = threading.Event()

    def _t(self, message_key: str, **params: object) -> str:
        return tr(self.locale, message_key, **params)

    def _sync_client_locale(self) -> None:
        """Keep provider-facing client errors in the agent's language."""
        if hasattr(self.client, "locale"):
            self.client.locale = self.locale

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.match_stats = MatchStats()
        self.hand_results = []
        self.turn_count = 0
        self.total_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_miss_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        self.last_error = ""
        self._trimmed = False
        self._compactions = 0
        self._estimated_keys.clear()
        self._nudged_keys.clear()
        self._estimated_this_turn = False
        self._estimate_calls_this_turn = 0
        self._estimate_iterations_this_turn = 0
        self._equity_calls_this_turn = 0
        self._equity_iterations_this_turn = 0
        self._talk_count_this_turn = 0
        self._cancel_event.clear()

    def cancel(self) -> None:
        """Ask an in-flight decision loop to stop making model calls."""
        self._cancel_event.set()

    def on_hand_finished(self, result: Optional[dict]) -> None:
        if not result:
            return
        self.match_stats.update(result, self.seat)
        self.hand_results.append(result)
        if len(self.hand_results) > _MAX_RECENT_HAND_RESULTS:
            del self.hand_results[0]
        if self.context.compacts:
            text = build_hand_summary_message(result, self.seat, locale=self.locale)
        else:
            text = build_hand_result_message(
                result, self.seat, self.name, locale=self.locale
            )
        self.messages.append({"role": "user", "content": text})

    def final_table_talk(
        self,
        result: Optional[dict],
        submit: Callable[[str], dict],
        emit,
    ) -> None:
        """Ask the model for one optional post-match table-talk message.

        Emits ``final_talk_response`` after every model response and exactly
        one ``final_talk_end`` once the call finishes.  The end event reports
        usage from the final-talk requests only, not the whole match.
        """
        if not self.table_talk_required or self._cancel_event.is_set():
            return
        self._sync_client_locale()
        mode = self.context.final_talk_context
        talk_usage = {key: 0 for key in self.total_usage}
        if mode == "fresh":
            messages: List[dict] = [
                {
                    "role": "system",
                    "content": build_final_talk_system_prompt(self.name, self.locale),
                },
                {
                    "role": "user",
                    "content": build_final_talk_fresh_message(
                        result,
                        self.seat,
                        self.hand_results,
                        locale=self.locale,
                        stats=self.match_stats,
                    ),
                },
            ]
        else:
            prompt = build_final_table_talk_message(
                result, self.seat, self.name, locale=self.locale
            )
            self.messages.append({"role": "user", "content": prompt})
            self._prepare_context(emit)
            messages = self.messages
        tools = [build_table_talk_tool(self.locale)]
        emit(
            {
                "type": "final_talk_start",
                "mode": mode,
                "message_count": len(messages),
                "estimated_tokens": self._messages_tokens(messages),
            }
        )

        def emit_end(ok: bool, reason: str = "", error: str = "") -> None:
            """Emit one closing event; usage covers final-talk calls only."""
            payload = {
                "type": "final_talk_end",
                "mode": mode,
                "ok": ok,
                "message_count": len(messages),
                "usage": dict(talk_usage),
            }
            if reason:
                payload["reason"] = reason
            if error:
                payload["error"] = error
            emit(payload)

        for attempt in range(2):
            if self._cancel_event.is_set():
                emit_end(False, reason="cancelled")
                return
            try:
                response = self.client.stream(
                    self.spec,
                    messages,
                    tools,
                    self.params,
                    lambda _event: None,
                    cancel=self._cancel_event.is_set,
                )
            except LLMError as exc:
                emit(
                    {
                        "type": "note",
                        "level": "warn",
                        "text": self._t("agent.final.not_generated", kind=exc.kind),
                    }
                )
                emit_end(False, reason="llm_error", error=exc.kind)
                return
            except Exception as exc:  # noqa: BLE001
                emit(
                    {
                        "type": "note",
                        "level": "warn",
                        "text": self._t(
                            "agent.final.failed",
                            error=f"{type(exc).__name__}: {exc}",
                        ),
                    }
                )
                emit_end(
                    False,
                    reason="exception",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            self.last_error = ""
            for key in talk_usage:
                amount = int(response.usage.get(key, 0))
                talk_usage[key] += amount
                self.total_usage[key] = self.total_usage.get(key, 0) + amount
            messages.append(
                response.assistant_message(
                    bool(self.spec.request.get("reasoning_echo", False))
                )
            )
            emit(
                {
                    "type": "final_talk_response",
                    "mode": mode,
                    "usage": dict(response.usage),
                    "latency_ms": round(response.latency_ms, 1),
                    "message_count": len(messages),
                }
            )
            talk_call: Optional[ToolCall] = None
            for item in response.tool_calls:
                if item.name == TABLE_TALK_TOOL_NAME and talk_call is None:
                    talk_call = item
                    continue
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.id,
                        "name": item.name,
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": self._t("agent.final.only_message"),
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            if talk_call is not None:
                parsed = talk_call.parsed_arguments()
                raw_text = (
                    parsed.get("text", parsed.get("message"))
                    if isinstance(parsed, dict)
                    else None
                )
                if isinstance(raw_text, str):
                    try:
                        submitted = submit(raw_text)
                    except Exception as exc:  # noqa: BLE001
                        submitted = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": talk_call.id,
                            "name": talk_call.name,
                            "content": json.dumps(submitted, ensure_ascii=False),
                        }
                    )
                    if submitted.get("ok"):
                        if mode == "full":
                            self.messages = messages
                        emit_end(True)
                        return
                    messages.append(
                        {
                            "role": "user",
                            "content": self._t(
                                "agent.final.rejected",
                                error=submitted.get("error", self._t("common.unknown")),
                            ),
                        }
                    )
                    continue
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": talk_call.id,
                        "name": talk_call.name,
                        "content": json.dumps(
                            {"ok": False, "error": self._t("agent.final.text_string")},
                            ensure_ascii=False,
                        ),
                    }
                )
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": self._t("agent.final.instruction"),
                    }
                )
        if mode == "full":
            self.messages = messages
        emit(
            {
                "type": "note",
                "level": "warn",
                "text": self._t("agent.final.no_result"),
            }
        )
        emit_end(False, reason="no_result")

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "provider_id": self.spec.provider_id,
            "provider_name": self.spec.provider_name,
            "model": self.spec.id,
            "model_name": self.spec.name,
            "params": dict(self.params),
            "turns": self.turn_count,
            "usage": dict(self.total_usage),
            "last_error": self.last_error,
            "table_talk": TABLE_TALK_TOOL_NAME in self.tool_names,
            "context": {
                "policy": self.context.policy,
                "high_water_tokens": self.context.high_water_tokens,
                "low_water_tokens": self.context.low_water_tokens,
                "keep_recent_hands": self.context.keep_recent_hands,
                "state_detail": self.context.state_detail,
                "tool_detail": self.context.tool_detail,
                "final_talk_context": self.context.final_talk_context,
            },
        }

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    def decide(self, view: AgentView, emit) -> Action:
        self._sync_client_locale()
        turn_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        usage_total = {key: 0 for key in self.total_usage}
        steps = 0
        failures = 0
        self.turn_count += 1
        self._estimated_this_turn = False
        self._estimate_calls_this_turn = 0
        self._estimate_iterations_this_turn = 0
        self._equity_calls_this_turn = 0
        self._equity_iterations_this_turn = 0
        self._talk_count_this_turn = 0
        if self._cancel_event.is_set():
            return Action(view.seat, ActionType.FOLD, source="llm", auto=True)

        emit(
            {
                "type": "turn_start",
                "turn_id": turn_id,
                "hand_id": view.hand_id,
                "hand_number": view.hand_number,
                "street": view.street,
                "pot": view.pot,
                "my_chips": view.my_chips,
                "opponent_chips": view.opponent_chips,
                "to_call": view.to_call,
                "legal_actions": view.legal_actions,
                "position": "button" if view.is_button else "big_blind",
                "table_talk_received": len(view.table_talk),
            }
        )
        if self.context.state_detail == "compact":
            turn_message = build_compact_turn_message(
                view,
                require_table_talk=self.table_talk_required,
                locale=self.locale,
            )
        else:
            turn_message = build_turn_message(
                view,
                require_table_talk=self.table_talk_required,
                locale=self.locale,
            )
        self.messages.append({"role": "user", "content": turn_message})
        emit({"type": "turn_state", "turn_id": turn_id, "text": turn_message})
        self._prepare_context(emit)

        while steps < self.max_steps:
            if self._cancel_event.is_set():
                return Action(view.seat, ActionType.FOLD, source="llm", auto=True)
            steps += 1
            self._prepare_context(emit)
            message_id = uuid.uuid4().hex[:12]
            emit(
                {
                    "type": "message_start",
                    "turn_id": turn_id,
                    "message_id": message_id,
                    "step": steps,
                }
            )
            emit(
                {
                    "type": "llm_request",
                    "turn_id": turn_id,
                    "message_id": message_id,
                    "step": steps,
                    "message_count": len(self.messages),
                    "estimated_context_tokens": self._context_tokens(),
                }
            )

            def on_stream(event: dict) -> None:
                event["turn_id"] = turn_id
                event["message_id"] = message_id
                emit(event)

            try:
                result = self.client.stream(
                    self.spec,
                    self.messages,
                    self.tools,
                    self.params,
                    on_stream,
                    cancel=self._cancel_event.is_set,
                )
                self.last_error = ""
            except LLMError as exc:
                if exc.kind == "cancelled":
                    return Action(view.seat, ActionType.FOLD, source="llm", auto=True)
                if exc.kind == "context_length":
                    before = len(self.messages)
                    self._trim_context(emit, force=True)
                    if len(self.messages) < before and steps < self.max_steps:
                        self.last_error = str(exc)
                        emit(
                            {
                                "type": "message_end",
                                "turn_id": turn_id,
                                "message_id": message_id,
                                "error": True,
                                "content": "",
                                "reasoning": "",
                                "tool_calls": [],
                                "estimated_usage": False,
                            }
                        )
                        continue
                self.last_error = str(exc)
                emit(
                    {
                        "type": "error",
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "kind": exc.kind,
                        "text": exc.detail,
                    }
                )
                emit(
                    {
                        "type": "message_end",
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "error": True,
                        "content": "",
                        "reasoning": "",
                        "tool_calls": [],
                    }
                )
                return self._fallback(
                    view,
                    emit,
                    turn_id,
                    started,
                    steps,
                    usage_total,
                    reason=self._t("agent.llm.failed", kind=exc.kind),
                    allow_check=True,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                emit(
                    {
                        "type": "error",
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "kind": "unknown",
                        "text": self.last_error,
                    }
                )
                emit(
                    {
                        "type": "message_end",
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "error": True,
                        "content": "",
                        "reasoning": "",
                        "tool_calls": [],
                    }
                )
                return self._fallback(
                    view,
                    emit,
                    turn_id,
                    started,
                    steps,
                    usage_total,
                    reason=self._t("agent.llm.exception"),
                    allow_check=True,
                )

            for key in usage_total:
                amount = int(result.usage.get(key, 0))
                usage_total[key] += amount
                self.total_usage[key] = self.total_usage.get(key, 0) + amount

            emit(
                {
                    "type": "message_end",
                    "turn_id": turn_id,
                    "message_id": message_id,
                    "content": result.content,
                    "reasoning": result.reasoning,
                    "finish_reason": result.finish_reason,
                    "usage": dict(result.usage),
                    "estimated_usage": bool(result.estimated_usage),
                    "latency_ms": round(result.latency_ms, 1),
                    "tool_calls": [
                        {
                            "index": index,
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for index, call in enumerate(result.tool_calls)
                    ],
                }
            )

            self.messages.append(
                result.assistant_message(bool(self.spec.request.get("reasoning_echo", False)))
            )

            if not result.tool_calls:
                failures += 1
                emit(
                    {
                        "type": "note",
                        "level": "warn",
                        "turn_id": turn_id,
                        "text": self._t(
                            "agent.no_tool",
                            failures=failures,
                            max=self.max_failures,
                        ),
                    }
                )
                if failures >= self.max_failures:
                    return self._fallback(
                        view,
                        emit,
                        turn_id,
                        started,
                        steps,
                        usage_total,
                        reason=self._t("agent.reason.no_action"),
                    )
                self.messages.append(
                    {
                        "role": "user",
                        "content": build_nudge_message(
                            self.table_talk_required, locale=self.locale
                        ),
                    }
                )
                continue

            chosen: Optional[Action] = None
            for index, call in enumerate(result.tool_calls):
                if chosen is not None:
                    self._append_tool_error(
                        call,
                        self._t("agent.note.action_submitted"),
                        emit,
                        turn_id,
                        index,
                        ignored=True,
                    )
                    continue
                tool_started = time.perf_counter()
                parsed = call.parsed_arguments()
                emit(
                    {
                        "type": "tool_call",
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "index": index,
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "parsed": parsed,
                        "parse_error": parsed is None,
                    }
                )

                if call.name not in self.tool_names:
                    failures += 1
                    separator = "、" if self.locale == "zh-CN" else ", "
                    available = separator.join(sorted(self.tool_names))
                    self._append_tool_error(
                        call,
                        self._t(
                            "agent.error.unknown_tool",
                            name=repr(call.name),
                            tools=available,
                        ),
                        emit,
                        turn_id,
                        index,
                    )
                    if failures >= self.max_failures:
                        return self._fallback(
                            view,
                            emit,
                            turn_id,
                            started,
                            steps,
                            usage_total,
                            reason=self._t("agent.reason.invalid_action"),
                        )
                    continue

                if call.name == "estimate_hand_probabilities":
                    if not self._handle_estimate(view, call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                    continue

                if call.name == "estimate_equity":
                    if not self._handle_equity(view, call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                    continue

                if call.name == "call_ev":
                    if not self._handle_call_ev(view, call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                    continue

                if call.name == "get_random_number":
                    if not self._handle_random(call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                    continue

                if call.name == TABLE_TALK_TOOL_NAME:
                    self._handle_table_talk(
                        call,
                        parsed,
                        emit,
                        turn_id,
                        index,
                        tool_started,
                    )
                    continue

                if call.name == "apply_action":
                    if parsed is None:
                        failures += 1
                        self._append_tool_error(
                            call,
                            self._t(
                                "agent.error.invalid_json",
                                args=call.arguments[:300],
                            ),
                            emit,
                            turn_id,
                            index,
                        )
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                        continue

                    if self.table_talk_required and self._talk_count_this_turn == 0:
                        self._append_tool_error(
                            call,
                            self._t("agent.error.talk_first"),
                            emit,
                            turn_id,
                            index,
                            extra={"required_tool": TABLE_TALK_TOOL_NAME},
                        )
                        continue

                    street_key = (view.hand_id, view.street)
                    board_complete = len(view.community) >= 5
                    requires_estimate = False
                    if "estimate_hand_probabilities" in self.tool_names:
                        if self.require_estimate_each_turn:
                            requires_estimate = not self._estimated_this_turn and not (
                                self.skip_estimate_on_complete_board and board_complete
                            )
                        elif self.require_estimate_first:
                            requires_estimate = (
                                street_key not in self._estimated_keys
                                and street_key not in self._nudged_keys
                                and not (
                                    self.skip_estimate_on_complete_board
                                    and board_complete
                                )
                            )
                    if requires_estimate:
                        if self.require_estimate_each_turn:
                            message = self._t("agent.error.estimate_required")
                        else:
                            # Soft workflow requirement: one probability check per
                            # street before the first action. Only nudged once, and
                            # it does not count toward the three-failure fallback.
                            self._nudged_keys.add(street_key)
                            message = self._t("agent.error.estimate_first_street")
                        self._append_tool_error(
                            call,
                            message,
                            emit,
                            turn_id,
                            index,
                            extra={"required_tool": "estimate_hand_probabilities"},
                        )
                        continue

                    action, error = self._build_action(view, parsed)
                    if error:
                        failures += 1
                        self._append_tool_error(
                            call,
                            error,
                            emit,
                            turn_id,
                            index,
                            extra=self._constraint_payload(view),
                        )
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                        continue

                    try:
                        validation_error = self.validate_action(action)
                    except Exception as exc:  # noqa: BLE001
                        validation_error = self._t(
                            "agent.error.validation",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    if validation_error:
                        failures += 1
                        self._append_tool_error(
                            call,
                            validation_error,
                            emit,
                            turn_id,
                            index,
                            extra=self._constraint_payload(view),
                        )
                        if failures >= self.max_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                            )
                        continue

                    chosen = action
                    payload = {
                        "ok": True,
                        "accepted": True,
                        "action": action.type.value,
                        "label": action_label(action.type.value, self.locale),
                        "amount": action.amount,
                        "message": self._t("agent.message.accepted"),
                    }
                    self._append_tool_result(call, payload, emit, turn_id, index, tool_started)
                    continue

            if chosen is not None:
                emit(
                    {
                        "type": "turn_end",
                        "turn_id": turn_id,
                        "action": chosen.to_dict(),
                        "auto": False,
                        "steps": steps,
                        "usage": usage_total,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
                return chosen

        return self._fallback(
            view,
            emit,
            turn_id,
            started,
            steps,
            usage_total,
            reason=self._t("agent.reason.max_steps", max=self.max_steps),
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _handle_estimate(
        self,
        view: AgentView,
        call: ToolCall,
        parsed: Optional[dict],
        emit,
        turn_id: str,
        index: int,
        started: float,
    ) -> bool:
        if parsed is None or not isinstance(parsed, dict):
            self._append_tool_error(
                call,
                self._t("agent.error.invalid_json", args=call.arguments[:300]),
                emit,
                turn_id,
                index,
            )
            return False
        if self._estimate_calls_this_turn >= self.max_estimate_calls:
            self._append_tool_error(
                call,
                self._t("agent.error.estimate_limit", max=self.max_estimate_calls),
                emit,
                turn_id,
                index,
            )
            return False
        remaining = self.max_estimate_iterations - self._estimate_iterations_this_turn
        if remaining <= 0:
            self._append_tool_error(
                call,
                self._t("agent.error.estimate_budget"),
                emit,
                turn_id,
                index,
            )
            return False
        if remaining < 2000:
            self._append_tool_error(
                call,
                self._t("agent.error.estimate_remaining"),
                emit,
                turn_id,
                index,
            )
            return False
        iterations = parsed.get("iterations", 30000)
        try:
            iterations = int(iterations)
        except (TypeError, ValueError):
            iterations = 30000
        iterations = max(2000, min(iterations, remaining))
        self._estimate_calls_this_turn += 1
        try:
            hole = [Card.from_str(code) for code in view.hole_cards]
            community = [Card.from_str(code) for code in view.community]
            result = self.analyzer.analyze(
                hole,
                community,
                hero_seat=view.seat,
                iterations=iterations,
                locale=self.locale,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_tool_error(
                call,
                self._t("agent.error.estimate_failed", error=exc),
                emit,
                turn_id,
                index,
            )
            return True

        self._estimate_iterations_this_turn += iterations
        self._estimated_this_turn = True
        self._estimated_keys.add((view.hand_id, view.street))
        payload = {"ok": True, **result.to_llm_dict(self.locale)}
        top = sorted(
            (item for item in result.distribution if item["probability"] > 0),
            key=lambda item: item["probability"],
            reverse=True,
        )[:4]
        summary = " · ".join(
            f"{hand_category_name(item['key'], self.locale, item.get('name', ''))} {item['probability'] * 100:.1f}%"
            for item in top
        )
        self._append_tool_result(
            call,
            payload,
            emit,
            turn_id,
            index,
            started,
            ui_result={"ok": True, **result.to_dict()},
            summary=summary,
        )
        return True

    def _handle_equity(
        self,
        view: AgentView,
        call: ToolCall,
        parsed: Optional[dict],
        emit,
        turn_id: str,
        index: int,
        started: float,
    ) -> bool:
        if parsed is None or not isinstance(parsed, dict):
            self._append_tool_error(
                call,
                self._t("agent.error.invalid_json", args=call.arguments[:300]),
                emit,
                turn_id,
                index,
            )
            return False
        try:
            result = self._compute_equity(view)
        except EquityBudgetError as exc:
            self._append_tool_error(call, str(exc), emit, turn_id, index)
            return False
        except Exception as exc:  # noqa: BLE001
            self._append_tool_error(
                call,
                self._t("agent.error.equity_failed", error=exc),
                emit,
                turn_id,
                index,
            )
            return True

        payload = {
            "ok": True,
            "street": view.street,
            **result.to_llm_dict(self.locale),
        }
        summary = self._t(
            "agent.summary.equity",
            equity=f"{result.equity * 100:.1f}",
            win=f"{result.win_rate * 100:.1f}",
            tie=f"{result.tie_rate * 100:.1f}",
        )
        self._append_tool_result(
            call,
            payload,
            emit,
            turn_id,
            index,
            started,
            ui_result={"ok": True, "street": view.street, **result.to_dict()},
            summary=summary,
        )
        return True

    def _handle_call_ev(
        self,
        view: AgentView,
        call: ToolCall,
        parsed: Optional[dict],
        emit,
        turn_id: str,
        index: int,
        started: float,
    ) -> bool:
        if parsed is None or not isinstance(parsed, dict):
            self._append_tool_error(
                call,
                self._t("agent.error.invalid_json", args=call.arguments[:300]),
                emit,
                turn_id,
                index,
            )
            return False

        if view.to_call <= 0 or view.call_amount <= 0 or "call" not in view.legal_actions:
            if view.to_call <= 0:
                reason = self._t("agent.error.call_ev_no_call")
            elif view.call_amount <= 0:
                reason = self._t("agent.error.call_ev_no_chips")
            else:
                reason = self._t("agent.error.call_ev_not_legal")
            payload = {
                "ok": True,
                "applicable": False,
                "reason": reason,
                "street": view.street,
                "pot": view.pot,
                "to_call": view.to_call,
                "call_amount": view.call_amount,
                "legal_actions": list(view.legal_actions),
            }
            self._append_tool_result(
                call,
                payload,
                emit,
                turn_id,
                index,
                started,
                summary=reason,
            )
            return True

        equity_result: Optional[EquityResult] = None
        equity_source = "model"
        raw_equity = parsed.get("equity")
        if raw_equity is None:
            try:
                equity_result = self._compute_equity(view)
            except EquityBudgetError as exc:
                self._append_tool_error(call, str(exc), emit, turn_id, index)
                return False
            except Exception as exc:  # noqa: BLE001
                self._append_tool_error(
                    call,
                    self._t("agent.error.equity_failed", error=exc),
                    emit,
                    turn_id,
                    index,
                )
                return True
            equity = equity_result.equity
            equity_source = "internal"
        else:
            try:
                equity = float(raw_equity)
            except (TypeError, ValueError):
                self._append_tool_error(
                    call,
                    self._t("agent.error.equity_range"),
                    emit,
                    turn_id,
                    index,
                )
                return False
            if not math.isfinite(equity) or not 0.0 <= equity <= 1.0:
                self._append_tool_error(
                    call,
                    self._t("agent.error.equity_range"),
                    emit,
                    turn_id,
                    index,
                )
                return False

        try:
            calc = analyze_call(
                view.pot,
                view.to_call,
                view.my_chips,
                equity,
                locale=self.locale,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_tool_error(
                call,
                self._t("agent.error.call_ev_failed", error=exc),
                emit,
                turn_id,
                index,
            )
            return True

        payload = {
            "ok": True,
            "equity_source": equity_source,
            "street": view.street,
            **calc.to_llm_dict(self.locale),
        }
        ui_result = {
            "ok": True,
            "equity_source": equity_source,
            "street": view.street,
            **calc.to_dict(),
        }
        if equity_result is not None:
            payload.update(
                {
                    "equity_method": equity_result.method,
                    "equity_samples": equity_result.samples,
                    "equity_std_error": round(equity_result.std_error, 4),
                }
            )
            ui_result.update(
                {
                    "equity_method": equity_result.method,
                    "equity_samples": equity_result.samples,
                    "equity_std_error": equity_result.std_error,
                    "equity_cached": equity_result.cached,
                }
            )
        else:
            payload["equity_method"] = "model"
            ui_result["equity_method"] = "model"

        summary = self._t(
            "agent.summary.call_ev",
            ev=f"{calc.ev_call:+.1f}",
            breakeven=f"{calc.breakeven_equity * 100:.1f}",
            equity=f"{equity * 100:.1f}",
        )
        self._append_tool_result(
            call,
            payload,
            emit,
            turn_id,
            index,
            started,
            ui_result=ui_result,
            summary=summary,
        )
        return True

    def _compute_equity(self, view: AgentView) -> EquityResult:
        if self._equity_calls_this_turn >= self.max_equity_calls:
            raise EquityBudgetError(
                self._t("agent.error.equity_limit", max=self.max_equity_calls)
            )
        remaining = self.max_equity_iterations - self._equity_iterations_this_turn
        if remaining < MIN_EQUITY_ITERATIONS:
            raise EquityBudgetError(self._t("agent.error.equity_budget"))
        iterations = min(DEFAULT_EQUITY_ITERATIONS, remaining)
        hole = [Card.from_str(code) for code in view.hole_cards]
        community = [Card.from_str(code) for code in view.community]
        result = self.equity_analyzer.analyze(
            hole,
            community,
            hero_seat=view.seat,
            iterations=iterations,
            locale=self.locale,
        )
        self._equity_calls_this_turn += 1
        if not result.cached and result.method == "monte_carlo":
            self._equity_iterations_this_turn += result.samples
        return result

    def _handle_random(
        self,
        call: ToolCall,
        parsed: Optional[dict],
        emit,
        turn_id: str,
        index: int,
        started: float,
    ) -> bool:
        if parsed is None or not isinstance(parsed, dict):
            self._append_tool_error(
                call,
                self._t("agent.error.invalid_json", args=call.arguments[:300]),
                emit,
                turn_id,
                index,
            )
            return False
        value = random.random()
        payload = {"ok": True, "value": value}
        self._append_tool_result(
            call,
            payload,
            emit,
            turn_id,
            index,
            started,
            summary=f"{value:.4f}",
        )
        return True

    def _handle_table_talk(
        self,
        call: ToolCall,
        parsed: Optional[dict],
        emit,
        turn_id: str,
        index: int,
        started: float,
    ) -> bool:
        """Handle the mandatory per-turn table-talk call."""
        if self.submit_table_talk is None:
            self._append_tool_error(
                call,
                self._t("agent.error.talk_disabled"),
                emit,
                turn_id,
                index,
            )
            return True
        if self._talk_count_this_turn >= self.table_talk_limits.max_per_turn:
            self._append_tool_error(
                call,
                self._t(
                    "agent.error.talk_limit",
                    max=self.table_talk_limits.max_per_turn,
                ),
                emit,
                turn_id,
                index,
            )
            return True
        if parsed is None or not isinstance(parsed, dict):
            self._append_tool_error(
                call,
                self._t("agent.error.invalid_json", args=call.arguments[:300]),
                emit,
                turn_id,
                index,
            )
            return True
        raw_text = parsed.get("text", parsed.get("message"))
        if not isinstance(raw_text, str):
            self._append_tool_error(
                call,
                self._t("agent.error.talk_text_type"),
                emit,
                turn_id,
                index,
            )
            return True

        try:
            result = self.submit_table_talk(raw_text, turn_id)
        except Exception as exc:  # noqa: BLE001
            self._append_tool_error(
                call,
                self._t(
                    "agent.error.talk_send_failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
                emit,
                turn_id,
                index,
            )
            return True
        if not isinstance(result, dict):
            result = {"ok": False, "error": self._t("agent.error.talk_invalid_result")}
        if not result.get("ok"):
            self._append_tool_error(
                call,
                str(result.get("error") or self._t("agent.error.talk_rejected")),
                emit,
                turn_id,
                index,
                extra={
                    "remaining_turn": result.get("remaining_turn"),
                    "max_chars": self.table_talk_limits.max_chars,
                },
            )
            return True

        self._talk_count_this_turn += 1
        payload = dict(result)
        self._append_tool_result(
            call,
            payload,
            emit,
            turn_id,
            index,
            started,
            ui_result=payload,
            summary=self._t("agent.summary.talk"),
        )
        return True

    def _build_action(self, view: AgentView, parsed: dict) -> tuple[Optional[Action], str]:
        raw = str(parsed.get("action", "")).strip().lower()
        action_type = _ACTION_ALIASES.get(raw, raw)
        if action_type not in {item.value for item in ActionType}:
            return None, self._t(
                "agent.error.unknown_action",
                action=repr(raw),
                actions=", ".join(view.legal_actions),
            )
        # The engine distinguishes an opening bet (BB option included) from a
        # raise. Accept either spelling from the model and normalize it.
        if action_type == "raise" and "raise" not in view.legal_actions and "bet" in view.legal_actions:
            action_type = "bet"
        elif action_type == "bet" and "bet" not in view.legal_actions and "raise" in view.legal_actions:
            action_type = "raise"
        if action_type not in view.legal_actions:
            return (
                None,
                self._t(
                    "agent.error.action_invalid",
                    action=repr(action_type),
                    actions=", ".join(view.legal_actions),
                    to_call=view.to_call,
                    min_raise_to=view.min_raise_to,
                    max_raise_to=view.max_raise_to,
                ),
            )
        amount = parsed.get("amount", parsed.get("raise_to", parsed.get("to", 0)))
        try:
            return (
                Action.from_payload(
                    view.seat,
                    {"type": action_type, "amount": amount},
                    source="llm",
                    locale=self.locale,
                ),
                "",
            )
        except ValueError as exc:
            return None, str(exc)

    def _append_tool_result(
        self,
        call: ToolCall,
        payload: dict,
        emit,
        turn_id: str,
        index: int,
        started: float,
        ui_result: Optional[dict] = None,
        summary: str = "",
    ) -> None:
        model_payload = (
            compact_tool_payload(call.name, payload)
            if self.context.tool_detail == "compact"
            else payload
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(model_payload, ensure_ascii=False),
            }
        )
        emit(
            {
                "type": "tool_result",
                "turn_id": turn_id,
                "index": index,
                "call_id": call.id,
                "name": call.name,
                "ok": True,
                "result": ui_result if ui_result is not None else payload,
                "summary": summary,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )

    def _append_tool_error(
        self,
        call: ToolCall,
        error: str,
        emit,
        turn_id: str,
        index: int,
        extra: Optional[dict] = None,
        ignored: bool = False,
    ) -> None:
        payload = {"ok": False, "error": error}
        if extra:
            payload.update(extra)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )
        emit(
            {
                "type": "tool_result",
                "turn_id": turn_id,
                "index": index,
                "call_id": call.id,
                "name": call.name,
                "ok": False,
                "ignored": ignored,
                "result": payload,
                "summary": error,
                "duration_ms": 0.0,
            }
        )

    @staticmethod
    def _constraint_payload(view: AgentView) -> dict:
        return {
            "legal_actions": view.legal_actions,
            "to_call": view.to_call,
            "call_amount": view.call_amount,
            "current_bet": view.current_bet,
            "your_street_bet": view.my_street_bet,
            "min_raise_to": view.min_raise_to,
            "max_raise_to": view.max_raise_to,
            "pot": view.pot,
            "your_chips": view.my_chips,
        }

    # ------------------------------------------------------------------
    # Fallback and context
    # ------------------------------------------------------------------

    def _fallback(
        self,
        view: AgentView,
        emit,
        turn_id: str,
        started: float,
        steps: int,
        usage: dict,
        reason: str,
        allow_check: bool = False,
    ) -> Action:
        legal = set(view.legal_actions)
        if "call" in legal:
            action = Action(view.seat, ActionType.CALL, source="llm", auto=True)
            note = self._t("agent.note.auto_call")
        elif allow_check and "check" in legal:
            action = Action(view.seat, ActionType.CHECK, source="llm", auto=True)
            note = self._t("agent.note.auto_check")
        else:
            action = Action(view.seat, ActionType.FOLD, source="llm", auto=True)
            note = self._t("agent.note.auto_fold")
        separator = "，" if self.locale == "zh-CN" else ", "
        suffix = "。" if self.locale == "zh-CN" else "."
        emit(
            {
                "type": "note",
                "level": "error",
                "turn_id": turn_id,
                "text": f"{reason}{separator}{note}{suffix}",
            }
        )
        emit(
            {
                "type": "turn_end",
                "turn_id": turn_id,
                "action": action.to_dict(),
                "auto": True,
                "steps": steps,
                "usage": usage,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "reason": reason,
            }
        )
        return action

    def _context_tokens(self) -> int:
        return self._messages_tokens(self.messages)

    def _messages_tokens(self, messages: Sequence[dict]) -> int:
        total = self._tools_tokens
        for message in messages:
            total += 6
            content = message.get("content")
            if isinstance(content, str):
                total += estimate_tokens(content)
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str):
                total += estimate_tokens(reasoning)
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str):
                total += estimate_tokens(tool_call_id)
            for call in message.get("tool_calls") or []:
                total += estimate_tokens(str(call.get("id") or ""))
                function = call.get("function") or {}
                total += estimate_tokens(str(function.get("name") or ""))
                total += estimate_tokens(str(function.get("arguments") or ""))
        return total

    def _prepare_context(self, emit, force: bool = False) -> None:
        """Apply the configured context policy before a model request."""
        if self.context.policy == "full":
            return
        if self.context.policy == "legacy":
            self._trim_context(emit, force=force)
            return
        self._compact_context(emit, force=force)

    def _compact_context(self, emit, force: bool = False) -> None:
        """Drop old hand detail blocks, keeping compact summaries and recent hands."""
        before_tokens = self._context_tokens()
        if not force and before_tokens <= self.context.high_water_tokens:
            return
        before_count = len(self.messages)
        keep = self.context.keep_recent_hands
        compacted, dropped = compact_history(
            self.messages,
            keep,
            self._turn_marker,
            self._summary_marker,
        )
        if not dropped:
            if force:
                self._trim_context(emit, force=True)
            return
        self.messages = compacted
        # If the recent window is still too large, shrink it one hand at a time.
        while (
            keep > 1
            and self._context_tokens() > self.context.high_water_tokens
        ):
            keep -= 1
            compacted, extra = compact_history(
                self.messages,
                keep,
                self._turn_marker,
                self._summary_marker,
            )
            if not extra:
                break
            self.messages = compacted
            dropped += extra
        if self._context_tokens() > self.context.high_water_tokens:
            # Last resort: reuse the legacy turn-level trim.  This still cuts at
            # complete turn boundaries and never edits a retained message.
            self._trim_context(emit, force=False)
        after_tokens = self._context_tokens()
        self._compactions += 1
        emit(
            {
                "type": "context_compacted",
                "policy": self.context.policy,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "before_messages": before_count,
                "after_messages": len(self.messages),
                "dropped_messages": dropped,
                "keep_recent_hands": keep,
            }
        )
        emit(
            {
                "type": "note",
                "level": "warn",
                "text": self._t(
                    "agent.note.context_compacted",
                    removed=dropped,
                    tokens=after_tokens,
                ),
            }
        )

    def _trim_context(self, emit, force: bool = False) -> None:
        before_tokens = self._context_tokens()
        before_count = len(self.messages)
        removed = 0
        while self._context_tokens() > self.max_context_tokens or (force and removed == 0):
            turn_starts = [
                index
                for index, message in enumerate(self.messages)
                if message.get("role") == "user"
                and str(message.get("content", "")).startswith(self._turn_marker)
            ]
            if len(turn_starts) <= 1:
                break
            # Drop the oldest complete turn, always cutting at a turn boundary.
            cut = turn_starts[1]
            removed += cut - 1
            self.messages = [self.messages[0]] + self.messages[cut:]
            force = False
        if removed:
            emit(
                {
                    "type": "context_trimmed",
                    "policy": "legacy",
                    "before_tokens": before_tokens,
                    "after_tokens": self._context_tokens(),
                    "before_messages": before_count,
                    "after_messages": len(self.messages),
                    "dropped_messages": removed,
                }
            )
        if removed and not self._trimmed:
            self._trimmed = True
            emit(
                {
                    "type": "note",
                    "level": "warn",
                    "text": self._t("agent.note.context_trimmed", removed=removed),
                }
            )


__all__ = [
    "LLMAgent",
    "TABLE_TALK_TOOL",
    "TOOLS",
    "TOOL_SCHEMAS",
    "select_tools",
]

