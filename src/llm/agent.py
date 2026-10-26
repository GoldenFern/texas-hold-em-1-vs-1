"""The tool-using LLM poker agent."""

from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, replace
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
from .retry import RetryBudget, RetryPolicy
from .context import (
    CheckpointCut,
    ContextSettings,
    MatchStats,
    TriggerBudget,
    checkpoint_message_indices,
    choose_checkpoint_cut,
    compact_history,
    compact_tool_payload,
    find_hand_starts,
    flatten_checkpoint_transcript,
    is_checkpoint_message,
    resolve_trigger_budget,
)
from .prompt import (
    HISTORY_TOOL_NAME,
    TABLE_TALK_TOOL_NAME,
    build_checkpoint_message,
    build_checkpoint_nudge,
    build_checkpoint_prompt,
    build_compact_turn_message,
    build_fallback_checkpoint,
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
    HISTORY_TOOL,
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


@dataclass
class ResumeOutcome:
    """Result of resolving an incomplete or output-limited reply."""

    action: Optional[Action] = None
    length_continuations: int = 0
    reused_step: bool = False


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
        history_provider: Optional[Callable[..., dict]] = None,
        table_talk_mode: str = "tool",
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
        self._checkpoint_marker = self._prompt_text["checkpoint_marker"]
        self.context = ContextSettings.from_runtime(runtime)
        self.client = client
        self.analyzer = analyzer
        self.equity_analyzer = equity_analyzer or EquityAnalyzer()
        self.validate_action = validate_action
        self.history_provider = history_provider
        self.max_steps = int(runtime.get("max_steps_per_turn", 12))
        self.max_failures = int(runtime.get("max_consecutive_failures", 3))
        self.retry_policy = RetryPolicy.from_runtime(runtime)
        self.partial_retry_mode = str(
            runtime.get("partial_retry_mode", "continue")
        ).strip().lower()
        if self.partial_retry_mode not in ("continue", "restart"):
            self.partial_retry_mode = "continue"
        self._incomplete_streams = 0
        self._retry_scheduled = 0
        self.require_estimate_first = bool(runtime.get("require_estimate_first", False))
        self.require_estimate_each_turn = bool(
            runtime.get("require_estimate_each_turn", False)
        )
        self.skip_estimate_on_complete_board = bool(
            runtime.get("skip_estimate_on_complete_board", False)
        )
        # Context budget: ``max_context_tokens_estimate=0`` means automatic,
        # i.e. the model window minus the reserved output tokens minus a
        # 2048-token safety margin.  A positive runtime value is a hard cap
        # that can only tighten the automatic model budget; when the model
        # window is unknown it is used as-is, with 220000 as the last resort.
        runtime_cap = int(runtime.get("max_context_tokens_estimate", 0) or 0)
        reserved_output = int(self.params.get("max_tokens", 32768))
        context_window = int(getattr(spec, "context_window", 0) or 0)
        if context_window > 0:
            model_budget = max(8000, context_window - reserved_output - 2048)
            if runtime_cap > 0:
                prompt_budget = min(runtime_cap, model_budget)
            else:
                prompt_budget = model_budget
        else:
            prompt_budget = runtime_cap if runtime_cap > 0 else 220000
        self.max_context_tokens = max(8000, prompt_budget)
        if self.context.uses_checkpoint:
            self.trigger_budget = resolve_trigger_budget(
                self.context, self.max_context_tokens, context_window
            )
            self.context = replace(
                self.context,
                high_water_tokens=self.trigger_budget.high_water_tokens,
                low_water_tokens=self.trigger_budget.low_water_tokens,
            )
        else:
            self.context = self.context.with_context_window(self.max_context_tokens)
            self.trigger_budget = TriggerBudget(
                high_water_tokens=self.context.high_water_tokens,
                hard_water_tokens=self.max_context_tokens,
                low_water_tokens=self.context.low_water_tokens,
                buffer_tokens=0,
                auto_ceiling_tokens=self.max_context_tokens,
                auto_enabled=True,
            )
        self.max_estimate_calls = int(runtime.get("max_estimate_calls_per_turn", 2))
        self.max_estimate_iterations = int(runtime.get("max_estimate_iterations", 50000))
        self.max_equity_calls = int(runtime.get("max_equity_calls_per_turn", 2))
        self.max_equity_iterations = int(runtime.get("max_equity_iterations", 50000))
        self.max_history_calls = int(runtime.get("max_history_calls_per_turn", 2))
        self.table_talk_limits = TableTalkLimits.from_runtime(runtime)
        self.submit_table_talk = submit_table_talk
        mode = str(table_talk_mode or "tool").strip().lower()
        if mode not in ("tool", "inline", "off"):
            mode = "tool"
        if submit_table_talk is None:
            mode = "off"
        self.table_talk_mode = mode
        self.table_talk_required = mode != "off"
        self.tools = (
            list(tools)
            if tools is not None
            else build_tools(
                self.locale,
                table_talk_in_action=(mode == "inline"),
            )
        )
        if self.history_provider is None:
            self.tools = [
                tool
                for tool in self.tools
                if tool["function"]["name"] != HISTORY_TOOL_NAME
            ]
        self.tool_names = {tool["function"]["name"] for tool in self.tools}
        if self.require_estimate_each_turn and "estimate_hand_probabilities" not in self.tool_names:
            raise ValueError(self._t("agent.constructor.estimate_required"))
        if self.table_talk_required and mode == "tool" and TABLE_TALK_TOOL_NAME not in self.tool_names:
            raise ValueError(self._t("agent.constructor.table_talk_tool_missing"))
        if self.table_talk_required and mode == "inline" and "apply_action" not in self.tool_names:
            raise ValueError(self._t("agent.constructor.table_talk_action_missing"))
        if self.table_talk_required and mode == "inline":
            apply_tool = next(
                (
                    tool
                    for tool in self.tools
                    if tool["function"]["name"] == "apply_action"
                ),
                None,
            )
            properties = (
                apply_tool["function"]["parameters"].get("properties", {})
                if apply_tool
                else {}
            )
            if "table_talk" not in properties:
                raise ValueError(
                    self._t("agent.constructor.table_talk_field_missing")
                )
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
                table_talk_mode=mode,
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
        self.context_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_miss_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._context_anchor: Optional[tuple[int, int]] = None
        self._checkpoint_through = 0
        self._checkpoint_retain_from = 0
        self._checkpoint_disabled = False
        self._summary_calls = 0
        self._summary_failures = 0
        self._summary_last_hand = 0
        self._summary_last_street = ""
        self._current_hand = 0
        self._current_street = ""
        self._history_calls_this_turn = 0
        self._last_checkpoint_error = ""
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
        self.context_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_miss_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._context_anchor = None
        self._checkpoint_through = 0
        self._checkpoint_retain_from = 0
        self._checkpoint_disabled = False
        self._summary_calls = 0
        self._summary_failures = 0
        self._summary_last_hand = 0
        self._summary_last_street = ""
        self._current_hand = 0
        self._current_street = ""
        self._history_calls_this_turn = 0
        self._last_checkpoint_error = ""
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
            self._prepare_context(emit, final=True)
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
            "context_usage": dict(self.context_usage),
            "last_error": self.last_error,
            "retry": {
                "mode": self.partial_retry_mode,
                "max_retries": self.retry_policy.max_retries,
                "scheduled": self._retry_scheduled,
                "incomplete_streams": self._incomplete_streams,
            },
            "table_talk": self.table_talk_required,
            "table_talk_mode": self.table_talk_mode,
            "history_tool": HISTORY_TOOL_NAME in self.tool_names,
            "context": {
                "policy": self.context.policy,
                "high_water_tokens": self.context.high_water_tokens,
                "low_water_tokens": self.context.low_water_tokens,
                "keep_recent_hands": self.context.keep_recent_hands,
                "keep_recent_tokens": self.context.keep_recent_tokens,
                "buffer_tokens": self.trigger_budget.buffer_tokens,
                "auto_ceiling_tokens": self.trigger_budget.auto_ceiling_tokens,
                "hard_water_tokens": self.trigger_budget.hard_water_tokens,
                "auto_trigger": self.trigger_budget.auto_enabled,
                "state_detail": self.context.state_detail,
                "tool_detail": self.context.tool_detail,
                "final_talk_context": self.context.final_talk_context,
                "compactions": self._compactions,
                "summary_calls": self._summary_calls,
                "summary_failures": self._summary_failures,
                "checkpoint_disabled": self._checkpoint_disabled,
                "checkpoint_through": self._checkpoint_through,
                "retained_from": self._checkpoint_retain_from,
                "last_checkpoint_error": self._last_checkpoint_error,
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
        length_continuations = 0
        retry_same_step = False
        continuation_start: Optional[int] = None
        continuation_active = False
        continuation_recovery_used = False
        self.turn_count += 1
        self._estimated_this_turn = False
        self._estimate_calls_this_turn = 0
        self._estimate_iterations_this_turn = 0
        self._equity_calls_this_turn = 0
        self._equity_iterations_this_turn = 0
        self._talk_count_this_turn = 0
        self._history_calls_this_turn = 0
        self._current_hand = view.hand_id
        self._current_street = view.street
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
        if self.context.uses_checkpoint and not self._checkpoint_disabled:
            self._maybe_checkpoint(emit, mid_hand=False)
        if self.context.state_detail == "compact":
            turn_message = build_compact_turn_message(
                view,
                require_table_talk=self.table_talk_required,
                table_talk_mode=self.table_talk_mode,
                locale=self.locale,
            )
        else:
            turn_message = build_turn_message(
                view,
                require_table_talk=self.table_talk_required,
                table_talk_mode=self.table_talk_mode,
                locale=self.locale,
            )
        self.messages.append({"role": "user", "content": turn_message})
        emit({"type": "turn_state", "turn_id": turn_id, "text": turn_message})
        self._prepare_context(emit)

        while steps < self.max_steps:
            if self._cancel_event.is_set():
                return Action(view.seat, ActionType.FOLD, source="llm", auto=True)
            if not retry_same_step:
                budget = RetryBudget(self.retry_policy)
            retry_same_step = False
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
                if event.get("type") == "retry_scheduled":
                    self._retry_scheduled += 1
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
                    retry_budget=budget,
                )
                self.last_error = ""
            except LLMError as exc:
                if exc.kind == "cancelled":
                    return Action(view.seat, ActionType.FOLD, source="llm", auto=True)
                if (
                    continuation_active
                    and not continuation_recovery_used
                    and exc.kind in (
                        "param",
                        "invalid_request",
                        "unknown",
                        "context_length",
                        "payload_too_large",
                    )
                ):
                    if (
                        continuation_start is not None
                        and 0 <= continuation_start <= len(self.messages)
                    ):
                        del self.messages[continuation_start:]
                    continuation_active = False
                    continuation_recovery_used = True
                    retry_same_step = True
                    steps -= 1
                    emit(
                        {
                            "type": "note",
                            "level": "warn",
                            "turn_id": turn_id,
                            "text": self._t("agent.note.continuation_rebuild"),
                        }
                    )
                    continue
                if exc.kind in ("context_length", "payload_too_large"):
                    before = len(self.messages)
                    if self.context.uses_checkpoint and not self._checkpoint_disabled:
                        self._maybe_checkpoint(emit, mid_hand=True, force=True)
                    if len(self.messages) == before:
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
                        retry_same_step = True
                        steps -= 1
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
                    allow_call=False,
                    reason_code=exc.kind or "llm_error",
                    auto_kind="api_error",
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
                    allow_call=False,
                    reason_code="exception",
                    auto_kind="api_error",
                )

            for key in usage_total:
                amount = int(result.usage.get(key, 0))
                usage_total[key] += amount
                self.total_usage[key] = self.total_usage.get(key, 0) + amount

            length_stop = result.finish_reason == "length"
            if not length_stop:
                length_continuations = 0
            incomplete = bool(
                result.incomplete
                or length_stop
                or result.finish_reason == "content_filter"
            )
            if result.incomplete:
                self._incomplete_streams += 1
            if not incomplete:
                continuation_active = False
                continuation_start = None
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
                    "error": bool(incomplete),
                    "incomplete": bool(incomplete),
                    "incomplete_reason": result.incomplete_reason,
                    "tool_calls": [
                        {
                            "index": index,
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                            "complete": bool(call.complete),
                        }
                        for index, call in enumerate(result.tool_calls)
                    ],
                }
            )

            if (
                result.finish_reason == "content_filter"
                or result.incomplete_reason == "content_filter"
            ):
                return self._fallback(
                    view,
                    emit,
                    turn_id,
                    started,
                    steps,
                    usage_total,
                    reason=self._t("agent.reason.content_filter"),
                    allow_check=True,
                    allow_call=False,
                    reason_code="content_filter",
                    auto_kind="content_filter",
                )

            action_hint = (
                self._find_valid_action(view, result)
                if (incomplete or length_stop)
                else None
            )
            if (
                incomplete
                and not length_stop
                and action_hint is None
                and self.partial_retry_mode == "restart"
            ):
                decision = budget.next_retry("incomplete_stream", retryable=True)
                if not decision.retry:
                    return self._fallback(
                        view,
                        emit,
                        turn_id,
                        started,
                        steps,
                        usage_total,
                        reason=self._t("agent.reason.incomplete"),
                        allow_check=True,
                        allow_call=False,
                        reason_code="incomplete_stream",
                        auto_kind="incomplete",
                    )
                self._emit_retry(
                    emit,
                    turn_id,
                    message_id,
                    decision,
                    "incomplete_stream",
                    scope="turn",
                )
                if not budget.wait(decision.delay, self._cancel_event.is_set):
                    return Action(view.seat, ActionType.FOLD, source="llm", auto=True)
                steps -= 1
                retry_same_step = True
                continue

            if incomplete and not continuation_active:
                continuation_start = len(self.messages)
            self.messages.append(
                result.assistant_message(
                    bool(self.spec.request.get("reasoning_echo", False)),
                    lower_reasoning=bool(
                        incomplete and not length_stop and not result.tool_calls
                    ),
                )
            )
            self._update_context_anchor(result)
            soft_failures = bool(incomplete or length_stop)

            if not result.tool_calls:
                if incomplete or length_stop:
                    resume = self._resume_partial(
                        view,
                        emit,
                        turn_id,
                        started,
                        steps,
                        usage_total,
                        budget,
                        message_id,
                        length_stop=length_stop,
                        length_continuations=length_continuations,
                    )
                    if resume.action is not None:
                        return resume.action
                    length_continuations = resume.length_continuations
                    if resume.reused_step:
                        steps -= 1
                        retry_same_step = True
                    continuation_active = True
                    continue
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
                if failures >= self.max_failures and not soft_failures:
                    return self._fallback(
                        view,
                        emit,
                        turn_id,
                        started,
                        steps,
                        usage_total,
                        reason=self._t("agent.reason.no_action"),
                        reason_code="no_action",
                        auto_kind="no_action",
                    )
                self.messages.append(
                    {
                        "role": "user",
                        "content": build_nudge_message(
                            self.table_talk_required,
                            table_talk_mode=self.table_talk_mode,
                            locale=self.locale,
                        ),
                    }
                )
                continue

            chosen: Optional[Action] = None
            failures_before = failures
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
                    if failures >= self.max_failures and not soft_failures:
                        return self._fallback(
                            view,
                            emit,
                            turn_id,
                            started,
                            steps,
                            usage_total,
                            reason=self._t("agent.reason.invalid_action"),
                            reason_code="invalid_action",
                            auto_kind="invalid_action",
                        )
                    continue

                if call.name == "estimate_hand_probabilities":
                    if not self._handle_estimate(view, call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
                            )
                    continue

                if call.name == "estimate_equity":
                    if not self._handle_equity(view, call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
                            )
                    continue

                if call.name == "call_ev":
                    if not self._handle_call_ev(view, call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
                            )
                    continue

                if call.name == "get_random_number":
                    if not self._handle_random(call, parsed, emit, turn_id, index, tool_started):
                        failures += 1
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
                            )
                    continue

                if call.name == HISTORY_TOOL_NAME:
                    self._handle_history(
                        call,
                        parsed,
                        emit,
                        turn_id,
                        index,
                        tool_started,
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
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
                            )
                        continue

                    inline_talk_entry: Optional[dict] = None
                    if self.table_talk_mode == "inline":
                        talk_ok, inline_talk_entry = self._submit_inline_table_talk(
                            call,
                            parsed,
                            emit,
                            turn_id,
                            index,
                        )
                        if not talk_ok:
                            failures += 1
                            if failures >= self.max_failures and not soft_failures:
                                return self._fallback(
                                    view,
                                    emit,
                                    turn_id,
                                    started,
                                    steps,
                                    usage_total,
                                    reason=self._t("agent.reason.invalid_action"),
                                    reason_code="invalid_action",
                                    auto_kind="invalid_action",
                                )
                            continue
                    elif self.table_talk_required and self._talk_count_this_turn == 0:
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
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
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
                        if failures >= self.max_failures and not soft_failures:
                            return self._fallback(
                                view,
                                emit,
                                turn_id,
                                started,
                                steps,
                                usage_total,
                                reason=self._t("agent.reason.invalid_action"),
                                reason_code="invalid_action",
                                auto_kind="invalid_action",
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
                    if inline_talk_entry:
                        payload["table_talk"] = inline_talk_entry
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

            if incomplete or length_stop:
                failures = failures_before
                resume = self._resume_partial(
                    view,
                    emit,
                    turn_id,
                    started,
                    steps,
                    usage_total,
                    budget,
                    message_id,
                    length_stop=length_stop,
                    length_continuations=length_continuations,
                )
                if resume.action is not None:
                    return resume.action
                length_continuations = resume.length_continuations
                if resume.reused_step:
                    steps -= 1
                    retry_same_step = True
                continuation_active = True
                continue

        return self._fallback(
            view,
            emit,
            turn_id,
            started,
            steps,
            usage_total,
            reason=self._t("agent.reason.max_steps", max=self.max_steps),
            reason_code="max_steps",
            auto_kind="max_steps",
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

    def _handle_history(
        self,
        call: ToolCall,
        parsed: Optional[dict],
        emit,
        turn_id: str,
        index: int,
        started: float,
    ) -> None:
        """Handle a public history-research request.

        The tool is read-only and optional.  Budget errors and malformed
        arguments are returned to the model without counting toward the
        three-failure action fallback.
        """
        if self.history_provider is None:
            self._append_tool_error(
                call,
                self._t("agent.error.history_disabled"),
                emit,
                turn_id,
                index,
            )
            return
        if parsed is None or not isinstance(parsed, dict):
            self._append_tool_error(
                call,
                self._t("agent.error.invalid_json", args=call.arguments[:300]),
                emit,
                turn_id,
                index,
            )
            return
        if self._history_calls_this_turn >= self.max_history_calls:
            self._append_tool_error(
                call,
                self._t(
                    "agent.error.history_budget",
                    max=self.max_history_calls,
                ),
                emit,
                turn_id,
                index,
            )
            return
        scope = str(parsed.get("scope", "both") or "both").strip().lower()
        if scope not in ("current_match", "completed_matches", "both"):
            scope = "both"
        try:
            last_hands = int(parsed.get("last_hands", 20))
        except (TypeError, ValueError):
            last_hands = 20
        last_hands = max(0, min(50, last_hands))
        focus = str(parsed.get("focus", "all") or "all").strip().lower()
        if focus not in (
            "all",
            "preflop",
            "postflop",
            "showdown",
            "result",
            "table_talk",
        ):
            focus = "all"
        self._history_calls_this_turn += 1
        try:
            report = self.history_provider(
                self.seat,
                scope=scope,
                last_hands=last_hands,
                focus=focus,
            )
            if not isinstance(report, dict):
                raise ValueError("history provider returned no report")
            payload = {"ok": True, **report}
        except Exception as exc:  # noqa: BLE001
            payload = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        current = payload.get("current_match") if isinstance(payload, dict) else {}
        hands = int((current or {}).get("hands", 0) or 0)
        opponent = (current or {}).get("opponent") or {}
        preflop = opponent.get("preflop") if isinstance(opponent, dict) else {}
        vpip = preflop.get("vpip") if isinstance(preflop, dict) else {}
        if not payload.get("ok", True):
            summary = str(
                payload.get("error") or self._t("agent.error.history_disabled")
            )
        elif isinstance(vpip, dict) and vpip.get("value") is not None:
            summary = self._t(
                "agent.summary.history",
                hands=hands,
                value=round(float(vpip["value"]) * 100, 1),
                n=int(vpip.get("n", 0) or 0),
            )
        else:
            summary = self._t("agent.summary.history_empty", hands=hands)
        self._append_tool_result(
            call,
            payload,
            emit,
            turn_id,
            index,
            started,
            summary=summary,
        )

    def _submit_inline_table_talk(
        self,
        call: ToolCall,
        parsed: dict,
        emit,
        turn_id: str,
        index: int,
    ) -> tuple[bool, Optional[dict]]:
        """Record the mandatory table talk carried by apply_action.

        Only the first apply_action of a turn needs to carry the talk. A
        retry after an invalid action may omit it or repeat it; a repeated
        talk is ignored so the per-turn cap does not block the retry.
        """
        if self._talk_count_this_turn > 0:
            return True, None
        raw_text = parsed.get("table_talk") if isinstance(parsed, dict) else None
        if not isinstance(raw_text, str) or not raw_text.strip():
            self._append_tool_error(
                call,
                self._t("agent.error.talk_inline_missing"),
                emit,
                turn_id,
                index,
                extra={"required_tool": "table_talk"},
            )
            return False, None
        if self.submit_table_talk is None:
            self._append_tool_error(
                call,
                self._t("agent.error.talk_disabled"),
                emit,
                turn_id,
                index,
            )
            return False, None
        result = self.submit_table_talk(raw_text, turn_id)
        if not isinstance(result, dict) or not result.get("ok"):
            error = (
                str(result.get("error") or "")
                if isinstance(result, dict)
                else ""
            )
            self._append_tool_error(
                call,
                error or self._t("agent.error.talk_inline_missing"),
                emit,
                turn_id,
                index,
                extra={"required_tool": "table_talk"},
            )
            return False, None
        self._talk_count_this_turn += 1
        entry = result.get("entry") if isinstance(result, dict) else None
        return True, entry if isinstance(entry, dict) else {"text": raw_text}

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
                "ok": bool(payload.get("ok", True)),
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

    def _find_valid_action(
        self,
        view: AgentView,
        result,
    ) -> Optional[Action]:
        """Return a fully valid action already present in a partial reply."""
        for call in result.tool_calls:
            if call.name != "apply_action" or not call.complete:
                continue
            parsed = call.parsed_arguments()
            if not isinstance(parsed, dict):
                continue
            action, error = self._build_action(view, parsed)
            if action is None or error:
                continue
            try:
                validation_error = self.validate_action(action)
            except Exception:  # noqa: BLE001
                continue
            if validation_error:
                continue
            return action
        return None

    def _emit_retry(
        self,
        emit,
        turn_id: str,
        message_id: str,
        decision,
        kind: str,
        *,
        scope: str = "turn",
    ) -> None:
        """Publish one retry countdown event."""
        self._retry_scheduled += 1
        emit(
            {
                "type": "retry_scheduled",
                "scope": scope,
                "turn_id": turn_id,
                "message_id": message_id,
                "attempt": decision.attempt,
                "max_retries": self.retry_policy.max_retries,
                "delay_ms": round(decision.delay * 1000.0),
                "at": round(time.time() * 1000.0 + decision.delay * 1000.0),
                "error_kind": kind,
                "text": self._t(
                    "agent.retry_scheduled",
                    kind=kind,
                    delay=f"{decision.delay:.1f}",
                    attempt=decision.attempt,
                ),
            }
        )

    def _resume_partial(
        self,
        view: AgentView,
        emit,
        turn_id: str,
        started: float,
        steps: int,
        usage: dict,
        budget: RetryBudget,
        message_id: str,
        *,
        length_stop: bool,
        length_continuations: int,
    ) -> ResumeOutcome:
        """Continue or stop after an incomplete or output-limited reply."""
        if length_stop:
            if length_continuations >= 2:
                action = self._fallback(
                    view,
                    emit,
                    turn_id,
                    started,
                    steps,
                    usage,
                    reason=self._t("agent.reason.output_limit"),
                    allow_check=True,
                    allow_call=False,
                    reason_code="output_limit",
                    auto_kind="output_limit",
                )
                return ResumeOutcome(
                    action=action,
                    length_continuations=length_continuations,
                )
            emit(
                {
                    "type": "note",
                    "level": "warn",
                    "turn_id": turn_id,
                    "text": self._t("agent.note.output_limit"),
                }
            )
            self.messages.append(
                {
                    "role": "user",
                    "content": self._t("agent.continuation.output_limit"),
                }
            )
            return ResumeOutcome(
                length_continuations=length_continuations + 1,
                reused_step=False,
            )

        decision = budget.next_retry("incomplete_stream", retryable=True)
        if not decision.retry:
            action = self._fallback(
                view,
                emit,
                turn_id,
                started,
                steps,
                usage,
                reason=self._t("agent.reason.incomplete"),
                allow_check=True,
                allow_call=False,
                reason_code="incomplete_stream",
                auto_kind="incomplete",
            )
            return ResumeOutcome(
                action=action,
                length_continuations=length_continuations,
            )
        self._emit_retry(
            emit,
            turn_id,
            message_id,
            decision,
            "incomplete_stream",
            scope="turn",
        )
        if not budget.wait(decision.delay, self._cancel_event.is_set):
            return ResumeOutcome(
                action=Action(view.seat, ActionType.FOLD, source="llm", auto=True),
                length_continuations=length_continuations,
            )
        self.messages.append(
            {
                "role": "user",
                "content": self._t("agent.continuation.incomplete"),
            }
        )
        return ResumeOutcome(
            length_continuations=length_continuations,
            reused_step=True,
        )

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
        allow_call: bool = True,
        reason_code: str = "",
        auto_kind: str = "",
    ) -> Action:
        legal = set(view.legal_actions)
        if allow_check and "check" in legal:
            action = Action(view.seat, ActionType.CHECK, source="llm", auto=True)
            note = self._t("agent.note.auto_check")
        elif allow_call and "call" in legal:
            action = Action(view.seat, ActionType.CALL, source="llm", auto=True)
            note = self._t("agent.note.auto_call")
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
                "reason_code": reason_code or auto_kind or "fallback",
                "auto_kind": auto_kind or "fallback",
            }
        )
        return action

    def _context_tokens(self) -> int:
        if self._context_anchor is not None:
            index, measured = self._context_anchor
            if 0 <= index <= len(self.messages):
                return int(measured) + self._estimate_messages_tokens(
                    self.messages[index:], include_tools=False
                )
        return self._estimate_messages_tokens(self.messages, include_tools=True)

    def _estimate_messages_tokens(
        self,
        messages: Sequence[dict],
        *,
        include_tools: bool = True,
    ) -> int:
        total = self._tools_tokens if include_tools else 0
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

    def _messages_tokens(self, messages: Sequence[dict]) -> int:
        return self._estimate_messages_tokens(messages, include_tools=True)

    def _update_context_anchor(self, result) -> None:
        """Anchor the next estimate on the provider's real prompt usage."""
        usage = result.usage or {}
        if result.estimated_usage or int(usage.get("input_tokens", 0) or 0) <= 0:
            self._context_anchor = None
            return
        measured = (
            int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("output_tokens", 0) or 0)
            + int(usage.get("reasoning_tokens", 0) or 0)
        )
        self._context_anchor = (len(self.messages), measured)

    def _invalidate_context_anchor(self) -> None:
        self._context_anchor = None

    def _add_context_usage(self, usage: dict) -> None:
        """Bill a compaction call separately from the game turn's usage."""
        for key in self.context_usage:
            amount = int((usage or {}).get(key, 0) or 0)
            self.context_usage[key] += amount
            self.total_usage[key] = int(self.total_usage.get(key, 0)) + amount

    def _prepare_context(
        self,
        emit,
        force: bool = False,
        *,
        mid_hand: bool = True,
        final: bool = False,
    ) -> None:
        """Apply the configured context policy before a model request."""
        if self.context.uses_checkpoint:
            if final:
                return
            if self._checkpoint_disabled:
                self._compact_context(emit, force=force)
                return
            self._maybe_checkpoint(emit, mid_hand=mid_hand, force=force)
            return
        if self.context.policy == "full":
            return
        if self.context.policy == "legacy":
            self._trim_context(emit, force=force)
            return
        self._compact_context(emit, force=force)

    def _checkpoint_stats(self) -> Optional[dict]:
        """Fetch the compact statistics block used by the summarizer."""
        if self.history_provider is None:
            return None
        try:
            return self.history_provider(
                self.seat,
                scope="both",
                last_hands=5,
                focus="all",
                checkpoint=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._last_checkpoint_error = f"{type(exc).__name__}: {exc}"
            return None

    def _maybe_checkpoint(
        self,
        emit,
        *,
        mid_hand: bool,
        force: bool = False,
    ) -> None:
        """Summarize completed hands into a checkpoint when the budget is due."""
        if self._checkpoint_disabled or not self.context.uses_checkpoint:
            return
        if self._cancel_event.is_set():
            return
        budget = self.trigger_budget
        if not budget.auto_enabled and not force:
            return
        current_hand = int(self._current_hand or 0)
        if current_hand <= 0:
            return
        before_tokens = self._context_tokens()
        threshold = (
            budget.hard_water_tokens if mid_hand else budget.high_water_tokens
        )
        if not force and before_tokens < threshold:
            return
        if (
            not force
            and self._summary_last_hand == current_hand
            and self._summary_last_street == self._current_street
        ):
            return
        cut = choose_checkpoint_cut(
            self.messages,
            current_hand,
            self.context.keep_recent_hands,
            self.context.keep_recent_tokens,
            self._turn_marker,
            self._summary_marker,
            lambda items: self._estimate_messages_tokens(
                items, include_tools=False
            ),
        )
        if cut is None:
            if force:
                self._trim_context(emit, force=True)
            return
        self._summary_last_hand = current_hand
        self._summary_last_street = self._current_street
        stats = self._checkpoint_stats()
        has_previous = any(
            is_checkpoint_message(str(message.get("content") or ""), self._checkpoint_marker)
            for message in self.messages
        )
        prompt = build_checkpoint_prompt(
            cut.through,
            cut.retain_from,
            self.locale,
            current_hand=current_hand,
            stats=stats,
            has_previous=has_previous,
        )
        emit(
            {
                "type": "context_summary_start",
                "policy": "checkpoint",
                "reason": (
                    "force"
                    if force
                    else ("mid_hand" if mid_hand else "hand_boundary")
                ),
                "before_tokens": before_tokens,
                "through_hand": cut.through,
                "retained_from": cut.retain_from,
                "current_hand": current_hand,
                "stats": bool(stats),
            }
        )
        calls_before = self._summary_calls
        result = self._run_checkpoint_summary(cut, prompt, emit)
        if self._cancel_event.is_set():
            return
        if result is None:
            if self._last_checkpoint_error == "cancelled":
                return
            self._summary_failures += 1
            self._last_checkpoint_error = (
                self._last_checkpoint_error or "summary_failed"
            )
            emit(
                {
                    "type": "context_summary_failed",
                    "kind": self._last_checkpoint_error,
                    "attempts": max(0, self._summary_calls - calls_before),
                    "fallback": True,
                }
            )
            emit(
                {
                    "type": "note",
                    "level": "warn",
                    "text": self._t(
                        "agent.note.context_checkpoint_failed",
                        kind=self._last_checkpoint_error,
                    ),
                }
            )
            fallback = build_fallback_checkpoint(stats, cut.through, self.locale)
            self._apply_checkpoint(
                cut,
                fallback,
                emit,
                fallback=True,
                usage={},
                attempts=0,
                latency_ms=0.0,
            )
            if self._summary_failures >= 2:
                self._checkpoint_disabled = True
            return
        summary, usage, attempts, latency_ms = result
        self._summary_failures = 0
        self._apply_checkpoint(
            cut,
            summary,
            emit,
            fallback=False,
            usage=usage,
            attempts=attempts,
            latency_ms=latency_ms,
        )

    def _summary_headings(self) -> list[str]:
        template = str(self._prompt_text.get("checkpoint_template", ""))
        return [
            line.strip()
            for line in template.splitlines()
            if line.strip().startswith("## ")
        ]

    def _summary_valid(self, text: str) -> tuple[bool, bool]:
        """Return ``(valid, looks_like_stats_dump)`` for a summary reply."""
        body = str(text or "").strip()
        if len(body) < 80:
            return False, False
        lowered = body.lower()
        stats_dump = (
            "vpip" in lowered
            and "pfr" in lowered
            and ("{" in body or "```" in body)
        )
        headings = self._summary_headings()
        filled = sum(1 for heading in headings if heading in body)
        return (filled >= 4 and not stats_dump), stats_dump

    def _summary_stream(
        self,
        messages: Sequence[dict],
        params: dict,
        budget: RetryBudget,
        emit,
        tool_choice: Optional[str],
    ):
        """Call the model once, forwarding retry countdown events."""

        def on_event(event: dict) -> None:
            if event.get("type") != "retry_scheduled":
                return
            payload = dict(event)
            payload["scope"] = "summary"
            emit(payload)

        def run(choice: Optional[str]):
            try:
                return self.client.stream(
                    self.spec,
                    list(messages),
                    self.tools,
                    params,
                    on_event,
                    cancel=self._cancel_event.is_set,
                    tool_choice=choice,
                    retry_budget=budget,
                )
            except TypeError:
                # Older/fake clients may not accept the optional keywords.
                try:
                    return self.client.stream(
                        self.spec,
                        list(messages),
                        self.tools,
                        params,
                        on_event,
                        cancel=self._cancel_event.is_set,
                        tool_choice=choice,
                    )
                except TypeError:
                    return self.client.stream(
                        self.spec,
                        list(messages),
                        self.tools,
                        params,
                        on_event,
                        cancel=self._cancel_event.is_set,
                    )

        try:
            return run(tool_choice)
        except LLMError as exc:
            if exc.kind == "param" and tool_choice is not None:
                return run(None)
            raise

    def _emit_summary_retry(self, emit, decision, kind: str) -> None:
        """Publish a retry countdown for a checkpoint summary request."""
        self._retry_scheduled += 1
        emit(
            {
                "type": "retry_scheduled",
                "scope": "summary",
                "attempt": decision.attempt,
                "max_retries": self.retry_policy.max_retries,
                "delay_ms": round(decision.delay * 1000.0),
                "at": round(time.time() * 1000.0 + decision.delay * 1000.0),
                "error_kind": kind,
                "text": self._t(
                    "agent.retry_scheduled",
                    kind=kind,
                    delay=f"{decision.delay:.1f}",
                    attempt=decision.attempt,
                ),
            }
        )

    def _summary_attempt(
        self,
        messages: Sequence[dict],
        params: dict,
        budget: RetryBudget,
        emit,
        usage_total: dict,
    ):
        """Send one summary request, retrying partial drafts under the budget."""
        while True:
            if self._cancel_event.is_set():
                return None, LLMError(
                    "cancelled", self._t("client.cancelled"), retryable=False
                )
            self._summary_calls += 1
            try:
                result = self._summary_stream(
                    messages, params, budget, emit, "none"
                )
            except LLMError as exc:
                return None, exc
            for key in usage_total:
                usage_total[key] += int(result.usage.get(key, 0) or 0)
            self._add_context_usage(result.usage)
            if result.finish_reason == "length" or result.incomplete_reason == "length":
                return None, LLMError(
                    "summary_length",
                    "summary reached the output token limit",
                    retryable=False,
                )
            if (
                result.finish_reason == "content_filter"
                or result.incomplete_reason == "content_filter"
            ):
                return None, LLMError(
                    "summary_content_filter",
                    "summary was blocked by the provider",
                    retryable=False,
                )
            if result.incomplete:
                decision = budget.next_retry(
                    "incomplete_stream", retryable=True
                )
                if not decision.retry:
                    return None, LLMError(
                        "incomplete_stream",
                        "summary stream was interrupted",
                        retryable=False,
                    )
                self._emit_summary_retry(emit, decision, "incomplete_stream")
                if not budget.wait(decision.delay, self._cancel_event.is_set):
                    return None, LLMError(
                        "cancelled", self._t("client.cancelled"), retryable=False
                    )
                continue
            return result, None

    def _summary_send(
        self,
        messages: Sequence[dict],
        params: dict,
        budget: RetryBudget,
        emit,
        usage_total: dict,
    ):
        """Send one candidate and allow one template-correction nudge."""
        result, error = self._summary_attempt(
            messages, params, budget, emit, usage_total
        )
        if result is None:
            return None, error, 1
        valid, stats_dump = self._summary_valid(result.content)
        if valid:
            return result, None, 1
        nudge = build_checkpoint_nudge(self.locale, stats_dump=stats_dump)
        nudged = list(messages) + [{"role": "user", "content": nudge}]
        retry, retry_error = self._summary_attempt(
            nudged, params, budget, emit, usage_total
        )
        if retry is None:
            return None, retry_error, 2
        valid, _ = self._summary_valid(retry.content)
        if valid:
            return retry, None, 2
        return None, LLMError(
            "summary_invalid",
            "summary did not match the required template",
            retryable=False,
        ), 2

    def _run_checkpoint_summary(
        self,
        cut: CheckpointCut,
        prompt: str,
        emit,
    ) -> Optional[tuple[str, dict, int, float]]:
        """Run the cache-friendly summary request with graded shrinking.

        The first attempt sends the unchanged conversation plus the summary
        prompt so the provider can reuse the normal request's prompt cache.
        On a context-length rejection it drops the retained tail from the
        summary request, then flattens older hands to text and shrinks through
        ``context_summary_retries``.  The newest hand is never removed.
        """
        started = time.perf_counter()
        start_calls = self._summary_calls
        params = dict(self.params)
        max_tokens = max(
            512,
            min(
                int(params.get("max_tokens", 32768) or 32768),
                int(self.context.summary_max_tokens),
            ),
        )
        params["max_tokens"] = max_tokens
        usage_total = {key: 0 for key in self.context_usage}
        last_error_kind = ""
        budget = RetryBudget(self.retry_policy)
        prompt_tokens = estimate_tokens(prompt)
        available = max(
            2_000,
            self.max_context_tokens - self._tools_tokens - prompt_tokens - 1_024,
        )
        full_messages = self.messages + [{"role": "user", "content": prompt}]
        before_tail = self.messages[: cut.index] + [
            {"role": "user", "content": prompt}
        ]
        candidates: list[list[dict]] = []
        if self._estimate_messages_tokens(self.messages, include_tools=False) <= available:
            candidates.append(full_messages)
        if self._estimate_messages_tokens(
            self.messages[: cut.index], include_tools=False
        ) <= available:
            candidates.append(before_tail)
        for step in (1.0,) + tuple(self.context.summary_retry_steps):
            target = max(2_000, int(available * float(step)))
            transcript = flatten_checkpoint_transcript(
                self.messages,
                cut.index,
                target,
                estimate_tokens,
                self._checkpoint_marker,
                self._turn_marker,
                self._summary_marker,
            )
            if not transcript:
                continue
            candidates.append(
                [
                    self.messages[0],
                    {"role": "user", "content": f"{transcript}\n\n{prompt}"},
                ]
            )
        if not candidates:
            self._last_checkpoint_error = "no_summary_candidate"
            return None
        for messages in candidates:
            if self._cancel_event.is_set():
                break
            result, error, _calls = self._summary_send(
                messages, params, budget, emit, usage_total
            )
            if result is not None:
                return (
                    str(result.content).strip(),
                    usage_total,
                    self._summary_calls - start_calls,
                    (time.perf_counter() - started) * 1000.0,
                )
            if error is None:
                continue
            last_error_kind = error.kind
            if error.kind in ("context_length", "payload_too_large"):
                continue
            break
        self._last_checkpoint_error = last_error_kind or "summary_invalid"
        return None

    def _apply_checkpoint(
        self,
        cut: CheckpointCut,
        summary: str,
        emit,
        *,
        fallback: bool,
        usage: dict,
        attempts: int,
        latency_ms: float,
    ) -> None:
        """Replace old hands with a checkpoint and retain the recent tail."""
        before_tokens = self._context_tokens()
        before_messages = len(self.messages)
        old_messages = list(self.messages)
        checkpoint_text = build_checkpoint_message(
            summary, cut.through, self.locale
        )
        checkpoint_message = {"role": "user", "content": checkpoint_text}
        tail = list(old_messages[cut.index:])
        self.messages = [old_messages[0], checkpoint_message, *tail]
        self._invalidate_context_anchor()
        self._checkpoint_through = cut.through
        retained_from = cut.retain_from
        after_tokens = self._context_tokens()
        # Use the low-water target to keep more recent hands verbatim.  The
        # checkpoint already covers them, so this only adds detail, not risk.
        low = self.trigger_budget.low_water_tokens
        if after_tokens < low and retained_from > 1:
            starts = find_hand_starts(
                old_messages, self._turn_marker, self._summary_marker
            )
            previous = sorted(hand for hand in starts if hand < retained_from)
            for hand in reversed(previous):
                candidate = [
                    old_messages[0],
                    checkpoint_message,
                    *old_messages[starts[hand] :],
                ]
                candidate_tokens = self._estimate_messages_tokens(
                    candidate, include_tools=True
                )
                if candidate_tokens > low:
                    break
                self.messages = candidate
                retained_from = hand
                after_tokens = candidate_tokens
        self._checkpoint_retain_from = retained_from
        self._compactions += 1
        summary_tokens = estimate_tokens(checkpoint_text)
        emit(
            {
                "type": "context_summary_end",
                "ok": True,
                "fallback": fallback,
                "usage": dict(usage),
                "attempts": attempts,
                "latency_ms": round(latency_ms, 1),
                "summary_tokens": summary_tokens,
                "after_tokens": after_tokens,
                "through_hand": cut.through,
                "text": summary,
            }
        )
        emit(
            {
                "type": "context_compacted",
                "policy": "checkpoint",
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "before_messages": before_messages,
                "after_messages": len(self.messages),
                "dropped_messages": max(0, before_messages - len(self.messages)),
                "keep_recent_hands": self.context.keep_recent_hands,
                "through_hand": cut.through,
                "retained_from": retained_from,
                "summary_tokens": summary_tokens,
                "fallback": fallback,
            }
        )
        kept = max(
            0,
            int(self._current_hand or retained_from) - retained_from,
        )
        emit(
            {
                "type": "note",
                "level": "warn",
                "text": self._t(
                    "agent.note.context_checkpoint",
                    through=cut.through,
                    kept=kept,
                    tokens=after_tokens,
                ),
            }
        )
        if self._context_tokens() > self.trigger_budget.hard_water_tokens:
            self._compact_context(emit, force=True)

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
            protected = [
                self.messages[index]
                for index in checkpoint_message_indices(
                    self.messages, self._checkpoint_marker
                )
            ]
            old_len = len(self.messages)
            self.messages = [self.messages[0]] + protected + self.messages[cut:]
            removed += max(0, old_len - len(self.messages))
            force = False
        if removed:
            emit(
                {
                    "type": "context_trimmed",
                    "policy": self.context.policy,
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
    "HISTORY_TOOL",
    "LLMAgent",
    "TABLE_TALK_TOOL",
    "TOOLS",
    "TOOL_SCHEMAS",
    "select_tools",
]

