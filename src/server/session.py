"""In-memory match orchestrator for the heads-up table."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.base import AgentView
from src.agents.project_bot import ProjectBot
from src.agents.rlcard_bot import RLCardBot
from src.analysis.equity import EquityAnalyzer
from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.analysis.history import (
    HistoryStats,
    build_match_record,
    build_public_hand_record,
)
from src.engine.actions import Action, ActionType
from src.engine.heads_up import HeadsUpGame, Street
from src.i18n import normalize_locale, tr
from src.llm.agent import LLMAgent, select_tools
from src.llm.client import LLMClient
from src.llm.prompt import (
    AGENT_TOOL_NAMES,
    GAME_TOOL_NAMES,
    build_system_prompt,
)
from src.llm.providers import ProviderRegistry
from src.llm.retry import RetryPolicy
from src.llm.table_talk import TableTalkLimits, normalize_table_talk
from src.llm.table_talk_styles import DEFAULT_STYLE_ID, resolve_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAME_CONFIG_PATH = PROJECT_ROOT / "config" / "game.json"

BOT_DELAY_DEFAULT = 3.0
BOT_DELAY_MIN = 0.0
BOT_DELAY_MAX = 10.0


@dataclass
class SeatConfig:
    kind: str
    name: str
    provider_id: str = ""
    model_id: str = ""
    params: dict = None  # type: ignore[assignment]
    bot_agent_type: str = "random"
    bot_model_path: Optional[str] = None
    bot_delay_seconds: float = BOT_DELAY_DEFAULT
    table_talk_style: str = ""

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


class GameSession:
    """Owns the engine, agents, transcript buffers and the game thread."""

    def __init__(
        self,
        socketio,
        registry: ProviderRegistry,
        client: LLMClient,
        analyzer: HandDistributionAnalyzer,
        equity_analyzer: Optional[EquityAnalyzer] = None,
        ui_locale: str = "zh-CN",
    ) -> None:
        self.socketio = socketio
        self.registry = registry
        self.client = client
        self.analyzer = analyzer
        self.equity_analyzer = equity_analyzer or EquityAnalyzer()
        self.ui_locale = normalize_locale(ui_locale, "zh-CN") or "zh-CN"
        self.agent_locale = "zh-CN"
        self.lock = threading.RLock()
        self._start_lock = threading.Lock()
        self.game: Optional[HeadsUpGame] = None
        self.agents: Dict[int, object] = {}
        self.seat_configs: Dict[int, SeatConfig] = {}
        self.match_id = ""
        self.god_view = True
        self.max_hands = 0
        self.hand_history: List[dict] = []
        self.current_stats = HistoryStats()
        self.completed_matches: List[dict] = []
        self.transcripts: Dict[int, List[dict]] = {0: [], 1: []}
        self.table_talk_enabled = False
        self.table_talk_mode = "off"
        self._default_talk_style = "off"
        self._seat_styles: Dict[int, str] = {0: "", 1: ""}
        self.table_talk: List[dict] = []
        self.event_seq = 0
        self.state_seq = 0
        self.match_error = ""
        self.aborted = False
        self.started_at = 0.0
        self._generation = 0
        self._start_epoch = 0
        self._stop_event = threading.Event()
        self._human_event = threading.Event()
        self._next_hand_event = threading.Event()
        self._human_token: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None
        self._last_result_hand_id = -1
        self._game_config = self._load_game_config()
        self._table_talk_limits = TableTalkLimits.from_runtime(
            self._game_config.get("llm_runtime", {})
        )
        self._table_talk_seq = 0
        self._table_talk_cursors: Dict[int, int] = {0: 0, 1: 0}
        self._table_talk_turn_counts: Dict[tuple[int, str], int] = {}
        self._final_talk_sent: Dict[int, bool] = {0: False, 1: False}

    def set_ui_locale(self, locale: str) -> None:
        normalized = normalize_locale(locale, self.ui_locale)
        if not normalized:
            return
        self.ui_locale = normalized
        # Human-facing engine errors follow the current interface language.
        # Agent prompts and tool messages keep their own fixed match locale.
        if self.game is not None:
            self.game.locale = normalized

    def _t(self, message_key: str, **params: object) -> str:
        return tr(self.ui_locale, message_key, **params)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @staticmethod
    def _load_game_config() -> dict:
        with GAME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _normalize_rules(self, raw: Optional[dict]) -> dict:
        defaults = dict(self._game_config.get("defaults", {}))
        limits = self._game_config.get("limits", {})
        raw = raw or {}
        rules = dict(defaults)
        rules.update({key: value for key, value in raw.items() if value is not None})

        def int_value(key: str, default: int) -> int:
            value = rules.get(key, default)
            if isinstance(value, bool):
                raise ValueError(self._t("session.rule.int", key=key))
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(self._t("session.rule.int", key=key))
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(self._t("session.rule.int", key=key)) from exc

        for key, default in (
            ("starting_chips", 1000),
            ("small_blind", 5),
            ("big_blind", 10),
            ("rebuy_amount", 1000),
        ):
            limit = limits.get(key, {})
            minimum = int(limit.get("min", 1))
            maximum = int(limit.get("max", 100000))
            value = int_value(key, default)
            if not minimum <= value <= maximum:
                raise ValueError(
                    self._t("session.rule.range", key=key, min=minimum, max=maximum)
                )
            rules[key] = value

        max_hands_limit = int(limits.get("max_hands", {}).get("max", 1000))
        max_hands = int_value("max_hands", 0)
        if not 0 <= max_hands <= max_hands_limit:
            raise ValueError(
                self._t("session.rule.max_hands_range", max=max_hands_limit)
            )
        rules["max_hands"] = max_hands

        blind_limit = limits.get("blind_increase_every_hands", {})
        blind_every = int_value("blind_increase_every_hands", 0)
        if not int(blind_limit.get("min", 0)) <= blind_every <= int(
            blind_limit.get("max", 1000)
        ):
            raise ValueError(
                self._t(
                    "session.rule.blind_increase_range",
                    min=int(blind_limit.get("min", 0)),
                    max=int(blind_limit.get("max", 1000)),
                )
            )
        rules["blind_increase_every_hands"] = blind_every

        factor_limit = limits.get("blind_increase_factor", {})
        raw_factor = rules.get("blind_increase_factor", 2.0)
        if isinstance(raw_factor, bool):
            raise ValueError(self._t("session.rule.blind_factor"))
        try:
            blind_factor = float(raw_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError(self._t("session.rule.blind_factor")) from exc
        if not float(factor_limit.get("min", 1.0)) <= blind_factor <= float(
            factor_limit.get("max", 10.0)
        ):
            raise ValueError(
                self._t(
                    "session.rule.blind_factor_range",
                    min=f"{float(factor_limit.get('min', 1.0)):g}",
                    max=f"{float(factor_limit.get('max', 10.0)):g}",
                )
            )
        rules["blind_increase_factor"] = blind_factor

        def bool_value(key: str, default: bool) -> bool:
            value = rules.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes", "on"):
                    return True
                if lowered in ("false", "0", "no", "off"):
                    return False
            raise ValueError(self._t("session.rule.bool", key=key))

        rules["auto_rebuy"] = bool_value("auto_rebuy", False)
        rules["god_view"] = bool_value("god_view", True)
        seed = rules.get("seed")
        if seed in ("", None):
            rules["seed"] = None
        else:
            if isinstance(seed, bool) or (isinstance(seed, float) and not seed.is_integer()):
                raise ValueError(self._t("session.rule.seed"))
            try:
                rules["seed"] = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError(self._t("session.rule.seed")) from exc
        if rules["big_blind"] < rules["small_blind"]:
            raise ValueError(self._t("session.rule.blinds"))
        if rules["starting_chips"] < rules["big_blind"]:
            raise ValueError(self._t("session.rule.starting_chips"))
        return rules

    def _parse_seats(self, payload: dict) -> List[SeatConfig]:
        seats_raw = payload.get("seats")
        if not isinstance(seats_raw, list) or len(seats_raw) != 2:
            raise ValueError(self._t("session.rule.two_seats"))
        seats: List[SeatConfig] = []
        for index, raw in enumerate(seats_raw):
            if not isinstance(raw, dict):
                raise ValueError(self._t("session.rule.seat_invalid", index=index))
            kind = str(raw.get("kind") or "human").lower()
            if kind not in ("human", "llm", "bot"):
                raise ValueError(self._t("session.rule.seat_kind", index=index))
            name = str(raw.get("name") or "").strip()
            if kind == "human":
                name = name or ("你" if self.ui_locale == "zh-CN" else "You")
            elif kind == "llm":
                provider_id = str(raw.get("provider_id") or "").strip()
                model_id = str(raw.get("model_id") or "").strip()
                if not provider_id or not model_id:
                    raise ValueError(self._t("session.rule.llm_ids", index=index))
                spec = self.registry.resolve(provider_id, model_id, locale=self.ui_locale)
                provider_name = (
                    spec.provider_name
                    if self.ui_locale == "zh-CN"
                    else spec.provider_name_en
                )
                name = name or f"{provider_name} · {spec.name}"
                table_talk_style = str(
                    raw.get("table_talk_style") or ""
                ).strip().lower()
                seats.append(
                    SeatConfig(
                        kind="llm",
                        name=name,
                        provider_id=provider_id,
                        model_id=model_id,
                        params=dict(raw.get("params") or {}),
                        table_talk_style=table_talk_style,
                    )
                )
                continue
            elif kind == "bot":
                bot_raw = raw.get("bot") if isinstance(raw.get("bot"), dict) else {}
                agent_type = str(bot_raw.get("agent_type") or "random").lower()
                if agent_type not in (
                    "random",
                    "dqn",
                    "nfsp",
                    "dmc",
                    "project_dqn",
                    "project_nfsp",
                ):
                    raise ValueError(
                        self._t("session.rule.bot_type", index=index, agent_type=agent_type)
                    )
                model_path = bot_raw.get("model_path")
                if agent_type != "random" and not model_path:
                    raise ValueError(
                        self._t("session.rule.bot_model", index=index, agent_type=agent_type)
                    )
                delay_raw = bot_raw.get("delay_seconds", BOT_DELAY_DEFAULT)
                if isinstance(delay_raw, bool):
                    raise ValueError(self._t("session.rule.bot_delay_number", index=index))
                try:
                    delay_seconds = float(delay_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        self._t("session.rule.bot_delay_number", index=index)
                    ) from exc
                if not BOT_DELAY_MIN <= delay_seconds <= BOT_DELAY_MAX:
                    raise ValueError(
                        self._t(
                            "session.rule.bot_delay_range",
                            index=index,
                            min=f"{BOT_DELAY_MIN:g}",
                            max=f"{BOT_DELAY_MAX:g}",
                        )
                    )
                name = name or ("Project Bot" if agent_type.startswith("project_") else "RLCard Bot")
                seats.append(
                    SeatConfig(
                        kind="bot",
                        name=name,
                        bot_agent_type=agent_type,
                        bot_model_path=str(model_path) if model_path else None,
                        bot_delay_seconds=delay_seconds,
                    )
                )
                continue
            seats.append(SeatConfig(kind="human", name=name))
        if sum(1 for seat in seats if seat.kind == "human") >= 2:
            raise ValueError(self._t("session.rule.two_humans"))
        if seats[0].name == seats[1].name:
            seats[1].name = seats[1].name + " B"
        return seats

    @staticmethod
    def _resolve_table_talk(
        seats: List[SeatConfig],
        requested: object,
        locale: str = "zh-CN",
    ) -> bool:
        """Return the effective table-talk setting for a new match."""
        if not isinstance(requested, bool):
            raise ValueError(tr(locale, "session.rule.table_talk_bool"))
        return requested and all(seat.kind == "llm" for seat in seats)

    @staticmethod
    def _resolve_table_talk_mode(runtime: dict) -> str:
        """Return the configured table-talk delivery mode."""
        mode = str(runtime.get("table_talk_mode", "inline")).strip().lower()
        return mode if mode in ("tool", "inline") else "inline"

    # ------------------------------------------------------------------
    # Match lifecycle
    # ------------------------------------------------------------------

    def _start_epoch_snapshot(self) -> int:
        """Capture the abort epoch before a start request begins waiting.

        Reading it after acquiring ``_start_lock`` lets a request that queued
        before an abort observe the new epoch and start anyway.
        """
        with self.lock:
            return self._start_epoch

    def start_match(self, payload: dict) -> dict:
        # The epoch must be captured before waiting for the start lock. An abort
        # while this request is queued invalidates the request.
        start_epoch = self._start_epoch_snapshot()
        # Serialize match starts so two requests cannot race on the same state.
        with self._start_lock:
            seats = self._parse_seats(payload)
            rules = self._normalize_rules(
                payload.get("rules") if isinstance(payload.get("rules"), dict) else {}
            )
            table_talk_requested = payload.get(
                "table_talk",
                self._game_config.get("defaults", {}).get("table_talk", True),
            )
            table_talk_enabled = self._resolve_table_talk(
                seats, table_talk_requested, locale=self.ui_locale
            )
            talk_mode = self._resolve_table_talk_mode(
                self._game_config.get("llm_runtime", {})
            )
            effective_talk_mode = talk_mode if table_talk_enabled else "off"
            default_talk_style = "off"
            seat_styles: Dict[int, str] = {0: "", 1: ""}
            if table_talk_enabled:
                requested_talk_style = payload.get(
                    "table_talk_style",
                    self._game_config.get("defaults", {}).get(
                        "table_talk_style", DEFAULT_STYLE_ID
                    ),
                )
                default_talk_style = resolve_style(
                    requested_talk_style,
                    self.ui_locale,
                ).id
                for index, seat in enumerate(seats):
                    if seat.kind != "llm":
                        continue
                    seat_styles[index] = resolve_style(
                        seat.table_talk_style or default_talk_style,
                        self.ui_locale,
                    ).id
            requested_agent_locale = payload.get("agent_locale")
            if requested_agent_locale in (None, "", "auto"):
                requested_agent_locale = self.ui_locale
            agent_locale = (
                normalize_locale(requested_agent_locale, self.ui_locale) or self.ui_locale
            )

            # Validate LLM keys before building anything.
            for seat in seats:
                if seat.kind == "llm":
                    spec = self.registry.resolve(
                        seat.provider_id, seat.model_id, locale=self.ui_locale
                    )
                    if not ProviderRegistry.key_present(spec):
                        provider_name = (
                            spec.provider_name
                            if self.ui_locale == "zh-CN"
                            else spec.provider_name_en
                        )
                        raise ValueError(
                            self._t(
                                "session.rule.missing_key",
                                provider=provider_name,
                                env=spec.api_key_env,
                            )
                        )

            # Build the new game and all agents in local variables. A failure
            # here leaves the running match untouched.
            new_game = HeadsUpGame(
                starting_chips=rules["starting_chips"],
                small_blind=rules["small_blind"],
                big_blind=rules["big_blind"],
                auto_rebuy=rules["auto_rebuy"],
                rebuy_amount=rules["rebuy_amount"],
                seed=rules["seed"],
                max_hands=rules["max_hands"],
                blind_increase_every_hands=rules["blind_increase_every_hands"],
                blind_increase_factor=rules["blind_increase_factor"],
                locale=self.ui_locale,
            )
            new_game.configure_players(
                {"name": seats[0].name, "kind": seats[0].kind},
                {"name": seats[1].name, "kind": seats[1].kind},
            )
            base_seed = rules.get("seed")
            if base_seed is None:
                base_seed = int(time.time())
            generation_holder = {"value": 0}
            new_agents: Dict[int, object] = {}
            llm_runtime = self._game_config.get("llm_runtime", {})
            llm_tool_names = (
                AGENT_TOOL_NAMES
                if table_talk_enabled and effective_talk_mode == "tool"
                else GAME_TOOL_NAMES
            )
            llm_tools = select_tools(
                llm_tool_names,
                locale=agent_locale,
                table_talk_in_action=(effective_talk_mode == "inline"),
                table_talk_max_per_turn=self._table_talk_limits.max_per_turn,
            )
            llm_system_prompts: Dict[int, str] = {}
            for seat_index, seat in enumerate(seats):
                if seat.kind != "llm":
                    continue
                llm_system_prompts[seat_index] = build_system_prompt(
                    llm_tool_names,
                    require_estimate_each_turn=bool(
                        llm_runtime.get("require_estimate_each_turn", False)
                    ),
                    table_talk_max_chars=self._table_talk_limits.chars_for(
                        agent_locale
                    ),
                    table_talk_max_per_turn=self._table_talk_limits.max_per_turn,
                    table_talk_mode=effective_talk_mode,
                    table_talk_style=seat_styles.get(
                        seat_index, DEFAULT_STYLE_ID
                    ),
                    locale=agent_locale,
                )
            for seat_index, seat in enumerate(seats):
                if seat.kind == "llm":
                    spec = self.registry.resolve(
                        seat.provider_id, seat.model_id, locale=self.ui_locale
                    )
                    new_agents[seat_index] = LLMAgent(
                        seat=seat_index,
                        name=seat.name,
                        spec=spec,
                        params=seat.params,
                        client=self.client,
                        analyzer=self.analyzer,
                        equity_analyzer=self.equity_analyzer,
                        validate_action=self._validator_for(
                            generation_holder, agent_locale
                        ),
                        runtime=llm_runtime,
                        tools=llm_tools,
                        system_prompt=llm_system_prompts.get(seat_index),
                        submit_table_talk=(
                            self._table_talk_submitter(
                                generation_holder, seat_index, agent_locale
                            )
                            if table_talk_enabled
                            else None
                        ),
                        history_provider=self._history_provider,
                        table_talk_mode=effective_talk_mode,
                        table_talk_style=seat_styles.get(
                            seat_index, DEFAULT_STYLE_ID
                        ),
                        locale=agent_locale,
                    )
                elif seat.kind == "bot":
                    if str(seat.bot_agent_type).startswith("project_"):
                        algorithm = str(seat.bot_agent_type).split("_", 1)[1]
                        model_path = seat.bot_model_path
                        try:
                            new_agents[seat_index] = ProjectBot(
                                name=seat.name,
                                model_path=model_path or "",
                                algorithm=algorithm,
                                locale=self.ui_locale,
                            )
                        except (FileNotFoundError, ValueError, RuntimeError) as exc:
                            raise ValueError(
                                self._t(
                                    "session.rule.project_bot_failed",
                                    index=seat_index,
                                    error=exc,
                                )
                            ) from exc
                    else:
                        bot = RLCardBot(
                            name=seat.name,
                            agent_type=seat.bot_agent_type,
                            model_path=seat.bot_model_path,
                            seed=int(base_seed) + seat_index * 97,
                            locale=self.ui_locale,
                        )
                        if not bot.describe().get("available"):
                            raise ValueError(
                                self._t(
                                    "session.rule.rlcard_bot_failed",
                                    index=seat_index,
                                    error=bot.describe().get("warning"),
                                )
                            )
                        new_agents[seat_index] = bot

            new_match_id = uuid.uuid4().hex[:12]
            with self.lock:
                if start_epoch != self._start_epoch:
                    raise ValueError(self._t("session.start_cancelled"))
                self._stop_current_thread()
                generation = self._generation
                generation_holder["value"] = generation
                self._archive_current_match_locked()
                self.match_id = new_match_id
                self.god_view = rules["god_view"]
                self.max_hands = rules["max_hands"]
                self.agent_locale = agent_locale
                if hasattr(self.client, "locale"):
                    self.client.locale = agent_locale
                self.hand_history = []
                self.current_stats = HistoryStats()
                self.transcripts = {0: [], 1: []}
                self.table_talk_enabled = table_talk_enabled
                self.table_talk_mode = effective_talk_mode
                self._default_talk_style = default_talk_style
                self._seat_styles = dict(seat_styles)
                self.table_talk = []
                self._table_talk_seq = 0
                self._table_talk_cursors = {0: 0, 1: 0}
                self._table_talk_turn_counts = {}
                self._final_talk_sent = {0: False, 1: False}
                self.event_seq = 0
                self.state_seq = 0
                self.match_error = ""
                self.aborted = False
                self.started_at = time.time()
                self._last_result_hand_id = -1
                self._human_token = None
                self._human_action = None
                self._human_event.clear()
                self._next_hand_event.clear()
                self.game = new_game
                self.seat_configs = {0: seats[0], 1: seats[1]}
                self._stop_event.clear()
                self.agents = new_agents
                self._thread = threading.Thread(target=self._run, args=(generation,), daemon=True)
                self._thread.start()

        # Give the thread a moment to deal the first hand.
        for _ in range(120):
            with self.lock:
                if start_epoch != self._start_epoch:
                    break
                if self.game and self.game.hand_id > 0:
                    break
            time.sleep(0.025)
        return self.get_state()

    def _validator_for(self, generation_holder: dict, locale: str = "zh-CN"):
        def validate(action: Action) -> Optional[str]:
            return self._validate_for_generation(
                action,
                int(generation_holder["value"]),
                locale,
            )

        return validate

    def _identity_for_config(self, config: Optional[SeatConfig]) -> str:
        """Return a stable public opponent identity for history grouping."""
        if config is None:
            return ""
        if config.kind == "llm":
            return f"llm:{config.provider_id}:{config.model_id}:{config.name}"
        if config.kind == "bot":
            return (
                f"bot:{config.bot_agent_type}:"
                f"{config.bot_model_path or ''}:{config.name}"
            )
        return f"human:{config.name}"

    def _archive_current_match_locked(self) -> None:
        """Archive the completed match before starting a new one."""
        if not self.hand_history or not self.game:
            return
        identities = [
            self._identity_for_config(self.seat_configs.get(0)),
            self._identity_for_config(self.seat_configs.get(1)),
        ]
        rules = {
            "starting_chips": self.game.starting_chips,
            "small_blind": self.game.small_blind,
            "big_blind": self.game.big_blind,
            "auto_rebuy": self.game.auto_rebuy,
        }
        record = build_match_record(
            self.match_id,
            self.hand_history,
            identities,
            started_at=self.started_at,
            ended_at=time.time(),
            rules=rules,
            table_talk_styles=dict(self._seat_styles),
        )
        self.completed_matches.append(record)
        if len(self.completed_matches) > 50:
            del self.completed_matches[:-50]

    def _history_provider(
        self,
        seat: int,
        *,
        scope: str = "both",
        last_hands: int = 20,
        focus: str = "all",
        checkpoint: bool = False,
    ) -> dict:
        """Return public history statistics for one LLM seat."""
        seat = 0 if int(seat) == 0 else 1
        with self.lock:
            game = self.game
            match_info = None
            if game is not None:
                match_info = {
                    "small_blind": game.small_blind,
                    "big_blind": game.big_blind,
                    "your_chips": game.players[seat].chips,
                    "opponent_chips": game.players[1 - seat].chips,
                }
            hero_identity = self._identity_for_config(
                self.seat_configs.get(seat)
            )
            opponent_identity = self._identity_for_config(
                self.seat_configs.get(1 - seat)
            )
            if checkpoint:
                return self.current_stats.checkpoint_snapshot(
                    self.completed_matches,
                    seat,
                    hero_identity=hero_identity,
                    opponent_identity=opponent_identity,
                    last_hands=5,
                    match_info=match_info,
                )
            return self.current_stats.report(
                self.completed_matches,
                seat,
                scope=scope,
                last_hands=last_hands,
                focus=focus,
                hero_identity=hero_identity,
                opponent_identity=opponent_identity,
                match_info=match_info,
            )

    def _table_talk_counts_for_hand(self, hand_id: int) -> dict:
        """Count turn table-talk messages per seat for one hand."""
        counts = {0: 0, 1: 0}
        for item in self.table_talk:
            if item.get("hand_id") != hand_id:
                continue
            if item.get("kind") != "turn":
                continue
            seat = item.get("seat")
            if seat in (0, 1):
                counts[int(seat)] += 1
        return counts

    def _table_talk_submitter(
        self,
        generation_holder: dict,
        seat: int,
        locale: str = "zh-CN",
    ):
        def submit(text: str, turn_id: str) -> dict:
            return self._record_table_talk(
                int(generation_holder["value"]),
                seat,
                text,
                turn_id,
                locale,
            )

        return submit

    def _record_table_talk(
        self,
        generation: int,
        seat: int,
        text: str,
        turn_id: str = "",
        locale: Optional[str] = None,
    ) -> dict:
        """Validate and store one table-talk message for the acting seat."""
        talk_locale = normalize_locale(locale, self.agent_locale) or self.agent_locale
        with self.lock:
            if not self._alive(generation) or not self.game:
                return {"ok": False, "error": tr(talk_locale, "session.table_talk.reset")}
            if not self.table_talk_enabled:
                return {"ok": False, "error": tr(talk_locale, "session.table_talk.disabled")}
            if self.game.street == Street.FINISHED:
                return {"ok": False, "error": tr(talk_locale, "session.table_talk.hand_finished")}
            if self.game.actor != seat:
                return {"ok": False, "error": tr(talk_locale, "session.table_talk.not_your_turn")}

            turn_key = (seat, str(turn_id or ""))
            used_turn = self._table_talk_turn_counts.get(turn_key, 0)
            remaining_turn = max(
                0,
                self._table_talk_limits.max_per_turn - used_turn,
            )
            if remaining_turn <= 0:
                return {
                    "ok": False,
                    "error": tr(
                        talk_locale,
                        "session.table_talk.limit",
                        max=self._table_talk_limits.max_per_turn,
                    ),
                    "remaining_turn": 0,
                }

            try:
                normalized = normalize_table_talk(
                    text,
                    self._table_talk_limits.chars_for(talk_locale),
                    locale=talk_locale,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "remaining_turn": remaining_turn,
                }

            player = self.game.players[seat]
            self._table_talk_seq += 1
            entry = {
                "id": uuid.uuid4().hex[:10],
                "seq": self._table_talk_seq,
                "hand_id": self.game.hand_id,
                "street": self.game.street.value,
                "seat": seat,
                "name": player.name,
                "text": normalized,
                "turn_id": str(turn_id or ""),
                "created_at": time.time(),
                "kind": "turn",
                "style": self._seat_styles.get(seat, ""),
            }
            self.table_talk.append(entry)
            if len(self.table_talk) > 500:
                del self.table_talk[:100]
            self._table_talk_turn_counts[turn_key] = used_turn + 1
            self._emit({"type": "table_talk", **entry}, generation=generation)
            return {
                "ok": True,
                "entry": entry,
                "remaining_turn": remaining_turn - 1,
            }

    def _record_final_table_talk(
        self,
        generation: int,
        seat: int,
        text: str,
        locale: Optional[str] = None,
    ) -> dict:
        """Store one optional post-match message from a seat."""
        talk_locale = normalize_locale(locale, self.agent_locale) or self.agent_locale
        with self.lock:
            if not self._alive(generation) or not self.game:
                return {
                    "ok": False,
                    "error": tr(talk_locale, "session.table_talk.match_reset"),
                }
            if not self.table_talk_enabled:
                return {
                    "ok": False,
                    "error": tr(talk_locale, "session.table_talk.disabled"),
                }
            if not self.game.match_over:
                return {
                    "ok": False,
                    "error": tr(talk_locale, "session.table_talk.match_not_finished"),
                }
            if self._final_talk_sent.get(seat):
                return {
                    "ok": False,
                    "error": tr(talk_locale, "session.table_talk.final_sent"),
                }
            try:
                normalized = normalize_table_talk(
                    text,
                    self._table_talk_limits.chars_for(talk_locale),
                    locale=talk_locale,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}

            player = self.game.players[seat]
            self._table_talk_seq += 1
            entry = {
                "id": uuid.uuid4().hex[:10],
                "seq": self._table_talk_seq,
                "hand_id": self.game.hand_id,
                "street": "finished",
                "seat": seat,
                "name": player.name,
                "text": normalized,
                "turn_id": "",
                "created_at": time.time(),
                "kind": "final",
                "style": self._seat_styles.get(seat, ""),
            }
            self.table_talk.append(entry)
            if len(self.table_talk) > 500:
                del self.table_talk[:100]
            self._final_talk_sent[seat] = True
            self._emit({"type": "table_talk", **entry}, generation=generation)
            return {"ok": True, "entry": entry}

    def _pending_table_talk(self, seat: int) -> List[dict]:
        """Return unread messages for one seat and advance its cursor."""
        if not self.table_talk_enabled:
            return []
        cursor = self._table_talk_cursors.get(seat, 0)
        pending = [
            item
            for item in self.table_talk
            if int(item.get("seq", 0)) > cursor
            and int(item.get("seat", -1)) != seat
            and item.get("kind") != "final"
        ]
        if not pending:
            return []
        batch = pending[: self._table_talk_limits.history_limit]
        self._table_talk_cursors[seat] = int(batch[-1]["seq"])
        return [dict(item) for item in batch]

    def _recent_own_table_talk(self, seat: int) -> List[dict]:
        """Return the seat's latest turn messages for anti-repetition."""
        limit = self._table_talk_limits.self_history_limit
        if limit <= 0:
            return []
        own = [
            item
            for item in self.table_talk
            if int(item.get("seat", -1)) == seat
            and item.get("kind") == "turn"
        ]
        return [dict(item) for item in own[-limit:]]

    def _stop_current_thread(self) -> None:
        with self.lock:
            for agent in self.agents.values():
                cancel = getattr(agent, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:  # noqa: BLE001
                        pass
            self._generation += 1
            self._stop_event.set()
            self._human_event.set()
            self._next_hand_event.set()
        # Do not join: an LLM request may still be in flight. The old thread
        # checks the generation before touching the new match.

    def shutdown(self) -> None:
        self._stop_current_thread()
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    def abort_match(self) -> dict:
        """Stop the running match without waiting for an in-flight request.

        The game thread is cancelled by generation; any response that arrives
        later is discarded by the existing ``_alive`` checks. The table stays
        visible with ``match.aborted`` set so the UI can show an explicit
        stopped state and offer a new match.
        """
        with self.lock:
            self._start_epoch += 1
            has_active_match = bool(self.game and not self.game.match_over)
            if has_active_match:
                self.aborted = True
                self.match_error = ""
                self.game.match_over = True
                self._human_token = None
                self._human_event.set()
                self._next_hand_event.set()
                self._stop_current_thread()
                self._emit(
                    {
                        "type": "match_aborted",
                        "text": self._t("session.event.match_aborted"),
                    }
                )
                self._broadcast_state()
            return self.get_state()

    def next_hand(self) -> dict:
        with self.lock:
            if not self.game:
                raise ValueError(self._t("session.not_started"))
            if self.aborted:
                raise ValueError(self._t("session.aborted"))
            if self.match_error or not self._thread_alive_locked():
                raise ValueError(self._t("session.thread_stopped"))
            if self.game.street != Street.FINISHED:
                raise ValueError(self._t("session.hand_not_finished"))
            if self.game.match_over:
                raise ValueError(self._t("session.match_over_new"))
            self._next_hand_event.set()
        return self.get_state()

    def set_god_view(self, enabled: bool) -> dict:
        with self.lock:
            self.god_view = bool(enabled)
            self._broadcast_state()
        return self.get_state()

    def human_action(self, payload: dict) -> dict:
        with self.lock:
            if not self.game:
                raise ValueError(self._t("session.not_started"))
            if self.aborted:
                raise ValueError(self._t("session.aborted"))
            if self.match_error or not self._thread_alive_locked():
                raise ValueError(self._t("session.thread_stopped"))
            actor = self.game.actor
            if actor is None:
                raise ValueError(self._t("session.action.no_actor"))
            player = self.game.players[actor]
            if player.kind != "human":
                raise ValueError(self._t("session.action.not_human"))
            token = (self.game.hand_id, len(self.game.actions), actor)
            if self._human_token != token:
                raise ValueError(self._t("session.action.state_changed"))
            decision_id = self._decision_id_locked()
            if not decision_id or payload.get("decision_id") != decision_id:
                raise ValueError(self._t("session.action.state_changed"))
            action = Action.from_payload(
                actor, payload, source="human", locale=self.ui_locale
            )
            result = self.game.apply(action)
            if not result.ok:
                return {"ok": False, "error": result.error, "state": self._snapshot_locked()}
            self._human_token = None
            self._human_event.set()
            snapshot = self._emit_state_locked()
            if self.game.street == Street.FINISHED:
                self._finish_hand_locked()
            return {"ok": True, "state": snapshot}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self, generation: int) -> None:
        try:
            with self.lock:
                if not self._alive(generation):
                    return
                self._next_hand_event.clear()
                before_blinds = (self.game.small_blind, self.game.big_blind)
                self.game.start_hand()
                self._emit_blinds_up_locked(before_blinds, generation=generation)
                self._broadcast_state()
                if self.game.street == Street.FINISHED:
                    self._finish_hand_locked(generation=generation)
            while self._alive(generation):
                with self.lock:
                    if not self._alive(generation) or not self.game:
                        return
                    finished = self.game.street == Street.FINISHED
                    match_over = self.game.match_over
                    actor = self.game.actor
                    kind = self.game.players[actor].kind if actor is not None else None
                if finished:
                    if match_over:
                        self._emit(
                            {
                                "type": "match_over",
                                "match_id": self.match_id,
                                "text": self._t("session.event.match_over"),
                            },
                            generation=generation,
                        )
                        self._run_final_table_talk(generation)
                        return
                    while self._alive(generation) and not self._next_hand_event.wait(0.4):
                        pass
                    if not self._alive(generation):
                        return
                    with self.lock:
                        # A new match may have replaced this one while the
                        # thread was waiting for the next-hand event.
                        if (
                            not self._alive(generation)
                            or not self.game
                            or self.game.street != Street.FINISHED
                        ):
                            continue
                        self._next_hand_event.clear()
                        before_blinds = (self.game.small_blind, self.game.big_blind)
                        try:
                            self.game.start_hand()
                        except Exception as exc:  # noqa: BLE001
                            self.match_error = self._t(
                                "session.event.next_hand_failed", error=exc
                            )
                            self.game.match_over = True
                            self._broadcast_state()
                            return
                        self._emit_blinds_up_locked(before_blinds, generation=generation)
                        self._broadcast_state()
                        if self.game.street == Street.FINISHED:
                            self._finish_hand_locked(generation=generation)
                    continue

                if actor is None:
                    time.sleep(0.02)
                    continue
                if kind == "human":
                    self._wait_for_human(generation, actor)
                else:
                    self._run_agent_turn(generation, actor)
        except Exception as exc:  # noqa: BLE001
            self.match_error = f"{type(exc).__name__}: {exc}"
            try:
                self._emit(
                    {
                        "type": "error",
                        "match_id": self.match_id,
                        "kind": "server",
                        "text": self.match_error,
                    },
                    generation=generation,
                )
                with self.lock:
                    self._broadcast_state()
            except Exception:
                pass

    def _run_final_table_talk(self, generation: int) -> None:
        """Let every LLM seat add one optional message after the match ends."""
        if not self.table_talk_enabled:
            return
        with self.lock:
            if not self._alive(generation) or not self.game:
                return
            result = self.game.result
        for seat in (0, 1):
            agent = self.agents.get(seat)
            if not isinstance(agent, LLMAgent):
                continue
            emit = self._make_emit(generation, seat)

            def submit(text: str, seat: int = seat) -> dict:
                return self._record_final_table_talk(
                    generation, seat, text, locale=self.agent_locale
                )

            try:
                agent.final_table_talk(result, submit, emit)
            except Exception as exc:  # noqa: BLE001
                emit(
                    {
                        "type": "note",
                        "level": "warn",
                        "text": self._t(
                            "session.event.final_talk_failed",
                            error=f"{type(exc).__name__}: {exc}",
                        ),
                    }
                )
            if not self._alive(generation):
                return

    def _wait_bot_delay(self, generation: int, seconds: float) -> bool:
        """Pause before a bot acts; returns False when the match was reset."""
        deadline = time.monotonic() + seconds
        while self._alive(generation):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._stop_event.wait(min(0.2, remaining))
        return False

    def _run_agent_turn(self, generation: int, seat: int) -> None:
        with self.lock:
            if not self._alive(generation) or not self.game:
                return
            view = self._build_view(seat)
            agent = self.agents.get(seat)
            if agent is None:
                action = self._safe_action(seat)
            else:
                action = None
            config = self.seat_configs.get(seat)
            delay = config.bot_delay_seconds if config and config.kind == "bot" else 0.0
            token = (self.game.hand_id, len(self.game.actions), seat)
        if delay > 0 and not self._wait_bot_delay(generation, delay):
            return
        if agent is not None:
            emit = self._make_emit(generation, seat)
            try:
                action = agent.decide(view, emit)
            except Exception as exc:  # noqa: BLE001
                emit(
                    {
                        "type": "error",
                        "kind": "agent",
                        "text": f"{type(exc).__name__}: {exc}",
                    }
                )
                action = None

        with self.lock:
            if not self._alive(generation) or not self.game:
                return
            if token != (self.game.hand_id, len(self.game.actions), self.game.actor):
                return
            if action is None:
                action = self._safe_action(seat, source="system")
            result = self.game.apply(action)
            if not result.ok:
                self._emit(
                    {
                        "type": "error",
                        "kind": "engine",
                        "match_id": self.match_id,
                        "text": self._t(
                            "session.event.agent_action_rejected", error=result.error
                        ),
                    },
                    generation=generation,
                )
                fallback = self._safe_action(seat, source="system")
                result = self.game.apply(fallback)
                if not result.ok:
                    self.match_error = self._t(
                        "session.event.fallback_failed", error=result.error
                    )
            self._broadcast_state()
            if self.game.street == Street.FINISHED:
                self._finish_hand_locked(generation=generation)

    def _wait_for_human(self, generation: int, seat: int) -> None:
        with self.lock:
            if not self._alive(generation) or not self.game:
                return
            # The state may have moved on between the loop's actor check and this
            # call. A human action that landed first already advanced the game,
            # so waiting here would stall the match forever.
            if self.game.actor != seat:
                return
            if self.game.players[seat].kind != "human":
                return
            self._human_token = (self.game.hand_id, len(self.game.actions), seat)
            self._human_event.clear()
            snapshot = self._snapshot_locked()
        self.socketio.emit("state", snapshot)
        while self._alive(generation):
            if self._human_event.wait(0.4):
                return

    # ------------------------------------------------------------------
    # State / views
    # ------------------------------------------------------------------

    def _snapshot_locked(self) -> dict:
        if not self.game:
            return {"has_match": False, "match": None}
        reveal_all = self.god_view
        human_seats = self.game.human_seats
        snapshot = self.game.snapshot(reveal_all=reveal_all, reveal_seats=human_seats)
        if self.aborted:
            snapshot["acting_seat"] = None
            snapshot["acting_kind"] = None
            snapshot["legal_actions"] = []
            for player in snapshot.get("players", []):
                player["is_acting"] = False
        snapshot["has_match"] = True
        snapshot["table_talk"] = [
            dict(item)
            for item in self.table_talk[-self._table_talk_limits.snapshot_limit :]
        ]
        snapshot["match"] = {
            "id": self.match_id,
            "hands_played": self.game.hands_played,
            "max_hands": self.game.max_hands,
            "match_over": self.game.match_over,
            "aborted": self.aborted,
            "god_view": self.god_view,
            "table_talk": self.table_talk_enabled,
            "table_talk_mode": self.table_talk_mode,
            "table_talk_style": self._default_talk_style,
            "agent_locale": self.agent_locale,
            "can_next_hand": self.game.street == Street.FINISHED and not self.game.match_over,
            "started_at": self.started_at,
            "error": self.match_error,
            "rules": {
                "starting_chips": self.game.starting_chips,
                "small_blind": self.game.small_blind,
                "big_blind": self.game.big_blind,
                "starting_small_blind": self.game.starting_small_blind,
                "starting_big_blind": self.game.starting_big_blind,
                "blind_level": self.game.blind_level,
                "auto_rebuy": self.game.auto_rebuy,
                "rebuy_amount": self.game.rebuy_amount,
                "max_hands": self.game.max_hands,
                "blind_increase_every_hands": self.game.blind_increase_every_hands,
                "blind_increase_factor": self.game.blind_increase_factor,
                "seed": self.game.seed,
            },
            "seats": [self._seat_public(seat) for seat in (0, 1)],
            "history": [self._history_item(item) for item in self.hand_history[-12:]],
            "last_result": self.game.result,
        }
        snapshot["seq"] = self.state_seq
        snapshot["server_time"] = time.time()
        snapshot["decision_id"] = self._decision_id_locked()
        return snapshot

    def _seat_public(self, seat: int) -> dict:
        config = self.seat_configs.get(seat)
        player = self.game.players[seat] if self.game else None
        agent = self.agents.get(seat)
        style_id = self._seat_styles.get(seat, "")
        info = {
            "seat": seat,
            "name": config.name if config else (player.name if player else f"Seat {seat}"),
            "kind": config.kind if config else (player.kind if player else "human"),
            "is_human": (config.kind if config else "") == "human",
            "provider_id": config.provider_id if config else "",
            "model_id": config.model_id if config else "",
            "model_name": "",
            "table_talk_style": style_id,
            "table_talk_style_name": "",
            "table_talk_style_intensity": 0,
            "agent": {},
        }
        if style_id:
            try:
                style = resolve_style(style_id, self.ui_locale)
                info["table_talk_style_name"] = style.localized_name(self.ui_locale)
                info["table_talk_style_intensity"] = style.intensity
            except ValueError:
                pass
        if config and config.kind == "llm":
            try:
                spec = self.registry.resolve(config.provider_id, config.model_id)
                info["model_name"] = spec.name
                info["provider_name"] = spec.provider_name
                info["icon"] = spec.model_icon or spec.provider_icon
                info["accent"] = spec.provider_accent
            except ValueError:
                pass
        if agent is not None and hasattr(agent, "describe"):
            info["agent"] = agent.describe()
        return info

    def _build_view(self, seat: int) -> AgentView:
        assert self.game is not None
        player = self.game.players[seat]
        opponent = self.game.players[1 - seat]
        constraints = self.game.action_constraints(seat)
        return AgentView(
            seat=seat,
            name=player.name,
            hand_id=self.game.hand_id,
            street=self.game.street.value,
            hole_cards=[card.code for card in player.hole_cards],
            community=[card.code for card in self.game.community],
            pot=self.game.pot,
            my_chips=player.chips,
            opponent_chips=opponent.chips,
            my_street_bet=player.street_bet,
            opponent_street_bet=opponent.street_bet,
            my_total_bet=player.total_bet,
            opponent_total_bet=opponent.total_bet,
            current_bet=self.game.current_bet,
            to_call=max(0, constraints.get("to_call", 0)),
            call_amount=int(constraints.get("call_amount", 0)),
            min_raise_to=int(constraints.get("min_raise_to", self.game.big_blind)),
            max_raise_to=int(constraints.get("max_raise_to", player.chips)),
            legal_actions=[action.value for action in self.game.legal_actions(seat)],
            action_history=list(self.game.actions[-12:]),
            opponent_last_action=opponent.last_action,
            button=self.game.button_index,
            small_blind=self.game.small_blind,
            big_blind=self.game.big_blind,
            hand_number=self.game.hand_id,
            match_note=self._match_note(seat),
            history_summary=[self._llm_history_item(item, seat) for item in self.hand_history[-3:]],
            table_talk=self._pending_table_talk(seat),
            own_table_talk=self._recent_own_table_talk(seat),
        )

    def _match_note(self, seat: int) -> str:
        assert self.game is not None
        total = len(self.hand_history)
        return tr(
            self.agent_locale,
            "agent.match_note",
            hand=self.game.hand_id,
            total=total,
            small=self.game.small_blind,
            big=self.game.big_blind,
            chips=self.game.starting_chips,
            mine=self.game.players[seat].chips,
            opponent=self.game.players[1 - seat].chips,
        )

    def _history_item(self, item: dict) -> dict:
        return {
            "hand_id": item.get("hand_id"),
            "reason": item.get("reason"),
            "winner_seats": item.get("winner_seats", []),
            "pot": item.get("pot", 0),
            "board": item.get("board", []),
            "players": item.get("players", []),
        }

    def _llm_history_item(self, item: dict, seat: int) -> dict:
        players = {entry.get("seat"): entry for entry in item.get("players", [])}
        mine = players.get(seat, {})
        return {
            "hand_id": item.get("hand_id"),
            "you_won": seat in (item.get("winner_seats") or []),
            "reason": item.get("reason"),
            "pot": item.get("pot", 0),
            "board": item.get("board", []),
            "your_net": mine.get("net", 0),
            "your_chips_after": mine.get("chips_after", 0),
        }

    def _safe_action(self, seat: int, source: str = "system") -> Action:
        assert self.game is not None
        legal = set(self.game.legal_actions(seat))
        if ActionType.CHECK.value in legal:
            return Action(seat, ActionType.CHECK, source=source, auto=True)
        if ActionType.FOLD.value in legal:
            return Action(seat, ActionType.FOLD, source=source, auto=True)
        if ActionType.CALL.value in legal:
            return Action(seat, ActionType.CALL, source=source, auto=True)
        return Action(seat, ActionType.FOLD, source=source, auto=True)

    def _validate_for_generation(
        self,
        action: Action,
        generation: int,
        locale: Optional[str] = None,
    ) -> Optional[str]:
        with self.lock:
            if not self._alive(generation) or not self.game:
                return tr(
                    locale or self.agent_locale,
                    "session.table_talk.reset",
                )
            return self.game.preview(action, locale=locale or self.agent_locale)

    # ------------------------------------------------------------------
    # Events / transcripts
    # ------------------------------------------------------------------

    def _alive(self, generation: int) -> bool:
        return generation == self._generation and not self._stop_event.is_set()

    def _thread_alive_locked(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _sync_human_token_locked(self) -> None:
        """Keep the action token in step with the state that was broadcast."""
        if not self.game or self.game.actor is None:
            self._human_token = None
            return
        actor = self.game.actor
        if self.game.players[actor].kind != "human":
            self._human_token = None
            return
        if self._human_token is None:
            self._human_token = (self.game.hand_id, len(self.game.actions), actor)

    def _decision_id_locked(self) -> Optional[str]:
        """Return the id of the decision the client is allowed to answer.

        The value is derived from the state the client last received. An action
        request must echo it, so a replayed or delayed request for an earlier
        decision cannot be applied to the current one.
        """
        if not self.game or self.game.actor is None:
            return None
        actor = self.game.actor
        if self.game.players[actor].kind != "human":
            return None
        return (
            f"{self.match_id}:{self.game.hand_id}:"
            f"{len(self.game.actions)}:{actor}"
        )

    def _make_emit(self, generation: int, seat: int):
        with self.lock:
            match_id = self.match_id

        def emit(event: dict) -> None:
            payload = dict(event)
            payload["seat"] = seat
            with self.lock:
                # Check the generation and write the transcript under the same
                # lock. Reading self.match_id outside this block lets a stale
                # thread append to a newly started match.
                if not self._alive(generation) or self.match_id != match_id:
                    return
                payload["match_id"] = match_id
                self.event_seq += 1
                payload["event_id"] = self.event_seq
                bucket = self.transcripts.setdefault(seat, [])
                bucket.append(payload)
                if len(bucket) > 6000:
                    del bucket[:1000]
            self.socketio.emit("agent_event", payload)

        return emit

    def _emit(self, event: dict, generation: Optional[int] = None) -> None:
        payload = dict(event)
        with self.lock:
            if generation is not None and not self._alive(generation):
                return
            payload.setdefault("match_id", self.match_id)
            self.event_seq += 1
            payload.setdefault("event_id", self.event_seq)
            seat = payload.get("seat", 0)
            try:
                seat = int(seat)
            except (TypeError, ValueError):
                seat = 0
            bucket = self.transcripts.setdefault(seat, [])
            stored = dict(payload)
            stored["seat"] = seat
            bucket.append(stored)
            if len(bucket) > 6000:
                del bucket[:1000]
        self.socketio.emit("agent_event", payload)

    def _emit_state_locked(self) -> dict:
        """Increment the state version, snapshot and emit while holding the lock."""
        self._sync_human_token_locked()
        self.state_seq += 1
        snapshot = self._snapshot_locked()
        self.socketio.emit("state", snapshot)
        return snapshot

    def _broadcast_state(self) -> None:
        with self.lock:
            self._emit_state_locked()

    def _emit_blinds_up_locked(
        self,
        before: tuple[int, int],
        generation: Optional[int] = None,
    ) -> None:
        """Announce a blind increase after a hand starts."""
        if not self.game:
            return
        after = (self.game.small_blind, self.game.big_blind)
        if after == before:
            return
        self._emit(
            {
                "type": "note",
                "level": "info",
                "text": self._t(
                    "session.event.blinds_up", small=after[0], big=after[1]
                ),
            },
            generation=generation,
        )

    def _finish_hand_locked(self, generation: Optional[int] = None) -> None:
        if not self.game or not self.game.result:
            return
        if self._last_result_hand_id == self.game.hand_id:
            return
        self._last_result_hand_id = self.game.hand_id
        result = self.game.result
        record = build_public_hand_record(
            self.game,
            result,
            self._table_talk_counts_for_hand(self.game.hand_id),
        )
        self.hand_history.append(record)
        self.current_stats.add(record)
        for agent in self.agents.values():
            if isinstance(agent, LLMAgent):
                try:
                    agent.on_hand_finished(record)
                except Exception:
                    pass
        self._emit(
            {
                "type": "hand_result",
                "hand_id": self.game.hand_id,
                "result": result,
                "match_over": self.game.match_over,
            },
            generation=generation,
        )
        self._broadcast_state()

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        with self.lock:
            snapshot = self._snapshot_locked()
            snapshot["transcript_counts"] = {
                str(seat): len(events) for seat, events in self.transcripts.items()
            }
            return snapshot

    def get_transcripts(self) -> dict:
        with self.lock:
            return {
                "match_id": self.match_id,
                "seats": {str(seat): list(events) for seat, events in self.transcripts.items()},
            }

    def analyze(self, seat: int, iterations: int = 30000) -> dict:
        with self.lock:
            if not self.game:
                raise ValueError(self._t("session.not_started"))
            if seat not in (0, 1):
                raise ValueError(self._t("session.analyze.seat"))
            player = self.game.players[seat]
            if player.kind != "human" and not self.god_view:
                raise ValueError(self._t("session.analyze.god_view"))
            if len(player.hole_cards) != 2:
                raise ValueError(self._t("session.analyze.no_hole"))
            hole = list(player.hole_cards)
            community = list(self.game.community)
            hand_id = self.game.hand_id
            street = self.game.street.value
        result = self.analyzer.analyze(
            hole,
            community,
            hero_seat=seat,
            iterations=iterations,
            locale=self.ui_locale,
        )
        payload = result.to_dict()
        payload["hand_id"] = hand_id
        payload["street"] = street
        payload["seat"] = seat
        return payload

    def test_llm(self, provider_id: str, model_id: str, params: Optional[dict]) -> dict:
        spec = self.registry.resolve(provider_id, model_id, locale=self.ui_locale)
        if not ProviderRegistry.key_present(spec):
            raise ValueError(
                self._t("session.llm_test.missing_key", env=spec.api_key_env)
            )
        normalized = spec.normalize_params(params or {})
        test_params = dict(normalized)
        test_params["max_tokens"] = min(int(test_params.get("max_tokens", 1024)), 2048)
        messages = [
            {"role": "system", "content": self._t("session.llm_test.system")},
            {"role": "user", "content": self._t("session.llm_test.user")},
        ]
        runtime = self._game_config.get("llm_runtime", {})
        test_client = LLMClient(
            self.registry,
            connect_timeout=min(10.0, float(runtime.get("connect_timeout_seconds", 20))),
            read_timeout=min(60.0, float(runtime.get("request_timeout_seconds", 300))),
            locale=self.ui_locale,
            retry_policy=RetryPolicy(max_retries=0),
        )
        try:
            result = test_client.stream(spec, messages, None, test_params, lambda event: None)
        finally:
            test_client.close()
        return {
            "ok": True,
            "provider_id": provider_id,
            "model_id": model_id,
            "text": (result.content or result.reasoning or "")[:300],
            "latency_ms": round(result.latency_ms, 1),
            "usage": result.usage,
            "estimated_usage": result.estimated_usage,
        }


__all__ = ["GameSession"]
