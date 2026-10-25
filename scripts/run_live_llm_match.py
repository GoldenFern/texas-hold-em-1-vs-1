"""Run a complete live LLM vs LLM match through the real Flask backend.

The script starts the normal application, injects an experiment-only provider
registry into the session, connects a Socket.IO client, starts a match through
the REST API, and advances hands until one stack is empty.

Examples
--------
Two-hand smoke test:

    python scripts/run_live_llm_match.py --max-hands 2 --seed 1

Full match on the DXH relay:

    python scripts/run_live_llm_match.py --seed 20261025

The browser can watch the match at ``http://127.0.0.1:5051`` while the driver
runs headless.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import requests
from werkzeug.serving import make_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client import LLMClient, LLMError
from src.llm.providers import ProviderRegistry
from src.llm.retry import RetryPolicy
from src.server.app import create_app

DEFAULT_PROVIDER_CONFIG = PROJECT_ROOT / "config" / "experiments" / "dxh_providers.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "live_match"

# Streaming deltas are not needed for the match report and would make the
# event log grow without bound. Keep one record per decision, tool result and
# context transition.
RECORDED_EVENT_TYPES = {
    "turn_start",
    "turn_state",
    "message_end",
    "tool_call",
    "tool_result",
    "turn_end",
    "hand_result",
    "match_over",
    "table_talk",
    "note",
    "error",
    "context_summary_start",
    "context_summary_end",
    "context_compacted",
    "context_trimmed",
    "final_talk_start",
    "final_talk_response",
    "final_talk_end",
}


class RetryingClient:
    """Compatibility wrapper; ``LLMClient`` owns the retry budget."""

    def __init__(self, client: LLMClient, attempts: int = 1, backoff: float = 2.0) -> None:
        self.client = client
        self.attempts = max(1, int(attempts))
        self.backoff = max(0.0, float(backoff))

    @property
    def locale(self) -> str:
        return self.client.locale

    @locale.setter
    def locale(self, value: str) -> None:
        self.client.locale = value

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.stream(*args, **kwargs)

    def close(self) -> None:
        self.client.close()


class JsonlWriter:
    """Append JSON lines under a lock and flush every record."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.handle.flush()
            self.count += 1

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.close()


class UsageMeter:
    """Sum billable usage from agent events."""

    KEYS = (
        "input_tokens",
        "cached_input_tokens",
        "cache_miss_input_tokens",
        "output_tokens",
        "reasoning_tokens",
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total = {key: 0 for key in self.KEYS}

    def add(self, usage: Optional[Dict[str, Any]]) -> None:
        if not isinstance(usage, dict):
            return
        with self.lock:
            for key in self.KEYS:
                self.total[key] += int(usage.get(key, 0) or 0)

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--provider", default="dxh")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--provider-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--seat-a-name", default="Luna A")
    parser.add_argument("--seat-b-name", default="Luna B")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--max-hands", type=int, default=0)
    parser.add_argument("--blind-increase-every-hands", type=int, default=0)
    parser.add_argument("--blind-increase-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--table-talk", dest="table_talk", action="store_true", default=True)
    parser.add_argument("--no-table-talk", dest="table_talk", action="store_false")
    parser.add_argument("--agent-locale", default="zh-CN")
    parser.add_argument("--ui-locale", default="zh-CN")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--llm-retries", type=int, default=10)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--next-hand-delay", type=float, default=3.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--final-talk-wait", type=float, default=300.0)
    parser.add_argument("--budget-cny", type=float, default=20.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--price-input", type=float, default=0.0198)
    parser.add_argument("--price-cache-input", type=float, default=0.0198)
    parser.add_argument("--price-output", type=float, default=0.099)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    return parser.parse_args()


def compact_state(snapshot: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
    match = snapshot.get("match") or {}
    players = snapshot.get("players") or []
    return {
        "ts": timestamp,
        "seq": snapshot.get("seq"),
        "hand_id": snapshot.get("hand_id"),
        "street": snapshot.get("street"),
        "pot": snapshot.get("pot"),
        "button": snapshot.get("button"),
        "small_blind": snapshot.get("small_blind"),
        "big_blind": snapshot.get("big_blind"),
        "blind_level": snapshot.get("blind_level"),
        "chips": [player.get("chips") for player in players],
        "hands_played": match.get("hands_played"),
        "match_over": match.get("match_over"),
        "can_next_hand": match.get("can_next_hand"),
        "acting_seat": snapshot.get("acting_seat"),
        "actions": len(snapshot.get("actions") or []),
        "error": match.get("error") or "",
    }


def state_signature(state: Dict[str, Any]) -> tuple:
    return (
        state.get("hand_id"),
        state.get("street"),
        tuple(state.get("chips") or []),
        state.get("hands_played"),
        state.get("match_over"),
        state.get("can_next_hand"),
        state.get("actions"),
    )


def usage_cost(usage: Dict[str, int], args: argparse.Namespace) -> float:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached = int(usage.get("cached_input_tokens", 0) or 0)
    uncached = max(0, input_tokens - cached)
    output = int(usage.get("output_tokens", 0) or 0)
    return (
        uncached * args.price_input
        + cached * args.price_cache_input
        + output * args.price_output
    ) / 1_000_000.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_session(session, stop_reason: str, wall_seconds: float) -> Dict[str, Any]:
    with session.lock:
        hand_history = copy.deepcopy(session.hand_history)
        table_talk = copy.deepcopy(session.table_talk)
        agents = {
            str(seat): agent.describe()
            for seat, agent in session.agents.items()
            if hasattr(agent, "describe")
        }
        metrics = {}
        position_metrics = {}
        for seat in (0, 1):
            try:
                metrics[str(seat)] = session.current_stats.metrics(seat)
                position_metrics[str(seat)] = {
                    "button": session.current_stats.metrics(seat, "button"),
                    "bb": session.current_stats.metrics(seat, "bb"),
                }
            except Exception as exc:  # noqa: BLE001
                metrics[str(seat)] = {"error": f"{type(exc).__name__}: {exc}"}
        final_state = None
        if session.game is not None:
            try:
                final_state = session.game.snapshot(reveal_all=True, reveal_seats=[])
                final_state["match_over"] = session.game.match_over
                final_state["hands_played"] = session.game.hands_played
            except Exception as exc:  # noqa: BLE001
                final_state = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "match_id": session.match_id,
            "started_at": session.started_at,
            "finished_at": time.time(),
            "wall_seconds": wall_seconds,
            "stop_reason": stop_reason,
            "match_error": session.match_error,
            "table_talk_enabled": session.table_talk_enabled,
            "agent_locale": session.agent_locale,
            "event_seq": session.event_seq,
            "state_seq": session.state_seq,
            "hand_history": hand_history,
            "table_talk": table_talk,
            "agents": agents,
            "metrics": metrics,
            "position_metrics": position_metrics,
            "final_state": final_state,
        }


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    provider_config = args.provider_config
    if not provider_config.is_absolute():
        provider_config = (PROJECT_ROOT / provider_config).resolve()
    args.provider_config = provider_config
    if not provider_config.is_file():
        raise SystemExit(f"provider config not found: {provider_config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or DEFAULT_OUTPUT_ROOT / f"{timestamp}_seed{args.seed}"
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Allow the local browser and Python client to use the experiment port.
    origins = os.environ.get("HU_ALLOWED_ORIGINS", "").strip()
    if not origins:
        origins = ",".join(
            [
                "http://127.0.0.1:5050",
                "http://localhost:5050",
                "http://127.0.0.1:5173",
                "http://localhost:5173",
                f"http://127.0.0.1:{args.port}",
                f"http://localhost:{args.port}",
            ]
        )
        os.environ["HU_ALLOWED_ORIGINS"] = origins
    elif f":{args.port}" not in origins:
        os.environ["HU_ALLOWED_ORIGINS"] = (
            origins + f",http://127.0.0.1:{args.port},http://localhost:{args.port}"
        )

    registry = ProviderRegistry(provider_config)
    spec = registry.resolve(args.provider, args.model, locale=args.ui_locale)
    if not ProviderRegistry.key_present(spec):
        raise SystemExit(f"missing API key: {spec.api_key_env}")

    app, socketio_server, session = create_app()
    base_client = LLMClient(
        registry,
        connect_timeout=args.connect_timeout,
        read_timeout=args.timeout,
        locale=args.ui_locale,
        retry_policy=RetryPolicy(
            max_retries=args.llm_retries,
            base_delay=args.retry_backoff,
            max_delay=10.0,
        ),
    )
    client = RetryingClient(base_client, attempts=1)
    old_client = session.client
    session.registry = registry
    session.client = client
    try:
        old_client.close()
    except Exception:  # noqa: BLE001
        pass

    server = make_server(args.host, args.port, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://{args.host}:{args.port}"
    http = requests.Session()
    http.trust_env = False
    headers = {"X-HU-Locale": args.ui_locale}

    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            response = http.get(f"{base_url}/api/health", timeout=2)
            if response.ok:
                break
        except requests.RequestException:
            time.sleep(0.3)
    else:
        raise SystemExit(f"server did not become ready at {base_url}")

    events = JsonlWriter(output / "events.jsonl")
    states = JsonlWriter(output / "states.jsonl")
    meter = UsageMeter()
    seen_events: set = set()
    last_state_signature: Optional[tuple] = None
    final_talk_ends: set = set()
    event_lock = threading.Lock()

    def record_agent_event(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if str(payload.get("type") or "") not in RECORDED_EVENT_TYPES:
            return
        event_id = payload.get("event_id")
        with event_lock:
            if event_id is not None:
                if event_id in seen_events:
                    return
                seen_events.add(event_id)
        record = dict(payload)
        record["_ts"] = time.time()
        events.write(record)
        event_type = record.get("type")
        if event_type == "message_end":
            meter.add(record.get("usage"))
        elif event_type == "context_summary_end":
            meter.add(record.get("usage"))
        elif event_type == "final_talk_end":
            meter.add(record.get("usage"))
            final_talk_ends.add(int(record.get("seat", -1)))

    def record_state(payload: Any) -> None:
        nonlocal last_state_signature
        if not isinstance(payload, dict):
            return
        state = compact_state(payload, time.time())
        signature = state_signature(state)
        if signature != last_state_signature:
            last_state_signature = signature
            states.write(state)

    # Capture events at the server boundary. Socket.IO delivery can drop a
    # batch when a polling client reconnects, and the browser connection is
    # not guaranteed to exist during a long unattended run.
    original_emit = socketio_server.emit

    def emit_wrapper(event: str, *emit_args: Any, **emit_kwargs: Any) -> Any:
        payload = emit_args[0] if emit_args else None
        if event == "agent_event":
            record_agent_event(payload)
        elif event == "state":
            record_state(payload)
        return original_emit(event, *emit_args, **emit_kwargs)

    socketio_server.emit = emit_wrapper

    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": True,
        "reasoning_effort": args.reasoning_effort,
    }
    payload = {
        "seats": [
            {
                "kind": "llm",
                "name": args.seat_a_name,
                "provider_id": args.provider,
                "model_id": args.model,
                "params": dict(params),
            },
            {
                "kind": "llm",
                "name": args.seat_b_name,
                "provider_id": args.provider,
                "model_id": args.model,
                "params": dict(params),
            },
        ],
        "rules": {
            "starting_chips": args.chips,
            "small_blind": args.sb,
            "big_blind": args.bb,
            "auto_rebuy": False,
            "max_hands": args.max_hands,
            "blind_increase_every_hands": args.blind_increase_every_hands,
            "blind_increase_factor": args.blind_increase_factor,
            "seed": args.seed,
            "god_view": True,
        },
        "table_talk": bool(args.table_talk),
        "agent_locale": args.agent_locale,
    }

    print(f"output: {output}")
    print(f"watch:  {base_url}")
    print(
        "match:  "
        f"{args.provider}/{args.model}, chips={args.chips}, blinds={args.sb}/{args.bb}, "
        f"max_hands={args.max_hands or 'until bust'}, table_talk={args.table_talk}, "
        f"blind_increase={args.blind_increase_every_hands or 'off'}"
    )
    started = time.time()
    stop_reason = "running"
    last_next_hand_for: Optional[int] = None
    finished_since: Optional[float] = None
    match_over_since: Optional[float] = None
    last_progress = 0.0
    poll_failures = 0

    try:
        response = http.post(
            f"{base_url}/api/game/new",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"start match failed: {response.status_code} {response.text[:500]}")
        start_state = response.json().get("state") or {}
        print(f"match id: {start_state.get('match', {}).get('id', '')}")

        while True:
            elapsed = time.time() - started
            try:
                state_response = http.get(
                    f"{base_url}/api/game/state",
                    headers=headers,
                    timeout=10,
                )
                state_response.raise_for_status()
                state = state_response.json()
                poll_failures = 0
            except requests.RequestException as exc:
                poll_failures += 1
                print(f"state poll failed ({poll_failures}): {exc}")
                if poll_failures >= 10:
                    stop_reason = "server_unreachable"
                    break
                time.sleep(args.poll_interval)
                continue

            match = state.get("match") or {}
            hands = int(match.get("hands_played", 0) or 0)
            match_over = bool(match.get("match_over"))
            can_next = bool(match.get("can_next_hand"))
            error = str(match.get("error") or "")

            if error:
                stop_reason = f"match_error: {error}"
                break

            usage = meter.snapshot()
            cost = usage_cost(usage, args)
            if time.time() - last_progress >= 30:
                last_progress = time.time()
                chips = [player.get("chips") for player in state.get("players") or []]
                print(
                    f"[{elapsed/60:6.1f}m] hand={hands:3d} stage={state.get('street')} "
                    f"chips={chips} cost=Y{cost:.4f} "
                    f"tokens={usage['input_tokens']}/{usage['output_tokens']}"
                )

            if match_over:
                if match_over_since is None:
                    match_over_since = time.time()
                    print(f"match over after {hands} hands; waiting for final talk")
                if not args.table_talk:
                    stop_reason = "match_over"
                    break
                if {0, 1}.issubset(final_talk_ends):
                    stop_reason = "match_over"
                    break
                if time.time() - match_over_since >= args.final_talk_wait:
                    stop_reason = "match_over_final_talk_timeout"
                    break
            elif can_next:
                if finished_since is None:
                    finished_since = time.time()
                if cost >= args.budget_cny:
                    stop_reason = "budget"
                    break
                if elapsed >= args.max_hours * 3600:
                    stop_reason = "max_hours"
                    break
                if (
                    last_next_hand_for != hands
                    and time.time() - finished_since >= args.next_hand_delay
                ):
                    try:
                        next_response = http.post(
                            f"{base_url}/api/game/next-hand",
                            json={},
                            headers=headers,
                            timeout=60,
                        )
                        if next_response.ok:
                            last_next_hand_for = hands
                            finished_since = None
                        else:
                            print(
                                "next-hand failed: "
                                f"{next_response.status_code} {next_response.text[:300]}"
                            )
                    except requests.RequestException as exc:
                        print(f"next-hand request failed: {exc}")
            else:
                finished_since = None

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        wall_seconds = time.time() - started
        try:
            session.shutdown()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
        collected = collect_session(session, stop_reason, wall_seconds)
        usage = meter.snapshot()
        cost = usage_cost(usage, args)
        hands = collected.get("hand_history") or []
        final_state = collected.get("final_state") or {}
        final_players = final_state.get("players") or []
        final_chips = [player.get("chips") for player in final_players]
        match_over = bool(final_state.get("match_over"))
        complete = bool(match_over and any(int(chip or 0) <= 0 for chip in final_chips))

        hands_writer = JsonlWriter(output / "hands.jsonl")
        for record in hands:
            hands_writer.write(record)
        hands_writer.close()
        write_json(output / "agents.json", collected.get("agents") or {})
        write_json(output / "table_talk.json", collected.get("table_talk") or [])
        write_json(output / "final_state.json", final_state)
        match_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "match_id": collected.get("match_id"),
            "started_at": collected.get("started_at"),
            "finished_at": collected.get("finished_at"),
            "wall_seconds": wall_seconds,
            "stop_reason": stop_reason,
            "complete": complete,
            "match_over": match_over,
            "hands_played": len(hands),
            "final_chips": final_chips,
            "blind_level": final_state.get("blind_level"),
            "match_error": collected.get("match_error"),
            "table_talk_enabled": collected.get("table_talk_enabled"),
            "agent_locale": collected.get("agent_locale"),
            "provider": args.provider,
            "model": args.model,
            "provider_config": str(provider_config),
            "port": args.port,
            "url": base_url,
            "rules": {
                "starting_chips": args.chips,
                "small_blind": final_state.get("small_blind", args.sb),
                "big_blind": final_state.get("big_blind", args.bb),
                "starting_small_blind": args.sb,
                "starting_big_blind": args.bb,
                "blind_level": final_state.get("blind_level", 0),
                "auto_rebuy": False,
                "max_hands": args.max_hands,
                "blind_increase_every_hands": args.blind_increase_every_hands,
                "blind_increase_factor": args.blind_increase_factor,
                "seed": args.seed,
            },
            "prices": {
                "input_per_m": args.price_input,
                "cache_input_per_m": args.price_cache_input,
                "output_per_m": args.price_output,
            },
            "params": params,
            "usage": usage,
            "cost_cny": cost,
            "event_count": events.count,
            "state_count": states.count,
            "metrics": collected.get("metrics"),
            "position_metrics": collected.get("position_metrics"),
            "agents": collected.get("agents"),
        }
        write_json(output / "match.json", match_payload)
        events.close()
        states.close()
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001
            pass
        print(
            f"finished: {len(hands)} hands, complete={complete}, "
            f"stop={stop_reason}, cost=Y{cost:.4f}, wall={wall_seconds/60:.1f}m"
        )
        print(f"report: {output / 'report.md'}")
        if not args.no_report:
            report_script = PROJECT_ROOT / "scripts" / "report_live_match.py"
            subprocess.run(
                [sys.executable, str(report_script), "--run", str(output)],
                check=False,
            )
        if args.keep_open:
            print(f"browser: {base_url} (press Enter to exit)")
            try:
                input()
            except EOFError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
