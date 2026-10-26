"""Flask application factory and HTTP/Socket.IO routes."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import secrets
import time
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, send_from_directory
from flask_socketio import SocketIO

from src.analysis.equity import EquityAnalyzer
from src.analysis.hand_distribution import HandDistributionAnalyzer
from src.i18n import DEFAULT_LOCALE, normalize_locale, tr
from src.llm.client import LLMClient, LLMError
from src.llm.providers import PROJECT_ROOT, ProviderRegistry
from src.llm.retry import RetryPolicy
from src.server.session import GameSession

APP_VERSION = "0.1.0"
STATIC_DIR = PROJECT_ROOT / "static"
DIST_DIR = STATIC_DIR / "dist"
GAME_CONFIG_PATH = PROJECT_ROOT / "config" / "game.json"
MODELS_DIR = PROJECT_ROOT / "models" / "rlcard"

logger = logging.getLogger("hu.server")


def _allowed_origins() -> list[str]:
    raw = os.environ.get("HU_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "http://127.0.0.1:5050",
        "http://localhost:5050",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://[::1]:5050",
        "http://[::1]:5173",
    ]


def _request_locale() -> str:
    raw = (
        request.headers.get("X-HU-Locale")
        or request.args.get("locale")
        or request.args.get("lang")
    )
    if not raw and request.path.startswith("/api/"):
        raw = request.headers.get("Accept-Language")
    return normalize_locale(raw, DEFAULT_LOCALE) or DEFAULT_LOCALE


def _json_object() -> dict:
    """Return a JSON object body, rejecting arrays and malformed JSON."""
    payload = request.get_json(silent=True)
    if payload is None:
        if request.get_data(cache=True):
            raise ValueError(tr(g.locale, "api.invalid_json"))
        return {}
    if not isinstance(payload, dict):
        raise ValueError(tr(g.locale, "api.json_object"))
    return payload


def _bool_field(payload: dict, key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    raise ValueError(tr(g.locale, "api.bool_field", key=key))


def create_app() -> tuple[Flask, SocketIO, GameSession]:
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = os.environ.get("HU_SECRET_KEY") or secrets.token_hex(16)
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    socketio = SocketIO(
        app,
        async_mode="threading",
        cors_allowed_origins=_allowed_origins(),
        logger=False,
        engineio_logger=False,
    )

    registry = ProviderRegistry()
    with GAME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        game_config = json.load(handle)
    runtime = game_config.get("llm_runtime", {})
    client = LLMClient(
        registry,
        connect_timeout=float(
            os.environ.get("HU_CONNECT_TIMEOUT", runtime.get("connect_timeout_seconds", 20))
        ),
        read_timeout=float(
            os.environ.get("HU_READ_TIMEOUT", runtime.get("request_timeout_seconds", 300))
        ),
        retry_policy=RetryPolicy.from_runtime(runtime),
    )
    analyzer = HandDistributionAnalyzer()
    equity_analyzer = EquityAnalyzer()
    session = GameSession(
        socketio,
        registry,
        client,
        analyzer,
        equity_analyzer,
        ui_locale=DEFAULT_LOCALE,
    )

    @app.before_request
    def apply_request_locale():
        g.locale = _request_locale()
        # Static assets and the shell must not flip the session language based
        # on the browser's Accept-Language. Only API calls carry the explicit
        # UI preference.
        if request.path.startswith("/api/"):
            session.set_ui_locale(g.locale)

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.get("/")
    def index():
        index_path = DIST_DIR / "index.html"
        if index_path.exists():
            return send_from_directory(str(DIST_DIR), "index.html")
        return (
            "<html><head><meta charset='utf-8'><title>Heads-Up Lab</title></head>"
            "<body style='background:#0b1512;color:#e8e3d7;font-family:system-ui;padding:48px'>"
            f"<h1>Heads-Up Lab</h1><p>{tr(g.locale, 'api.frontend_missing')}</p>"
            "<pre style='background:#13211d;padding:16px;border-radius:12px'>"
            "cd frontend\nnpm install\nnpm run build\n</pre>"
            f"<p>{tr(g.locale, 'api.frontend_after_build')}</p></body></html>"
        )

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(
            {
                "app": {
                    "name": "Heads-Up Lab",
                    "version": APP_VERSION,
                    "description": tr(g.locale, "bootstrap.description"),
                },
                "providers": registry.list_providers(),
                "game": game_config,
                "rlcard": _rlcard_info(),
                "env": {
                    "god_view_default": bool(game_config.get("defaults", {}).get("god_view", True)),
                },
                "server_time": time.time(),
            }
        )

    # ------------------------------------------------------------------
    # Game
    # ------------------------------------------------------------------

    @app.get("/api/game/state")
    def game_state():
        return jsonify(session.get_state())

    @app.get("/api/game/transcripts")
    def game_transcripts():
        return jsonify(session.get_transcripts())

    @app.post("/api/game/new")
    def game_new():
        try:
            payload = _json_object()
            state = session.start_match(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception:  # noqa: BLE001
            logger.exception("start_match failed")
            return jsonify({"ok": False, "error": tr(g.locale, "api.internal_error")}), 500
        return jsonify({"ok": True, "state": state})

    @app.post("/api/game/action")
    def game_action():
        try:
            payload = _json_object()
            result = session.human_action(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if not result.get("ok"):
            return jsonify(result), 400
        return jsonify(result)

    @app.post("/api/game/abort")
    def game_abort():
        try:
            state = session.abort_match()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "state": state})

    @app.post("/api/game/next-hand")
    def game_next_hand():
        try:
            state = session.next_hand()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "state": state})

    @app.post("/api/game/god-view")
    def game_god_view():
        try:
            payload = _json_object()
            enabled = _bool_field(payload, "enabled", True)
            state = session.set_god_view(enabled)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "state": state})

    @app.post("/api/game/analyze")
    def game_analyze():
        try:
            payload = _json_object()
            seat = int(payload.get("seat", 0))
            iterations = int(payload.get("iterations", 30000))
            result = session.analyze(seat, iterations)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "analysis": result})

    # ------------------------------------------------------------------
    # LLM utilities
    # ------------------------------------------------------------------

    @app.post("/api/llm/test")
    def llm_test():
        try:
            payload = _json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        provider_id = str(payload.get("provider_id") or "")
        model_id = str(payload.get("model_id") or "")
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        try:
            result = session.test_llm(provider_id, model_id, params)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except LLMError as exc:
            return jsonify({"ok": False, "error": exc.detail, "kind": exc.kind}), 400
        except Exception:  # noqa: BLE001
            logger.exception("llm test failed")
            return jsonify({"ok": False, "error": tr(g.locale, "api.internal_error")}), 500
        return jsonify(result)

    @app.post("/api/preferences")
    def preferences():
        try:
            payload = _json_object()
            locale = normalize_locale(payload.get("locale"), g.locale) or g.locale
            session.set_ui_locale(locale)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "locale": locale})

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "version": APP_VERSION})

    # ------------------------------------------------------------------
    # Socket.IO
    # ------------------------------------------------------------------

    @socketio.on("connect")
    def on_connect():
        raw = (
            request.args.get("locale")
            or request.headers.get("X-HU-Locale")
            or request.headers.get("Accept-Language")
        )
        session.set_ui_locale(normalize_locale(raw, session.ui_locale) or session.ui_locale)
        socketio.emit("state", session.get_state(), to=request.sid)

    @socketio.on("request_state")
    def on_request_state():
        socketio.emit("state", session.get_state(), to=request.sid)

    @socketio.on("request_transcripts")
    def on_request_transcripts():
        socketio.emit("transcripts", session.get_transcripts(), to=request.sid)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/") or request.path.startswith("/static/"):
            return jsonify(
                {"ok": False, "error": tr(getattr(g, "locale", DEFAULT_LOCALE), "api.not_found")}
            ), 404
        return index()

    @app.errorhandler(500)
    def server_error(error):
        return jsonify(
            {"ok": False, "error": tr(getattr(g, "locale", DEFAULT_LOCALE), "api.internal_error")}
        ), 500

    logger.info("Heads-Up Lab ready. Static dist: %s", DIST_DIR)
    return app, socketio, session


def _rlcard_info() -> dict:
    available = importlib.util.find_spec("rlcard") is not None
    models = []
    if MODELS_DIR.exists():
        for path in sorted(MODELS_DIR.glob("*")):
            if path.suffix.lower() in (".pth", ".tar", ".pt"):
                models.append(
                    {
                        "name": path.name,
                        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "size_kb": round(path.stat().st_size / 1024, 1),
                    }
                )
    return {
        "available": available,
        "agents": [
            {
                "id": "random",
                "name": "RandomAgent",
                "description": "RLCard 官方随机策略，无需模型文件，适合验证链路。",
                "description_en": "Official RLCard random policy; no model file needed, useful for verifying the pipeline.",
                "needs_model": False,
            },
            {
                "id": "dqn",
                "name": "DQN",
                "description": "加载 RLCard DQN checkpoint。",
                "description_en": "Load a RLCard DQN checkpoint.",
                "needs_model": True,
            },
            {
                "id": "nfsp",
                "name": "NFSP",
                "description": "加载 RLCard NFSP checkpoint。",
                "description_en": "Load a RLCard NFSP checkpoint.",
                "needs_model": True,
            },
            {
                "id": "dmc",
                "name": "DMC",
                "description": "加载 RLCard Deep Monte-Carlo checkpoint。",
                "description_en": "Load a RLCard deep Monte-Carlo checkpoint.",
                "needs_model": True,
            },
            {
                "id": "project_dqn",
                "name": "Project DQN",
                "description": "本项目规则训练的 73 维 / 10 动作 DQN。",
                "description_en": "73-dim / 10-action DQN trained on this project's rules.",
                "needs_model": True,
            },
            {
                "id": "project_nfsp",
                "name": "Project NFSP",
                "description": "本项目规则训练的 73 维 / 10 动作 NFSP。",
                "description_en": "73-dim / 10-action NFSP trained on this project's rules.",
                "needs_model": True,
            },
        ],
        "models": models,
        "model_dir": str(MODELS_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


__all__ = ["create_app"]
