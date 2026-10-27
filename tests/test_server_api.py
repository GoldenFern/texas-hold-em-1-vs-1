"""REST robustness tests using Flask's test client."""

from __future__ import annotations

from src.server.app import create_app


def make_client():
    app, _socketio, session = create_app()
    return app.test_client(), session


def test_non_object_json_is_rejected_with_400() -> None:
    client, session = make_client()
    try:
        for path in (
            "/api/game/new",
            "/api/game/action",
            "/api/game/god-view",
            "/api/game/analyze",
            "/api/llm/test",
        ):
            response = client.post(path, json=[1, 2, 3])
            assert response.status_code == 400, path
            assert response.get_json()["ok"] is False
    finally:
        session.shutdown()


def test_malformed_json_is_rejected_without_internal_text() -> None:
    client, session = make_client()
    try:
        response = client.post(
            "/api/game/new",
            data="{",
            content_type="application/json",
        )
        assert response.status_code == 400
        body = response.get_json()
        assert "AttributeError" not in body["error"]
        assert "Traceback" not in body["error"]
    finally:
        session.shutdown()


def test_god_view_accepts_only_explicit_booleans() -> None:
    client, session = make_client()
    try:
        response = client.post("/api/game/god-view", json={"enabled": "false"})
        assert response.status_code == 200
        assert session.god_view is False

        response = client.post("/api/game/god-view", json={"enabled": "maybe"})
        assert response.status_code == 400
    finally:
        session.shutdown()


def test_table_talk_requires_boolean_and_two_llm_seats() -> None:
    client, session = make_client()
    try:
        payload = {
            "seats": [
                {"kind": "human", "name": "You"},
                {"kind": "bot", "name": "Bot", "bot": {"agent_type": "random"}},
            ],
            "rules": {"starting_chips": 1000},
            "table_talk": "yes",
        }
        response = client.post("/api/game/new", json=payload)
        assert response.status_code == 400
        assert "table_talk" in response.get_json()["error"]

        payload["table_talk"] = True
        payload["table_talk_style"] = "not-a-style"
        response = client.post("/api/game/new", json=payload)
        assert response.status_code == 200
        assert session.table_talk_enabled is False
        state = response.get_json()["state"]
        assert state["match"]["table_talk"] is False
        assert state["match"]["table_talk_mode"] == "off"
        assert state["match"]["table_talk_style"] == "off"
    finally:
        session.shutdown()


def test_abort_endpoint_stops_match() -> None:
    client, session = make_client()
    try:
        payload = {
            "seats": [
                {"kind": "human", "name": "You"},
                {"kind": "bot", "name": "Bot", "bot": {"agent_type": "random"}},
            ],
            "rules": {"starting_chips": 1000},
        }
        response = client.post("/api/game/new", json=payload)
        assert response.status_code == 200
        response = client.post("/api/game/abort")
        assert response.status_code == 200
        state = response.get_json()["state"]
        assert state["match"]["aborted"] is True
        assert state["match"]["match_over"] is True
    finally:
        session.shutdown()


def test_bootstrap_includes_english_provider_names() -> None:
    client, session = make_client()
    try:
        response = client.get("/api/bootstrap", headers={"X-HU-Locale": "en-US"})
        assert response.status_code == 200
        payload = response.get_json()
        providers = {item["id"]: item for item in payload["providers"]}
        assert providers["doubao"]["name_en"] == "Volcengine"
        assert providers["qwen"]["models"][0]["provider_name_en"] == "Qwen"
        styles = {item["id"]: item for item in payload["table_talk_styles"]}
        assert styles["gentle"]["name_en"] == "The Gentleman"
        assert styles["gentle"]["default"] is True
        assert styles["tieba_l3"]["name_en"] == "Forum Trash Talker (Hard)"
        assert styles["tieba_l3"]["intensity"] == 3
    finally:
        session.shutdown()


def test_api_errors_follow_locale_header() -> None:
    client, session = make_client()
    try:
        response = client.post(
            "/api/game/god-view",
            json={"enabled": "maybe"},
            headers={"X-HU-Locale": "en-US"},
        )
        assert response.status_code == 400
        assert "must be a boolean" in response.get_json()["error"]

        response = client.post(
            "/api/game/god-view",
            json={"enabled": "maybe"},
            headers={"X-HU-Locale": "zh-CN"},
        )
        assert response.status_code == 400
        assert "必须是布尔值" in response.get_json()["error"]
    finally:
        session.shutdown()


def test_preferences_endpoint_updates_session_locale() -> None:
    client, session = make_client()
    try:
        response = client.post(
            "/api/preferences",
            json={"locale": "zh-CN"},
            headers={"X-HU-Locale": "en-US"},
        )
        assert response.status_code == 200
        assert response.get_json()["locale"] == "zh-CN"
        assert session.ui_locale == "zh-CN"
    finally:
        session.shutdown()


def test_invalid_rules_return_400() -> None:
    client, session = make_client()
    try:
        payload = {
            "seats": [
                {"kind": "human", "name": "You"},
                {"kind": "bot", "name": "Bot", "bot": {"agent_type": "random"}},
            ],
            "rules": {"starting_chips": 50},
        }
        response = client.post("/api/game/new", json=payload)
        assert response.status_code == 400
        assert "starting_chips" in response.get_json()["error"]
    finally:
        session.shutdown()
