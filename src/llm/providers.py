"""Provider registry and vendor-specific request shaping."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.i18n import tr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "providers.json"


@dataclass(frozen=True)
class ModelSpec:
    provider_id: str
    provider_name: str
    provider_name_en: str
    provider_icon: str
    model_icon: str
    provider_accent: str
    base_url: str
    api_key_env: str
    docs_url: str
    request: Dict[str, Any]
    id: str
    name: str
    tagline: str
    tagline_en: str
    context_window: int
    max_output: int
    defaults: Dict[str, Any]
    controls: Dict[str, Any]

    def default_params(self) -> dict:
        params = dict(self.defaults)
        params.setdefault("temperature", 0.8)
        params.setdefault("top_p", 0.95)
        params.setdefault("max_tokens", 32768)
        params.setdefault("thinking", True)
        params.setdefault("reasoning_effort", "")
        return params

    def normalize_params(self, raw: Optional[dict]) -> dict:
        """Clamp user parameters to values this model accepts."""
        params = self.default_params()
        raw = raw or {}
        controls = self.controls or {}

        def numeric(key: str, value: Any) -> None:
            control = controls.get(key, {})
            if not control.get("show", False):
                return
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            lo = float(control.get("min", 0))
            hi = float(control.get("max", number))
            number = max(lo, min(hi, number))
            params[key] = int(number) if key == "max_tokens" else number

        if "temperature" in raw:
            numeric("temperature", raw["temperature"])
        if "top_p" in raw:
            numeric("top_p", raw["top_p"])
        if "max_tokens" in raw:
            numeric("max_tokens", raw["max_tokens"])

        thinking_control = controls.get("thinking", {})
        if thinking_control.get("mode") == "toggle" and "thinking" in raw:
            params["thinking"] = bool(raw["thinking"])
        elif thinking_control.get("mode") == "always":
            params["thinking"] = True
        elif thinking_control.get("mode") == "off":
            params["thinking"] = False

        effort_control = controls.get("reasoning_effort", {})
        options = list(effort_control.get("options") or [])
        if options:
            effort = str(raw.get("reasoning_effort") or params.get("reasoning_effort") or options[0])
            if effort not in options:
                effort = options[0]
            params["reasoning_effort"] = effort
        else:
            params["reasoning_effort"] = ""

        # The provider counts prompt + completion against the same window.
        # Keep a small prompt reserve even when the model advertises a very
        # large max output (Doubao Seed 2.1 lists 256K output on a 256K window).
        if self.context_window > 0:
            safe_max = max(1024, int(self.context_window) - 8192)
            params["max_tokens"] = min(
                int(params.get("max_tokens", 32768)),
                safe_max,
            )

        # Some models require a slightly higher temperature in thinking mode.
        if self.provider_id == "qwen" and params.get("thinking"):
            params["temperature"] = max(float(params.get("temperature", 0.7)), 0.6)
        if self.provider_id == "longcat" and params.get("thinking"):
            params["temperature"] = 1.0
        return params

    def public_dict(self, key_present: bool) -> dict:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "provider_name_en": self.provider_name_en,
            "icon": self.model_icon or self.provider_icon,
            "accent": self.provider_accent,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "key_present": key_present,
            "docs_url": self.docs_url,
            "id": self.id,
            "name": self.name,
            "tagline": self.tagline,
            "tagline_en": self.tagline_en,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "defaults": self.default_params(),
            "controls": self.controls,
        }


class ProviderRegistry:
    """Loads provider metadata and builds vendor-specific chat payloads."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self._models: Dict[str, Dict[str, ModelSpec]] = {}
        self._providers: List[dict] = []
        self._load()

    def _load(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self._providers = []
        self._models = {}
        for provider in config.get("providers", []):
            public_provider = {
                "id": provider["id"],
                "name": provider["name"],
                "name_en": provider.get("name_en", provider["name"]),
                "icon": provider.get("icon", ""),
                "accent": provider.get("accent", "#888888"),
                "base_url": provider.get("base_url", ""),
                "api_key_env": provider.get("api_key_env", ""),
                "docs_url": provider.get("docs_url", ""),
                "models": [],
            }
            model_map: Dict[str, ModelSpec] = {}
            for model in provider.get("models", []):
                spec = ModelSpec(
                    provider_id=provider["id"],
                    provider_name=provider["name"],
                    provider_name_en=provider.get("name_en", provider["name"]),
                    provider_icon=provider.get("icon", ""),
                    model_icon=model.get("icon", ""),
                    provider_accent=provider.get("accent", "#888888"),
                    base_url=provider.get("base_url", ""),
                    api_key_env=provider.get("api_key_env", ""),
                    docs_url=provider.get("docs_url", ""),
                    request=dict(provider.get("request", {})),
                    id=model["id"],
                    name=model.get("name", model["id"]),
                    tagline=model.get("tagline", ""),
                    tagline_en=model.get("tagline_en", model.get("tagline", "")),
                    context_window=int(model.get("context_window", 128000)),
                    max_output=int(model.get("max_output", 8192)),
                    defaults=dict(model.get("defaults", {})),
                    controls=dict(model.get("controls", {})),
                )
                model_map[spec.id] = spec
                public_provider["models"].append(
                    spec.public_dict(self.key_present(spec))
                )
            self._models[provider["id"]] = model_map
            self._providers.append(public_provider)

    @staticmethod
    def key_present(spec: ModelSpec) -> bool:
        return bool(os.environ.get(spec.api_key_env, "").strip())

    @staticmethod
    def get_api_key(spec: ModelSpec, locale: str = "zh-CN") -> str:
        key = os.environ.get(spec.api_key_env, "").strip()
        if not key:
            raise ValueError(tr(locale, "providers.missing_key", env=spec.api_key_env))
        return key

    def list_providers(self) -> List[dict]:
        providers = []
        for provider in self._providers:
            models = []
            for model in provider["models"]:
                # Refresh key presence on each request.
                spec = self._models[provider["id"]][model["id"]]
                refreshed = dict(model)
                refreshed["key_present"] = self.key_present(spec)
                models.append(refreshed)
            providers.append({**provider, "models": models})
        return providers

    def resolve(
        self,
        provider_id: str,
        model_id: str,
        locale: str = "zh-CN",
    ) -> ModelSpec:
        if provider_id not in self._models:
            raise ValueError(tr(locale, "providers.unknown_provider", provider=provider_id))
        models = self._models[provider_id]
        if model_id not in models:
            raise ValueError(
                tr(locale, "providers.unknown_model", provider=provider_id, model=model_id)
            )
        return models[model_id]

    def build_payload(
        self,
        spec: ModelSpec,
        messages: List[dict],
        tools: Optional[List[dict]],
        params: dict,
        stream: bool,
    ) -> dict:
        payload: Dict[str, Any] = {
            "model": spec.id,
            "messages": messages,
            "stream": stream,
            "max_tokens": int(params.get("max_tokens", 32768)),
        }
        # MiniMax M3.1 Flash ignores sampling parameters entirely.
        if spec.provider_id != "minimax":
            payload["temperature"] = float(params.get("temperature", 0.8))
            payload["top_p"] = float(params.get("top_p", 0.95))

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        thinking = bool(params.get("thinking", True))
        effort = str(params.get("reasoning_effort") or "")
        style = spec.request.get("thinking_style", "none")
        if style == "thinking":
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
        elif style == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
        elif style == "enable_thinking":
            payload["enable_thinking"] = thinking
            extra = spec.request.get("extra_body")
            if isinstance(extra, dict):
                payload.update(extra)

        effort_style = spec.request.get("reasoning_effort_style", "none")
        if effort and thinking and effort_style != "none":
            payload["reasoning_effort"] = effort

        if spec.request.get("tool_stream"):
            payload["tool_stream"] = True
        if spec.request.get("stream_usage") and stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def endpoint(spec: ModelSpec) -> str:
        return spec.base_url.rstrip("/") + "/chat/completions"


__all__ = ["ModelSpec", "ProviderRegistry", "PROJECT_ROOT"]
