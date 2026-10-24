"""Official context window and output cap regression tests for provider specs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from src.llm.providers import PROJECT_ROOT, ProviderRegistry

EXPECTED_LIMITS = {
    ("deepseek", "deepseek-flash"): (1048576, 393216),
    ("doubao", "doubao-seed-2-1-turbo-260628"): (256000, 256000),
    ("minimax", "MiniMax-M3.1-Flash"): (1000000, 131072),
    ("qwen", "qwen3.8-flash"): (1000000, 131072),
    ("zhipu", "glm-5.3-flash"): (1048576, 131072),
    ("kimi", "kimi-k2.7-code"): (262144, 32768),
    ("longcat", "LongCat-2.5-Preview"): (1048576, 131072),
    ("mimo", "mimo-v2.6-flash"): (1000000, 131072),
    # Official hard window is 1,050,000, but prompts above 272K input are
    # billed at 2x input / 1.5x output, so the project budgets 272K.
    ("4router", "gpt-6-luna"): (272000, 128000),
}

DXH_CONFIG = PROJECT_ROOT / "config" / "experiments" / "dxh_providers.json"


@pytest.fixture(scope="module")
def registry() -> ProviderRegistry:
    return ProviderRegistry()


def _iter_models(registry: ProviderRegistry) -> Iterator[tuple[str, dict[str, Any]]]:
    for provider in registry.list_providers():
        for model in provider["models"]:
            yield provider["id"], model


@pytest.mark.parametrize(
    ("provider_id", "model_id"),
    sorted(EXPECTED_LIMITS),
)
def test_context_metadata_matches_official_docs(
    registry: ProviderRegistry, provider_id: str, model_id: str
) -> None:
    expected_context, expected_output = EXPECTED_LIMITS[(provider_id, model_id)]
    spec = registry.resolve(provider_id, model_id)

    assert spec.context_window == expected_context
    assert spec.max_output == expected_output


def test_pricing_sensitive_models_use_expected_limits(
    registry: ProviderRegistry,
) -> None:
    gpt = registry.resolve("4router", "gpt-6-luna")
    kimi = registry.resolve("kimi", "kimi-k2.7-code")
    deepseek = registry.resolve("deepseek", "deepseek-flash")
    doubao = registry.resolve("doubao", "doubao-seed-2-1-turbo-260628")

    assert (gpt.context_window, gpt.max_output) == (272000, 128000)
    assert (kimi.context_window, kimi.max_output) == (262144, 32768)
    assert (deepseek.context_window, deepseek.max_output) == (1048576, 393216)
    assert (doubao.context_window, doubao.max_output) == (256000, 256000)


def test_all_models_have_sane_limits(registry: ProviderRegistry) -> None:
    for provider_id, model in _iter_models(registry):
        label = f"{provider_id}/{model['id']}"
        assert model["context_window"] > 0, label
        assert model["max_output"] > 0, label
        assert model["max_output"] <= model["context_window"], label


def test_max_tokens_control_matches_max_output(registry: ProviderRegistry) -> None:
    for provider_id, model in _iter_models(registry):
        max_tokens = (model.get("controls") or {}).get("max_tokens") or {}
        if "max" not in max_tokens:
            continue
        assert max_tokens["max"] == model["max_output"], f"{provider_id}/{model['id']}"


def test_max_tokens_never_consumes_the_whole_context_window(
    registry: ProviderRegistry,
) -> None:
    doubao = registry.resolve("doubao", "doubao-seed-2-1-turbo-260628")
    params = doubao.normalize_params({"max_tokens": 256000})
    assert params["max_tokens"] == doubao.context_window - 8192

    gpt = registry.resolve("4router", "gpt-6-luna")
    params = gpt.normalize_params({"max_tokens": 128000})
    assert params["max_tokens"] == 128000


def test_dxh_experiment_config_uses_same_budget() -> None:
    registry = ProviderRegistry(DXH_CONFIG)
    spec = registry.resolve("dxh", "gpt-6-luna")

    assert (spec.context_window, spec.max_output) == (272000, 128000)
    max_tokens = spec.controls.get("max_tokens") or {}
    assert max_tokens.get("max") == spec.max_output
