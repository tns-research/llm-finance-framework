"""Tests for the raw LLM response cache."""

import os

from src.response_cache import (
    build_cache_key,
    load_cached_response,
    store_cached_response,
)


def test_cache_key_changes_when_prompt_changes():
    base_kwargs = {
        "provider": "openrouter",
        "model_name": "demo-model",
        "system_prompt": "system prompt",
        "user_prompt": "user prompt",
        "cache_context": {"provider_mode": "openrouter"},
    }
    key_one = build_cache_key(**base_kwargs)
    key_two = build_cache_key(**{**base_kwargs, "user_prompt": "different user prompt"})

    assert key_one != key_two


def test_cache_round_trip(tmp_path):
    cache_dir = tmp_path / "llm-cache"
    response = "BUY\n0.73\nCached response"

    path = store_cached_response(
        provider="openrouter",
        model_name="demo-model",
        system_prompt="system prompt",
        user_prompt="user prompt",
        response=response,
        cache_context={"provider_mode": "openrouter"},
        cache_dir=str(cache_dir),
    )

    assert os.path.exists(path)
    assert (
        load_cached_response(
            provider="openrouter",
            model_name="demo-model",
            system_prompt="system prompt",
            user_prompt="user prompt",
            cache_context={"provider_mode": "openrouter"},
            cache_dir=str(cache_dir),
        )
        == response
    )


def test_generate_response_uses_disk_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "llm-cache"
    monkeypatch.setenv("LLM_RESPONSE_CACHE_DIR", str(cache_dir))

    import src.model_router as model_router

    calls = {"count": 0}

    def fake_openrouter(model_name, system_prompt, user_prompt):
        calls["count"] += 1
        return "SELL\n0.66\nLive response"

    monkeypatch.setattr(model_router, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(model_router, "call_openrouter", fake_openrouter)

    first = model_router.generate_response("demo-model", "system prompt", "user prompt")
    second = model_router.generate_response(
        "demo-model", "system prompt", "user prompt"
    )

    assert first == second == "SELL\n0.66\nLive response"
    assert calls["count"] == 1
