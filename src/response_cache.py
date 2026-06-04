"""Disk cache for raw LLM responses.

The cache is keyed by provider, model, prompts, and provider-specific context so
repeat runs can reuse the exact same raw response without requerying the model.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_SCHEMA = "llm-response-cache/v1"
CACHE_ENV_VAR = "LLM_RESPONSE_CACHE_DIR"


def _stable_json(payload: dict) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_root(cache_dir: str | None = None) -> str:
    if cache_dir:
        return cache_dir

    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return override

    return os.path.join(PROJECT_ROOT, "results", "llm_cache")


def build_cache_key(
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    cache_context: dict | None = None,
) -> str:
    payload = {
        "schema": CACHE_SCHEMA,
        "provider": provider,
        "model_name": model_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "cache_context": cache_context or {},
    }
    return _sha256(_stable_json(payload))


def cache_path(
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    cache_context: dict | None = None,
    cache_dir: str | None = None,
) -> str:
    key = build_cache_key(
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache_context=cache_context,
    )
    return os.path.join(_cache_root(cache_dir), provider, f"{key}.json")


def load_cached_response(
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    cache_context: dict | None = None,
    cache_dir: str | None = None,
) -> str | None:
    path = cache_path(
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache_context=cache_context,
        cache_dir=cache_dir,
    )
    if not os.path.exists(path):
        return None

    try:
        with open(path) as f:
            entry = json.load(f)
    except Exception:
        return None

    expected_key = build_cache_key(
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache_context=cache_context,
    )
    if entry.get("cache_key") != expected_key:
        return None

    response = entry.get("response")
    if not isinstance(response, str) or not response.strip():
        return None

    return response


def store_cached_response(
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    cache_context: dict | None = None,
    cache_dir: str | None = None,
) -> str:
    path = cache_path(
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cache_context=cache_context,
        cache_dir=cache_dir,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {
        "schema": CACHE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_key": build_cache_key(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cache_context=cache_context,
        ),
        "provider": provider,
        "model_name": model_name,
        "system_prompt_sha256": _sha256(system_prompt),
        "user_prompt_sha256": _sha256(user_prompt),
        "cache_context": cache_context or {},
        "response": response,
    }

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)
    return path
