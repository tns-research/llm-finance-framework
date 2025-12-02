# src/openrouter_model.py

import os
from typing import Optional

import requests

from .config_compat import OPENROUTER_API_BASE


def get_openrouter_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY in environment.\n"
            "Set it, for example:\n"
            "  export OPENROUTER_API_KEY='sk_...'\n"
            "or in PowerShell:\n"
            "  $env:OPENROUTER_API_KEY = 'sk_...'"
        )
    return api_key


def call_openrouter(model_name: str, system_prompt: str, user_prompt: str) -> str:
    """
    Call OpenRouter for a given model name and return the assistant message content.
    """

    api_key = get_openrouter_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "llm-finance-experiment",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 16384,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response format: {data}") from e

    return content
