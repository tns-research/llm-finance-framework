# src/model_router.py
"""
Single dispatch point for LLM calls, so every call site (trading decisions and
report narratives) routes through one place based on LLM_PROVIDER. The router
also persists raw responses to disk so identical reruns do not requery the
provider.

Providers:
- "openrouter"            -> OpenRouter HTTP API (needs OPENROUTER_API_KEY).
- "claude_code"           -> Claude Code subscription CLI, single-shot.
- "claude_code_subagents" -> Claude Code subscription CLI, multi-agent (analysts).

The "dummy" provider is handled separately by each call site (it does not need a
router model), so it never reaches here.
"""

from .claude_code_model import call_claude_code
from .config import ANALYST_AGENTS, CLAUDE_CODE_MODEL, LLM_PROVIDER
from .openrouter_model import call_openrouter
from .response_cache import load_cached_response, store_cached_response


def generate_response(router_model: str, system_prompt: str, user_prompt: str) -> str:
    """Route a (system, user) prompt to the active provider and return text."""
    effective_model = router_model or CLAUDE_CODE_MODEL
    cache_context = {"provider_mode": LLM_PROVIDER}

    if LLM_PROVIDER == "openrouter":
        cache_context["sampling"] = {"temperature": 0.0, "max_tokens": 50000}
    elif LLM_PROVIDER == "claude_code_subagents":
        cache_context["analyst_agents"] = ANALYST_AGENTS

    cached_response = load_cached_response(
        LLM_PROVIDER,
        effective_model,
        system_prompt,
        user_prompt,
        cache_context=cache_context,
    )
    if cached_response is not None:
        return cached_response

    if LLM_PROVIDER == "openrouter":
        response = call_openrouter(effective_model, system_prompt, user_prompt)
    elif LLM_PROVIDER == "claude_code":
        response = call_claude_code(system_prompt, user_prompt, model=effective_model)
    elif LLM_PROVIDER == "claude_code_subagents":
        response = call_claude_code(
            system_prompt,
            user_prompt,
            model=effective_model,
            agents=ANALYST_AGENTS,
        )
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Expected one of: 'dummy', 'openrouter', 'claude_code', "
            "'claude_code_subagents'."
        )

    store_cached_response(
        LLM_PROVIDER,
        effective_model,
        system_prompt,
        user_prompt,
        response,
        cache_context=cache_context,
    )
    return response
