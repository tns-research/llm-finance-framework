# Model Providers

The framework treats a model as a simple contract:

```
f(system_prompt, user_prompt) -> str
```

Every provider returns the decision text in the same shape, so they are
interchangeable. You pick one in `src/config.py`:

```python
LLM_PROVIDER = "dummy"  # "dummy" | "openrouter" | "claude_code" | "claude_code_subagents"
```

`USE_DUMMY_MODEL` is derived from `LLM_PROVIDER` (it is `True` only when the
provider is `"dummy"`). Do not set it by hand.

## Available providers

### `dummy`
Deterministic local stub. No network, no API key, no cost. Use it for
development, tests, and to exercise the full pipeline (features, prompts,
baselines, reports) without calling a real model.

### `openrouter`
Calls a model over the OpenRouter HTTP API. Requires `OPENROUTER_API_KEY` in
your environment. Requests are metered per token. Because calls are plain HTTP,
this provider parallelizes well and is the right choice for large backtests
(hundreds of days) and for comparing many different models.

```python
LLM_PROVIDER = "openrouter"
# export OPENROUTER_API_KEY=...   (in your shell / .env, never committed)
```

### `claude_code`
Routes each decision through a locally installed
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI, using your
own Claude subscription rather than a metered API key. The CLI is invoked in a
neutral working directory with all agentic tools disabled
(`Bash`, `Edit`, `Write`, `Read`, web access, etc.), so the model behaves like a
plain completion endpoint and produces no side effects.

```python
LLM_PROVIDER = "claude_code"
CLAUDE_CODE_MODEL = "sonnet"  # alias "sonnet" / "opus" / "haiku", or a full model id
```

Requirements and intended use:
- The `claude` CLI must be installed and signed in to a Claude account on the
  machine that runs the experiment. The provider only works where that login
  exists.
- Calls consume your Claude subscription quota (rate limits), not a per-token
  API balance. Use it the way the subscription is meant to be used: interactive,
  human-scale research, small experiments, short windows, or as a high-quality
  reference model. For unattended, high-volume backtests, use `openrouter`.

### `claude_code_subagents`
Same Claude Code CLI, but the lead model consults a set of specialist analysts
(defined in `ANALYST_AGENTS` in `src/config.py`, e.g. a technical analyst and a
risk analyst) before committing to a decision. This is a distinct research arm:
the deliberation structure can change the decision relative to a single-shot
call on the same data. It costs more quota and latency than `claude_code`
because the lead delegates to each analyst and then synthesizes.

```python
LLM_PROVIDER = "claude_code_subagents"
```

## How it is wired

- `src/config.py` is the single source of truth for the active provider and its
  options.
- `src/model_router.py` exposes `generate_response(router_model, system_prompt,
  user_prompt)` and dispatches to the right backend based on `LLM_PROVIDER`.
- `src/claude_code_model.py` builds and runs the Claude Code CLI command, blocks
  agentic tools, and parses the result.
- `src/openrouter_model.py` implements the OpenRouter HTTP call.

Call sites (the trading engine and the reporting layer) only ever call
`model_router.generate_response(...)`; they never talk to a specific provider
directly. Adding a new provider means adding one branch to the router.

## Choosing a provider

| Provider                 | Cost model            | Parallel | Best for |
|--------------------------|-----------------------|----------|----------|
| `dummy`                  | Free                  | n/a      | Dev, tests, pipeline smoke checks |
| `openrouter`             | Per token             | Yes      | Large backtests, multi-model comparisons |
| `claude_code`            | Subscription quota    | No       | Small, high-quality experiments; reference model |
| `claude_code_subagents`  | Subscription quota    | No       | Multi-analyst deliberation as a research arm |

A backtest of N days is N sequential decisions. For the Claude Code providers,
prefer short windows or a sampled subset of days; for volume, stay on
OpenRouter.
