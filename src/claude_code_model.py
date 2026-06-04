# src/claude_code_model.py
"""
Claude Code provider.

Calls the official Claude Code CLI in headless print mode (`claude -p`), using
the user's Claude Code SUBSCRIPTION (OAuth credentials in ~/.claude). This means
no per-token API billing: calls are charged against the subscription plan and
consume its rate limits.

Intended use:
- This relies on the official, documented Claude Code headless mode. It is meant
  for INTERACTIVE-SCALE research: small experiments, short windows, a handful of
  tickers, or as a high-quality reference model. It is NOT a high-volume API
  replacement. For large automated backtests (hundreds+ of sequential days), use
  the OpenRouter provider, or Anthropic's API with an API key (pay per token),
  which is the path Anthropic intends for programmatic application backends.
- All file/web/shell tools are disabled on every call. The model reasons only
  from the market data in the prompt. Web access is deliberately blocked to avoid
  lookahead bias (the model must not be able to look up what actually happened).

Two modes:
- single-shot: one model call returns the decision (like a plain completion).
- subagents: a lead delegates to specialist analyst subagents (technical, risk,
  ...) via the Task tool, then synthesizes the decision. Richer, slower, costs
  more, and the deliberation structure can change the decision.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile

# Tools blocked on every call. Web tools are blocked to prevent lookahead bias.
_BLOCKED_TOOLS = "Bash Edit Write Read WebFetch WebSearch NotebookEdit Glob Grep"


def find_claude_cli() -> str:
    """Locate the claude CLI binary. Raises a clear error if not found."""
    explicit = os.environ.get("CLAUDE_CLI_PATH")
    if explicit and os.path.exists(explicit):
        return explicit

    on_path = shutil.which("claude")
    if on_path:
        return on_path

    # Fall back to the Claude Agent SDK bundled binary (version-agnostic glob).
    candidates = glob.glob(
        "/usr/lib/node_modules/**/@anthropic-ai/claude-agent-sdk-*/claude",
        recursive=True,
    ) + glob.glob(
        "/usr/local/lib/node_modules/**/@anthropic-ai/claude-agent-sdk-*/claude",
        recursive=True,
    )
    if candidates:
        return candidates[0]

    raise RuntimeError(
        "claude CLI not found. The Claude Code provider needs the Claude Code "
        "subscription CLI on PATH (or set CLAUDE_CLI_PATH). It only works on a "
        "machine where Claude Code is logged in (OAuth in ~/.claude)."
    )


def call_claude_code(
    system_prompt: str,
    user_prompt: str,
    model: str = "sonnet",
    agents: dict | None = None,
    timeout: int = 180,
) -> str:
    """
    Run one Claude Code headless call and return the assistant's text result.

    Args:
        system_prompt: Trader persona / instructions (fully replaces Claude Code's
            default system prompt, for a clean research identity).
        user_prompt: The market data + decision request.
        model: Claude model alias ("sonnet", "opus", "haiku") or a full model id.
        agents: Optional dict of subagent definitions. When provided, the lead
            may delegate to them via the Task tool (multi-agent mode).
        timeout: Hard timeout in seconds for the subprocess.

    Returns:
        The model's text output (the `result` field of the CLI JSON).
    """
    claude = find_claude_cli()

    cmd = [
        claude,
        "-p",
        user_prompt,
        "--system-prompt",
        system_prompt,
        "--model",
        model,
        "--output-format",
        "json",
    ]

    if agents:
        # Subagent mode: expose the analyst team and let the lead delegate via
        # Task. Web/shell/file tools stay blocked (no lookahead, no side effects).
        cmd += ["--agents", json.dumps(agents)]
        cmd += ["--disallowedTools", _BLOCKED_TOOLS]
    else:
        # Single-shot: pure completion. Also block Task so the model cannot spawn
        # agents unprompted.
        cmd += ["--disallowedTools", _BLOCKED_TOOLS + " Task"]

    # Run from a neutral temp dir so no project CLAUDE.md / files leak into context.
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Claude Code call timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"Claude Code CLI error (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[:500]}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Unexpected Claude Code output (not JSON): {proc.stdout[:500]}"
        ) from e

    if data.get("is_error"):
        raise RuntimeError(f"Claude Code returned an error: {data.get('result')}")

    result = data.get("result")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(f"Claude Code returned empty result: {data}")

    return result
