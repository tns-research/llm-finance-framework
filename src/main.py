# src/main.py

import os

from .config import (
    CLAUDE_CODE_MODEL,
    LLM_MODELS,
    LLM_PROVIDER,
    RANDOM_SEED,
    TEST_MODE,
    get_current_config_summary,
    get_experiment_suffix,
)
from .data_prep import prepare_features
from .data_sources import data_manager
from .logging_config import setup_logging
from .prompts import build_prompts
from .reproducibility import set_global_seeds, write_run_manifest
from .trading_engine import run_single_model


def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # Route diagnostics (progress, warnings) through logging; report blocks
    # stay on stdout via print(). Level from LOG_LEVEL env var (default INFO).
    setup_logging()

    # Seed every stochastic component up front so the run replays identically.
    set_global_seeds(RANDOM_SEED)

    # NEW: Use data source manager instead of hardcoded CSV
    print("Step 1: Fetch raw data")
    try:
        raw_df = data_manager.get_data()

        # Save to temporary location for processing
        raw_path = os.path.join(base_dir, "data", "raw", "current_data.csv")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        raw_df.to_csv(raw_path, index=False)

    except Exception as e:
        print(f"ERROR: Failed to fetch data: {e}")
        raise

    features_path = os.path.join(base_dir, "data", "processed", "features.csv")
    prompts_path = os.path.join(base_dir, "data", "processed", "prompts.csv")

    # Cost / provider awareness
    if LLM_PROVIDER == "openrouter":
        print("[PROVIDER] REAL API MODE: OpenRouter")
        print("This will make actual OpenRouter API calls and incur per-token costs.")
        print("Estimated cost: ~$0.01-0.05 per model call depending on prompt length.")
        print("Press Ctrl+C within 3 seconds to cancel...")
        try:
            import time

            time.sleep(3)
            print("Proceeding with real API calls...")
        except KeyboardInterrupt:
            print('\nCancelled. Set LLM_PROVIDER = "dummy" to use dummy models.')
            exit(1)
    elif LLM_PROVIDER in ("claude_code", "claude_code_subagents"):
        # Fail early with a clear message if the subscription CLI is missing.
        from .claude_code_model import find_claude_cli

        find_claude_cli()
        mode = (
            "multi-agent (analysts)"
            if LLM_PROVIDER == "claude_code_subagents"
            else "single-shot"
        )
        print(
            f"[PROVIDER] Claude Code via subscription, {mode}, model={CLAUDE_CODE_MODEL}"
        )
        print("Calls are billed to your Claude Code subscription (no per-token cost)")
        print("and consume its rate limits. Intended use: interactive-scale research.")
        if not TEST_MODE:
            print("[NOTICE] TEST_MODE is OFF: a full backtest = many sequential")
            print("subscription calls (slow, heavy on rate limits). For large runs,")
            print('consider LLM_PROVIDER = "openrouter". Ctrl+C within 3s to cancel...')
            try:
                import time

                time.sleep(3)
            except KeyboardInterrupt:
                print("\nCancelled.")
                exit(1)

    # Show current experiment configuration
    config_summary = get_current_config_summary()
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"  Active experiment: {config_summary['experiment']}")
    print(f"  Description: {config_summary['description']}")
    print(
        f"  Settings: dates={config_summary['show_dates']}, "
        f"memory={config_summary['strategic_journal']}, "
        f"feeling={config_summary['feeling_log']}"
    )

    # Debug: Show actual runtime value of DEBUG_SHOW_FULL_PROMPT
    from .config import DEBUG_SHOW_FULL_PROMPT

    print(f"  Debug settings: DEBUG_SHOW_FULL_PROMPT={DEBUG_SHOW_FULL_PROMPT}")
    print("=" * 70 + "\n")

    print("Step 2: Prepare features")
    features = prepare_features(raw_path, features_path)
    print(f"Features shape: {features.shape}")

    print("Step 3: Build prompts")
    prompts = build_prompts(features_path, prompts_path)
    print(f"Prompts shape: {prompts.shape}")

    # Get experiment suffix for model tags
    exp_suffix = get_experiment_suffix()

    # Build the run plan (model_tag, router_model) for the active provider.
    if LLM_PROVIDER == "dummy":
        run_plan = [(f"dummy_model{exp_suffix}", None)]
    elif LLM_PROVIDER in ("claude_code", "claude_code_subagents"):
        base = CLAUDE_CODE_MODEL.replace("/", "-")
        suffix = "-subagents" if LLM_PROVIDER == "claude_code_subagents" else ""
        run_plan = [(f"claude-{base}{suffix}{exp_suffix}", CLAUDE_CODE_MODEL)]
    else:  # openrouter
        run_plan = [(f"{m['tag']}{exp_suffix}", m["router_model"]) for m in LLM_MODELS]

    # Emit the reproducibility manifest before running (seed, deps, data
    # checksum, git SHA, config). See docs/REPRODUCIBILITY.md.
    manifest_path = write_run_manifest(base_dir, model_tags=[t for t, _ in run_plan])
    print(f"[REPRO] Run manifest: {manifest_path}")

    for model_tag, router_model in run_plan:
        run_single_model(model_tag, router_model, prompts, raw_path)


if __name__ == "__main__":
    run_pipeline()
