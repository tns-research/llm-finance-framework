# src/main.py

import os
from .config_compat import (
    USE_DUMMY_MODEL,
    LLM_MODELS,
    ACTIVE_EXPERIMENT,
    get_experiment_suffix,
    get_current_config_summary,
    list_experiments,
)
from .data_prep import prepare_features
from .prompts import build_prompts
from .trading_engine import run_single_model


def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(__file__))

    raw_path = os.path.join(base_dir, "data", "raw", "sp500.csv")
    features_path = os.path.join(base_dir, "data", "processed", "features.csv")
    prompts_path = os.path.join(base_dir, "data", "processed", "prompts.csv")

    # FORCE DUMMY MODE FOR SAFETY - Never call real OpenRouter models during testing
    from .config_compat import USE_DUMMY_MODEL

    if not USE_DUMMY_MODEL:
        print(
            "WARNING: USE_DUMMY_MODEL is False! Forcing to True to prevent accidental API calls."
        )
        print("Set USE_DUMMY_MODEL = True in config.py for testing.")
        # Override to ensure safety
        import src.config

        src.config.USE_DUMMY_MODEL = True

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
    print("=" * 70 + "\n")

    print("Step 1  prepare features")
    features = prepare_features(raw_path, features_path)
    print(f"Features shape  {features.shape}")

    print("Step 2  build prompts")
    prompts = build_prompts(features_path, prompts_path)
    print(f"Prompts shape  {prompts.shape}")

    # Get experiment suffix for model tags
    exp_suffix = get_experiment_suffix()

    if USE_DUMMY_MODEL:
        model_tag = f"dummy_model{exp_suffix}"
        run_single_model(model_tag, None, prompts, raw_path)
    else:
        for model_conf in LLM_MODELS:
            model_tag = f"{model_conf['tag']}{exp_suffix}"
            run_single_model(model_tag, model_conf["router_model"], prompts, raw_path)


if __name__ == "__main__":
    run_pipeline()
