"""Reproducibility helpers: deterministic seeding and per-run manifests.

A run is reproducible when the same code, data, dependencies, and seed produce
the same outputs. ``set_global_seeds`` pins the stochastic components; the manifest
records everything needed to recreate the environment (git SHA, dependency
versions, data checksum, config, seed). See docs/REPRODUCIBILITY.md.
"""

import importlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone

_TRACKED_DEPS = ["pandas", "numpy", "scipy", "matplotlib", "requests"]


def set_global_seeds(seed: int) -> None:
    """Seed every stochastic component so a run replays identically.

    Seeds Python's ``random`` (the dummy model) and NumPy's global RNG (the
    statistical bootstraps). Safe to call when NumPy is absent.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _git_sha(base_dir: str) -> str:
    """Short git SHA of the working tree, suffixed ``-dirty`` if uncommitted."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if rev.returncode != 0:
            return "unknown"
        sha = rev.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0 and status.stdout.strip():
            sha += "-dirty"
        return sha
    except Exception:
        return "unknown"


def _dep_versions() -> dict:
    versions = {}
    for mod in _TRACKED_DEPS:
        try:
            versions[mod] = importlib.import_module(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    return versions


def _data_block(base_dir: str) -> dict:
    from . import config

    block = {
        "data_source": config.DATA_SOURCE,
        "date_start": config.DATA_START,
        "date_end": config.DATA_END,
    }
    if config.DATA_SOURCE == "vendored":
        manifest_path = os.path.join(base_dir, config.VENDORED_MANIFEST_PATH)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                m = json.load(f)
            block.update(
                {
                    "file": m.get("file"),
                    "sha256": m.get("sha256"),
                    "rows": m.get("rows"),
                    "symbol": m.get("symbol"),
                    "snapshot_range": [m.get("date_start"), m.get("date_end")],
                }
            )
    return block


def _provider_block() -> dict:
    from . import config

    block = {"llm_provider": config.LLM_PROVIDER}
    if config.LLM_PROVIDER == "openrouter":
        block["openrouter_models"] = [m.get("router_model") for m in config.LLM_MODELS]
        # Mirrors the payload in src/openrouter_model.py.
        block["sampling"] = {"temperature": 0.0, "max_tokens": 50000}
    elif config.LLM_PROVIDER in ("claude_code", "claude_code_subagents"):
        block["claude_code_model"] = config.CLAUDE_CODE_MODEL
        if config.LLM_PROVIDER == "claude_code_subagents":
            block["analyst_agents"] = list(config.ANALYST_AGENTS.keys())
    return block


def build_run_manifest(base_dir: str, model_tags) -> dict:
    """Assemble the run manifest dict (does not write it)."""
    from . import config

    summary = config.get_current_config_summary()
    return {
        "schema": "llm-finance-run-manifest/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(base_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": getattr(config, "RANDOM_SEED", None),
        "dependencies": _dep_versions(),
        "provider": _provider_block(),
        "experiment": {
            "active_experiment": config.ACTIVE_EXPERIMENT,
            "active_personality": config.ACTIVE_PERSONALITY,
            "show_date_to_llm": summary["show_dates"],
            "strategic_journal": summary["strategic_journal"],
            "feeling_log": summary["feeling_log"],
            "chain_of_thought": summary["chain_of_thought"],
            "technical_indicators": config.ENABLE_TECHNICAL_INDICATORS,
            "full_trading_history": config.ENABLE_FULL_TRADING_HISTORY,
        },
        "run": {
            "test_mode": config.TEST_MODE,
            "test_limit": config.TEST_LIMIT,
            "start_row": config.START_ROW,
            "model_tags": list(model_tags),
        },
        "data": _data_block(base_dir),
    }


def write_run_manifest(base_dir: str, model_tags) -> str:
    """Build and write results/RUN_MANIFEST.json; return its path."""
    manifest = build_run_manifest(base_dir, model_tags)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "RUN_MANIFEST.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path
