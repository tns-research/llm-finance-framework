"""Tests for the reproducibility layer: deterministic seeding and run manifests."""

import json
import os

from src import config
from src.dummy_model import dummy_call_model
from src.reproducibility import (
    build_run_manifest,
    set_global_seeds,
    write_run_manifest,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _dummy_sequence(seed, n=8):
    set_global_seeds(seed)
    return [dummy_call_model("sys", f"day {i}") for i in range(n)]


def test_seed_makes_dummy_model_deterministic():
    """Same seed -> identical dummy decision stream; different seed -> different."""
    assert _dummy_sequence(123) == _dummy_sequence(123)
    assert _dummy_sequence(123) != _dummy_sequence(999)


def test_set_global_seeds_seeds_numpy():
    import numpy as np

    set_global_seeds(7)
    first = np.random.rand(5).tolist()
    set_global_seeds(7)
    second = np.random.rand(5).tolist()
    assert first == second


def test_manifest_has_required_keys_and_records_seed():
    manifest = build_run_manifest(PROJECT_ROOT, model_tags=["dummy_model_test"])
    for key in (
        "schema",
        "git_sha",
        "python",
        "dependencies",
        "random_seed",
        "provider",
        "experiment",
        "run",
        "data",
    ):
        assert key in manifest, f"manifest missing key: {key}"

    assert manifest["random_seed"] == config.RANDOM_SEED
    assert manifest["run"]["model_tags"] == ["dummy_model_test"]
    assert manifest["provider"]["llm_provider"] == config.LLM_PROVIDER
    assert set(manifest["dependencies"]) >= {"pandas", "numpy", "scipy"}


def test_manifest_records_vendored_data_checksum():
    """When the vendored snapshot is the source, the manifest carries its SHA-256."""
    if config.DATA_SOURCE != "vendored":
        return
    manifest = build_run_manifest(PROJECT_ROOT, model_tags=["dummy"])
    with open(os.path.join(PROJECT_ROOT, config.VENDORED_MANIFEST_PATH)) as f:
        expected = json.load(f)["sha256"]
    assert manifest["data"]["sha256"] == expected


def test_write_run_manifest_emits_valid_json(tmp_path):
    path = write_run_manifest(str(tmp_path), model_tags=["dummy"])
    assert os.path.exists(path)
    assert path == os.path.join(str(tmp_path), "results", "RUN_MANIFEST.json")
    with open(path) as f:
        reloaded = json.load(f)
    assert reloaded["run"]["model_tags"] == ["dummy"]
