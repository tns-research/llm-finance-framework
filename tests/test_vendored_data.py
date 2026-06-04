"""Integrity tests for the vendored, frozen dataset.

These guard the reproducibility contract: the committed snapshot must match the
checksum recorded in its manifest, and the VendoredDataSource must refuse to load
a snapshot that has been modified or corrupted.
"""

import hashlib
import json
import os

import pytest

from src.data_sources import DataSourceError, VendoredDataSource

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "spy_daily.csv")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "MANIFEST.json")


def _load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def test_snapshot_and_manifest_exist():
    assert os.path.exists(DATA_PATH), f"vendored snapshot missing: {DATA_PATH}"
    assert os.path.exists(MANIFEST_PATH), f"manifest missing: {MANIFEST_PATH}"


def test_snapshot_sha256_matches_manifest():
    manifest = _load_manifest()
    actual = hashlib.sha256(open(DATA_PATH, "rb").read()).hexdigest()
    assert actual == manifest["sha256"], (
        "Vendored snapshot checksum does not match MANIFEST.json. "
        "If you intentionally refreshed the data, regenerate the manifest "
        "(scripts/refresh_data.py) and commit both files together."
    )


def test_snapshot_row_count_matches_manifest():
    manifest = _load_manifest()
    with open(DATA_PATH) as f:
        data_rows = sum(1 for _ in f) - 1  # minus header
    assert data_rows == manifest["rows"]


def test_vendored_source_loads_expected_columns():
    df = VendoredDataSource().fetch_data("SPY", "2015-01-01", "2023-12-31")
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(df) == _load_manifest()["rows"]


def test_vendored_source_rejects_corrupted_file(tmp_path, monkeypatch):
    """A snapshot whose bytes do not match the manifest checksum must fail loudly."""
    import src.data_sources as ds

    bad_csv = tmp_path / "spy_daily.csv"
    bad_csv.write_text("Date,Open,High,Low,Close,Volume\n2020-01-02,1,1,1,1,1\n")
    monkeypatch.setattr(ds.config, "VENDORED_DATA_PATH", str(bad_csv), raising=False)
    monkeypatch.setattr(
        ds.config, "VENDORED_MANIFEST_PATH", MANIFEST_PATH, raising=False
    )

    with pytest.raises(DataSourceError, match="checksum mismatch"):
        VendoredDataSource().fetch_data("SPY", "2015-01-01", "2023-12-31")
