# Reproducibility contract

A run is reproducible when the same **code**, **data**, **dependencies**, and
**seed** produce the same outputs. This document is the contract; the pieces that
enforce it live in the repo.

## The four pillars

| Pillar | How it is pinned | Where |
|--------|------------------|-------|
| Code | git SHA recorded per run | `results/RUN_MANIFEST.json` |
| Data | frozen snapshot verified by SHA-256 on load | `data/raw/spy_daily.csv` + `data/raw/MANIFEST.json` |
| Dependencies | exact `==` pins, tested on Python 3.11.2 | `requirements.txt` |
| Randomness | single seed applied to every stochastic component | `RANDOM_SEED` in `src/config.py` |

## Seeding

`RANDOM_SEED` (default `42`, in `src/config.py`) is applied once at the start of
every run by `set_global_seeds` (`src/reproducibility.py`), before any decision
is made. It seeds:

- Python's `random`: used by the dummy model (`src/dummy_model.py`).
- NumPy's global RNG: used by the statistical bootstraps
  (`src/statistical_validation.py`) and the random baseline.

With the dummy provider this makes the whole pipeline bit-for-bit
deterministic: two runs with the same config produce identical
`results/parsed/*.csv`. Real LLM providers are not guaranteed identical even at
`temperature = 0`; for those, reproducibility comes from the recorded manifest
(model + version + sampling params) plus the frozen prompts/data.

## The run manifest

Every run writes `results/RUN_MANIFEST.json` (`write_run_manifest`) capturing:

- `git_sha` (suffixed `-dirty` if the tree has uncommitted changes)
- `python`, `platform`, and `dependencies` (pandas / numpy / scipy / matplotlib / requests versions)
- `random_seed`
- `provider` (provider, model(s), and sampling params where they apply)
- `experiment` (active experiment, personality, and feature toggles)
- `run` (test mode/limit, start row, model tags produced)
- `data` (source, date range, and, for the vendored snapshot, its SHA-256, row count, and symbol)

Commit or archive this file next to the results it describes.

## Raw LLM response cache

The router also caches raw provider responses on disk in `results/llm_cache/`.
The cache key includes the provider, model, prompts, and provider-specific
context, so identical reruns can reuse the exact same raw text without another
API call. Set `LLM_RESPONSE_CACHE_DIR` to move the cache, or delete the cache
directory to force fresh calls.

## Reproducing a past run

1. Check out the `git_sha` from the manifest.
2. `pip install -r requirements.txt` (matches the recorded dependency versions).
3. Confirm the data checksum: `pytest tests/test_vendored_data.py` (verifies
   `spy_daily.csv` still matches `MANIFEST.json`).
4. Set the same `RANDOM_SEED`, provider, and experiment toggles in `src/config.py`.
5. `python -m src.main`.

## Refreshing pins or data (deliberate actions)

- **Dependencies:** bump a pin in `requirements.txt`, re-run `pytest`, and update
  the "Tested with" note here if the Python version changed.
- **Data:** run `python scripts/refresh_data.py`, which regenerates
  `spy_daily.csv` and `MANIFEST.json` together; then update
  `data/raw/PROVENANCE.md`. Refreshing changes the snapshot's SHA-256, so commit
  the snapshot and its manifest in the same commit.
