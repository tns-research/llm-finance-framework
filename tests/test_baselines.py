"""Guard tests for the baseline strategy registries.

BASELINE_REGISTRY (name -> strategy function) and STRATEGY_METADATA
(name -> reporting metadata) are two parallel dicts keyed by the same
strategy names. They can silently drift if a strategy is added to one but
not the other. These tests fail loudly when that happens.
"""

from src.baselines import BASELINE_REGISTRY, STRATEGY_METADATA


def test_registry_and_metadata_cover_same_strategies():
    assert set(BASELINE_REGISTRY) == set(STRATEGY_METADATA), (
        "BASELINE_REGISTRY and STRATEGY_METADATA have drifted; every strategy "
        "needs an entry in both."
    )


def test_every_registry_entry_is_callable():
    for name, fn in BASELINE_REGISTRY.items():
        assert callable(fn), f"BASELINE_REGISTRY[{name!r}] is not callable"


def test_every_metadata_entry_has_required_fields():
    for name, meta in STRATEGY_METADATA.items():
        assert "category" in meta, f"{name!r} metadata missing 'category'"
        assert "indicators" in meta, f"{name!r} metadata missing 'indicators'"
        assert "description" in meta, f"{name!r} metadata missing 'description'"
