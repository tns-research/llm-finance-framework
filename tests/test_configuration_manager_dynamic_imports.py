#!/usr/bin/env python3
"""
Test ConfigurationManager Dynamic Import Behavior

Tests to verify ConfigurationManager dynamic import reliability and behavior
between test and production environments.
"""

import os
import tempfile
from unittest.mock import patch

from src.configuration_manager import ConfigurationManager


def test_dynamic_import_with_different_config_states():
    """Test ConfigurationManager dynamic import with various config.py states"""

    print("Testing ConfigurationManager dynamic import behavior...")

    # Test 1: Normal operation with ENABLE_CHAIN_OF_THOUGHT = True
    print("\n1. Testing normal operation (ENABLE_CHAIN_OF_THOUGHT = True)")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
ENABLE_CHAIN_OF_THOUGHT = True
ENABLE_STRATEGIC_JOURNAL = True
ENABLE_FEELING_LOG = True
ACTIVE_EXPERIMENT = "memory_feeling"
SYMBOL = "^GSPC"
DATA_START = "2015-01-01"
DATA_END = "2023-12-31"
""")
        temp_config_path = f.name

    try:
        config_manager = ConfigurationManager(config_file=temp_config_path)
        flags = config_manager.get_feature_flags()

        print(f"  ENABLE_CHAIN_OF_THOUGHT: {flags.get('ENABLE_CHAIN_OF_THOUGHT')}")
        print(f"  ENABLE_STRATEGIC_JOURNAL: {flags.get('ENABLE_STRATEGIC_JOURNAL')}")
        print(f"  ENABLE_FEELING_LOG: {flags.get('ENABLE_FEELING_LOG')}")

        assert (
            flags.get("ENABLE_CHAIN_OF_THOUGHT") == True
        ), "Should read ENABLE_CHAIN_OF_THOUGHT = True"
        assert (
            flags.get("ENABLE_STRATEGIC_JOURNAL") == True
        ), "Should read ENABLE_STRATEGIC_JOURNAL = True"
        assert (
            flags.get("ENABLE_FEELING_LOG") == True
        ), "Should read ENABLE_FEELING_LOG = True"
        print("  ✓ Normal operation test passed")

    finally:
        os.unlink(temp_config_path)

    # Test 2: Chain of thought disabled
    print("\n2. Testing with ENABLE_CHAIN_OF_THOUGHT = False")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
ENABLE_CHAIN_OF_THOUGHT = False
ENABLE_STRATEGIC_JOURNAL = True
ENABLE_FEELING_LOG = False
ACTIVE_EXPERIMENT = "memory_only"
SYMBOL = "^GSPC"
DATA_START = "2015-01-01"
DATA_END = "2023-12-31"
""")
        temp_config_path = f.name

    try:
        config_manager = ConfigurationManager(config_file=temp_config_path)
        flags = config_manager.get_feature_flags()

        print(f"  ENABLE_CHAIN_OF_THOUGHT: {flags.get('ENABLE_CHAIN_OF_THOUGHT')}")
        print(f"  ENABLE_STRATEGIC_JOURNAL: {flags.get('ENABLE_STRATEGIC_JOURNAL')}")
        print(f"  ENABLE_FEELING_LOG: {flags.get('ENABLE_FEELING_LOG')}")

        assert (
            flags.get("ENABLE_CHAIN_OF_THOUGHT") == False
        ), "Should read ENABLE_CHAIN_OF_THOUGHT = False"
        assert (
            flags.get("ENABLE_STRATEGIC_JOURNAL") == True
        ), "Should read ENABLE_STRATEGIC_JOURNAL = True"
        assert (
            flags.get("ENABLE_FEELING_LOG") == False
        ), "Should read ENABLE_FEELING_LOG = False"
        print("  ✓ Disabled chain of thought test passed")

    finally:
        os.unlink(temp_config_path)

    # Test 3: Test fallback behavior when import fails
    print("\n3. Testing fallback behavior with invalid config file")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
# Invalid Python syntax to trigger import failure
INVALID SYNTAX HERE
ENABLE_CHAIN_OF_THOUGHT = True
""")
        temp_config_path = f.name

    try:
        config_manager = ConfigurationManager(config_file=temp_config_path)
        flags = config_manager.get_feature_flags()

        print("  Config loading failed (expected), testing fallback behavior")
        print(
            f"  ENABLE_CHAIN_OF_THOUGHT: {flags.get('ENABLE_CHAIN_OF_THOUGHT')} (should be False)"
        )
        print(
            f"  ENABLE_STRATEGIC_JOURNAL: {flags.get('ENABLE_STRATEGIC_JOURNAL')} (should be True)"
        )
        print(
            f"  ENABLE_FEELING_LOG: {flags.get('ENABLE_FEELING_LOG')} (should be True)"
        )

        # These should be the default fallback values from _get_feature_flags()
        assert (
            flags.get("ENABLE_CHAIN_OF_THOUGHT") == False
        ), "Should fallback to ENABLE_CHAIN_OF_THOUGHT = False"
        assert (
            flags.get("ENABLE_STRATEGIC_JOURNAL") == True
        ), "Should fallback to ENABLE_STRATEGIC_JOURNAL = True"
        assert (
            flags.get("ENABLE_FEELING_LOG") == True
        ), "Should fallback to ENABLE_FEELING_LOG = True"
        print("  ✓ Fallback behavior test passed")

    finally:
        os.unlink(temp_config_path)

    # Test 4: Test feature flag override from legacy config
    print("\n4. Testing feature flag override from legacy config")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
ENABLE_CHAIN_OF_THOUGHT = True   # This gets applied to override experiment default
ENABLE_STRATEGIC_JOURNAL = False # This gets applied to override experiment default
ENABLE_FEELING_LOG = True        # This gets applied to override experiment default
ACTIVE_EXPERIMENT = "baseline"    # Baseline experiment defaults: SJ=False, FL=False, COT=False
SYMBOL = "^GSPC"
DATA_START = "2015-01-01"
DATA_END = "2023-12-31"
""")
        temp_config_path = f.name

    try:
        config_manager = ConfigurationManager(config_file=temp_config_path)
        flags = config_manager.get_feature_flags()

        print("  Legacy config overrides: COT=True, SJ=False, FL=True")
        print(
            f"  Experiment: {config_manager._config.active_experiment} (defaults: COT=False, SJ=False, FL=False)"
        )
        print(
            f"  Result ENABLE_CHAIN_OF_THOUGHT: {flags.get('ENABLE_CHAIN_OF_THOUGHT')}"
        )
        print(
            f"  Result ENABLE_STRATEGIC_JOURNAL: {flags.get('ENABLE_STRATEGIC_JOURNAL')}"
        )
        print(f"  Result ENABLE_FEELING_LOG: {flags.get('ENABLE_FEELING_LOG')}")

        # All flags should match what was set in legacy config, overriding experiment defaults
        assert (
            flags.get("ENABLE_CHAIN_OF_THOUGHT") == True
        ), "Chain of thought should be overridden to True"
        assert (
            flags.get("ENABLE_STRATEGIC_JOURNAL") == False
        ), "Strategic journal should remain False (matches legacy config)"
        assert (
            flags.get("ENABLE_FEELING_LOG") == True
        ), "Feeling log should be overridden to True"
        print("  ✓ Feature flag override behavior test passed")

    finally:
        os.unlink(temp_config_path)

    print("\n✅ All ConfigurationManager dynamic import tests passed!")


def test_feature_flag_pipeline_consistency():
    """Test that feature flags flow consistently through the entire pipeline"""

    print("\nTesting feature flag pipeline consistency...")

    # NOTE: This test uses the global config module which is initialized with the default config.
    # In a real application, there should only be one ConfigurationManager instance.
    # For testing purposes, we verify that the global config values are consistent.

    # Import config to get the global instance
    from src import config

    # Get flags from the global ConfigurationManager instance
    cm_flags = config._config_manager.get_feature_flags()

    print(
        f"ConfigurationManager flags: COT={cm_flags['ENABLE_CHAIN_OF_THOUGHT']}, SJ={cm_flags['ENABLE_STRATEGIC_JOURNAL']}, FL={cm_flags['ENABLE_FEELING_LOG']}"
    )

    # Test config exposure (should match ConfigurationManager)
    compat_flags = {
        "ENABLE_CHAIN_OF_THOUGHT": config.ENABLE_CHAIN_OF_THOUGHT,
        "ENABLE_STRATEGIC_JOURNAL": config.ENABLE_STRATEGIC_JOURNAL,
        "ENABLE_FEELING_LOG": config.ENABLE_FEELING_LOG,
    }

    print(
        f"config flags: COT={compat_flags['ENABLE_CHAIN_OF_THOUGHT']}, SJ={compat_flags['ENABLE_STRATEGIC_JOURNAL']}, FL={compat_flags['ENABLE_FEELING_LOG']}"
    )

    # Verify consistency between ConfigurationManager and config
    assert (
        cm_flags["ENABLE_CHAIN_OF_THOUGHT"] == compat_flags["ENABLE_CHAIN_OF_THOUGHT"]
    ), "Chain of thought flag mismatch"
    assert (
        cm_flags["ENABLE_STRATEGIC_JOURNAL"] == compat_flags["ENABLE_STRATEGIC_JOURNAL"]
    ), "Strategic journal flag mismatch"
    assert (
        cm_flags["ENABLE_FEELING_LOG"] == compat_flags["ENABLE_FEELING_LOG"]
    ), "Feeling log flag mismatch"

    # Test parsing function access with current global configuration
    from src.backtest import parse_response_text

    # Create a test response that matches the current configuration
    expected_lines = 3  # Base
    if compat_flags["ENABLE_CHAIN_OF_THOUGHT"]:
        expected_lines += 1
    if compat_flags["ENABLE_STRATEGIC_JOURNAL"]:
        expected_lines += 1
    if compat_flags["ENABLE_FEELING_LOG"]:
        expected_lines += 1

    print(f"Current configuration expects {expected_lines} lines per response")

    # Create appropriate test response
    response_lines = []
    if compat_flags["ENABLE_CHAIN_OF_THOUGHT"]:
        response_lines.append(
            "Analyzing market conditions: RSI shows neutral, MACD weak momentum."
        )

    response_lines.extend(["BUY", "0.75", "Market indicators suggest upward momentum."])

    if compat_flags["ENABLE_STRATEGIC_JOURNAL"]:
        response_lines.append("Reviewing recent performance and adjusting strategy.")
    if compat_flags["ENABLE_FEELING_LOG"]:
        response_lines.append("Feeling cautiously optimistic.")

    test_response = "\n".join(response_lines)

    try:
        result = parse_response_text(test_response)
        (
            decision,
            prob,
            explanation,
            chain_of_thought,
            strategic_journal,
            feeling_log,
        ) = result

        print(f"Parsing result: decision={decision}, prob={prob}")
        print(
            f"Chain of thought present: {chain_of_thought != 'Chain of thought reasoning disabled.'}"
        )
        print(
            f"Strategic journal present: {strategic_journal != 'Strategic journal disabled in this configuration.'}"
        )
        print(
            f"Feeling log present: {feeling_log != 'Feeling log disabled in this configuration.'}"
        )

        # Verify parsing worked correctly for current configuration
        assert decision == "BUY", "Decision should be BUY"
        assert prob == 0.75, "Probability should be 0.75"
        assert ("Analyzing market conditions" in chain_of_thought) == compat_flags[
            "ENABLE_CHAIN_OF_THOUGHT"
        ], "Chain of thought presence should match config"
        assert ("disabled" in strategic_journal) == (
            not compat_flags["ENABLE_STRATEGIC_JOURNAL"]
        ), "Strategic journal status should match config"
        assert ("disabled" in feeling_log) == (
            not compat_flags["ENABLE_FEELING_LOG"]
        ), "Feeling log status should match config"

        print("  ✓ Pipeline consistency test passed")

    except Exception as e:
        print(f"  ✗ Parsing failed: {e}")
        raise


if __name__ == "__main__":
    test_dynamic_import_with_different_config_states()
    test_feature_flag_pipeline_consistency()
    print("\n🎉 All ConfigurationManager dynamic import tests completed successfully!")
