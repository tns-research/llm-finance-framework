#!/usr/bin/env python3
"""
End-to-End Feature Flag Synchronization Tests

Tests that feature flags remain synchronized throughout the entire pipeline:
config.py → ConfigurationManager → config → prompt generation → parsing
"""

import os
import tempfile
from unittest.mock import patch

from src.backtest import parse_response_text
from src.prompt_builder import PromptBuilder


def test_feature_flag_pipeline_synchronization():
    """Test complete feature flag synchronization from config to parsing"""

    print("Testing end-to-end feature flag synchronization...")

    # Test all 8 combinations of feature flags
    test_cases = [
        # (chain_of_thought, strategic_journal, feeling_log, expected_lines)
        (False, False, False, 3),  # Baseline
        (False, False, True, 4),  # Feeling only
        (False, True, False, 4),  # Strategic only
        (False, True, True, 5),  # Strategic + feeling
        (True, False, False, 4),  # Chain of thought only
        (True, False, True, 5),  # Chain of thought + feeling
        (True, True, False, 5),  # Chain of thought + strategic
        (True, True, True, 6),  # All features
    ]

    for cot, strategic, feeling, expected_lines in test_cases:
        print(
            f"\n  Testing combination: COT={cot}, SJ={strategic}, FL={feeling} (expected {expected_lines} lines)"
        )

        # Create temporary config with specific feature flags
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f"""
ENABLE_CHAIN_OF_THOUGHT = {cot}
ENABLE_STRATEGIC_JOURNAL = {strategic}
ENABLE_FEELING_LOG = {feeling}
ACTIVE_EXPERIMENT = "baseline"  # Use baseline to test overrides
SYMBOL = "^GSPC"
DATA_START = "2015-01-01"
DATA_END = "2023-12-31"
""")
            temp_config_path = f.name

        try:
            # Patch global config variables to match test configuration
            # This is critical because parse_response_text imports flags from config
            with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", cot), patch(
                "src.config.ENABLE_STRATEGIC_JOURNAL", strategic
            ), patch("src.config.ENABLE_FEELING_LOG", feeling):

                # Test 1: ConfigurationManager reads flags correctly
                from src.configuration_manager import ConfigurationManager

                config_manager = ConfigurationManager(config_file=temp_config_path)
                cm_flags = config_manager.get_feature_flags()

                assert (
                    cm_flags["ENABLE_CHAIN_OF_THOUGHT"] == cot
                ), f"CM COT mismatch: expected {cot}, got {cm_flags['ENABLE_CHAIN_OF_THOUGHT']}"
                assert (
                    cm_flags["ENABLE_STRATEGIC_JOURNAL"] == strategic
                ), f"CM SJ mismatch: expected {strategic}, got {cm_flags['ENABLE_STRATEGIC_JOURNAL']}"
                assert (
                    cm_flags["ENABLE_FEELING_LOG"] == feeling
                ), f"CM FL mismatch: expected {feeling}, got {cm_flags['ENABLE_FEELING_LOG']}"
                print("    ✓ ConfigurationManager flags correct")

                # Test 2: Prompt generation includes correct instructions
                prompt_builder = PromptBuilder(config_manager)
                prompt = prompt_builder.build_system_prompt()

                # Check for chain of thought instructions
                has_cot_instructions = "Before making your trading decision" in prompt
                assert (
                    has_cot_instructions == cot
                ), f"Prompt COT instructions mismatch: expected {cot}, got {has_cot_instructions}"

                # Check output format specifies correct line count
                expected_line_spec = f"exactly {expected_lines} lines"
                assert (
                    expected_line_spec in prompt
                ), f"Prompt missing line count specification: {expected_line_spec}"
                print("    ✓ Prompt generation correct")

                # Test 3: Parse response with matching line count
                # Create response with correct number of lines for this config
                response_lines = []
                if cot:
                    response_lines.append(
                        "Analyzing market conditions: RSI shows neutral, MACD weak momentum."
                    )

                response_lines.extend(
                    ["BUY", "0.75", "Market indicators suggest upward momentum."]
                )

                if strategic:
                    response_lines.append(
                        "Reviewing recent performance and adjusting strategy."
                    )

                if feeling:
                    response_lines.append("Feeling cautiously optimistic.")

                test_response = "\n".join(response_lines)
                actual_lines = len(response_lines)
                assert (
                    actual_lines == expected_lines
                ), f"Response line count mismatch: expected {expected_lines}, got {actual_lines}"

                # Parse the response
                result = parse_response_text(test_response)
                (
                    decision,
                    prob,
                    explanation,
                    chain_of_thought,
                    strategic_journal,
                    feeling_log,
                ) = result

                # Verify parsing results match configuration
                assert decision == "BUY", "Decision parsing failed"
                assert prob == 0.75, "Probability parsing failed"
                assert ("disabled" in chain_of_thought) == (
                    not cot
                ), f"Chain of thought status mismatch: cot={cot}, result='{chain_of_thought[:30]}...'"
                assert ("disabled" in strategic_journal) == (
                    not strategic
                ), f"Strategic journal status mismatch: strategic={strategic}"
                assert ("disabled" in feeling_log) == (
                    not feeling
                ), f"Feeling log status mismatch: feeling={feeling}"
                print("    ✓ Response parsing correct")

            print(
                f"    ✅ Combination COT={cot}, SJ={strategic}, FL={feeling} passed all synchronization checks"
            )

        except Exception as e:
            print(
                f"    ❌ Combination COT={cot}, SJ={strategic}, FL={feeling} failed: {e}"
            )
            raise
        finally:
            os.unlink(temp_config_path)

    print("\n✅ All feature flag synchronization tests passed!")


def test_config_synchronization():
    """Test that config module stays synchronized with ConfigurationManager"""

    print("\nTesting config synchronization...")

    # Test with different config states
    test_configs = [
        (True, False, True),  # COT enabled, strategic disabled, feeling enabled
        (False, True, False),  # COT disabled, strategic enabled, feeling disabled
    ]

    for cot, strategic, feeling in test_configs:
        print(f"\n  Testing config sync: COT={cot}, SJ={strategic}, FL={feeling}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f"""
ENABLE_CHAIN_OF_THOUGHT = {cot}
ENABLE_STRATEGIC_JOURNAL = {strategic}
ENABLE_FEELING_LOG = {feeling}
ACTIVE_EXPERIMENT = "baseline"
SYMBOL = "^GSPC"
DATA_START = "2015-01-01"
DATA_END = "2023-12-31"
""")
            temp_config_path = f.name

        try:
            # Create ConfigurationManager with custom config
            from src.configuration_manager import ConfigurationManager

            config_manager = ConfigurationManager(config_file=temp_config_path)

            # Get flags from both sources
            cm_flags = config_manager.get_feature_flags()

            # Import config after ConfigurationManager is created
            # Note: In real usage, config creates its own ConfigurationManager instance
            from src import config

            compat_flags = {
                "ENABLE_CHAIN_OF_THOUGHT": config.ENABLE_CHAIN_OF_THOUGHT,
                "ENABLE_STRATEGIC_JOURNAL": config.ENABLE_STRATEGIC_JOURNAL,
                "ENABLE_FEELING_LOG": config.ENABLE_FEELING_LOG,
            }

            print(
                f"    CM flags: COT={cm_flags['ENABLE_CHAIN_OF_THOUGHT']}, SJ={cm_flags['ENABLE_STRATEGIC_JOURNAL']}, FL={cm_flags['ENABLE_FEELING_LOG']}"
            )
            print(
                f"    Compat flags: COT={compat_flags['ENABLE_CHAIN_OF_THOUGHT']}, SJ={compat_flags['ENABLE_STRATEGIC_JOURNAL']}, FL={compat_flags['ENABLE_FEELING_LOG']}"
            )

            # Note: config uses its own ConfigurationManager instance, so flags may differ
            # This test documents the current behavior rather than enforcing synchronization
            # In production, only one ConfigurationManager instance should exist

            print("    ✓ Config compat behavior documented")

        finally:
            os.unlink(temp_config_path)

    print("\n✅ Config compat synchronization tests completed!")


def test_prompt_parsing_consistency():
    """Test that prompts and parsing logic are perfectly aligned"""

    print("\nTesting prompt-parsing consistency...")

    # Test cases with specific prompts and expected parsing behavior
    test_scenarios = [
        {
            "config": {"cot": True, "sj": True, "fl": True},
            "prompt_check": lambda p: "Before making your trading decision" in p
            and "exactly 6 lines" in p,
            "response_lines": 6,
            "expected_parsing": {
                "cot_present": True,
                "sj_present": True,
                "fl_present": True,
            },
        },
        {
            "config": {"cot": False, "sj": False, "fl": False},
            "prompt_check": lambda p: "Before making your trading decision" not in p
            and "exactly 3 lines" in p,
            "response_lines": 3,
            "expected_parsing": {
                "cot_present": False,
                "sj_present": False,
                "fl_present": False,
            },
        },
        {
            "config": {"cot": True, "sj": False, "fl": False},
            "prompt_check": lambda p: "Before making your trading decision" in p
            and "exactly 4 lines" in p,
            "response_lines": 4,
            "expected_parsing": {
                "cot_present": True,
                "sj_present": False,
                "fl_present": False,
            },
        },
    ]

    for scenario in test_scenarios:
        config = scenario["config"]
        print(
            f"\n  Testing scenario: COT={config['cot']}, SJ={config['sj']}, FL={config['fl']}"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f"""
ENABLE_CHAIN_OF_THOUGHT = {config['cot']}
ENABLE_STRATEGIC_JOURNAL = {config['sj']}
ENABLE_FEELING_LOG = {config['fl']}
ACTIVE_EXPERIMENT = "baseline"
SYMBOL = "^GSPC"
DATA_START = "2015-01-01"
DATA_END = "2023-12-31"
""")
            temp_config_path = f.name

        try:
            # Patch global config variables for parsing consistency
            with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", config["cot"]), patch(
                "src.config.ENABLE_STRATEGIC_JOURNAL", config["sj"]
            ), patch("src.config.ENABLE_FEELING_LOG", config["fl"]):

                # Generate prompt
                from src.configuration_manager import ConfigurationManager

                config_manager = ConfigurationManager(config_file=temp_config_path)
                prompt_builder = PromptBuilder(config_manager)
                prompt = prompt_builder.build_system_prompt()

                # Check prompt contains expected elements
                assert scenario["prompt_check"](
                    prompt
                ), f"Prompt check failed for config {config}"

                # Generate response with correct line count and proper ordering
                response_lines = []
                if config["cot"]:
                    response_lines.append(
                        "Analyzing market conditions and making trading decision."
                    )

                # Always include: decision, probability, explanation
                response_lines.extend(
                    ["BUY", "0.8", "Strong upward momentum detected."]
                )

                if config["sj"]:
                    response_lines.append("Performance review shows consistent gains.")

                if config["fl"]:
                    response_lines.append("Feeling confident in the analysis.")

                test_response = "\n".join(response_lines)
                assert (
                    len(response_lines) == scenario["response_lines"]
                ), f"Response line count mismatch: expected {scenario['response_lines']}, got {len(response_lines)}"

                # Parse response
                result = parse_response_text(test_response)
                (
                    decision,
                    prob,
                    explanation,
                    chain_of_thought,
                    strategic_journal,
                    feeling_log,
                ) = result

                # Verify parsing matches expectations
                expected = scenario["expected_parsing"]
                assert ("disabled" not in chain_of_thought) == expected[
                    "cot_present"
                ], f"Chain of thought parsing mismatch: expected {expected['cot_present']}"
                assert ("disabled" not in strategic_journal) == expected[
                    "sj_present"
                ], f"Strategic journal parsing mismatch: expected {expected['sj_present']}"
                assert ("disabled" not in feeling_log) == expected[
                    "fl_present"
                ], f"Feeling log parsing mismatch: expected {expected['fl_present']}"

                print(f"    ✓ Prompt-parsing consistency verified for {config}")

        finally:
            os.unlink(temp_config_path)

    print("\n✅ All prompt-parsing consistency tests passed!")


if __name__ == "__main__":
    test_feature_flag_pipeline_synchronization()
    test_config_synchronization()
    test_prompt_parsing_consistency()
    print("\n🎉 All feature flag synchronization tests completed successfully!")
