# src/dummy_model.py

import random


def dummy_call_model(system_prompt: str, user_prompt: str) -> str:
    """
    Dummy model that returns random decisions for testing.
    Generates realistic-looking outputs with random BUY/HOLD/SELL decisions.
    Supports all feature flag combinations for comprehensive testing.
    """
    # Imported at call time (not module level) so tests can patch these flags
    # on src.config and have the patched values take effect here.
    from .config import (
        ENABLE_CHAIN_OF_THOUGHT,
        ENABLE_FEELING_LOG,
        ENABLE_STRATEGIC_JOURNAL,
    )

    # Generate random decision and probability
    decisions = ["BUY", "HOLD", "SELL"]
    decision = random.choice(decisions)
    prob = round(random.uniform(0.4, 0.9), 2)

    # Validate decision and probability
    if decision not in decisions:
        raise ValueError(f"Invalid decision generated: {decision}")
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"Invalid probability generated: {prob}")

    # Build response lines in correct order based on enabled features
    response_lines = []

    # Add chain of thought if enabled (Line 1 when present)
    if ENABLE_CHAIN_OF_THOUGHT:
        chain_of_thought = "Analyzing market conditions: RSI shows neutral conditions, MACD indicates weak momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning."
        response_lines.append(chain_of_thought)

    # Add decision, probability, explanation (shifted indices when chain of thought enabled)
    explanations = {
        "BUY": "Market indicators suggest upward momentum. Technical analysis shows bullish patterns.",
        "HOLD": "Market conditions are uncertain. Waiting for clearer signals before taking a position.",
        "SELL": "Risk indicators are elevated. Taking a defensive position to protect capital.",
    }
    explanation = explanations[decision]

    response_lines.extend([decision, f"{prob:.2f}", explanation])

    # Add strategic journal if enabled
    if ENABLE_STRATEGIC_JOURNAL:
        strategic_responses = [
            "Reviewing recent performance and adjusting strategy accordingly.",
            "Maintaining consistent approach while monitoring market conditions.",
            "Strategic position confirmed based on current analysis.",
        ]
        response_lines.append(random.choice(strategic_responses))

    # Add feeling log if enabled
    if ENABLE_FEELING_LOG:
        feeling_responses = [
            "Feeling cautiously optimistic about current market conditions.",
            "Maintaining disciplined approach despite uncertainty.",
            "Confident in analytical process and risk management.",
        ]
        response_lines.append(random.choice(feeling_responses))

    response = "\n".join(response_lines)

    # Validation: Ensure correct number of lines based on enabled features
    expected_lines = 3  # Base: decision, prob, explanation
    if ENABLE_CHAIN_OF_THOUGHT:
        expected_lines += 1
    if ENABLE_STRATEGIC_JOURNAL:
        expected_lines += 1
    if ENABLE_FEELING_LOG:
        expected_lines += 1

    actual_lines = len(response.splitlines())
    if actual_lines != expected_lines:
        raise ValueError(
            f"Dummy model output {actual_lines} lines, expected {expected_lines}. "
            f"Config: COT={ENABLE_CHAIN_OF_THOUGHT}, Journal={ENABLE_STRATEGIC_JOURNAL}, Feeling={ENABLE_FEELING_LOG}"
        )

    return response
