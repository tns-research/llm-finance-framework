# src/dummy_model.py

import random

from .config_compat import ENABLE_FEELING_LOG, ENABLE_STRATEGIC_JOURNAL


def dummy_call_model(system_prompt: str, user_prompt: str) -> str:
    """
    Dummy model that returns random decisions for testing.
    Generates realistic-looking outputs with random BUY/HOLD/SELL decisions.
    """
    # Random decision
    decisions = ["BUY", "HOLD", "SELL"]
    decision = random.choice(decisions)

    # Random probability between 0.4 and 0.9 (realistic confidence range)
    prob = round(random.uniform(0.4, 0.9), 2)

    # Simple explanations based on decision
    explanations = {
        "BUY": "Market indicators suggest upward momentum. Technical analysis shows bullish patterns.",
        "HOLD": "Market conditions are uncertain. Waiting for clearer signals before taking a position.",
        "SELL": "Risk indicators are elevated. Taking a defensive position to protect capital.",
    }

    explanation = explanations[decision]

    # Build response with appropriate number of lines based on config
    response_lines = [decision, f"{prob:.2f}", explanation]

    if ENABLE_STRATEGIC_JOURNAL:
        strategic_journal = f"Reviewing recent performance and adjusting strategy accordingly. Current market regime suggests {decision.lower()} positioning."
        response_lines.append(strategic_journal)

    if ENABLE_FEELING_LOG:
        feelings = [
            "Feeling cautiously optimistic about current market conditions.",
            "Feeling uncertain but maintaining discipline in decision-making.",
            "Feeling confident in the analysis but aware of potential risks.",
        ]
        feeling = random.choice(feelings)
        response_lines.append(feeling)

    response = "\n".join(response_lines)
    return response
