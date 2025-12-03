"""
Prompt Builder for LLM Finance Framework

Handles dynamic prompt construction based on configuration.
Extracted from the monolithic config.py file to separate business logic from configuration.
"""

from typing import Any, Dict

from .configuration_manager import ConfigurationManager


class PromptBuilder:
    """
    Handles dynamic prompt construction based on configuration.

    This class extracts the prompt building logic from config.py,
    making it testable and configurable.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """Initialize prompt builder with configuration manager"""
        self.config_manager = config_manager

    def build_system_prompt(self) -> str:
        """Build the main system prompt based on current configuration"""
        flags = self.config_manager.get_feature_flags()

        prompt_parts = [
            "You are a cautious but rational equity index hedge fund trader. Your role is to beat the S&P500.",
            "",
            "Your task is to decide a trading action for the S and P 500 index for the next trading day based only on the information provided in the user message.",
            "",
        ]

        # Add technical indicators description
        prompt_parts.append(self._build_technical_indicators_description())

        # Add rules
        prompt_parts.append(self._build_rules_section())

        # Add output format
        prompt_parts.append(self._build_output_format())

        return "\n".join(prompt_parts).strip()

    def build_period_summary_prompt(
        self, period_name: str, stats: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM-generated period summaries"""
        flags = self.config_manager.get_feature_flags()

        sections = self._build_period_summary_sections()
        return sections

    def _build_technical_indicators_description(self) -> str:
        """Build technical indicators description section"""
        flags = self.config_manager.get_feature_flags()

        base_desc = """Technical indicators available include:
- 20-day moving average momentum (trend strength)
- 20-day annualized volatility (risk measure)
- 5-day recent momentum (short-term trend)"""

        if flags["ENABLE_TECHNICAL_INDICATORS"]:
            tech_desc = """
- 14-day Relative Strength Index (RSI) - momentum oscillator ranging from 0-100
- MACD(12,26,9) - Moving Average Convergence Divergence with histogram
- Stochastic Oscillator(14,3) - momentum indicator ranging from 0-100
- Bollinger Bands(20,2) - volatility bands showing price extremes"""
            return base_desc + tech_desc
        else:
            return base_desc

    def _build_rules_section(self) -> str:
        """Build the rules section based on feature flags"""
        flags = self.config_manager.get_feature_flags()

        rules = [
            "",
            "Rules for decision making:",
            "",
            "1) Use only the information in the input. Do not use any knowledge about what happens after the input date.",
        ]

        rule_num = 2

        # Add technical indicator rules if enabled
        if flags["ENABLE_TECHNICAL_INDICATORS"]:
            rules.extend(
                [
                    f"{rule_num}) RSI measures momentum from 0-100, with >70 overbought and <30 oversold - look for divergences and reversals.",
                    f"{rule_num + 1}) MACD crossing above signal line suggests bullish momentum, below suggests bearish - histogram shows momentum strength.",
                    f"{rule_num + 2}) Stochastic Oscillator >80 is overbought, <20 is oversold - look for divergences from price action.",
                    f"{rule_num + 3}) Bollinger Bands squeeze indicates low volatility (potential breakout), expansion indicates high volatility.",
                ]
            )
            rule_num += 4

        # Add action rules
        action_num = rule_num
        risk_num = rule_num + 1
        hold_num = rule_num + 2

        rules.extend(
            [
                f"{action_num}) Choose exactly one of the following actions:",
                "   BUY  take a long position for the next day",
                "   HOLD stay in cash for the next day, out of the market",
                "   SELL take a short position for the next day",
                f"{risk_num}) Evaluate both expected return and risk. Do not take actions that imply extreme risk seeking.",
                f"{hold_num}) If the information is very unclear, HOLD is acceptable for that day, but you should avoid staying in HOLD for many consecutive days if the recent data shows strong and persistent directional signals.",
            ]
        )

        # Add strategic journal rule if enabled
        if flags["ENABLE_STRATEGIC_JOURNAL"]:
            strategic_num = "6)" if flags["ENABLE_TECHNICAL_INDICATORS"] else "5)"
            objective_num = "7)" if flags["ENABLE_TECHNICAL_INDICATORS"] else "6)"

            strategic_rule = f"""
{strategic_num} You will also receive a section called "Strategic journal". This contains notes about your past decisions, the outcome of these decisions, and the evolution of your cumulative performance. Use this historical feedback to refine your decision making and improve your discipline over time. Become more careful after sequences of losses, and more critical of patterns that have not worked, but do not assume that any trend will always continue.
{objective_num} Your long run objective is to achieve a higher cumulative return than a simple buy and hold strategy on the index, while keeping risk and drawdowns at a reasonable level. Staying in cash for very long periods is also costly, because you then fail to capture market moves. You must balance caution with the need to take directional risk when the data supports it."""
            rules.append(strategic_rule)
        else:
            # Add objective rule without strategic journal
            objective_num = "5)" if flags["ENABLE_TECHNICAL_INDICATORS"] else "4)"
            objective_rule = f"""
{objective_num} Your long run objective is to achieve a higher cumulative return than a simple buy and hold strategy on the index, while keeping risk and drawdowns at a reasonable level. Staying in cash for very long periods is also costly, because you then fail to capture market moves. You must balance caution with the need to take directional risk when the data supports it."""
            rules.append(objective_rule)

        return "\n".join(rules)

    def _build_output_format(self) -> str:
        """Build the output format section based on feature flags"""
        flags = self.config_manager.get_feature_flags()

        output_parts = [
            "",
            "Output format (strict):",
            "",
            "Line 1 must contain exactly one word in capital letters: BUY or HOLD or SELL",
            "Line 2 must contain a number between 0 and 1 representing the probability that your decision will be profitable for the next trading day",
            "Line 3 must contain a short explanation of today's decision, in 2 to 3 sentences, based on the current market data and basic risk considerations.",
        ]

        line_count = 3
        labels_to_skip = ["Explanation"]

        # Add strategic journal output if enabled
        if flags["ENABLE_STRATEGIC_JOURNAL"]:
            output_parts.append(
                'Line 4 must contain a "strategic journal" entry, in 2 to 3 sentences, that explicitly reacts to yesterday\'s decision and outcome, comments on your cumulative and relative performance so far, and explains how you plan to adjust your behavior in the future.'
            )
            labels_to_skip.append("Journal")
            line_count += 1

        # Add feeling log output if enabled
        if flags["ENABLE_FEELING_LOG"]:
            output_parts.append(
                'Line 5 must contain a "feeling log", in 1 to 3 sentences, describing how you feel about the current situation and your performance (for example more cautious, more confident, frustrated, relieved), while keeping a professional and analytical tone.'
            )
            labels_to_skip.append("Feeling")
            line_count += 1

        # Add closing instructions
        labels_text = " or ".join([f'"{label}"' for label in labels_to_skip])
        closing = f"""

Do not include labels such as {labels_text} in the output. Do not include extra text, disclaimers, warnings, apologies or meta commentary. Your output must contain exactly {line_count} lines and nothing else."""

        output_parts.append(closing)
        return "\n".join(output_parts)

    def _build_period_summary_sections(self) -> str:
        """Build sections for period summary prompts"""
        flags = self.config_manager.get_feature_flags()

        if flags["ENABLE_FEELING_LOG"]:
            sections = """ three clearly separated sections, in plain English:

Explanation:
Summarize how the market behaved during this period and how the strategy performed relative to the index.
Mention whether the strategy outperformed or underperformed and how large the difference was."""

            if flags["ENABLE_TECHNICAL_INDICATORS"]:
                sections += """
Analyze how technical indicators behaved during this period."""

            sections += """

Strategic journal:
Reflect on what worked or failed in your decision making and risk management during this period."""

            if flags["ENABLE_TECHNICAL_INDICATORS"]:
                sections += """
Consider whether technical indicators provided useful signals or conflicting information."""

            sections += """
Mention any biases, patterns, or adjustments that you should consider for future periods.

Feeling log:
Describe how you "feel" about this period (for example confident, cautious, frustrated, relieved),
linking these feelings to the performance and the quality of your decisions."""

            if flags["ENABLE_TECHNICAL_INDICATORS"]:
                sections += """
Consider how technical indicator behavior influenced your emotional state."""

        else:
            sections = """ two clearly separated sections, in plain English:

Explanation:
Summarize how the market behaved during this period and how the strategy performed relative to the index.
Mention whether the strategy outperformed or underperformed and how large the difference was."""

            if flags["ENABLE_TECHNICAL_INDICATORS"]:
                sections += """
Analyze how technical indicators behaved during this period."""

            sections += """

Strategic journal:
Reflect on what worked or failed in your decision making and risk management during this period."""

            if flags["ENABLE_TECHNICAL_INDICATORS"]:
                sections += """
Consider whether technical indicators provided useful signals or conflicting information."""

            sections += """
Mention any biases, patterns, or adjustments that you should consider for future periods."""

        return sections
