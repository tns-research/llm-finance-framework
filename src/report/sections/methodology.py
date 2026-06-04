# src/report/sections/methodology.py
"""Methodology section: static technical-implementation narrative (markdown/HTML renderers).

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


def generate_methodology_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate comprehensive methodology section explaining technical indicators and dual criteria HOLD evaluation."""
    lines = [
        "## 🔬 Methodology & Technical Implementation",
        "",
        "### Technical Indicator Framework",
        "- **RSI(14)**: Momentum oscillator for overbought/oversold conditions (thresholds: 30/70)",
        "- **MACD(12,26,9)**: Trend-following momentum indicator with signal line crossovers",
        "- **Stochastic(14,3)**: Price momentum relative to recent trading range (thresholds: 20/80)",
        "- **Bollinger Bands(20,2)**: Volatility-based support/resistance levels with position tracking",
        "",
        "### Enhanced HOLD Evaluation",
        "**Previous Method**: Simple next-day return > 0 (resulted in 0% success rate)",
        "",
        "**New Dual Criteria**:",
        "1. **Quiet Market Success**: Performance in low-volatility environments (<0.2% daily moves)",
        "2. **Risk Avoidance**: Protection against significant losses in uncertain conditions (>2% potential loss)",
        "3. **Context Adjustment**: Volatility and regime-aware evaluation with weighted scoring",
        "",
        "### Signal Integration Approach",
        "- **Consensus Framework**: Multiple indicators must align for high-confidence signals",
        "- **Weighting System**: Different indicators weighted by historical effectiveness",
        "- **Contrarian Filtering**: System identifies when to fade vs. follow technical extremes",
        "- **Adaptive Thresholds**: Dynamic signal strength based on market volatility",
        "",
        "### Decision Framework Architecture",
        "- **Multi-Modal Input**: Technical indicators + LLM reasoning + risk metrics",
        "- **Probabilistic Outputs**: Confidence scores for BUY/HOLD/SELL decisions",
        "- **Context Awareness**: Market regime detection and volatility adjustment",
        "- **Memory Integration**: Historical performance feedback for adaptive learning",
        "",
        "---",
        "",
    ]

    return lines


def generate_methodology_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate HTML methodology section explaining technical indicators and dual criteria HOLD evaluation."""
    html = f"""
    <div class="section">
        <h2>🔬 Methodology & Technical Implementation</h2>

        <h3>Technical Indicator Framework</h3>
        <ul>
            <li><strong>RSI(14):</strong> Momentum oscillator for overbought/oversold conditions (thresholds: 30/70)</li>
            <li><strong>MACD(12,26,9):</strong> Trend-following momentum indicator with signal line crossovers</li>
            <li><strong>Stochastic(14,3):</strong> Price momentum relative to recent trading range (thresholds: 20/80)</li>
            <li><strong>Bollinger Bands(20,2):</strong> Volatility-based support/resistance levels with position tracking</li>
        </ul>

        <h3>Enhanced HOLD Evaluation</h3>
        <p><strong>Previous Method:</strong> Simple next-day return > 0 (resulted in 0% success rate)</p>

        <p><strong>New Dual Criteria:</strong></p>
        <ol>
            <li><strong>Quiet Market Success:</strong> Performance in low-volatility environments (<0.2% daily moves)</li>
            <li><strong>Risk Avoidance:</strong> Protection against significant losses in uncertain conditions (>2% potential loss)</li>
            <li><strong>Context Adjustment:</strong> Volatility and regime-aware evaluation with weighted scoring</li>
        </ol>

        <h3>Signal Integration Approach</h3>
        <ul>
            <li><strong>Consensus Framework:</strong> Multiple indicators must align for high-confidence signals</li>
            <li><strong>Weighting System:</strong> Different indicators weighted by historical effectiveness</li>
            <li><strong>Contrarian Filtering:</strong> System identifies when to fade vs. follow technical extremes</li>
            <li><strong>Adaptive Thresholds:</strong> Dynamic signal strength based on market volatility</li>
        </ul>

        <h3>Decision Framework Architecture</h3>
        <ul>
            <li><strong>Multi-Modal Input:</strong> Technical indicators + LLM reasoning + risk metrics</li>
            <li><strong>Probabilistic Outputs:</strong> Confidence scores for BUY/HOLD/SELL decisions</li>
            <li><strong>Context Awareness:</strong> Market regime detection and volatility adjustment</li>
            <li><strong>Memory Integration:</strong> Historical performance feedback for adaptive learning</li>
        </ul>
    </div>
    """

    return html
