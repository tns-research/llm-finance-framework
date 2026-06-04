# src/report/sections/insights.py
"""Insights and recommendations section: markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


def generate_insights_recommendations(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate insights and recommendations section."""
    lines = [
        "## 💡 Key Insights & Recommendations",
        "",
    ]

    # Extract insights from statistical validation
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]

        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]

            lines.append("### Key Findings")
            for finding in sa.get("key_findings", []):
                lines.append(f"- {finding}")

            lines.append("")
            lines.append("### Recommendations")
            for rec in sa.get("recommendations", []):
                lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "### Overall Assessment",
        ]
    )

    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]
            lines.extend(
                [
                    f"- **Assessment**: {sa.get('overall_assessment', 'Unknown').upper()}",
                    f"- **Confidence Level**: {sa.get('confidence_level', 'Unknown').upper()}",
                ]
            )

    lines.extend(["", "---", ""])

    return lines


def generate_insights_recommendations_html(data_sources: Dict, model_tag: str) -> str:
    """Generate insights and recommendations section in HTML."""
    html = """            <div class="section">
                <h2>💡 Key Insights & Recommendations</h2>
"""

    # Extract insights from statistical validation
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]

        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]

            html += """
                <h3>Key Findings</h3>
                <ul class="insights-list">
"""
            for finding in sa.get("key_findings", []):
                html += f"                    <li>{finding}</li>\n"
            html += "                </ul>\n"

            html += """
                <h3>Recommendations</h3>
                <ul class="insights-list">
"""
            for rec in sa.get("recommendations", []):
                html += f"                    <li>{rec}</li>\n"
            html += "                </ul>\n"

    html += """
                <h3>Overall Assessment</h3>
                <div class="metric-grid">
"""

    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]
            html += f"""
                    <div class="metric-card">
                        <div class="label">Assessment</div>
                        <div class="value">{sa.get('overall_assessment', 'Unknown').upper()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Confidence Level</div>
                        <div class="value">{sa.get('confidence_level', 'Unknown').upper()}</div>
                    </div>
"""

    html += """
                </div>
            </div>
"""

    return html
