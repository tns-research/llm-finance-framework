"""Statistical validation section.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from io import StringIO
from typing import Dict, List

from ...statistical_validation import print_validation_report


def _capture_validation_report_text(stat_validation, model_tag):
    """Capture print_validation_report's console output as a string.

    Shared by the markdown and HTML statistical-validation renderers.
    """
    import sys

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    try:
        print_validation_report(stat_validation, model_tag)
        return captured_output.getvalue()
    finally:
        sys.stdout = old_stdout


def generate_statistical_validation_section(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate statistical validation section."""
    lines = [
        "## 📊 Statistical Validation",
        "",
    ]

    if "statistical_validation" in data_sources:
        validation_text = _capture_validation_report_text(
            data_sources["statistical_validation"], model_tag
        )

        # Convert the console output to markdown
        lines.extend(validation_text.split("\n"))
        lines.append("")

    # Include statistical visualization if available
    if (
        "statistical_plots" in data_sources
        and "statistical_validation" in data_sources["statistical_plots"]
    ):
        lines.extend(
            [
                f"![Statistical Validation Visualization]({data_sources['statistical_plots']['statistical_validation']})",
                "*Figure 3: Bootstrap distribution and confidence intervals*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_statistical_validation_section_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate statistical validation section in HTML."""
    html = """            <div class="section">
                <h2>📊 Statistical Validation</h2>
"""

    if "statistical_validation" in data_sources:
        validation_text = _capture_validation_report_text(
            data_sources["statistical_validation"], model_tag
        )

        # Convert the console output to HTML
        html += f"""
                <div class="code-block">{validation_text}</div>
"""

    # Include statistical visualization if available
    if (
        "statistical_plots" in data_sources
        and "statistical_validation" in data_sources["statistical_plots"]
    ):
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['statistical_plots']['statistical_validation']}" alt="Statistical Validation Visualization">
                    <div class="chart-caption">Figure 3: Bootstrap distribution and confidence intervals</div>
                </div>
"""

    html += "            </div>\n"
    return html
