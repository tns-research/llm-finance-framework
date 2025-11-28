#!/usr/bin/env python3
"""
Simple wrapper script to generate comprehensive reports.
Run from the project root directory.
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from report_generator import generate_comprehensive_report

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive experiment report')
    parser.add_argument('model_tag', nargs='?', default='dummy_model_memory_only',
                       help='Model identifier (default: dummy_model_memory_only)')
    parser.add_argument('--format', '-f', choices=['markdown', 'html'], default='markdown',
                       help='Output format (default: markdown)')
    parser.add_argument('--no-charts', action='store_true',
                       help='Skip generating additional charts')

    args = parser.parse_args()

    print(f"Generating {args.format.upper()} report for model: {args.model_tag}")

    try:
        report_path = generate_comprehensive_report(
            model_tag=args.model_tag,
            include_additional_charts=not args.no_charts,
            output_format=args.format
        )
        print(f"✓ Comprehensive report generated: {report_path}")

        # If HTML, suggest opening it
        if args.format == 'html':
            print(f"🌐 Open the report in your browser: file://{os.path.abspath(report_path)}")

    except Exception as e:
        print(f"✗ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
