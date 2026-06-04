# Master HTML report stylesheet.
#
# Extracted verbatim from report_generator.generate_master_report_html so the
# CSS lives in one place. Interpolated back into the HTML header f-string; the
# golden snapshot (tests/test_report_snapshot.py) guards byte-for-byte parity.

REPORT_CSS = """\
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .section {
            margin-bottom: 50px;
            padding: 30px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #4f46e5;
        }

        .section h2 {
            color: #1e293b;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section h3 {
            color: #334155;
            font-size: 1.4rem;
            font-weight: 500;
            margin: 25px 0 15px 0;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }

        .metric-card .label {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 5px;
        }

        .metric-card .value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
        }

        .metric-card .value.positive {
            color: #10b981;
        }

        .metric-card .value.negative {
            color: #ef4444;
        }

        .chart-container {
            margin: 30px 0;
            text-align: center;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }

        .chart-caption {
            font-style: italic;
            color: #64748b;
            font-size: 0.9rem;
        }

        .data-sources {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .data-source-item {
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .data-source-item::before {
            content: "📊";
            font-size: 1.2rem;
        }

        .code-block {
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 20px 0;
            white-space: pre-wrap;
        }

        .insights-list {
            list-style: none;
            padding: 0;
        }

        .insights-list li {
            padding: 12px 0;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }

        .insights-list li::before {
            content: "💡";
            font-size: 1.1rem;
            flex-shrink: 0;
        }

        .insights-list li:last-child {
            border-bottom: none;
        }

        .footer {
            background: #f1f5f9;
            padding: 30px;
            text-align: center;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
        }

        .footer p {
            margin: 5px 0;
            font-size: 0.9rem;
        }

        /* Beautiful Table Styling */
        .table-container {
            overflow-x: auto;
            margin: 25px 0;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            background: white;
        }

        .performance-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            background: white;
        }

        .performance-table thead th {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            font-weight: 600;
            padding: 14px 18px;
            text-align: left;
            border: none;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .performance-table tbody td {
            padding: 12px 18px;
            border-bottom: 1px solid #f1f5f9;
            transition: background 0.2s ease;
        }

        .performance-table tbody tr:hover {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        .performance-table .positive {
            color: #10b981;
            font-weight: 600;
        }

        .performance-table .negative {
            color: #ef4444;
            font-weight: 600;
        }

        .performance-table .neutral {
            color: #64748b;
        }

        /* Enhanced insights cards */
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }

        .insight-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }

        .insight-card h4 {
            color: #1e293b;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .insight-card ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .insight-card li {
            padding: 8px 0;
            border-bottom: 1px solid #f1f5f9;
            color: #475569;
        }

        .insight-card li:last-child {
            border-bottom: none;
        }

        .insight-card li::before {
            content: "•";
            color: #4f46e5;
            font-weight: bold;
            margin-right: 8px;
        }

        /* Executive Highlights Cards */
        .executive-highlights {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
            padding: 25px;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 16px;
            border: 1px solid #e2e8f0;
        }

        .highlight-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .highlight-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }

        .highlight-card .metric {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4f46e5;
            margin-bottom: 8px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        .highlight-card .label {
            font-size: 1rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 4px;
        }

        .highlight-card .subtext {
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 500;
        }

        @media (max-width: 768px) {
            .header {
                padding: 20px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .content {
                padding: 20px;
            }

            .section {
                padding: 20px;
            }

            .performance-table {
                font-size: 0.8rem;
            }

            .performance-table th,
            .performance-table td {
                padding: 10px 12px;
            }

            .table-container {
                margin: 15px -10px;
            }

            .insights-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }
        }
"""
