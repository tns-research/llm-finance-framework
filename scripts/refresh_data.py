#!/usr/bin/env python3
"""
Regenerate the vendored, frozen dataset (data/raw/spy_daily.csv) and its manifest.

Refreshing is a DELIBERATE action: it changes the file's SHA-256, so commit the
new spy_daily.csv and MANIFEST.json together and update data/raw/PROVENANCE.md.

Default source is yfinance (no API key, split/dividend-adjusted). If you prefer a
keyed provider (Tiingo, Alpha Vantage), fetch into the same
Date,Open,High,Low,Close,Volume schema and reuse the write/manifest logic below.

Usage:
    python scripts/refresh_data.py                # SPY, 2015-01-01..2024-01-01
    python scripts/refresh_data.py --symbol QQQ --start 2016-01-01 --end 2024-01-01
"""

import argparse
import hashlib
import json
import os
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "spy_daily.csv")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "MANIFEST.json")


def fetch_yfinance(symbol: str, start: str, end: str):
    import pandas as pd
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise SystemExit(f"yfinance returned no data for {symbol} ({start}..{end})")
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(4)
    df["Volume"] = df["Volume"].astype("int64")
    return df, yf.__version__


def main():
    parser = argparse.ArgumentParser(description="Refresh the vendored dataset.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    args = parser.parse_args()

    df, yf_version = fetch_yfinance(args.symbol, args.start, args.end)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False, lineterminator="\n")

    raw = open(DATA_PATH, "rb").read()
    manifest = {
        "file": os.path.basename(DATA_PATH),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": int(len(df)),
        "symbol": args.symbol,
        "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
        "date_start": df["Date"].iloc[0],
        "date_end": df["Date"].iloc[-1],
        "adjustment": "split_and_dividend_adjusted",
        "source": "yfinance",
        "source_version": yf_version,
        "frozen_at": date.today().isoformat(),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(df)} rows -> {DATA_PATH}")
    print(f"  range : {manifest['date_start']} .. {manifest['date_end']}")
    print(f"  sha256: {manifest['sha256']}")
    print("Remember to update data/raw/PROVENANCE.md and commit both files.")


if __name__ == "__main__":
    main()
