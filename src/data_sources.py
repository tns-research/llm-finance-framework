"""
Data source abstraction for financial data.
Supports a vendored offline snapshot (default), Stooq, and CSV fallback.

LEGAL NOTICE:
- Stooq data is owned by Stooq and subject to their Terms of Service
- This code only provides access mechanism, no data redistribution
- Users must comply with Stooq's terms when using this functionality
"""

import hashlib
import json
import logging
import os
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)

# Canonical OHLCV column names returned by every data source.
_CANONICAL_OHLCV_COLUMNS = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


class DataSourceError(Exception):
    """Base exception for data source errors."""

    pass


class VendoredDataSource:
    """
    Reads the committed, frozen snapshot (default data source).

    The snapshot ships in the repo so a clean clone runs offline and
    deterministically. The file is verified against the SHA-256 recorded in
    MANIFEST.json on every load: a corrupted or modified snapshot fails loudly
    rather than silently changing results.
    """

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        data_path = getattr(config, "VENDORED_DATA_PATH", "data/raw/spy_daily.csv")
        manifest_path = getattr(
            config, "VENDORED_MANIFEST_PATH", "data/raw/MANIFEST.json"
        )

        if not os.path.exists(data_path):
            raise DataSourceError(
                f"Vendored dataset not found: {data_path}. "
                "It should be committed to the repo (see data/raw/PROVENANCE.md)."
            )
        if not os.path.exists(manifest_path):
            raise DataSourceError(f"Vendored manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        expected = manifest.get("sha256")
        actual = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
        if expected != actual:
            raise DataSourceError(
                f"Vendored dataset checksum mismatch for {data_path}.\n"
                f"  expected (manifest): {expected}\n"
                f"  actual (file):       {actual}\n"
                "The snapshot was modified or corrupted. Restore it from git, or "
                "run scripts/refresh_data.py to regenerate it and update the manifest."
            )

        from .data_prep import load_raw_csv

        df = load_raw_csv(data_path)
        df = df.rename(columns=_CANONICAL_OHLCV_COLUMNS)
        return df


class CSVDataSource:
    """Data source for local CSV files."""

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from local CSV file."""
        # Use new variable name with backward compatibility
        csv_path = getattr(
            config,
            "CSV_DATA_PATH",
            getattr(config, "CSV_FALLBACK_PATH", "data/raw/sp500.csv"),
        )

        if not os.path.exists(csv_path):
            raise DataSourceError(f"CSV file not found: {csv_path}")

        # Use existing load_raw_csv logic
        from .data_prep import load_raw_csv

        df = load_raw_csv(csv_path)

        # load_raw_csv returns lowercase columns for internal processing,
        # but we need to return the format expected by the CSV saving process
        # Convert back to capital column names for consistency
        df = df.rename(columns=_CANONICAL_OHLCV_COLUMNS)

        return df


class StooqDataSource:
    """Data source for Stooq (reliable financial data)."""

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Stooq API."""
        try:
            # Use symbol directly (user specifies exact Stooq symbol in config)
            # Add .us suffix if not present (Stooq convention)
            stooq_symbol = symbol.lower()
            if not stooq_symbol.endswith(".us") and "-" not in stooq_symbol:
                stooq_symbol += ".us"

            # Stooq HISTORICAL CSV API (q/d/l endpoint)
            # Use date range if provided, otherwise get full history
            if start_date and end_date:
                # Convert dates to YYYYMMDD format for Stooq API
                start_fmt = start_date.replace("-", "")
                end_fmt = end_date.replace("-", "")
                url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&d1={start_fmt}&d2={end_fmt}&i=d"
            else:
                # Full historical range
                url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"

            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                raise DataSourceError(
                    f"Stooq API returned status {response.status_code}"
                )

            csv_data = response.text

            # Validate response - historical endpoint doesn't include 'Symbol' in CSV header
            if not csv_data.strip() or "Date" not in csv_data:
                raise DataSourceError(
                    f"Invalid response from Stooq for symbol {symbol}"
                )

            # Parse CSV
            df = pd.read_csv(StringIO(csv_data))

            if df.empty:
                raise DataSourceError(
                    f"No data returned from Stooq for symbol {symbol}"
                )

            # Validate required columns - historical format: Date,Open,High,Low,Close,Volume
            # (no Symbol or Time columns)
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise DataSourceError(f"Stooq response missing columns: {missing_cols}")

            # Transform to our expected format (load_raw_csv expects capital column names)
            df_our_format = pd.DataFrame(
                {
                    "Date": pd.to_datetime(df["Date"]),
                    "Open": df["Open"],
                    "High": df["High"],
                    "Low": df["Low"],
                    "Close": df["Close"],
                    "Volume": df["Volume"],
                }
            )

            return df_our_format

        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Network error accessing Stooq: {str(e)}")
        except Exception as e:
            raise DataSourceError(f"Stooq error: {str(e)}")


class DataSourceManager:
    """Manages data sources with fallback support."""

    def __init__(self):
        self.sources = {
            "vendored": VendoredDataSource(),
            "csv": CSVDataSource(),
            "stooq": StooqDataSource(),
        }

    def get_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get data from configured source with fallback."""
        # Use config defaults if not provided
        primary_source = getattr(config, "DATA_SOURCE", "csv")

        # Show active configuration at runtime
        if primary_source == "vendored":
            vendored_path = getattr(
                config, "VENDORED_DATA_PATH", "data/raw/spy_daily.csv"
            )
            logger.info("Loading vendored snapshot: %s", vendored_path)
        elif primary_source == "stooq":
            stooq_symbol = getattr(config, "STOOQ_SYMBOL", "SPY")
            date_start = start_date or getattr(config, "DATA_START", "2015-01-01")
            date_end = end_date or getattr(config, "DATA_END", "2023-12-31")
            logger.info(
                "Fetching historical data: %s (%s to %s)",
                stooq_symbol,
                date_start,
                date_end,
            )
        else:
            csv_path = getattr(
                config,
                "CSV_DATA_PATH",
                getattr(config, "CSV_FALLBACK_PATH", "data/raw/sp500.csv"),
            )
            logger.info("Loading CSV data from: %s", csv_path)

        # For Stooq, use STOOQ_SYMBOL config instead of generic SYMBOL
        if primary_source == "stooq":
            symbol = getattr(config, "STOOQ_SYMBOL", "SPY")
        elif symbol is None:
            symbol = getattr(config, "SYMBOL", "^GSPC")

        if start_date is None:
            start_date = getattr(config, "DATA_START", "2015-01-01")
        if end_date is None:
            end_date = getattr(config, "DATA_END", "2023-12-31")

        # Determine fallback source
        fallback_source = "stooq" if primary_source == "csv" else "csv"

        # Try primary source
        try:
            source = self.sources[primary_source]
            logger.info("Fetching from %s: %s", primary_source, symbol)
            df = source.fetch_data(symbol, start_date, end_date)
            logger.info("Success: %d rows from %s", len(df), primary_source)
            return df

        except DataSourceError as e:
            logger.warning("Primary source %s failed: %s", primary_source, e)

            # Vendored is offline-by-design: never silently fall back to network.
            if primary_source == "vendored":
                raise DataSourceError(f"Vendored data source failed: {e}")

            # Only try fallback if primary source wasn't Stooq
            if primary_source == "stooq":
                raise DataSourceError(f"Stooq data source failed: {e}")

            # Try fallback (only for CSV primary source)
            try:
                logger.info("Trying fallback: %s", fallback_source)
                # For fallback, use the generic symbol (not STOOQ_SYMBOL)
                fallback_symbol = getattr(config, "SYMBOL", "^GSPC")
                df = self.sources[fallback_source].fetch_data(
                    fallback_symbol, start_date, end_date
                )
                logger.info(
                    "Fallback success: %d rows from %s", len(df), fallback_source
                )
                return df
            except DataSourceError as fallback_e:
                logger.error("Fallback also failed: %s", fallback_e)
                raise DataSourceError("All data sources failed")


# Global instance for easy access
data_manager = DataSourceManager()
