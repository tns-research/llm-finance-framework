"""
Test Stooq data source integration.
Stooq provides reliable financial data without SSL issues.
"""

from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_sources import StooqDataSource

# Realistic multi-row historical CSV in Stooq's q/d/l format (no Symbol/Time cols).
_FAKE_STOOQ_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-15,185.50,187.25,184.80,186.90,45236789\n"
    "2024-01-16,187.00,188.50,186.20,187.80,38920456\n"
    "2024-01-17,187.90,189.10,186.50,188.40,41002314\n"
)


def _mock_stooq_response(text=_FAKE_STOOQ_CSV, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


class TestStooqOfflineParsing:
    """Deterministic, offline mirror of the live checks above.

    These mock the HTTP layer and exercise the real ``StooqDataSource`` parsing
    path, so the properties the live tests guarded (datetime dates, no nulls,
    numeric OHLCV) stay covered without network access.
    """

    @patch("src.data_sources.requests.get")
    def test_parses_multirow_historical(self, mock_get):
        mock_get.return_value = _mock_stooq_response()
        df = StooqDataSource().fetch_data("AAPL", "2024-01-01", "2024-01-31")
        assert len(df) == 3
        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]

    @patch("src.data_sources.requests.get")
    def test_date_column_is_datetime(self, mock_get):
        mock_get.return_value = _mock_stooq_response()
        df = StooqDataSource().fetch_data("AAPL", "2024-01-01", "2024-01-31")
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert not df["Date"].isna().any()

    @patch("src.data_sources.requests.get")
    def test_ohlcv_numeric_and_complete(self, mock_get):
        mock_get.return_value = _mock_stooq_response()
        df = StooqDataSource().fetch_data("AAPL", "2024-01-01", "2024-01-31")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"
            assert df[col].isna().sum() == 0, f"{col} has nulls"
        assert (df["High"] >= df["Low"]).all()


@pytest.mark.network
class TestStooqDirectAPI:
    """Test Stooq data via direct API calls (most reliable).

    Live: hits stooq.com. Skipped by default; run with ``pytest --run-network``.
    """

    def test_stooq_api_connection(self):
        """Test that Stooq historical API is accessible."""
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"

        response = requests.get(url, timeout=10)
        assert response.status_code == 200, f"Stooq API returned {response.status_code}"

        # Should return CSV data with historical format (no Symbol/Time columns)
        csv_data = response.text
        assert (
            "Date,Open,High,Low,Close,Volume" in csv_data
        ), "Unexpected historical CSV format"

        print("✓ Stooq historical API connection successful")

    def test_aapl_data_retrieval(self):
        """Test retrieving AAPL historical data from Stooq."""
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"

        response = requests.get(url, timeout=10)
        csv_data = response.text

        # Parse CSV
        df = pd.read_csv(StringIO(csv_data))

        # Validate structure - historical format has no Symbol/Time columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Historical data should have multiple rows (not just 1 current quote)
        assert not df.empty, "No data returned"
        assert len(df) >= 10, f"Expected at least 10 historical rows, got {len(df)}"

        # Validate data types
        assert pd.api.types.is_numeric_dtype(df["Open"]), "Open should be numeric"
        assert pd.api.types.is_numeric_dtype(df["Close"]), "Close should be numeric"

        print(f"✓ AAPL historical data retrieved: {len(df)} rows")
        print(f"  Latest: {df.iloc[0]['Close']}")

    def test_spy_data_retrieval(self):
        """Test retrieving SPY (S&P 500 ETF) historical data from Stooq."""
        url = "https://stooq.com/q/d/l/?s=spy.us&i=d"

        response = requests.get(url, timeout=10)
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        # Validate structure - historical format has no Symbol column
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        assert not df.empty, "No SPY data returned"
        assert len(df) >= 10, f"Expected at least 10 historical rows, got {len(df)}"

        print(f"✓ SPY historical data retrieved: {len(df)} rows")
        print(f"  Latest: {df.iloc[0]['Close']}")

    def test_multiple_symbols(self):
        """Test retrieving historical data for multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "SPY"]

        for symbol in symbols:
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"

            response = requests.get(url, timeout=10)
            assert response.status_code == 200, f"Failed for {symbol}"

            csv_data = response.text
            df = pd.read_csv(StringIO(csv_data))

            assert not df.empty, f"No data for {symbol}"
            assert (
                len(df) >= 5
            ), f"Expected at least 5 historical rows for {symbol}, got {len(df)}"

            print(f"✓ {symbol}: {len(df)} historical rows")

    def test_date_parsing(self):
        """Test that historical dates are properly formatted."""
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"

        response = requests.get(url, timeout=10)
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        # Parse date column
        df["parsed_date"] = pd.to_datetime(df["Date"])

        # Should not have NaT values
        assert not df["parsed_date"].isna().any(), "Some dates failed to parse"

        # Historical data should have dates (latest should be recent)
        latest_date = df["parsed_date"].max()
        oldest_date = df["parsed_date"].min()

        # Should span at least a few days of historical data
        assert (
            latest_date - oldest_date
        ).days >= 3, "Historical data should span multiple days"
        assert latest_date >= datetime.now() - timedelta(
            days=30
        ), f"Latest data too old: {latest_date}"

        print(
            f"✓ Date parsing successful: {oldest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
        )

    def test_data_quality(self):
        """Test that historical data values are reasonable."""
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"

        response = requests.get(url, timeout=10)
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        # Historical data comes oldest first, so check most recent data (last few rows)
        # and some middle data to ensure quality across the time series
        recent_indices = [-1, -2, -3]  # Last 3 rows (most recent)

        for i in recent_indices:
            if abs(i) <= len(df):  # Make sure we don't go out of bounds
                idx = len(df) + i if i < 0 else i  # Convert negative index to positive

                # AAPL current prices should be reasonable ($150-300 range typically)
                close_price = df.iloc[idx]["Close"]
                assert (
                    100 <= close_price <= 500
                ), f"Unreasonable recent AAPL price: ${close_price} at row {idx}"

                # Volume should be positive
                volume = df.iloc[idx]["Volume"]
                assert volume > 0, f"Volume should be positive: {volume} at row {idx}"

                # High should be >= Low
                assert (
                    df.iloc[idx]["High"] >= df.iloc[idx]["Low"]
                ), f"High should be >= Low at row {idx}"

                # Open and Close should be within High-Low range
                high = df.iloc[idx]["High"]
                low = df.iloc[idx]["Low"]
                open_price = df.iloc[idx]["Open"]
                close_price = df.iloc[idx]["Close"]

                assert (
                    low <= open_price <= high
                ), f"Open price {open_price} not within range [{low}, {high}] at row {idx}"
                assert (
                    low <= close_price <= high
                ), f"Close price {close_price} not within range [{low}, {high}] at row {idx}"

        print("✓ Historical data quality checks passed")


class TestStooqDataTransformation:
    """Test transforming Stooq data to our expected format."""

    def test_column_mapping(self):
        """Test mapping Stooq historical columns to our expected OHLCV format."""
        # Simulate historical Stooq data (no Symbol/Time columns)
        stooq_data = {
            "Date": ["2024-01-15", "2024-01-16"],
            "Open": [185.50, 187.00],
            "High": [187.25, 188.50],
            "Low": [184.80, 186.20],
            "Close": [186.90, 187.80],
            "Volume": [45236789, 38920456],
        }
        df_stooq = pd.DataFrame(stooq_data)

        # Transform to our format (what our data source would do)
        df_our_format = pd.DataFrame(
            {
                "Date": pd.to_datetime(df_stooq["Date"]),
                "Open": df_stooq["Open"],
                "High": df_stooq["High"],
                "Low": df_stooq["Low"],
                "Close": df_stooq["Close"],
                "Volume": df_stooq["Volume"],
            }
        )

        # Validate our expected columns exist
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in expected_cols:
            assert col in df_our_format.columns, f"Missing column: {col}"

        # Validate data types
        assert pd.api.types.is_datetime64_any_dtype(
            df_our_format["Date"]
        ), "Date should be datetime"
        assert pd.api.types.is_numeric_dtype(
            df_our_format["Close"]
        ), "Close should be numeric"

        print("✓ Historical column mapping to our format successful")

    def test_symbol_cleaning(self):
        """Test cleaning Stooq symbol format."""
        stooq_symbol = "AAPL.US"
        our_symbol = stooq_symbol.replace(".US", "")

        assert our_symbol == "AAPL", f"Symbol cleaning failed: {our_symbol}"

        print("✓ Symbol cleaning successful")


@pytest.mark.network
class TestStooqHistoricalData:
    """Test retrieving historical data from Stooq.

    Live: hits stooq.com. Skipped by default; run with ``pytest --run-network``.
    """

    def test_historical_api(self):
        """Test Stooq's historical data API."""
        # Note: Stooq's free API gives daily data, not minute-level
        # The q/d/l endpoint provides actual historical time series

        # Test with historical endpoint
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"

        response = requests.get(url, timeout=10)
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        # Should have multiple days of historical trading data
        assert (
            len(df) >= 10
        ), f"Should have at least 10 historical data points, got {len(df)}"

        print(f"✓ Historical API works: {len(df)} data points available")

    def test_data_completeness(self):
        """Test that historical data is complete OHLCV."""
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"

        response = requests.get(url, timeout=10)
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        # Check for null values in critical columns across historical data
        critical_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in critical_cols:
            null_count = df[col].isna().sum()
            assert (
                null_count == 0
            ), f"Column {col} has {null_count} null values in historical data"

        print("✓ Historical data completeness check passed")


# Standalone test function
def run_stooq_tests():
    """Run basic Stooq historical data tests manually."""
    print("Testing Stooq historical data source...")

    # Test basic connectivity with historical data
    url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        print(f"SUCCESS: Got {len(df)} rows of AAPL historical data")
        print("Sample data:")
        print(df.head())

        # Test SPY historical data
        url_spy = "https://stooq.com/q/d/l/?s=spy.us&i=d"
        response_spy = requests.get(url_spy, timeout=10)

        if response_spy.status_code == 200:
            csv_data_spy = response_spy.text
            df_spy = pd.read_csv(StringIO(csv_data_spy))
            print(f"SPY historical data: {len(df_spy)} rows")
            print("SUCCESS: Stooq provides reliable historical data!")
        else:
            print("SPY request failed")
    else:
        print(f"Stooq API failed: {response.status_code}")


if __name__ == "__main__":
    run_stooq_tests()
