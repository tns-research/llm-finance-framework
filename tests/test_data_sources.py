"""Test the data source abstraction layer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_sources import (
    CSVDataSource,
    DataSourceError,
    DataSourceManager,
    StooqDataSource,
)


class TestDataSourceManager:
    """Test the data source manager."""

    def test_init(self):
        """Test manager initializes correctly."""
        manager = DataSourceManager()
        assert "csv" in manager.sources
        assert "stooq" in manager.sources

    def test_csv_fallback_only(self):
        """Test CSV source works."""
        manager = DataSourceManager()

        # Mock CSV data
        test_df = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [103, 104],
                "Volume": [1000, 1100],
            }
        )

        with patch.object(CSVDataSource, "fetch_data", return_value=test_df):
            with patch("src.data_sources.config") as mock_config:
                mock_config.DATA_SOURCE = "csv"
                mock_config.SYMBOL = "TEST"
                result = manager.get_data("TEST", "2023-01-01", "2023-01-02")
                assert len(result) == 2
                assert list(result.columns) == [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]


class TestStooqDataSource:
    """Test Stooq data source specifically."""

    @patch("src.data_sources.requests.get")
    def test_stooq_successful_fetch(self, mock_get):
        """Test successful Stooq historical data fetch."""
        # Mock successful response with historical format (no Symbol/Time columns)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-15,185.50,187.25,184.80,186.90,45236789
2024-01-16,187.00,188.50,186.20,187.80,38920456"""
        mock_get.return_value = mock_response

        stooq_source = StooqDataSource()
        result = stooq_source.fetch_data("AAPL", "2024-01-01", "2024-01-31")

        assert len(result) == 2  # Historical data returns multiple rows
        assert list(result.columns) == [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
        assert result.iloc[0]["Close"] == 186.90
        assert result.iloc[1]["Close"] == 187.80

    @patch("src.data_sources.requests.get")
    def test_stooq_network_error(self, mock_get):
        """Test Stooq network error handling."""
        mock_get.side_effect = Exception("Network timeout")

        stooq_source = StooqDataSource()

        with pytest.raises(DataSourceError, match="Stooq error: Network timeout"):
            stooq_source.fetch_data("AAPL", "2024-01-01", "2024-01-31")

    @patch("src.data_sources.requests.get")
    def test_stooq_symbol_formatting(self, mock_get):
        """Test Stooq symbol formatting (.us suffix added when needed)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-15,485.50,487.25,484.80,486.90,45236789"""
        mock_get.return_value = mock_response

        stooq_source = StooqDataSource()

        # Test symbol without .us suffix
        result = stooq_source.fetch_data("SPY", "2024-01-01", "2024-01-31")
        mock_get.assert_called_with(
            "https://stooq.com/q/d/l/?s=spy.us&d1=20240101&d2=20240131&i=d", timeout=10
        )

        # Reset mock
        mock_get.reset_mock()

        # Test symbol with .us suffix already
        result = stooq_source.fetch_data("SPY.US", "2024-01-01", "2024-01-31")
        mock_get.assert_called_with(
            "https://stooq.com/q/d/l/?s=spy.us&d1=20240101&d2=20240131&i=d", timeout=10
        )

        # Reset mock
        mock_get.reset_mock()

        # Test crypto symbol (should not add .us)
        result = stooq_source.fetch_data("BTC-USD", "2024-01-01", "2024-01-31")
        mock_get.assert_called_with(
            "https://stooq.com/q/d/l/?s=btc-usd&d1=20240101&d2=20240131&i=d", timeout=10
        )


class TestDataSourceIntegration:
    """Test integration between data sources."""

    @patch("src.data_sources.requests.get")
    def test_stooq_no_fallback_on_failure(self, mock_get):
        """Test that Stooq does NOT fallback to CSV when it fails."""
        manager = DataSourceManager()

        # Mock Stooq failure
        mock_get.side_effect = Exception("Stooq network error")

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "SPY"

            # Should raise error immediately, not try CSV fallback
            with pytest.raises(DataSourceError, match="Stooq data source failed"):
                manager.get_data()


if __name__ == "__main__":
    print("Running data source tests...")

    # Run basic tests
    manager = DataSourceManager()
    print(
        f"Data source manager initialized with sources: {list(manager.sources.keys())}"
    )

    # Test CSV mock
    print("Testing CSV fallback...")
    # Add more manual tests as needed
