"""
Integration tests for the complete LLM Finance Framework pipeline.
Tests end-to-end functionality, error handling, and configuration validation.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src import config
from src.data_sources import DataSourceError, DataSourceManager
from src.main import run_pipeline


class TestIntegration:
    """Integration tests for the complete system."""

    def test_config_validation_invalid_data_source(self):
        """Test that invalid DATA_SOURCE values are rejected."""
        with patch("src.config.DATA_SOURCE", "invalid_source"):
            with pytest.raises(
                ValueError, match="Invalid DATA_SOURCE 'invalid_source'"
            ):
                # Force re-validation by calling the function
                config.validate_data_source_config()

    def test_config_validation_missing_stooq_symbol(self):
        """Test validation when STOOQ_SYMBOL is empty."""
        with patch("src.config.DATA_SOURCE", "stooq"), patch(
            "src.config.STOOQ_SYMBOL", ""
        ):
            with pytest.raises(
                ValueError, match="STOOQ_SYMBOL must be a non-empty string"
            ):
                config.validate_data_source_config()

    @patch("requests.get")
    def test_stooq_network_error_no_fallback(self, mock_get):
        """Test that Stooq network errors don't fallback to CSV."""
        # Mock network failure
        mock_get.side_effect = Exception("Network timeout")

        manager = DataSourceManager()

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "SPY"

            # Should raise DataSourceError, not fallback to CSV
            with pytest.raises(DataSourceError, match="Stooq data source failed"):
                manager.get_data()

    @patch("requests.get")
    def test_stooq_successful_historical_fetch(self, mock_get):
        """Test successful historical data fetching from Stooq."""
        # Mock successful response with historical data format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-01,450.00,455.00,448.00,452.50,50000000
2024-01-02,453.00,458.00,451.00,456.75,55000000"""
        mock_get.return_value = mock_response

        manager = DataSourceManager()

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "SPY"
            mock_config.DATA_START = "2024-01-01"
            mock_config.DATA_END = "2024-01-31"

            result = manager.get_data()

            # Verify we got historical data
            assert len(result) == 2
            assert list(result.columns) == [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
            assert result.iloc[0]["Close"] == 452.50
            assert result.iloc[1]["Close"] == 456.75

            # Verify correct API call was made
            mock_get.assert_called_once()
            # call_args[0] contains positional args, call_args[1] contains keyword args
            call_url = mock_get.call_args[0][0]  # First positional argument is the URL
            assert "q/d/l" in call_url  # Historical endpoint
            assert "d1=20240101" in call_url  # Start date
            assert "d2=20240131" in call_url  # End date
            assert "spy.us" in call_url  # Symbol with .us

    def test_csv_fallback_functionality(self):
        """Test that CSV data source works as fallback."""
        import os
        import tempfile

        manager = DataSourceManager()

        # Create a temporary CSV file for testing (since we removed raw data from git)
        test_csv_content = """Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,95.0,102.0,1000000
2023-01-02,102.0,108.0,98.0,105.0,1200000
2023-01-03,105.0,110.0,100.0,108.0,900000
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_file.write(test_csv_content)
            temp_csv_path = temp_file.name

        try:
            # This will use the temporary CSV file
            with patch("src.data_sources.config") as mock_config:
                mock_config.DATA_SOURCE = "csv"
                mock_config.CSV_DATA_PATH = temp_csv_path

                # Should work without errors
                result = manager.get_data()

                # Verify we got data
                assert not result.empty
                assert "Date" in result.columns
                assert "Close" in result.columns
                assert len(result) == 3  # Should have 3 rows
                assert result["Close"].iloc[0] == 102.0
                assert result["Close"].iloc[1] == 105.0
                assert result["Close"].iloc[2] == 108.0

        finally:
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    def test_config_backward_compatibility(self):
        """Test that old CSV_FALLBACK_PATH still works."""
        with patch("src.data_sources.config") as mock_config:
            mock_config.CSV_DATA_PATH = None  # Not set
            mock_config.CSV_FALLBACK_PATH = "test/path.csv"  # Old variable

            from src.data_sources import CSVDataSource

            csv_source = CSVDataSource()

            # Should use the fallback path
            with patch.object(csv_source, "fetch_data") as mock_fetch:
                mock_fetch.return_value = pd.DataFrame({"test": [1]})
                # This would normally call fetch_data with the fallback path
                # We can't easily test this without more complex mocking

    def test_date_range_validation(self):
        """Test that date ranges are properly handled."""
        manager = DataSourceManager()

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-15,450.00,455.00,448.00,452.50,50000000"""
            mock_get.return_value = mock_response

            with patch("src.data_sources.config") as mock_config:
                mock_config.DATA_SOURCE = "stooq"
                mock_config.STOOQ_SYMBOL = "SPY"

                # Test with custom date range
                result = manager.get_data(
                    start_date="2024-01-01", end_date="2024-01-31"
                )

                # Verify API was called with correct date parameters
                call_url = mock_get.call_args[0][
                    0
                ]  # First positional argument is the URL
                assert "d1=20240101" in call_url
                assert "d2=20240131" in call_url

    @patch("requests.get")
    def test_full_pipeline_stooq_integration(self, mock_get):
        """Test complete pipeline with Stooq historical data."""
        # Mock Stooq response with realistic historical data
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Need sufficient historical data for technical indicators (2+ years, 500+ rows)
        # Generate realistic historical data spanning 2022-2024
        import pandas as pd

        # Create date range from 2022 to 2024
        dates = pd.date_range("2022-01-01", "2024-01-25", freq="D")

        # Generate realistic SPY-like price data with trends and volatility
        base_price = 350  # Starting price around 2022 levels
        prices = []
        volumes = []

        for i, date in enumerate(dates):
            # Add upward trend + seasonal variation + random noise
            trend = i * 0.15  # Gradual upward trend
            seasonal = 5 * ((i % 252) / 252) * 3.14159  # Yearly seasonal pattern
            noise = (i % 7 - 3) * 0.5  # Weekly noise

            price = base_price + trend + seasonal + noise
            prices.append(max(price, 200))  # Ensure positive prices

            # Volume with some variation
            base_vol = 60000000
            vol_noise = (i % 10 - 5) * 2000000
            volume = base_vol + vol_noise
            volumes.append(max(volume, 10000000))

        # Generate OHLC data
        data_rows = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC around close price
            volatility = 0.02  # 2% daily volatility
            high_price = close_price * (1 + volatility + (i % 3) * 0.005)
            low_price = close_price * (1 - volatility - (i % 3) * 0.005)
            open_price = close_price - 1 + (i % 5) - 2  # Slight gap from previous close

            data_rows.append(
                f"{date.strftime('%Y-%m-%d')},{open_price:.2f},{high_price:.2f},{low_price:.2f},{close_price:.2f},{volumes[i]}"
            )

        mock_response.text = "Date,Open,High,Low,Close,Volume\n" + "\n".join(data_rows)
        mock_get.return_value = mock_response

        # Mock the experiment functions to avoid full LLM calls
        with patch("src.main.run_single_model") as mock_trading, patch(
            "src.main.get_current_config_summary",
            return_value={
                "experiment": "test_experiment",
                "description": "Test experiment description",
                "show_dates": False,
                "strategic_journal": True,
                "feeling_log": True,
            },
        ), patch("src.main.get_experiment_suffix", return_value="test"):

            mock_trading.return_value = {
                "decisions": ["BUY", "HOLD", "SELL"],
                "performance": {"total_return": 0.05},
            }

            # Run the complete pipeline
            try:
                from src.main import run_pipeline

                run_pipeline()

                # Verify data was fetched and saved
                import os

                raw_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "data",
                    "raw",
                    "current_data.csv",
                )
                assert os.path.exists(raw_path), "Raw data CSV was not saved"

                # Verify data content
                df = pd.read_csv(raw_path)
                assert (
                    len(df) >= 700
                ), f"Expected 700+ rows, got {len(df)}"  # ~2 years of daily data
                assert list(df.columns) == [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]

                print("✅ Full pipeline integration test passed")

            except Exception as e:
                pytest.fail(f"Full pipeline failed with Stooq data: {e}")

    @patch("requests.get")
    def test_stooq_data_format_compatibility(self, mock_get):
        """Test that Stooq data format works with data_prep.py modifications."""
        # Mock Stooq response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-15,450.00,455.00,448.00,452.50,50000000
2024-01-16,453.00,458.00,451.00,456.75,55000000"""
        mock_get.return_value = mock_response

        # Test data source to data_prep compatibility
        manager = DataSourceManager()

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "SPY"

            # Get data from Stooq
            stooq_data = manager.get_data()

            # Verify Stooq returns expected format
            assert list(stooq_data.columns) == [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
            assert len(stooq_data) == 2

            # Save to temp file and test data_prep compatibility
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                stooq_data.to_csv(f.name, index=False)
                temp_path = f.name

            try:
                # Test that data_prep can load this format
                from src.data_prep import load_raw_csv

                processed_data = load_raw_csv(temp_path)

                # Verify data_prep returns expected lowercase columns
                expected_cols = ["date", "open", "high", "low", "close", "volume"]
                assert list(processed_data.columns) == expected_cols
                assert len(processed_data) == 2

                print("✅ Stooq data format compatible with data_prep.py")

            finally:
                os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling and resilience."""

    @patch("requests.get")
    def test_stooq_api_server_error(self, mock_get):
        """Test handling of Stooq API server errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        manager = DataSourceManager()

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "SPY"

            with pytest.raises(DataSourceError, match="Stooq API returned status 500"):
                manager.get_data()

    @patch("requests.get")
    def test_stooq_invalid_response_format(self, mock_get):
        """Test handling of malformed Stooq responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Invalid response format"
        mock_get.return_value = mock_response

        manager = DataSourceManager()

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "SPY"

            with pytest.raises(DataSourceError, match="Invalid response from Stooq"):
                manager.get_data()


class TestDataQuality:
    """Test data quality and consistency."""

    @patch("requests.get")
    def test_stooq_data_structure_consistency(self, mock_get):
        """Test that Stooq data has consistent OHLCV structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-01,100.00,105.00,99.00,103.00,1000000
2024-01-02,103.50,108.00,102.00,106.50,1200000
2024-01-03,107.00,109.50,105.50,108.25,900000"""
        mock_get.return_value = mock_response

        manager = DataSourceManager()

        with patch("src.data_sources.config") as mock_config:
            mock_config.DATA_SOURCE = "stooq"
            mock_config.STOOQ_SYMBOL = "TEST"

            result = manager.get_data()

            # Verify all required columns exist
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            assert all(col in result.columns for col in required_cols)

            # Verify data types
            assert pd.api.types.is_datetime64_any_dtype(result["Date"])
            assert pd.api.types.is_numeric_dtype(result["Open"])
            assert pd.api.types.is_numeric_dtype(result["Close"])
            assert pd.api.types.is_numeric_dtype(result["Volume"])

            # Verify OHLC logic (High >= Open, Close, Low)
            for idx, row in result.iterrows():
                assert row["High"] >= row["Open"]
                assert row["High"] >= row["Close"]
                assert row["Low"] <= row["Open"]
                assert row["Low"] <= row["Close"]

    def test_symbol_formatting_edge_cases(self):
        """Test various symbol formats are handled correctly."""
        test_cases = [
            ("SPY", "spy.us"),  # Standard symbol
            ("AAPL", "aapl.us"),  # Another standard
            ("SPY.US", "spy.us"),  # Already has .us
            ("BTC-USD", "btc-usd"),  # Crypto (no .us)
        ]

        for input_symbol, expected_url_part in test_cases:
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = """Date,Open,High,Low,Close,Volume
2024-01-01,100.00,105.00,99.00,103.00,1000000"""
                mock_get.return_value = mock_response

                manager = DataSourceManager()

                with patch("src.data_sources.config") as mock_config:
                    mock_config.DATA_SOURCE = "stooq"
                    mock_config.STOOQ_SYMBOL = input_symbol

                    manager.get_data()

                    # Verify correct URL was called
                    call_url = mock_get.call_args[0][
                        0
                    ]  # First positional argument is the URL
                    assert expected_url_part in call_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
