# src/data_prep.py

import os

import numpy as np
import pandas as pd

from . import config


def load_raw_csv(raw_path: str) -> pd.DataFrame:
    """
    Lit un CSV type Stooq avec au moins les colonnes:
    Date, Open, High, Low, Close, Volume

    On est robuste:
    - autodétection du séparateur
    - on enlève les espaces dans les noms de colonnes
    """

    if not os.path.exists(raw_path):
        raise RuntimeError(
            f"Raw file {raw_path} not found. "
            "Place here a CSV with columns like Date, Open, High, Low, Close, Volume."
        )

    # sep=None + engine="python" pour autodétecter , ou ; ou tab
    df = pd.read_csv(raw_path, sep=None, engine="python")

    # Nettoyage des noms de colonnes
    df.columns = df.columns.astype(str).str.strip()

    print("Loaded raw CSV with columns:", list(df.columns))

    if "Date" not in df.columns:
        raise RuntimeError("CSV must contain a 'Date' column")

    # Certaines sources ont 'Close', d autres 'Adj Close'
    price_col = None
    if "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"

    if price_col is None:
        raise RuntimeError(
            "CSV must contain a 'Close' or 'Adj Close' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Conversion date
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Tri chronologique
    df = df.sort_values("date").reset_index(drop=True)

    # Keep OHLCV data for advanced indicators
    required_cols = ["date"]
    ohlc_cols = []

    # Add OHLC columns if available
    if "Open" in df.columns:
        ohlc_cols.append("Open")
    if "High" in df.columns:
        ohlc_cols.append("High")
    if "Low" in df.columns:
        ohlc_cols.append("Low")
    if "Close" in df.columns or price_col != "Close":
        ohlc_cols.append(price_col)
    if "Volume" in df.columns:
        ohlc_cols.append("Volume")

    df = df[required_cols + ohlc_cols]

    # Rename columns to standard format
    rename_dict = {price_col: "close"}
    if "Open" in df.columns:
        rename_dict["Open"] = "open"
    if "High" in df.columns:
        rename_dict["High"] = "high"
    if "Low" in df.columns:
        rename_dict["Low"] = "low"
    if "Volume" in df.columns:
        rename_dict["Volume"] = "volume"

    df = df.rename(columns=rename_dict)

    # Fill missing OHLC columns with close price if not available
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]

    print("DataFrame after selecting date and close, head:")
    print(df.head())

    if df["close"].isna().all():
        raise RuntimeError("Column 'close' is all NaN. Check your CSV file.")

    return df


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over specified window

    Args:
        prices: Series of price data
        window: Period for RSI calculation (default 14)

    Returns:
        Series with RSI values (0-100 scale)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Avoid division by zero by replacing 0 losses with NaN
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Series of price data
        period: Period for EMA calculation

    Returns:
        Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    fast_ema = compute_ema(prices, fast)
    slow_ema = compute_ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 1,
) -> tuple:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        k_period: Period for %K calculation
        d_period: Period for %D smoothing
        smooth_k: Smoothing period for %K (optional)

    Returns:
        tuple: (%K, %D)
    """
    # Raw %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Smooth %K if requested
    k_percent = raw_k.rolling(window=smooth_k).mean()

    # %D is SMA of %K
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


def compute_bollinger_bands(
    prices: pd.Series, window: int = 20, std_dev: float = 2
) -> tuple:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of price data
        window: Period for moving average and standard deviation
        std_dev: Number of standard deviations for bands

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rendement quotidien en pourcentage
    df["return_1d"] = df["close"].pct_change() * 100.0

    # Retours décalés (batch operation to avoid DataFrame fragmentation)
    return_lags = {
        f"ret_lag_{k}": df["return_1d"].shift(k)
        for k in range(1, config.PAST_RET_LAGS + 1)
    }
    df = pd.concat([df, pd.DataFrame(return_lags)], axis=1)

    # Momentum 20 jours
    df["ma20_pct"] = df["return_1d"].rolling(config.MA20_WINDOW).sum()

    # Volatilité réalisée 20 jours annualisée
    daily_vol_20 = df["return_1d"].rolling(config.VOL20_WINDOW).std()
    df["vol20_annualized"] = daily_vol_20 * np.sqrt(252)

    # RSI (Relative Strength Index) - always calculated for baselines and analysis
    df["rsi_14"] = compute_rsi(df["close"], config.RSI_WINDOW)

    # Advanced Technical Indicators - always calculated for baselines and analysis
    # MACD
    macd_line, macd_signal, macd_hist = compute_macd(
        df["close"], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
    )
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_hist

    # Stochastic Oscillator
    stoch_k, stoch_d = compute_stochastic(
        df["high"],
        df["low"],
        df["close"],
        config.STOCH_K,
        config.STOCH_D,
        config.STOCH_SMOOTH_K,
    )
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(
        df["close"], config.BB_WINDOW, config.BB_STD
    )
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower
    # Bollinger Band position (%B)
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    # Historical Technical Indicators (only when enabled)
    if config.ENABLE_TECHNICAL_INDICATORS:
        # Batch operation to avoid DataFrame fragmentation
        tech_lags = {}
        for k in range(1, config.PAST_RET_LAGS + 1):
            tech_lags[f"rsi_lag_{k}"] = df["rsi_14"].shift(k)
            tech_lags[f"macd_hist_lag_{k}"] = df["macd_histogram"].shift(k)
            tech_lags[f"stoch_k_lag_{k}"] = df["stoch_k"].shift(k)
            tech_lags[f"bb_position_lag_{k}"] = df["bb_position"].shift(k)
        df = pd.concat([df, pd.DataFrame(tech_lags)], axis=1)

    # Momentum 5 jours
    df["ret_5d"] = df["return_1d"].rolling(config.RET_5D_WINDOW).sum()

    # Label: rendement du jour suivant
    df["next_return_1d"] = df["return_1d"].shift(-1)

    # On enlève les lignes incomplètes
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "Feature DataFrame is empty after processing. Check your raw CSV."
        )

    print("Features DataFrame head:")
    print(df.head())

    return df


def prepare_features(raw_path: str, processed_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    raw = load_raw_csv(raw_path)

    # NE PAS faire ça:
    # raw = raw.iloc[20:].reset_index(drop=True)

    features = compute_features(raw)
    features.to_csv(processed_path, index=False)
    return features
