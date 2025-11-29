# src/data_prep.py

import os
import numpy as np
import pandas as pd
from .config import (
    PAST_RET_LAGS,
    MA20_WINDOW,
    VOL20_WINDOW,
    RET_5D_WINDOW,
    RSI_WINDOW,
)


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

    # On garde juste date et prix, et on renomme en 'close'
    df = df[["date", price_col]].rename(columns={price_col: "close"})

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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rendement quotidien en pourcentage
    df["return_1d"] = df["close"].pct_change() * 100.0

    # Retours décalés
    for k in range(1, PAST_RET_LAGS + 1):
        df[f"ret_lag_{k}"] = df["return_1d"].shift(k)

    # Momentum 20 jours
    df["ma20_pct"] = df["return_1d"].rolling(MA20_WINDOW).sum()

    # Volatilité réalisée 20 jours annualisée
    daily_vol_20 = df["return_1d"].rolling(VOL20_WINDOW).std()
    df["vol20_annualized"] = daily_vol_20 * np.sqrt(252)

    # RSI (Relative Strength Index)
    df["rsi_14"] = compute_rsi(df["close"], RSI_WINDOW)

    # Momentum 5 jours
    df["ret_5d"] = df["return_1d"].rolling(RET_5D_WINDOW).sum()

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
