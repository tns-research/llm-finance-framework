# src/constants.py
"""Shared numeric constants for the analytics layer.

Kept dependency-free (no intra-package imports) so any module can import it
without risking an import cycle.
"""

# Trading days in a calendar year, used to annualize daily returns and
# volatility (np.sqrt(TRADING_DAYS_PER_YEAR)) and daily means (* TRADING_DAYS_PER_YEAR).
TRADING_DAYS_PER_YEAR = 252

# Rolling lookback (trading days) for the ~1-month annualized volatility series
# used in regime classification.
VOL_WINDOW = 20
