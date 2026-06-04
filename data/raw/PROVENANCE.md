# Data Provenance: `spy_daily.csv`

This is the **canonical, frozen dataset** the framework runs on by default
(`DATA_SOURCE = "vendored"`). It is committed to the repo so a clean clone runs
`python -m src.main` offline and deterministically, with no API key and no network.

| Field | Value |
|-------|-------|
| File | `data/raw/spy_daily.csv` |
| Symbol | SPY (SPDR S&P 500 ETF, used as S&P 500 proxy) |
| Date range | 2015-01-02 to 2023-12-29 |
| Rows | 2264 daily bars |
| Columns | `Date, Open, High, Low, Close, Volume` |
| Adjustment | Split- and dividend-adjusted OHLC (yfinance `auto_adjust=True`) |
| Source | [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance) |
| Source version | yfinance 1.4.1 |
| SHA-256 | `b3b90a1d0d9c16de85718b0d7eebfddcf0d5532393f22c859cf91568ba0b1811` |
| Frozen at | 2026-06-01 |

The SHA-256 and other metadata are recorded in machine-readable form in
`data/raw/MANIFEST.json`. The `VendoredDataSource` verifies the file against this
checksum on every load and refuses to run on a corrupted or modified snapshot. A
test (`tests/test_vendored_data.py`) asserts the committed file still matches the
manifest.

## Why adjusted prices

Returns are computed as `close.pct_change()`. Using split/dividend-adjusted closes
avoids artificial price gaps on ex-dividend dates, so daily returns reflect actual
total return rather than spurious drops. The provenance records the adjustment so
results stay interpretable.

## Refreshing / replacing the snapshot

The committed snapshot is intentionally frozen. To regenerate or extend it, run
`python scripts/refresh_data.py` (re-fetches via yfinance, rewrites the CSV and the
manifest checksum). Refreshing is a deliberate action: it changes the SHA-256, so
commit the new `spy_daily.csv` **and** `MANIFEST.json` together, and update this
file's date/range/version fields.

> Yahoo Finance data is provided under Yahoo's terms. This snapshot is vendored for
> reproducibility of this research project; respect the upstream terms for any
> redistribution or commercial use.
