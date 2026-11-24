"""Price data loading: Yahoo Finance, a local CSV, or synthetic GBM prices.

The synthetic option exists so every script in this repo runs end-to-end
with no internet or API keys - useful for testing and demos.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_prices(tickers=None, start="2018-01-01", end=None,
                csv: str | None = None, synthetic: bool = False,
                seed: int = 1) -> pd.DataFrame:
    """Return a DataFrame of daily close prices, one column per asset."""
    if synthetic:
        return synthetic_prices(tickers or ["AAA", "BBB", "CCC", "DDD", "EEE"],
                                seed=seed)
    if csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        return df.sort_index()
    import yfinance as yf  # optional dependency, imported lazily
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.dropna()


def synthetic_prices(names, n_days: int = 2520, seed: int = 1) -> pd.DataFrame:
    """Correlated geometric Brownian motion prices for testing."""
    rng = np.random.default_rng(seed)
    k = len(names)
    mu = rng.uniform(0.06, 0.11, k) / TRADING_DAYS
    vol = rng.uniform(0.12, 0.25, k) / np.sqrt(TRADING_DAYS)
    corr = 0.15 * np.ones((k, k)) + 0.85 * np.eye(k)
    cov = np.outer(vol, vol) * corr
    shocks = rng.multivariate_normal(mu - 0.5 * vol ** 2, cov, size=n_days)
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=names)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()
