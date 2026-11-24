"""Markowitz efficient frontier: random portfolios, the optimised frontier,
the maximum-Sharpe and minimum-variance portfolios, and a diversification
curve showing how portfolio volatility falls as assets are added.

Usage:
    python src/frontier.py --synthetic
    python src/frontier.py --tickers AAPL MSFT JNJ XOM KO --start 2019-01-01
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from data import TRADING_DAYS, load_prices, log_returns

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def annualised_stats(rets: pd.DataFrame):
    mu = rets.mean().to_numpy() * TRADING_DAYS
    cov = rets.cov().to_numpy() * TRADING_DAYS
    return mu, cov


def port_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def min_vol_for_return(target, mu, cov):
    k = len(mu)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: w @ mu - target})
    res = minimize(lambda w: w @ cov @ w, np.full(k, 1 / k),
                   bounds=[(0, 1)] * k, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else None


def max_sharpe(mu, cov, rf=0.02):
    k = len(mu)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w),
                   np.full(k, 1 / k), bounds=[(0, 1)] * k,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"max-Sharpe optimisation failed: {res.message}")
    return res.x


def diversification_curve(rets: pd.DataFrame, n_draws: int = 300, seed: int = 0):
    """Average equal-weight portfolio vol as a function of number of assets."""
    rng = np.random.default_rng(seed)
    cols = list(rets.columns)
    out = []
    for n in range(1, len(cols) + 1):
        vols = []
        for _ in range(n_draws):
            picks = rng.choice(cols, size=n, replace=False)
            w = np.full(n, 1 / n)
            _, cov = annualised_stats(rets[picks])
            vols.append(np.sqrt(w @ cov @ w))
        out.append(np.mean(vols))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rf", type=float, default=0.02)
    args = ap.parse_args()

    prices = load_prices(args.tickers, start=args.start, csv=args.csv,
                         synthetic=args.synthetic)
    rets = log_returns(prices)
    mu, cov = annualised_stats(rets)
    names = list(prices.columns)

    # Random portfolio cloud
    rng = np.random.default_rng(1)
    W = rng.dirichlet(np.ones(len(mu)), size=4000)
    cloud = np.array([port_perf(w, mu, cov) for w in W])
    sharpe_cloud = (cloud[:, 0] - args.rf) / cloud[:, 1]

    # Optimised frontier
    targets = np.linspace(mu.min(), mu.max(), 40)
    frontier = []
    for t in targets:
        w = min_vol_for_return(t, mu, cov)
        if w is not None:
            frontier.append(port_perf(w, mu, cov))
    frontier = np.array(frontier)

    w_ms = max_sharpe(mu, cov, args.rf)
    ms_ret, ms_vol = port_perf(w_ms, mu, cov)

    # Global minimum-variance portfolio: the left tip of the frontier
    i_gmv = int(np.argmin(frontier[:, 1]))
    gmv_ret, gmv_vol = frontier[i_gmv]

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(cloud[:, 1], cloud[:, 0], c=sharpe_cloud, cmap="viridis",
                    s=8, alpha=0.45)
    ax.plot(frontier[:, 1], frontier[:, 0], "r-", lw=2.2,
            label="Efficient frontier")
    ax.plot(ms_vol, ms_ret, "r*", ms=16, label="Max Sharpe")
    ax.plot(gmv_vol, gmv_ret, "ks", ms=8, label="Minimum variance")
    # Capital allocation line: risk-free asset blended with the tangency
    # portfolio dominates every risky-only mix below it
    cal_x = np.array([0.0, cloud[:, 1].max()])
    cal_y = args.rf + (ms_ret - args.rf) / ms_vol * cal_x
    ax.plot(cal_x, cal_y, "k--", lw=1.3,
            label=f"Capital allocation line (rf {args.rf:.0%})")
    ax.set_xlim(left=0)
    fig.colorbar(sc, label="Sharpe ratio")
    ax.set(xlabel="Annualised volatility", ylabel="Annualised return",
           title="Efficient frontier vs 4,000 random portfolios")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "efficient_frontier.png", dpi=150)

    div = diversification_curve(rets)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(range(1, len(div) + 1), div, "o-", lw=1.8)
    ax2.set(xlabel="Number of assets (equal weight)",
            ylabel="Average annualised volatility",
            title="Diversification: volatility falls as assets are added")
    ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(FIGURES / "diversification.png", dpi=150)
    plt.close("all")

    weights = {n: round(float(w), 4) for n, w in zip(names, w_ms)}
    summary = {"max_sharpe_weights": weights,
               "max_sharpe_return": round(ms_ret, 4),
               "max_sharpe_vol": round(ms_vol, 4),
               "sharpe": round((ms_ret - args.rf) / ms_vol, 3),
               "min_variance_return": round(float(gmv_ret), 4),
               "min_variance_vol": round(float(gmv_vol), 4)}
    out = ROOT / "reports" / "frontier_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figures -> {FIGURES}")
    return w_ms, prices


if __name__ == "__main__":
    main()
