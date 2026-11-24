"""Monte Carlo simulation of future portfolio value using geometric Brownian
motion fitted to historical returns - the 'stochastic stock simulation' half
of the project.

Usage:
    python src/simulate.py --synthetic --horizon 252 --paths 2000
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data import TRADING_DAYS, load_prices, log_returns
from frontier import annualised_stats, max_sharpe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def simulate_gbm(mu_ann, vol_ann, s0=100.0, horizon=252, n_paths=2000, seed=11):
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    drift = (mu_ann - 0.5 * vol_ann ** 2) * dt
    diff = vol_ann * np.sqrt(dt)
    shocks = rng.standard_normal((horizon, n_paths))
    log_paths = np.cumsum(drift + diff * shocks, axis=0)
    return s0 * np.exp(np.vstack([np.zeros(n_paths), log_paths]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument("--paths", type=int, default=2000)
    args = ap.parse_args()

    prices = load_prices(args.tickers, csv=args.csv, synthetic=args.synthetic)
    rets = log_returns(prices)
    mu, cov = annualised_stats(rets)
    w = max_sharpe(mu, cov)
    port_mu = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))

    paths = simulate_gbm(port_mu, port_vol, horizon=args.horizon,
                         n_paths=args.paths)
    t = np.arange(paths.shape[0])
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t, paths[:, :60], color="steelblue", alpha=0.15, lw=0.7)
    labels = ["5th", "25th", "median", "75th", "95th"]
    styles = [":", "--", "-", "--", ":"]
    for row, lab, st in zip(pct, labels, styles):
        ax.plot(t, row, "k" + st, lw=1.6, label=f"{lab} percentile")
    ax.set(xlabel="Trading days ahead",
           ylabel="Portfolio value (start = 100)",
           title=f"Max-Sharpe portfolio: {args.paths} simulated GBM paths "
                 f"(mu={port_mu:.1%}, vol={port_vol:.1%})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "simulated_paths.png", dpi=150)
    plt.close(fig)

    final = paths[-1]
    print(f"Portfolio mu={port_mu:.2%}, vol={port_vol:.2%}")
    print(f"After {args.horizon} days: median {np.median(final):.1f}, "
          f"5-95% range [{np.percentile(final, 5):.1f}, "
          f"{np.percentile(final, 95):.1f}]")
    print(f"P(loss) = {(final < 100).mean():.1%}")
    print(f"Figure -> {FIGURES / 'simulated_paths.png'}")


if __name__ == "__main__":
    main()
