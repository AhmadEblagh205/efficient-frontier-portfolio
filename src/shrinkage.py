"""One mitigation for covariance-estimation noise: Ledoit-Wolf shrinkage
of the covariance matrix. Sample covariances estimated on limited data
carry noisy extreme eigenvalues, and the optimiser loves exactly those
errors; shrinking toward a structured target trades a little bias for a
lot of variance, and the honest test is out of sample.

Usage:
    python src/shrinkage.py --synthetic
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from data import TRADING_DAYS, load_prices, log_returns
from frontier import annualised_stats, max_sharpe

ROOT = Path(__file__).resolve().parents[1]


def lw_cov(returns) -> tuple[np.ndarray, float]:
    lw = LedoitWolf().fit(returns.to_numpy())
    return lw.covariance_ * TRADING_DAYS, float(lw.shrinkage_)


def out_of_sample_sharpe(weights, test_returns) -> float:
    port = test_returns.to_numpy() @ weights
    return float(port.mean() / port.std() * np.sqrt(TRADING_DAYS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for seed in range(args.seeds):
        prices = load_prices(args.tickers, synthetic=True, seed=seed) \
            if args.synthetic or not args.tickers else \
            load_prices(args.tickers)
        rets = log_returns(prices)
        half = len(rets) // 2
        train, test = rets.iloc[:half], rets.iloc[half:]

        mu, cov_sample = annualised_stats(train)
        cov_lw, intensity = lw_cov(train)
        w_s = max_sharpe(mu, cov_sample)
        w_l = max_sharpe(mu, cov_lw)
        rows.append({
            "shrinkage_intensity": round(intensity, 3),
            "oos_sharpe_sample": round(out_of_sample_sharpe(w_s, test), 3),
            "oos_sharpe_ledoit": round(out_of_sample_sharpe(w_l, test), 3),
        })

    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
    summary = {"per_seed": rows, "mean": mean,
               "note": ("Half sample estimation, half sample evaluation. "
                        "With only five assets in this synthetic experiment, "
                        "the sample covariance is already stable enough that "
                        "the difference is small. This does not imply the same "
                        "result in a higher-dimensional or shorter-sample setting.")}
    out = ROOT / "reports" / "shrinkage_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
