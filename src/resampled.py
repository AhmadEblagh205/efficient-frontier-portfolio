"""The resampled frontier, estimation error made visible: bootstrap the
return history, refit the max-Sharpe portfolio on every draw, and watch
the "optimal" weights swing. A simple resampled-weight average is then
compared with the point-estimate optimum as a stability diagnostic.  This
is an educational resampling experiment, not a claim that averaging bootstrap
optimisers is universally optimal.

Usage:
    python src/resampled.py --draws 300
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data import log_returns, synthetic_prices
from frontier import annualised_stats, max_sharpe

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["TECH", "BANK", "ENERGY", "HEALTH", "GOLD"]


def resample_weights(rets, n_draws: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = len(rets)
    draws = []
    for _ in range(n_draws):
        sample = rets.iloc[rng.integers(0, T, T)]
        mu, cov = annualised_stats(sample)
        draws.append(max_sharpe(mu, cov))
    return np.array(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    args = ap.parse_args()
    rets = log_returns(synthetic_prices(NAMES))
    mu, cov = annualised_stats(rets)
    w_point = max_sharpe(mu, cov)
    W = resample_weights(rets, args.draws)
    w_avg = W.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, name in enumerate(NAMES):
        lo, hi = np.quantile(W[:, i], [0.05, 0.95])
        ax.hlines(i, lo, hi, lw=2, color="#0b3d91")
        ax.plot(w_point[i], i, "s", color="#c1440e", ms=7,
                label="point estimate" if i == 0 else None)
        ax.plot(w_avg[i], i, "o", color="#0b3d91",
                label="resampled average" if i == 0 else None)
    ax.set_yticks(range(len(NAMES)), NAMES)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(xlabel="max-Sharpe weight (5th to 95th pct across bootstraps)",
           title="The optimal portfolio, under resampling")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "resampled_weights.png",
                dpi=150)
    plt.close(fig)

    summary = {
        "point_estimate": dict(zip(NAMES, np.round(w_point, 3).tolist())),
        "resampled_average": dict(zip(NAMES, np.round(w_avg, 3).tolist())),
        "max_abs_weight_point": round(float(np.abs(w_point).max()), 3),
        "max_abs_weight_resampled": round(float(np.abs(w_avg).max()), 3),
        "mean_weight_sd_across_draws": round(float(W.std(axis=0).mean()), 3),
        "note": ("The averaged allocation is less concentrated than the point "
                 "optimum in this run, while the dispersion across bootstrap "
                 "draws makes the estimator's sampling sensitivity visible. "
                 "This is a stability diagnostic, not a universal optimality "
                 "claim for resampled weights."),
    }
    (ROOT / "reports" / "resampled_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
