"""Known-truth benchmark for mean-variance estimation error.

The usual efficient-frontier picture hides a hard problem: expected returns and
covariances are estimated.  This module generates a synthetic Gaussian
log-return world with known annual moments, then asks which input error actually
drives long-only maximum-Sharpe instability.

For each simulated history it compares:

* both inputs estimated from the sample;
* expected returns estimated but the true covariance supplied;
* covariance estimated but the true expected returns supplied;
* equal weights as a parameter-free benchmark;
* the oracle portfolio using the true moments.

Performance is evaluated under the known true distribution, so this is a
controlled estimation-error experiment rather than a backtest claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import TRADING_DAYS
from frontier import max_sharpe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def known_truth_world() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed annual log-return means and covariance for five assets."""
    mu = np.array([0.04, 0.06, 0.10, 0.08, 0.14], dtype=float)
    vol = np.array([0.12, 0.18, 0.16, 0.20, 0.24], dtype=float)
    corr = np.array([
        [1.00, 0.25, 0.15, 0.20, 0.10],
        [0.25, 1.00, 0.30, 0.20, 0.20],
        [0.15, 0.30, 1.00, 0.25, 0.15],
        [0.20, 0.20, 0.25, 1.00, 0.30],
        [0.10, 0.20, 0.15, 0.30, 1.00],
    ])
    cov = np.outer(vol, vol) * corr
    return mu, cov


def true_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                rf: float = 0.02) -> float:
    var = float(weights @ cov @ weights)
    if var <= 0:
        raise ValueError("portfolio variance must be positive")
    return float((weights @ mu - rf) / np.sqrt(var))


def total_variation_weight_error(weights: np.ndarray,
                                 oracle: np.ndarray) -> float:
    """Half-L1 distance between two fully invested long-only portfolios."""
    return float(0.5 * np.abs(weights - oracle).sum())


def _sample_moments(daily_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = daily_returns.mean(axis=0) * TRADING_DAYS
    cov = np.cov(daily_returns, rowvar=False) * TRADING_DAYS
    return mu, cov


def run_benchmark(years=(1, 3, 10, 30), n_worlds: int = 200,
                  seed: int = 202600, rf: float = 0.02) -> dict:
    """Run paired known-truth worlds at increasing sample horizons."""
    years = tuple(int(y) for y in years)
    if any(y <= 0 for y in years):
        raise ValueError("sample years must be positive")
    if n_worlds < 1:
        raise ValueError("n_worlds must be positive")

    mu_true, cov_true = known_truth_world()
    oracle = max_sharpe(mu_true, cov_true, rf=rf)
    equal = np.full(len(mu_true), 1 / len(mu_true))
    oracle_sharpe = true_sharpe(oracle, mu_true, cov_true, rf)
    equal_sharpe = true_sharpe(equal, mu_true, cov_true, rf)

    rows: dict[int, list[dict]] = {y: [] for y in years}
    max_days = max(years) * TRADING_DAYS
    mean_daily = mu_true / TRADING_DAYS
    cov_daily = cov_true / TRADING_DAYS

    # Each world is simulated once to the longest horizon, then shorter
    # experiments use prefixes. This makes the horizon comparison paired.
    for world in range(n_worlds):
        rng = np.random.default_rng(seed + world)
        full = rng.multivariate_normal(mean_daily, cov_daily, size=max_days)
        for y in years:
            sample = full[: y * TRADING_DAYS]
            mu_hat, cov_hat = _sample_moments(sample)

            w_both = max_sharpe(mu_hat, cov_hat, rf=rf)
            w_mean = max_sharpe(mu_hat, cov_true, rf=rf)
            w_cov = max_sharpe(mu_true, cov_hat, rf=rf)

            rows[y].append({
                "weight_error_both": total_variation_weight_error(w_both, oracle),
                "weight_error_mean_only": total_variation_weight_error(w_mean, oracle),
                "weight_error_cov_only": total_variation_weight_error(w_cov, oracle),
                "true_sharpe_both": true_sharpe(w_both, mu_true, cov_true, rf),
                "true_sharpe_mean_only": true_sharpe(w_mean, mu_true, cov_true, rf),
                "true_sharpe_cov_only": true_sharpe(w_cov, mu_true, cov_true, rf),
                "equal_weight_beats_sample_mvo": (
                    equal_sharpe > true_sharpe(w_both, mu_true, cov_true, rf)
                ),
            })

    summary_rows = []
    for y in years:
        block = rows[y]
        def med(key):
            return float(np.median([r[key] for r in block]))
        summary_rows.append({
            "years": y,
            "median_weight_error_both": med("weight_error_both"),
            "median_weight_error_mean_only": med("weight_error_mean_only"),
            "median_weight_error_cov_only": med("weight_error_cov_only"),
            "median_true_sharpe_both": med("true_sharpe_both"),
            "median_true_sharpe_mean_only": med("true_sharpe_mean_only"),
            "median_true_sharpe_cov_only": med("true_sharpe_cov_only"),
            "p_equal_weight_beats_sample_mvo": float(np.mean([
                r["equal_weight_beats_sample_mvo"] for r in block
            ])),
        })

    return {
        "provenance": "independent synthetic known-truth experiment",
        "n_worlds": n_worlds,
        "risk_free_rate": rf,
        "oracle_weights": oracle.tolist(),
        "oracle_true_sharpe": oracle_sharpe,
        "equal_weight_true_sharpe": equal_sharpe,
        "rows": summary_rows,
        "interpretation": (
            "In this stated synthetic world, expected-return estimation is the "
            "dominant source of long-only maximum-Sharpe weight error. "
            "Covariance-only error is much smaller. Equal weight is a useful "
            "sanity benchmark because sample optimisation can amplify noisy "
            "expected-return estimates."
        ),
    }


def make_figure(summary: dict) -> Path:
    years = np.array([r["years"] for r in summary["rows"]], dtype=float)
    both = np.array([r["median_weight_error_both"] for r in summary["rows"]])
    mean = np.array([r["median_weight_error_mean_only"] for r in summary["rows"]])
    cov = np.array([r["median_weight_error_cov_only"] for r in summary["rows"]])
    sharpe = np.array([r["median_true_sharpe_both"] for r in summary["rows"]])
    p_eq = np.array([r["p_equal_weight_beats_sample_mvo"] for r in summary["rows"]])

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].plot(years, both, "o-", label="estimate mean + covariance")
    axes[0].plot(years, mean, "o--", label="estimate mean only")
    axes[0].plot(years, cov, "o--", label="estimate covariance only")
    axes[0].set_xscale("log")
    axes[0].set_xticks(years, [str(int(y)) for y in years])
    axes[0].set(xlabel="Years of daily observations",
                ylabel="Median weight error (half-L1)",
                title="Where optimiser instability comes from")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(years, sharpe, "o-", label="sample max-Sharpe")
    axes[1].axhline(summary["equal_weight_true_sharpe"], linestyle="--",
                    label="equal weight")
    axes[1].axhline(summary["oracle_true_sharpe"], linestyle=":",
                    label="oracle")
    axes[1].set_xscale("log")
    axes[1].set_xticks(years, [str(int(y)) for y in years])
    axes[1].set(xlabel="Years of daily observations",
                ylabel="True Sharpe in known world",
                title="Optimisation can amplify noisy means")
    ax2 = axes[1].twinx()
    ax2.plot(years, p_eq, "s-.", alpha=0.7, label="P(equal weight wins)")
    ax2.set_ylabel("Fraction of worlds equal weight wins")
    ax2.set_ylim(0, 1)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, fontsize=8,
                   loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "estimation_error.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=200)
    args = ap.parse_args()
    summary = run_benchmark(n_worlds=args.worlds)
    out = ROOT / "reports" / "estimation_error.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    fig = make_figure(summary)
    print(json.dumps(summary, indent=2))
    print(f"Figure -> {fig}")


if __name__ == "__main__":
    main()
