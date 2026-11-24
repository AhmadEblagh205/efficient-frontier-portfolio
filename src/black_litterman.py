"""Black-Litterman return shrinkage with explicit excess-return accounting.

Reverse optimisation recovers equilibrium *excess* returns
    pi = delta * Sigma * w_market.
A view is then blended with those equilibrium excess returns.  The posterior is
converted back to total expected returns before it enters the ordinary
maximum-Sharpe optimiser.

Usage:
    python src/black_litterman.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data import log_returns, synthetic_prices
from frontier import annualised_stats, max_sharpe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
NAMES = ["TECH", "BANK", "ENERGY", "HEALTH", "GOLD"]


def equilibrium_returns(cov, w_mkt, delta: float = 2.5):
    """Equilibrium annual excess returns from reverse optimisation."""
    cov = np.asarray(cov, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    return delta * cov @ w_mkt


def posterior_returns(pi, cov, P, Q, view_confidence: float,
                      tau: float = 0.05):
    """Black-Litterman posterior expected *excess* returns.

    ``view_confidence`` is in [0, 1].  Zero confidence returns ``pi`` exactly.
    For this compact demonstration, confidence maps to view variance via
    Omega = ((1-c)/c) * diag(P tau Sigma P'), so confidence one is the
    zero-view-variance limit and enforces the stated linear view.
    """
    pi = np.asarray(pi, dtype=float)
    cov = np.asarray(cov, dtype=float)
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    conf = float(view_confidence)
    if not 0.0 <= conf <= 1.0:
        raise ValueError("view_confidence must lie in [0, 1]")
    if conf == 0.0:
        return pi.copy()

    tau_cov = tau * cov
    view_cov = P @ tau_cov @ P.T
    if conf == 1.0:
        omega = np.zeros_like(view_cov)
    else:
        omega = ((1.0 - conf) / conf) * np.diag(np.diag(view_cov))
    gain = tau_cov @ P.T @ np.linalg.inv(view_cov + omega)
    return pi + gain @ (Q - P @ pi)


def implied_market_weights(pi, cov, delta: float = 2.5):
    """Recover normalised market weights from equilibrium excess returns."""
    raw = np.linalg.solve(cov, pi) / delta
    return raw / raw.sum()


def main():
    rets = log_returns(synthetic_prices(NAMES))
    _, cov = annualised_stats(rets)
    w_mkt = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    rf = 0.02
    pi = equilibrium_returns(cov, w_mkt)

    # One excess-return view: TECH outperforms ENERGY by 4 percentage points.
    P = np.zeros(len(NAMES)); P[0], P[2] = 1.0, -1.0
    Q = np.array([0.04])

    rows = {}
    for conf in [0.0, 0.25, 1.0]:
        post_excess = posterior_returns(pi, cov, P, Q, conf)
        # max_sharpe expects total expected returns and subtracts rf internally.
        total_returns = rf + post_excess
        w = max_sharpe(total_returns, cov, rf=rf)
        rows[f"confidence_{conf}"] = {
            "posterior_excess_returns": dict(zip(NAMES, np.round(post_excess, 4))),
            "weights": dict(zip(NAMES, np.round(w, 3))),
            "view_spread": round(float(P @ post_excess), 4),
        }

    # A dense confidence path makes the prior-to-view transition visible.
    grid = np.linspace(0.0, 1.0, 21)
    path_weights = []
    path_spread = []
    for conf in grid:
        post = posterior_returns(pi, cov, P, Q, float(conf))
        w = max_sharpe(rf + post, cov, rf=rf)
        path_weights.append(w)
        path_spread.append(float(P @ post))
    path_weights = np.asarray(path_weights)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(NAMES):
        ax.plot(grid, path_weights[:, i], label=name)
    ax.set(xlabel="View confidence", ylabel="Long-only max-Sharpe weight",
           title="Black-Litterman: confidence tilts the equilibrium portfolio")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "black_litterman_confidence.png", dpi=160)
    plt.close(fig)

    summary = {
        "market_weights": dict(zip(NAMES, w_mkt)),
        "equilibrium_excess_returns": dict(zip(NAMES, np.round(pi, 4))),
        "view": "TECH excess return minus ENERGY excess return equals 4 percentage points",
        "risk_free_rate": rf,
        "results": rows,
        "confidence_path": {
            "confidence": np.round(grid, 3).tolist(),
            "view_spread": np.round(path_spread, 5).tolist(),
        },
        "note": (
            "Zero confidence reproduces the market portfolio (up to numerical "
            "optimisation tolerance). Increasing confidence tilts the posterior "
            "toward the stated relative-return view; confidence one enforces "
            "the 4 percentage-point spread in this one-view demonstration."
        ),
    }
    out = ROOT / "reports" / "black_litterman.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
