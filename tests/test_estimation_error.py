import numpy as np
import pytest

from black_litterman import equilibrium_returns, posterior_returns
from data import log_returns, synthetic_prices
from estimation_error import known_truth_world, run_benchmark, true_sharpe
from frontier import annualised_stats, max_sharpe


def test_black_litterman_zero_confidence_reproduces_equilibrium_market():
    names = ["TECH", "BANK", "ENERGY", "HEALTH", "GOLD"]
    rets = log_returns(synthetic_prices(names, seed=1))
    _, cov = annualised_stats(rets)
    w_mkt = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    rf = 0.02
    pi = equilibrium_returns(cov, w_mkt)
    post = posterior_returns(pi, cov, np.array([1, 0, -1, 0, 0]),
                             np.array([0.04]), 0.0)
    w = max_sharpe(rf + post, cov, rf=rf)
    assert np.allclose(post, pi)
    assert np.allclose(w, w_mkt, atol=2e-3)


def test_black_litterman_full_confidence_enforces_linear_view():
    names = ["TECH", "BANK", "ENERGY", "HEALTH", "GOLD"]
    rets = log_returns(synthetic_prices(names, seed=1))
    _, cov = annualised_stats(rets)
    w_mkt = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    pi = equilibrium_returns(cov, w_mkt)
    P = np.array([1.0, 0.0, -1.0, 0.0, 0.0])
    Q = np.array([0.04])
    post = posterior_returns(pi, cov, P, Q, 1.0)
    assert np.isclose(float(P @ post), 0.04, atol=1e-10)


def test_black_litterman_rejects_invalid_confidence():
    mu, cov = known_truth_world()
    with pytest.raises(ValueError):
        posterior_returns(mu, cov, np.eye(len(mu))[0], np.array([0.01]), 1.2)


def test_known_truth_oracle_beats_equal_weight():
    mu, cov = known_truth_world()
    oracle = max_sharpe(mu, cov, rf=0.02)
    equal = np.full(len(mu), 1 / len(mu))
    assert true_sharpe(oracle, mu, cov, 0.02) > true_sharpe(equal, mu, cov, 0.02)


def test_estimation_error_benchmark_identifies_noisy_means_as_main_driver():
    summary = run_benchmark(years=(3,), n_worlds=20, seed=202600)
    row = summary["rows"][0]
    assert row["median_weight_error_mean_only"] > 5 * row["median_weight_error_cov_only"]
    assert row["median_true_sharpe_cov_only"] > row["median_true_sharpe_mean_only"]


def test_more_history_reduces_sample_mvo_weight_error():
    summary = run_benchmark(years=(1, 30), n_worlds=10, seed=202600)
    one, thirty = summary["rows"]
    assert thirty["median_weight_error_both"] < one["median_weight_error_both"]
    assert thirty["median_true_sharpe_both"] > one["median_true_sharpe_both"]
