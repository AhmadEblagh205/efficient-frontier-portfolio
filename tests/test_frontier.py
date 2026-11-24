import numpy as np

from data import log_returns, synthetic_prices
from frontier import annualised_stats, max_sharpe, min_vol_for_return, port_perf


def _rets():
    prices = synthetic_prices(["A", "B", "C"], n_days=600, seed=3)
    return log_returns(prices)


def test_synthetic_prices_are_positive_and_shaped():
    prices = synthetic_prices(["A", "B"], n_days=100, seed=0)
    assert prices.shape == (100, 2)
    assert (prices > 0).all().all()


def test_max_sharpe_weights_are_a_valid_portfolio():
    mu, cov = annualised_stats(_rets())
    w = max_sharpe(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-8).all()


def test_frontier_point_matches_target_return():
    mu, cov = annualised_stats(_rets())
    target = float(mu.mean())
    w = min_vol_for_return(target, mu, cov)
    ret, _ = port_perf(w, mu, cov)
    assert abs(ret - target) < 1e-4


def test_ledoit_wolf_shrinks_and_stays_valid():
    from shrinkage import lw_cov
    from frontier import max_sharpe
    rets = _rets()
    cov, intensity = lw_cov(rets)
    assert 0.0 < intensity <= 1.0
    assert np.allclose(cov, cov.T)
    assert (np.linalg.eigvalsh(cov) > 0).all()
    mu, _ = annualised_stats(rets)
    w = max_sharpe(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_black_litterman_limits_and_tilt():
    import numpy as np
    from black_litterman import equilibrium_returns, posterior_returns
    rng = np.random.default_rng(0)
    A = rng.normal(size=(60, 4))
    cov = np.cov(A.T) / 10
    w_mkt = np.array([0.4, 0.3, 0.2, 0.1])
    pi = equilibrium_returns(cov, w_mkt)
    P = np.array([1.0, 0.0, -1.0, 0.0])
    Q = np.array([0.05])
    near_zero = posterior_returns(pi, cov, P, Q, view_confidence=1e-8)
    assert np.allclose(near_zero, pi, atol=1e-4)     # no view, no change
    strong = posterior_returns(pi, cov, P, Q, view_confidence=1.0)
    spread_pi = pi[0] - pi[2]
    spread_post = strong[0] - strong[2]
    assert spread_post > spread_pi                    # tilts toward the view
    assert abs(spread_post - 0.05) < abs(spread_pi - 0.05)


def test_resampled_weights_are_valid_variable_and_average_is_valid():
    import numpy as np
    from data import log_returns, synthetic_prices
    from frontier import annualised_stats, max_sharpe
    from resampled import resample_weights
    rets = log_returns(synthetic_prices(["A", "B", "C", "D"], n_days=800,
                                        seed=4))
    mu, cov = annualised_stats(rets)
    w_point = max_sharpe(mu, cov)
    W = resample_weights(rets, n_draws=120, seed=1)
    w_avg = W.mean(axis=0)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)
    assert (W >= -1e-8).all()
    assert W.std(axis=0).mean() > 0.01     # the sampling sensitivity is real
    assert abs(w_avg.sum() - 1) < 1e-6
    # In the fixed default world the averaged allocation is less concentrated,
    # but this is a reproducible observation rather than a universal theorem.
    assert np.abs(w_avg).max() <= np.abs(w_point).max() + 1e-9
