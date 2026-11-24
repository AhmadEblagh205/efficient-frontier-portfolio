# Methodology

This note documents the mathematical objects used by the repository and the distinction between a textbook efficient frontier and an estimation-error experiment.

## 1. Mean-variance portfolio

For annual expected returns `mu`, covariance `Sigma`, weights `w`, and risk-free rate `r_f`:

- expected return: `w' mu`
- variance: `w' Sigma w`
- volatility: `sqrt(w' Sigma w)`
- Sharpe ratio: `(w' mu - r_f) / sqrt(w' Sigma w)`

All optimisation in the repository is long-only and fully invested:

`w_i >= 0`, `sum_i w_i = 1`.

The maximum-Sharpe portfolio is found numerically with SLSQP. A point on the efficient frontier minimises variance subject to a target expected return and the same weight constraints.

## 2. Why estimation error matters

The optimiser never sees the true moments in real applications. It sees estimates `mu_hat` and `Sigma_hat`. Because the optimum is a nonlinear function of these inputs, sampling noise can turn into large weight changes.

`src/estimation_error.py` makes that mechanism observable with known truth.

The fixed annual synthetic moments are:

`mu = [0.04, 0.06, 0.10, 0.08, 0.14]`

with annual volatilities:

`[0.12, 0.18, 0.16, 0.20, 0.24]`

and a fixed positive correlation matrix. Daily Gaussian log returns are simulated from `N(mu/252, Sigma/252)`.

For each of 200 random worlds, a 30-year path is generated once. The 1-, 3-, 10- and 30-year studies use prefixes of that same path, which makes the sample-length comparison paired.

Four portfolios matter:

1. **oracle:** true `mu`, true `Sigma`;
2. **both estimated:** `mu_hat`, `Sigma_hat`;
3. **mean-only error:** `mu_hat`, true `Sigma`;
4. **covariance-only error:** true `mu`, `Sigma_hat`.

The half-L1 / total-variation distance between estimated and oracle portfolios is:

`0.5 * sum_i |w_i - w_i,oracle|`.

For two fully invested non-negative weight vectors this lies between 0 and 1 and has an intuitive interpretation as the minimum fraction of capital that must be reallocated to turn one portfolio into the other.

Every estimated portfolio is evaluated under the **true** `mu` and `Sigma`, avoiding the circularity of judging an optimiser only by the same noisy estimates that generated its weights.

## 3. Equal weight as a benchmark

`1/N` is not treated as theoretically optimal. It is a low-parameter benchmark. If a high-variance optimiser cannot reliably beat it in a known-truth experiment, that is evidence that input estimation is dominating the nominal optimisation advantage.

The benchmark therefore reports the fraction of simulated worlds in which equal weight has a higher true Sharpe than the sample-optimised portfolio.

## 4. Bootstrap weight stability

`src/resampled.py` samples rows of the observed return history with replacement, recomputes the moments and refits the maximum-Sharpe portfolio. The distribution of the resulting weights is a sampling-sensitivity diagnostic.

The current implementation uses an iid bootstrap. It is appropriate for the repository's iid synthetic default but would need a block/bootstrap variant if serial dependence were an explicit feature of the data.

The average bootstrap weight is shown as a stabilised descriptive summary. No theorem in this repository claims that this average is an optimal portfolio.

## 5. Ledoit-Wolf covariance shrinkage

The Ledoit-Wolf estimator shrinks the sample covariance toward a structured target. The trade-off is intentional: introduce some bias to reduce estimation variance.

The repository compares sample and shrunk covariance estimates on one half of a synthetic history and evaluates the resulting portfolios on the second half. Both portfolios use the same estimated mean vector, so this experiment isolates the effect of changing the covariance estimator only imperfectly; the known-truth decomposition separately demonstrates that expected-return noise remains the larger problem in the default world.

## 6. Black-Litterman accounting

Reverse optimisation gives equilibrium **excess** returns:

`pi = delta Sigma w_market`.

For view matrix `P`, view vector `Q`, scalar `tau` and view covariance `Omega`, the posterior excess-return mean can be written:

`mu_BL = pi + tau Sigma P' (P tau Sigma P' + Omega)^(-1) (Q - P pi)`.

This repository maps confidence `c in [0,1]` to:

`Omega = ((1-c)/c) diag(P tau Sigma P')`

for `0 < c < 1`, with exact endpoint handling:

- `c = 0`: return `pi`;
- `c = 1`: zero view variance in the stated linear view.

Because `pi` and `mu_BL` are excess returns, the ordinary `max_sharpe` routine receives `r_f + mu_BL` as total expected returns and subtracts `r_f` once. The earlier implementation passed excess returns directly and then subtracted `r_f` again; that was inconsistent and prevented zero confidence from recovering the market portfolio.

## 7. Scenario simulation

The GBM module simulates a fitted one-dimensional portfolio process with constant annual drift and volatility. It illustrates the distribution implied by those assumptions; it is not an empirical forecast or risk model for live investment decisions.
