# Changelog

## 0.5.0 — defensibility and estimation-error audit
- Added a 200-world known-truth Monte Carlo benchmark that decomposes maximum-Sharpe weight error into expected-return and covariance-estimation components across 1, 3, 10 and 30 years of observations.
- Added an equal-weight benchmark and true-distribution evaluation so the optimiser is not judged by the same noisy moments used to fit it.
- Corrected the Black-Litterman accounting: equilibrium quantities are excess returns and are converted back to total returns before the Sharpe optimiser applies the risk-free rate.
- Replaced the former confidence scaling with explicit endpoint behaviour: zero confidence returns the equilibrium prior, while confidence one enforces the stated linear view.
- Added a Black-Litterman confidence-path figure and mathematical methodology note.
- Reframed resampled weights as a bootstrap stability diagnostic rather than a universal Michaud-optimality claim or proof that any specific allocation is a literal corner solution.
- Hardened the max-Sharpe optimiser to fail loudly if numerical optimisation fails.
- Expanded tests around Black-Litterman limits and the estimation-error decomposition.

## 0.4.0 — update round 8
Added bootstrap resampling of the return history and repeated maximum-Sharpe refits to visualise allocation sensitivity to sampling noise.

## 0.3.0 — update round 6
Added Black-Litterman reverse optimisation and a stated relative-return view. (The excess-return / risk-free accounting was corrected in 0.5.0.)

## 0.2.0 — update round 1
Added a Ledoit-Wolf covariance shrinkage experiment: half-sample estimation, half-sample out-of-sample Sharpe comparison against the sample covariance, with validity tests.

## 0.1.0
Initial release: Markowitz frontier, tangency, minimum variance, capital allocation line and Monte Carlo fan.
