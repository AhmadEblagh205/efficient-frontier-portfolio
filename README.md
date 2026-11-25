# Portfolio Optimisation Under Estimation Error

[![CI](https://github.com/AhmadEblagh205/efficient-frontier-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/efficient-frontier-portfolio/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this to see what happens to a mean-variance optimiser when the expected returns and covariance matrix it receives are only estimates. It starts from the textbook Markowitz frontier and then measures the estimation error directly.

The project is an independent analytical rebuild of work I first explored in a UCL project (graded 92%). All headline results use synthetic data where the true returns and covariance are known. They are not an investment track record.

## What it does

- Computes the long-only efficient frontier, the maximum-Sharpe and minimum-variance portfolios, the capital allocation line and an equal-weight curve (`src/frontier.py`).
- Simulates 200 paired five-asset histories and compares estimated portfolios with the oracle (`src/estimation_error.py`).
- Bootstraps one return history to see how much the optimal weights move (`src/resampled.py`).
- Compares sample covariance with Ledoit-Wolf shrinkage, and runs a Black-Litterman prior and view demonstration.
- Draws a GBM fan chart for the fitted portfolio, as a scenario illustration and not a forecast (`src/simulate.py`).

## Results

All results are from simulated data. The estimation-error study recomputes the long-only maximum-Sharpe portfolio after 1, 3, 10 and 30 years of daily data. At 3 years, the median half-L1 distance from the oracle weights is:

| Inputs estimated | Median weight error |
|---|---:|
| expected returns and covariance | 0.523 |
| expected returns only | 0.514 |
| covariance only | 0.045 |

Evaluated under the true distribution, the median sample-optimised portfolio has a Sharpe ratio of 0.502. The covariance-only portfolio reaches 0.663 against 0.664 for the oracle. An equal-weight portfolio beats the sample maximum-Sharpe portfolio in 76.5% of the 3-year simulations, and in 18.5% with 30 years of data. Most of the error in this setup comes from the expected returns.

- The bootstrap gives a mean per-asset weight standard deviation of 0.107.
- Ledoit-Wolf shrinkage intensity is small (mean 0.025). The out-of-sample Sharpe ratio is 0.736 for sample covariance and 0.733 for Ledoit-Wolf.
- With zero confidence in the Black-Litterman view the optimiser recovers the stated market weights. With full confidence the posterior enforces the stated view.

Figures are in `reports/figures/` and the numbers are in `reports/*.json`.

## How to run

```bash
pip install -r requirements.txt

python src/frontier.py --synthetic
python src/estimation_error.py --worlds 200
python src/resampled.py --draws 300
python src/shrinkage.py --synthetic
python src/black_litterman.py
python src/simulate.py --synthetic --horizon 252 --paths 2000

pytest -q
```

The frontier can also read a local price CSV or download prices through `yfinance`. The committed results and figures use synthetic data, so the repository runs offline. The tests cover portfolio constraints, frontier target returns, Ledoit-Wolf covariance validity, bootstrap variability, Black-Litterman limits and the estimation-error decomposition. CI runs them on push and pull request.

## Notes

The Black-Litterman code works in excess-return space, and the posterior is converted back to total returns before the Sharpe optimiser subtracts the risk-free rate. An earlier version subtracted the risk-free rate twice and failed its own zero-view check. The market weights and the view are illustrative assumptions. Equations and experiment design are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Licence

MIT, see [LICENSE](LICENSE).
