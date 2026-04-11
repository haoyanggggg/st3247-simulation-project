"""
Regression Adjustment for ABC Rejection on the Adaptive-Network SIR Model

Date: 2026-04-11

Description
-----------
This script applies Beaumont et al. (2002) local linear regression adjustment
to the accepted posterior samples produced by `abc_rejection.py`.

The workflow:
1. Load accepted rejection-ABC samples and summaries
2. Compute kernel weights from accepted ABC distances
3. Fit weighted linear regressions for each parameter
4. Adjust posterior samples toward the observed summaries
5. Compare raw and adjusted posteriors visually
6. Save adjusted parameter samples for downstream use

Key Design Choices
------------------
- Input source:
    Accepted samples saved by `abc_rejection.py`

- Adjustment method:
    Local linear regression with distance-based kernel weighting

- Weighting:
    Epanechnikov-style weights using the accepted ABC distances

Outputs
-------
- Regression-adjusted posterior samples
- Before/after posterior histogram comparisons

Notes
-----
- This is a post-processing step on top of rejection ABC, not a standalone
  inference method.
- The quality of the adjustment depends on the accepted sample size and the
  local linear approximation around the observed summaries.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from abc_rejection import REGRESSION_ADJUSTMENT_DIR

REGRESSION_DIAGNOSTICS_DIR = "outputs/regression_adjustment"

def compute_weights(distances, epsilon):
    """
    Epanechnikov-style kernel weights.
    Assumes distances are already accepted distances with d_i <= epsilon.
    """
    u = distances / epsilon
    w = 1.0 - u**2
    w[u > 1.0] = 0.0
    return w


def regression_adjustment(theta_accept, summaries_accept, s_obs, accepted_distances):
    """
    Beaumont et al. (2002) local linear regression adjustment.

    Parameters
    ----------
    theta_accept : ndarray of shape (n_accept, n_params)
        Accepted parameter samples.
    summaries_accept : ndarray of shape (n_accept, n_summaries)
        Corresponding accepted simulated summaries.
    s_obs : ndarray of shape (n_summaries,)
        Observed summary statistics.
    accepted_distances : ndarray of shape (n_accept,)
        Distances for accepted samples.

    Returns
    -------
    theta_adj : ndarray of shape (n_accept, n_params)
        Regression-adjusted parameter samples.
    """
    theta_adj = np.zeros_like(theta_accept, dtype=float)

    epsilon = np.max(accepted_distances)
    weights = compute_weights(accepted_distances, epsilon)

    for j in range(theta_accept.shape[1]):
        y = theta_accept[:, j]
        X = summaries_accept

        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)
        b = model.coef_

        theta_adj[:, j] = y - (X - s_obs) @ b

    return theta_adj


def posterior_summary(x):
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }


def print_posterior_comparison(theta_accept, theta_adj):
    param_names = ["beta", "gamma", "rho"]

    for j, name in enumerate(param_names):
        raw_stats = posterior_summary(theta_accept[:, j])
        adj_stats = posterior_summary(theta_adj[:, j])

        print(f"\n{name.upper()}")
        print("Raw ABC:")
        print(raw_stats)
        print("Adjusted ABC:")
        print(adj_stats)


def plot_before_after(theta_accept, theta_adj, output_dir):
    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    param_names = ["beta", "gamma", "rho"]

    for j, name in enumerate(param_names):
        plt.figure(figsize=(7, 4))
        plt.hist(theta_accept[:, j], bins=17, density=True, alpha=0.5, label="Rejection ABC")
        plt.hist(theta_adj[:, j], bins=17, density=True, alpha=0.5, label="Regression-adjusted ABC")
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.title(f"Posterior comparison: {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_before_after_regression_adjustment.png"), dpi=200)
        plt.show()
        plt.close()


def save_adjusted_samples(theta_adj, output_dir):
    output_dir = os.path.join(output_dir, "param_estimates")
    os.makedirs(output_dir, exist_ok=True)
    adjusted_samples_df = pd.DataFrame(theta_adj, columns=["beta", "gamma", "rho"])
    adjusted_samples_df.to_csv(
        os.path.join(output_dir, "abc_regression_adjusted_output.csv"),
        index=False
    )


def main():
    data = np.load(f"{REGRESSION_ADJUSTMENT_DIR}/abc_rejection_output.npz")

    accepted_parameters = data["accepted_parameters"]
    accepted_summaries = data["accepted_summaries"]
    observed_summary = data["observed_summary"]
    accepted_distances = data["accepted_distances"]

    theta_adj = regression_adjustment(
        theta_accept=accepted_parameters,
        summaries_accept=accepted_summaries,
        s_obs=observed_summary,
        accepted_distances=accepted_distances,
    )

    print_posterior_comparison(accepted_parameters, theta_adj)
    plot_before_after(accepted_parameters, theta_adj, REGRESSION_DIAGNOSTICS_DIR)
    save_adjusted_samples(theta_adj, REGRESSION_DIAGNOSTICS_DIR)

    print("\nSaved adjusted samples and before/after plots.")


if __name__ == "__main__":
    main()
