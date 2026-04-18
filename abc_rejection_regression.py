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
- Regression-adjusted posterior overlays across multiple ABC tolerances

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

from abc_rejection import REFERENCE_RESULTS_PATH, REFERENCE_SUMMARY_SET_NAME
from runtime_summary import RUNTIME_SUMMARY_PATH, write_runtime_summary

REGRESSION_DIAGNOSTICS_DIR = "outputs/regression_adjustment"
OVERLAY_EPSILONS = (0.01, 0.05, 0.10)
DIAGNOSTIC_EPSILON = 0.01


def load_basic_abc_runtime_metrics():
    """
    Load the already-recorded basic ABC runtime row so regression adjustment
    inherits the same simulation budget in the combined CSV summary.
    """
    if not RUNTIME_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing runtime summary file: {RUNTIME_SUMMARY_PATH}. "
            "Run abc_rejection.py first so abc_rejection_regression.py can reuse "
            "the same simulation counts and wall-clock time."
        )

    runtime_df = pd.read_csv(RUNTIME_SUMMARY_PATH)
    basic_rows = runtime_df.loc[runtime_df["method_name"] == "abc_rejection"]
    if basic_rows.empty:
        raise ValueError(
            "outputs/runtime_summary.csv does not contain an abc_rejection row. "
            "Run abc_rejection.py first so regression adjustment can reuse the "
            "same simulation counts and wall-clock time."
        )

    basic_row = basic_rows.iloc[0]
    return {
        "total_simulator_calls": int(basic_row["total_simulator_calls"]),
        "wall_clock_seconds": float(basic_row["wall_clock_seconds"]),
    }

def compute_weights(distances, epsilon):
    """
    Epanechnikov-style kernel weights.
    Assumes distances are already accepted distances with d_i <= epsilon.
    """
    u = distances / epsilon
    w = 1.0 - u**2
    w[u > 1.0] = 0.0
    return w


def fit_regression_models(theta_accept, summaries_accept, accepted_distances):
    """
    Fit the weighted local linear regression models used for Beaumont adjustment.
    """
    epsilon = np.max(accepted_distances)
    weights = compute_weights(accepted_distances, epsilon)
    models = []

    for j in range(theta_accept.shape[1]):
        y = theta_accept[:, j]
        X = summaries_accept

        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)
        models.append(model)

    return models


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
    models = fit_regression_models(theta_accept, summaries_accept, accepted_distances)

    for j, model in enumerate(models):
        y = theta_accept[:, j]
        X = summaries_accept
        b = model.coef_

        theta_adj[:, j] = y - (X - s_obs) @ b

    return theta_adj, models


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


def get_accepted_subset_by_epsilon(reference_parameters,
                                   reference_summaries,
                                   reference_distances,
                                   acceptance_epsilon):
    """
    Select accepted rejection samples at the requested ABC quantile threshold.
    """
    reference_parameters = np.asarray(reference_parameters, dtype=np.float64)
    reference_summaries = np.asarray(reference_summaries, dtype=np.float64)
    reference_distances = np.asarray(reference_distances, dtype=np.float64)

    finite_mask = np.isfinite(reference_distances)
    finite_distances = reference_distances[finite_mask]
    if finite_distances.size == 0:
        raise ValueError("The saved rejection reference file contains no finite distances.")

    distance_threshold = float(np.quantile(finite_distances, acceptance_epsilon))
    accepted_mask = finite_mask & (reference_distances <= distance_threshold)
    if accepted_mask.sum() == 0:
        raise ValueError(
            f"No accepted samples found at epsilon={acceptance_epsilon:.2f}."
        )

    return {
        "acceptance_epsilon": float(acceptance_epsilon),
        "distance_threshold": distance_threshold,
        "accepted_parameters": reference_parameters[accepted_mask],
        "accepted_summaries": reference_summaries[accepted_mask],
        "accepted_distances": reference_distances[accepted_mask],
    }


def plot_before_after(theta_accept, theta_adj, output_dir):
    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    param_names = ["beta", "gamma", "rho"]

    for j, name in enumerate(param_names):
        plt.figure(figsize=(7, 4))
        plt.hist(
            theta_accept[:, j],
            bins=30,
            density=True,
            alpha=0.5,
            label=f"Basic ABC ({REFERENCE_SUMMARY_SET_NAME})",
        )
        plt.hist(
            theta_adj[:, j],
            bins=30,
            density=True,
            alpha=0.5,
            label="Regression-adjusted ABC",
        )
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.title(f"Posterior comparison: {name} ({REFERENCE_SUMMARY_SET_NAME})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_before_after_regression_adjustment.png"), dpi=200)
        plt.show()
        plt.close()


def plot_basic_vs_adjusted_overlay(theta_accept, theta_adj, output_dir):
    """
    Combined three-panel comparison of Reduced set J basic ABC vs regression adjustment.
    """
    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    param_names = ["beta", "gamma", "rho"]

    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 4.2))

    for param_idx, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(
            theta_accept[:, param_idx],
            bins=30,
            density=True,
            alpha=0.45,
            label=f"Basic ABC ({REFERENCE_SUMMARY_SET_NAME})",
        )
        ax.hist(
            theta_adj[:, param_idx],
            bins=30,
            density=True,
            alpha=0.45,
            label="Regression-adjusted ABC",
        )
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"{name} ({REFERENCE_SUMMARY_SET_NAME})")
        ax.legend()

    fig.suptitle("Regression-adjusted posterior vs basic ABC", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "basic_abc_vs_regression_adjusted_overlay.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def plot_adjusted_overlay_by_epsilon(adjusted_results_by_epsilon, output_dir):
    """
    Overlay regression-adjusted posterior histograms across multiple epsilon values.
    """
    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    param_names = ["beta", "gamma", "rho"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 4.2))

    for param_idx, (ax, name) in enumerate(zip(axes, param_names)):
        for color, result in zip(colors, adjusted_results_by_epsilon):
            theta_adj = result["theta_adj"]
            acceptance_epsilon = result["acceptance_epsilon"]
            ax.hist(
                theta_adj[:, param_idx],
                bins=30,
                density=True,
                alpha=0.4,
                color=color,
                label=f"ε={acceptance_epsilon:.2f} (n={theta_adj.shape[0]})",
            )
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"{name} ({REFERENCE_SUMMARY_SET_NAME})")
        ax.legend()

    fig.suptitle("Regression-adjusted posterior overlay across ABC tolerances", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "posterior_overlay_regression_adjustment_across_epsilons.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def plot_regression_projection_diagnostic(diagnostic_result, output_dir):
    """
    Plot accepted parameters against the fitted summary-discrepancy projection.

    This is the multivariate analogue of the one-summary Beaumont-style scatter:
    the x-axis is the projection of (s_i - s_obs) onto the fitted regression
    direction for each parameter.

    We generate this diagnostic only at ε = 0.01 because:
    1. it is the project's reference rejection tolerance used for the main
       regression-adjusted posterior and downstream methods; and
    2. it is the tightest tolerance among the compared values, so it gives the
       most relevant local-linear diagnostic around the observed summaries.
    """
    output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    param_names = ["beta", "gamma", "rho"]

    acceptance_epsilon = diagnostic_result["acceptance_epsilon"]
    theta_accept = diagnostic_result["accepted_parameters"]
    summaries_accept = diagnostic_result["accepted_summaries"]
    observed_summary = diagnostic_result["observed_summary"]
    theta_adj = diagnostic_result["theta_adj"]
    models = diagnostic_result["models"]

    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 4.2))

    for param_idx, (ax, name, model) in enumerate(zip(axes, param_names, models)):
        centered_summaries = summaries_accept - observed_summary
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        coefficient_norm = float(np.linalg.norm(coefficients))
        intercept_at_observed = float(model.predict(observed_summary.reshape(1, -1))[0])
        adjusted_mean = float(np.mean(theta_adj[:, param_idx]))

        if coefficient_norm <= 1e-12:
            projected_discrepancy = np.zeros(centered_summaries.shape[0], dtype=np.float64)
            regression_slope = 0.0
        else:
            direction = coefficients / coefficient_norm
            projected_discrepancy = centered_summaries @ direction
            regression_slope = coefficient_norm

        ax.scatter(
            projected_discrepancy,
            theta_accept[:, param_idx],
            s=6,
            alpha=0.15,
            color="#5fa8ff",
            edgecolors="none",
        )

        x_min = float(np.min(projected_discrepancy))
        x_max = float(np.max(projected_discrepancy))
        if abs(x_max - x_min) <= 1e-12:
            x_min -= 1.0
            x_max += 1.0
        x_line = np.linspace(x_min, x_max, 200)
        y_line = intercept_at_observed + regression_slope * x_line

        ax.plot(
            x_line,
            y_line,
            color="#f5a623",
            linewidth=2.0,
            label=f"Regression (|β̂|={regression_slope:.2f})",
        )
        ax.axhline(
            adjusted_mean,
            color="red",
            linestyle="--",
            linewidth=1.6,
            label="Adjusted post. mean",
        )
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Projected summary discrepancy")
        ax.set_ylabel(name)
        ax.set_title(f"{name} (ε={acceptance_epsilon:.2f})")
        ax.legend()

    fig.suptitle(
        f"Accepted samples regression diagnostic ({REFERENCE_SUMMARY_SET_NAME}, ε={acceptance_epsilon:.2f})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"regression_projection_diagnostic_eps-{acceptance_epsilon:.4f}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def save_adjusted_samples(theta_adj, output_dir, filename="abc_regression_adjusted_output.csv"):
    output_dir = os.path.join(output_dir, "param_estimates")
    os.makedirs(output_dir, exist_ok=True)
    adjusted_samples_df = pd.DataFrame(theta_adj, columns=["beta", "gamma", "rho"])
    adjusted_samples_df.to_csv(
        os.path.join(output_dir, filename),
        index=False
    )


def main():
    data = np.load(REFERENCE_RESULTS_PATH)
    basic_runtime_metrics = load_basic_abc_runtime_metrics()

    if not all(key in data for key in ("reference_parameters", "reference_summaries", "distances")):
        raise ValueError(
            "The saved rejection reference file is missing the full reference pool needed "
            "to build regression-adjusted overlays at epsilon=0.05 and 0.10. "
            "Re-run abc_rejection.py first."
        )

    reference_parameters = data["reference_parameters"]
    reference_summaries = data["reference_summaries"]
    reference_distances = data["distances"]
    accepted_parameters = data["accepted_parameters"]
    accepted_summaries = data["accepted_summaries"]
    observed_summary = data["observed_summary"]
    accepted_distances = data["accepted_distances"]

    theta_adj, _ = regression_adjustment(
        theta_accept=accepted_parameters,
        summaries_accept=accepted_summaries,
        s_obs=observed_summary,
        accepted_distances=accepted_distances,
    )

    print_posterior_comparison(accepted_parameters, theta_adj)
    plot_before_after(accepted_parameters, theta_adj, REGRESSION_DIAGNOSTICS_DIR)
    plot_basic_vs_adjusted_overlay(accepted_parameters, theta_adj, REGRESSION_DIAGNOSTICS_DIR)
    save_adjusted_samples(theta_adj, REGRESSION_DIAGNOSTICS_DIR)

    raw_results_by_epsilon = []
    adjusted_results_by_epsilon = []
    for acceptance_epsilon in OVERLAY_EPSILONS:
        accepted_subset = get_accepted_subset_by_epsilon(
            reference_parameters,
            reference_summaries,
            reference_distances,
            acceptance_epsilon,
        )
        theta_adj_eps, models_eps = regression_adjustment(
            theta_accept=accepted_subset["accepted_parameters"],
            summaries_accept=accepted_subset["accepted_summaries"],
            s_obs=observed_summary,
            accepted_distances=accepted_subset["accepted_distances"],
        )
        raw_results_by_epsilon.append(accepted_subset)
        adjusted_results_by_epsilon.append(
            {
                "acceptance_epsilon": accepted_subset["acceptance_epsilon"],
                "distance_threshold": accepted_subset["distance_threshold"],
                "accepted_parameters": accepted_subset["accepted_parameters"],
                "accepted_summaries": accepted_subset["accepted_summaries"],
                "observed_summary": observed_summary,
                "theta_adj": theta_adj_eps,
                "models": models_eps,
            }
        )
        save_adjusted_samples(
            theta_adj_eps,
            REGRESSION_DIAGNOSTICS_DIR,
            filename=f"abc_regression_adjusted_output_eps-{acceptance_epsilon:.4f}.csv",
        )

    diagnostic_result = next(
        result for result in adjusted_results_by_epsilon
        if abs(result["acceptance_epsilon"] - DIAGNOSTIC_EPSILON) <= 1e-12
    )

    plot_adjusted_overlay_by_epsilon(adjusted_results_by_epsilon, REGRESSION_DIAGNOSTICS_DIR)
    plot_regression_projection_diagnostic(diagnostic_result, REGRESSION_DIAGNOSTICS_DIR)

    write_runtime_summary(
        method_name="abc_rejection_regression",
        total_simulator_calls=basic_runtime_metrics["total_simulator_calls"],
        wall_clock_seconds=basic_runtime_metrics["wall_clock_seconds"],
        posterior_sample_size=int(theta_adj.shape[0]),
        acceptance_rate=(
            float(theta_adj.shape[0] / basic_runtime_metrics["total_simulator_calls"])
            if basic_runtime_metrics["total_simulator_calls"] > 0 else None
        ),
    )

    print(f"\nSaved adjusted samples and before/after plots for {REFERENCE_SUMMARY_SET_NAME}.")


if __name__ == "__main__":
    main()
