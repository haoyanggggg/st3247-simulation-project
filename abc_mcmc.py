"""
Approximate Bayesian Computation MCMC (ABC-MCMC) for the adaptive SIR model.

Date: 2026-04-11

Description
-----------
This script extends the existing ABC rejection workflow to an ABC-MCMC sampler
for the posterior distribution of (beta, gamma, rho). The ABC calibration for
the chosen reference summary set is loaded from the saved rejection reference run rather than
recomputed inside this script.

Key Design Choices
------------------
- Summary statistics:
    Reference summary set from `abc_rejection.py`

- Distance function:
    Euclidean distance on standardized summaries

- Normalization:
    Simulation-based mean and standard deviation loaded from the same
    prior-predictive reference run used for rejection ABC

- ABC threshold:
    Fixed as the empirical acceptance quantile loaded from that same rejection
    reference run

- Proposal:
    Symmetric Gaussian random walk inside the same uniform prior support

Outputs
-------
- Reference rejection samples under the same chosen summary-set configuration
- Posterior samples after burn-in
- Trace plots for beta, gamma, rho
- Rejection-vs-MCMC posterior comparison histograms
- Saved chain diagnostics and acceptance rate
"""

###############
#
#
# 1. IMPORT LIBRARIES
#
#
###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from simulator import simulate
from abc_rejection import (
    N,
    PARAMETER_NAMES,
    SUMMARY_SET_INDICES,
    REFERENCE_SUMMARY_SET_NAME,
    REFERENCE_RESULTS_PATH,
    summary_statistics_name,
    epsilon,
    degrees,
    seed,
    build_simulation_context,
    compute_linear_slope,
    early_time_window_min,
    early_time_window_max,
    early_time_points_centered,
    early_time_points_denom,
    late_time_window_min,
    late_time_window_max,
    late_time_points_centered,
    late_time_points_denom,
)


###############
#
#
# 2. GLOBAL VARIABLES
#
#
###############
SUMMARY_SET_NAME = REFERENCE_SUMMARY_SET_NAME
SUMMARY_INDICES = SUMMARY_SET_INDICES[SUMMARY_SET_NAME]
SUMMARY_NAMES = [summary_statistics_name[idx] for idx in SUMMARY_INDICES]

# `N_proposals` counts Metropolis proposals. The stored chain also includes
# the initial accepted state at iteration 0.
N_proposals = 30_000
N_mcmc = N_proposals + 1
burn_in = 3000
thinning = 1
proposal_scales = np.array([0.035, 0.012, 0.06], dtype=np.float64)
PRIOR_LOWER = np.array([0.05, 0.02, 0.0], dtype=np.float64)
PRIOR_UPPER = np.array([0.5, 0.2, 0.8], dtype=np.float64)

BASE_DIR = Path(__file__).resolve().parent
ABC_MCMC_DIR = BASE_DIR / "outputs" / "abc_mcmc"
ABC_MCMC_PLOTS_DIR = ABC_MCMC_DIR / "plots"
ABC_MCMC_PARAM_DIR = ABC_MCMC_DIR / "param_estimates"
###############
#
#
# 3. HELPER FUNCTIONS
#
#
###############
def load_reference_rejection_results(reference_path: Path = REFERENCE_RESULTS_PATH) -> dict:
    """
    Load the saved rejection calibration used as the MCMC reference.
    """
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing rejection reference file: {reference_path}. "
            "Run abc_rejection.py first to generate it."
        )

    required_keys = (
        "reference_parameters",
        "reference_summaries",
        "accepted_parameters",
        "accepted_summaries",
        "accepted_distances",
        "acceptance_epsilon",
        "observed_summary",
        "standardized_observed",
        "summary_mu",
        "summary_sigma",
        "zero_sigma_mask",
        "distances",
        "distance_threshold",
        "initial_parameters",
        "initial_summary",
        "initial_distance",
        "summary_indices",
    )

    with np.load(reference_path, allow_pickle=True) as data:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                "The saved rejection reference file is missing required fields: "
                f"{missing_keys}. Re-run abc_rejection.py to regenerate "
                f"{reference_path.name} with the full {SUMMARY_SET_NAME} calibration."
            )

        summary_indices = np.asarray(data["summary_indices"], dtype=np.int64)
        if not np.array_equal(summary_indices, np.asarray(SUMMARY_INDICES, dtype=np.int64)):
            raise ValueError(
                f"The saved rejection reference file does not match {SUMMARY_SET_NAME}. "
                "Re-run abc_rejection.py for the current summary-set configuration."
            )

        reference_results = {
            "reference_parameters": np.asarray(data["reference_parameters"], dtype=np.float64),
            "reference_summaries": np.asarray(data["reference_summaries"], dtype=np.float64),
            "accepted_parameters": np.asarray(data["accepted_parameters"], dtype=np.float64),
            "accepted_summaries": np.asarray(data["accepted_summaries"], dtype=np.float64),
            "accepted_distances": np.asarray(data["accepted_distances"], dtype=np.float64),
            "acceptance_epsilon": float(np.asarray(data["acceptance_epsilon"]).item()),
            "observed_summary_subset": np.asarray(data["observed_summary"], dtype=np.float64),
            "standardized_observed": np.asarray(data["standardized_observed"], dtype=np.float64),
            "summary_mu": np.asarray(data["summary_mu"], dtype=np.float64),
            "summary_sigma": np.asarray(data["summary_sigma"], dtype=np.float64),
            "zero_sigma_mask": np.asarray(data["zero_sigma_mask"], dtype=bool),
            "distances": np.asarray(data["distances"], dtype=np.float64),
            "distance_threshold": float(np.asarray(data["distance_threshold"]).item()),
            "initial_parameters": np.asarray(data["initial_parameters"], dtype=np.float64),
            "initial_summary": np.asarray(data["initial_summary"], dtype=np.float64),
            "initial_distance": float(np.asarray(data["initial_distance"]).item()),
        }

    if reference_results["accepted_parameters"].shape[0] == 0:
        raise ValueError("The saved rejection reference file contains no accepted samples.")

    return reference_results


def in_prior_support(parameters: np.ndarray) -> bool:
    """Return True when parameters lie inside the rectangular prior support."""
    return bool(np.all(parameters >= PRIOR_LOWER) and np.all(parameters <= PRIOR_UPPER))


def simulate_summary_statistics(parameters: np.ndarray,
                                rng: np.random.Generator,
                                simulation_context: dict) -> np.ndarray:
    """
    Simulate the model at a fixed parameter vector and compute the full summary vector.
    """
    beta, gamma, rho = parameters

    infected_fraction, rewire_counts, degree_histogram = simulate(
        beta=float(beta),
        gamma=float(gamma),
        rho=float(rho),
        rng=rng,
        simulation_context=simulation_context,
    )

    max_infection_frac = np.max(infected_fraction)
    time_to_peak = np.argmax(infected_fraction)

    early_log_infection_fraction = np.log(
        infected_fraction[early_time_window_min:early_time_window_max] + epsilon
    )
    slope_early_infection = compute_linear_slope(
        early_log_infection_fraction,
        early_time_points_centered,
        early_time_points_denom,
    )

    early_log_rewire_counts = np.log(
        rewire_counts[early_time_window_min:early_time_window_max] + epsilon
    )
    slope_rewire = compute_linear_slope(
        early_log_rewire_counts,
        early_time_points_centered,
        early_time_points_denom,
    )

    mean_degree = np.sum(degrees * degree_histogram) / N
    mean_degree_sq = np.sum((degrees ** 2) * degree_histogram) / N
    var_degree = mean_degree_sq - mean_degree ** 2

    late_log_infection_fraction = np.log(
        infected_fraction[late_time_window_min:late_time_window_max] + epsilon
    )
    slope_late_infection = compute_linear_slope(
        late_log_infection_fraction,
        late_time_points_centered,
        late_time_points_denom,
    )

    rewiring_per_infection = (
        np.sum(rewire_counts, dtype=np.float64) /
        (np.sum(infected_fraction, dtype=np.float64) + epsilon)
    )

    peak_val = np.max(infected_fraction)
    half_peak = 0.5 * peak_val
    above_half = np.where(infected_fraction >= half_peak)[0]
    peak_width_half_max = float(above_half[-1] - above_half[0]) if above_half.size >= 2 else 0.0

    full_summary = np.array(
        [
            max_infection_frac,
            time_to_peak,
            slope_early_infection,
            slope_rewire,
            var_degree,
            slope_late_infection,
            rewiring_per_infection,
            peak_width_half_max,
        ],
        dtype=np.float64,
    )

    return full_summary[list(SUMMARY_INDICES)]

def standardize_with_reference(summary_statistics: np.ndarray,
                               summary_mu: np.ndarray,
                               summary_sigma: np.ndarray,
                               zero_sigma_mask: np.ndarray) -> np.ndarray:
    """Standardize summaries using the rejection-reference scaling terms."""
    summary_statistics = np.asarray(summary_statistics, dtype=np.float64)
    standardized_summary = (summary_statistics - summary_mu) / summary_sigma
    standardized_summary[..., zero_sigma_mask] = 0.0
    return standardized_summary


def compute_distance(summary_statistics: np.ndarray,
                     standardized_observed: np.ndarray,
                     summary_mu: np.ndarray,
                     summary_sigma: np.ndarray,
                     zero_sigma_mask: np.ndarray) -> float:
    """Compute the ABC distance under the fixed rejection-reference scaling."""
    standardized_summary = standardize_with_reference(
        summary_statistics,
        summary_mu,
        summary_sigma,
        zero_sigma_mask,
    )
    return float(np.linalg.norm(standardized_summary - standardized_observed))


def run_abc_mcmc(rng: np.random.Generator,
                 simulation_context: dict,
                 standardized_observed: np.ndarray,
                 summary_mu: np.ndarray,
                 summary_sigma: np.ndarray,
                 zero_sigma_mask: np.ndarray,
                 distance_threshold: float,
                 initial_parameters: np.ndarray,
                 initial_summary: np.ndarray,
                 initial_distance: float) -> dict:
    """
    Run ABC-MCMC with a symmetric Gaussian random walk.

    With a uniform prior on a rectangular support and a symmetric proposal,
    the Metropolis-Hastings ratio is 1 whenever the proposal stays inside the
    support and satisfies the ABC distance threshold.
    """
    chain = np.zeros((N_mcmc, len(PARAMETER_NAMES)), dtype=np.float64)
    chain_summaries = np.zeros((N_mcmc, len(SUMMARY_INDICES)), dtype=np.float64)
    distances = np.zeros(N_mcmc, dtype=np.float64)
    accepted_moves = np.zeros(N_mcmc, dtype=bool)

    current_parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    current_summary = np.asarray(initial_summary, dtype=np.float64).copy()
    current_distance = float(initial_distance)

    chain[0] = current_parameters
    chain_summaries[0] = current_summary
    distances[0] = current_distance

    for iteration in tqdm(range(1, N_mcmc), total=N_proposals, desc="Running ABC-MCMC"):
        proposed_parameters = current_parameters + rng.normal(
            loc=0.0,
            scale=proposal_scales,
            size=len(PARAMETER_NAMES),
        )

        if in_prior_support(proposed_parameters):
            proposed_summary = simulate_summary_statistics(
                proposed_parameters,
                rng,
                simulation_context,
            )
            proposed_distance = compute_distance(
                proposed_summary,
                standardized_observed,
                summary_mu,
                summary_sigma,
                zero_sigma_mask,
            )

            if np.isfinite(proposed_distance) and proposed_distance <= distance_threshold:
                current_parameters = proposed_parameters
                current_summary = proposed_summary
                current_distance = proposed_distance
                accepted_moves[iteration] = True

        chain[iteration] = current_parameters
        chain_summaries[iteration] = current_summary
        distances[iteration] = current_distance

    acceptance_rate = accepted_moves[1:].mean()

    return {
        "chain": chain,
        "chain_summaries": chain_summaries,
        "distances": distances,
        "accepted_moves": accepted_moves,
        "acceptance_rate": float(acceptance_rate),
    }


def get_posterior_samples(chain: np.ndarray) -> np.ndarray:
    """Apply burn-in and thinning to obtain the retained posterior samples."""
    return np.asarray(chain, dtype=np.float64)[burn_in::thinning]


def plot_trace(chain: np.ndarray, output_dir: Path, acceptance_rate: float) -> None:
    """Save trace plots for the ABC-MCMC chain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(PARAMETER_NAMES), 1, figsize=(10, 8), sharex=True)

    for param_idx, (ax, param_name) in enumerate(zip(axes, PARAMETER_NAMES)):
        ax.plot(chain[:, param_idx], linewidth=0.8)
        ax.axvline(burn_in, color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel(param_name)
        ax.set_title(f"Trace plot: {param_name}")

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(
        f"ABC-MCMC traces ({SUMMARY_SET_NAME}, acceptance rate = {acceptance_rate:.3f})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "abc_mcmc_trace_plots.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_posterior_comparison(reference_accepted_parameters: np.ndarray,
                              posterior_samples: np.ndarray,
                              output_dir: Path,
                              acceptance_rate: float) -> None:
    """Save rejection-vs-MCMC posterior comparison histograms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(15, 4.5))

    for param_idx, (ax, param_name) in enumerate(zip(axes, PARAMETER_NAMES)):
        ax.hist(
            reference_accepted_parameters[:, param_idx],
            bins=30,
            density=True,
            alpha=0.45,
            label=f"ABC rejection (n={reference_accepted_parameters.shape[0]})",
        )
        ax.hist(
            posterior_samples[:, param_idx],
            bins=30,
            density=True,
            alpha=0.45,
            label=f"ABC-MCMC (n={posterior_samples.shape[0]})",
        )
        ax.set_title(f"Posterior comparison: {param_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(
        f"ABC rejection vs ABC-MCMC ({SUMMARY_SET_NAME}, acceptance rate = {acceptance_rate:.3f})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        output_dir / "abc_rejection_vs_mcmc_posterior_histograms.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def save_chain_outputs(chain_results: dict,
                       reference_results: dict) -> None:
    """Save posterior samples, diagnostic plots, and chain metadata."""
    ABC_MCMC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ABC_MCMC_PARAM_DIR.mkdir(parents=True, exist_ok=True)

    reference_accepted_parameters = np.asarray(
        reference_results["accepted_parameters"],
        dtype=np.float64,
    )
    posterior_samples = get_posterior_samples(chain_results["chain"])
    reference_accepted_df = pd.DataFrame(reference_accepted_parameters, columns=PARAMETER_NAMES)
    reference_accepted_df.to_csv(
        ABC_MCMC_PARAM_DIR / "abc_rejection_reference_output.csv",
        index=False,
    )
    posterior_samples_df = pd.DataFrame(posterior_samples, columns=PARAMETER_NAMES)
    posterior_samples_df.to_csv(
        ABC_MCMC_PARAM_DIR / "abc_mcmc_output.csv",
        index=False,
    )

    diagnostics_df = pd.DataFrame(
        {
            "metric": [
                "summary_set",
                "reference_simulations",
                "reference_accepted_samples",
                "mcmc_proposals",
                "mcmc_chain_states",
                "burn_in",
                "thinning",
                "acceptance_epsilon",
                "distance_threshold",
                "acceptance_rate",
                "posterior_sample_size",
            ],
            "value": [
                SUMMARY_SET_NAME,
                reference_results["reference_parameters"].shape[0],
                reference_accepted_parameters.shape[0],
                N_proposals,
                N_mcmc,
                burn_in,
                thinning,
                reference_results["acceptance_epsilon"],
                chain_results["distance_threshold"],
                chain_results["acceptance_rate"],
                posterior_samples.shape[0],
            ],
        }
    )
    diagnostics_df.to_csv(ABC_MCMC_DIR / "abc_mcmc_diagnostics.csv", index=False)

    np.savez(
        ABC_MCMC_DIR / "abc_mcmc_chain.npz",
        chain=chain_results["chain"],
        chain_summaries=chain_results["chain_summaries"],
        distances=chain_results["distances"],
        accepted_moves=chain_results["accepted_moves"],
        reference_parameters=reference_results["reference_parameters"],
        reference_summaries=reference_results["reference_summaries"],
        rejection_accepted_parameters=reference_accepted_parameters,
        rejection_accepted_summaries=reference_results["accepted_summaries"],
        rejection_accepted_distances=reference_results["accepted_distances"],
        posterior_samples=posterior_samples,
        observed_summary=reference_results["observed_summary_subset"],
        standardized_observed=reference_results["standardized_observed"],
        summary_mu=reference_results["summary_mu"],
        summary_sigma=reference_results["summary_sigma"],
        zero_sigma_mask=reference_results["zero_sigma_mask"],
        summary_indices=np.asarray(SUMMARY_INDICES, dtype=np.int64),
        summary_names=np.asarray(SUMMARY_NAMES, dtype=object),
        distance_threshold=chain_results["distance_threshold"],
        acceptance_rate=chain_results["acceptance_rate"],
        burn_in=burn_in,
        thinning=thinning,
    )

    plot_trace(
        chain_results["chain"],
        ABC_MCMC_PLOTS_DIR,
        chain_results["acceptance_rate"],
    )
    plot_posterior_comparison(
        reference_accepted_parameters,
        posterior_samples,
        ABC_MCMC_PLOTS_DIR,
        chain_results["acceptance_rate"],
    )
def autocorrelation(x, max_lag=200):
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n

    acf = []
    for lag in range(1, max_lag):
        if lag >= n:
            break
        c = np.dot(x[:-lag], x[lag:]) / (n - lag)
        acf.append(c / var)
    return np.array(acf)

def effective_sample_size(x):
    acf = autocorrelation(x)
    
    # Geyer truncation (stop when autocorrelation becomes negative)
    positive_acf = acf[acf > 0]
    
    if len(positive_acf) == 0:
        return len(x)
    
    tau = 1 + 2 * np.sum(positive_acf)
    return len(x) / tau

###############
#
#
# 4. MAIN FUNCTION
#
#
###############
def main() -> None:
    """
    Execute the ABC-MCMC workflow using the chosen reference summary statistics.
    """
    rng = np.random.default_rng(seed)
    simulation_context = build_simulation_context(N)
    reference_results = load_reference_rejection_results()

    chain_results = run_abc_mcmc(
        rng=rng,
        simulation_context=simulation_context,
        standardized_observed=reference_results["standardized_observed"],
        summary_mu=reference_results["summary_mu"],
        summary_sigma=reference_results["summary_sigma"],
        zero_sigma_mask=reference_results["zero_sigma_mask"],
        distance_threshold=reference_results["distance_threshold"],
        initial_parameters=reference_results["initial_parameters"],
        initial_summary=reference_results["initial_summary"],
        initial_distance=reference_results["initial_distance"],
    )
    chain_results["distance_threshold"] = reference_results["distance_threshold"]

    save_chain_outputs(chain_results, reference_results)
    
    posterior_samples = get_posterior_samples(chain_results["chain"])
    print()
    for i, name in enumerate(PARAMETER_NAMES):
        ess = effective_sample_size(posterior_samples[:, i])
        print(f"{name}: ESS ≈ {ess:.1f} out of {len(posterior_samples)}\n")

    print(f"\nSummary set: {SUMMARY_SET_NAME}")
    print(f"ABC threshold quantile: {reference_results['acceptance_epsilon']:.4f}")
    print(f"Loaded reference file: {REFERENCE_RESULTS_PATH}")
    print(f"Reference rejection simulations: {reference_results['reference_parameters'].shape[0]}")
    print(f"Reference accepted samples: {reference_results['accepted_parameters'].shape[0]}")
    print(f"Distance threshold: {reference_results['distance_threshold']:.6f}")
    print(f"MCMC proposals: {N_proposals}")
    print(f"Stored chain states: {N_mcmc}")
    print(f"Acceptance rate: {chain_results['acceptance_rate']:.4f}")
    print(f"Saved posterior samples: {posterior_samples.shape[0]}")
    print("Saved reference rejection samples, ABC-MCMC outputs, and comparison plots.")


###############
#
#
# 5. RUN MAIN FUNCTION
#
#
###############
if __name__ == "__main__":
    main()
