"""
Sequential Monte Carlo ABC (SMC-ABC) for the adaptive-network SIR model.

Date: 2026-04-12

Description
-----------
This script implements the Population Monte Carlo ABC algorithm described by
Beaumont et al. for the posterior distribution of (beta, gamma, rho).

The workflow:
1. Load the saved rejection-ABC calibration for the chosen reference summary set from `abc_rejection.py`
2. Build a decreasing sequence of ABC distance thresholds
3. Generate the first SMC population by sampling from the prior until the
   simulated-observed distance is below the stage-1 distance threshold
4. Propagate later populations by weighted resampling and Gaussian perturbation
5. Reweight particles using the Beaumont proposal-mixture correction
6. Compare the final SMC posterior against the rejection-ABC posterior visually
7. Save diagnostic outputs and posterior samples for downstream use

Key Design Choices
------------------
- Summary statistics:
    Reference summary set from `abc_rejection.py`

- Distance function:
    Euclidean distance on standardized summaries

- Normalization:
    Simulation-based mean and standard deviation loaded from the same
    rejection reference run used elsewhere in the project

- Tolerance schedule:
    Stage 1 uses a calibrated starting threshold, then later thresholds are
    tightened adaptively from the evolving SMC populations

- Perturbation kernel:
    Multivariate Gaussian random walk with covariance
    2 x empirical covariance of the current particle population

Outputs
-------
- Final weighted SMC particle population
- Final resampled SMC posterior samples
- Rejection-vs-SMC posterior comparison plots
- Stage-by-stage posterior progression plots
- Diagnostics for each SMC stage

Notes
-----
- The saved rejection-ABC output is used only for calibration quantities
  such as the summary scaling terms, the initial threshold calibration, and
  the comparison posterior plots.
- The SMC stage-1 population itself is generated from the prior, following
  the Beaumont PMC-ABC algorithm.
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
    epsilon as log_offset,
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

N_particles = 300
FINAL_RESAMPLE_SIZE = N_particles
INTERMEDIATE_TOLERANCE_QUANTILES = [0.10, 0.05, 0.02]

PRIOR_LOWER = np.array([0.05, 0.02, 0.0], dtype=np.float64)
PRIOR_UPPER = np.array([0.5, 0.2, 0.8], dtype=np.float64)
PRIOR_VOLUME = float(np.prod(PRIOR_UPPER - PRIOR_LOWER))

BASE_DIR = Path(__file__).resolve().parent
SMC_ABC_DIR = BASE_DIR / "outputs" / "smc_abc"
SMC_ABC_PLOTS_DIR = SMC_ABC_DIR / "plots"
SMC_ABC_PARAM_DIR = SMC_ABC_DIR / "param_estimates"
###############
#
#
# 3. HELPER FUNCTIONS
#
#
###############
def load_reference_rejection_results(reference_path: Path = REFERENCE_RESULTS_PATH) -> dict:
    """
    Load the saved rejection calibration used by the SMC workflow.
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
            "target_quantile": float(np.asarray(data["acceptance_epsilon"]).item()),
            "observed_summary_subset": np.asarray(data["observed_summary"], dtype=np.float64),
            "standardized_observed": np.asarray(data["standardized_observed"], dtype=np.float64),
            "summary_mu": np.asarray(data["summary_mu"], dtype=np.float64),
            "summary_sigma": np.asarray(data["summary_sigma"], dtype=np.float64),
            "zero_sigma_mask": np.asarray(data["zero_sigma_mask"], dtype=bool),
            "distances": np.asarray(data["distances"], dtype=np.float64),
            "distance_threshold": float(np.asarray(data["distance_threshold"]).item()),
        }

    finite_mask = np.isfinite(reference_results["distances"])
    if finite_mask.sum() == 0:
        raise ValueError("The saved rejection reference file contains no finite distances.")
    if reference_results["accepted_parameters"].shape[0] == 0:
        raise ValueError("The saved rejection reference file contains no accepted samples.")

    reference_results["finite_mask"] = finite_mask
    return reference_results


def build_tolerance_quantile_schedule(reference_results: dict) -> np.ndarray:
    """
    Build the decreasing quantile schedule that guides threshold tightening.
    """
    final_quantile = float(reference_results["target_quantile"])
    candidate_quantiles = list(INTERMEDIATE_TOLERANCE_QUANTILES)
    candidate_quantiles.append(final_quantile)

    quantile_schedule = []
    for quantile in candidate_quantiles:
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Invalid ABC quantile {quantile}. Must lie in (0, 1).")
        if quantile_schedule and abs(quantile - quantile_schedule[-1]) <= 1e-12:
            continue
        quantile_schedule.append(quantile)

    if any(
        quantile_schedule[i] <= quantile_schedule[i + 1]
        for i in range(len(quantile_schedule) - 1)
    ):
        raise ValueError(
            "Tolerance quantiles must be strictly decreasing. "
            f"Received {quantile_schedule}."
        )

    return np.asarray(quantile_schedule, dtype=np.float64)


def compute_initial_distance_threshold(reference_results: dict,
                                       initial_quantile: float) -> float:
    """
    Calibrate the stage-1 threshold from the saved rejection-reference distances.

    Stage 1 has no previous SMC population yet, so we use the same prior-predictive
    reference run already produced by `abc_rejection.py`.
    """
    finite_distances = reference_results["distances"][reference_results["finite_mask"]]
    final_quantile = float(reference_results["target_quantile"])

    if abs(initial_quantile - final_quantile) <= 1e-12:
        return float(reference_results["distance_threshold"])

    return float(np.quantile(finite_distances, initial_quantile))


def compute_adaptive_distance_threshold(previous_population: dict,
                                        previous_quantile: float,
                                        current_quantile: float) -> float:
    """
    Tighten the next tolerance using the evolving SMC population.

    We interpret the quantile schedule cumulatively, so moving from q_(t-1) to q_t
    corresponds to taking the relative quantile q_t / q_(t-1) of the previous
    population's accepted distances.
    """
    previous_quantile = float(previous_quantile)
    current_quantile = float(current_quantile)

    if not 0.0 < current_quantile < previous_quantile < 1.0:
        raise ValueError(
            "Adaptive SMC thresholds require 0 < current_quantile < previous_quantile < 1."
        )

    relative_quantile = current_quantile / previous_quantile
    previous_distances = np.asarray(previous_population["distances"], dtype=np.float64)
    adaptive_threshold = float(np.quantile(previous_distances, relative_quantile))
    previous_threshold = float(previous_population["distance_threshold"])

    if not np.isfinite(adaptive_threshold):
        raise ValueError("Adaptive SMC threshold became non-finite.")

    if adaptive_threshold >= previous_threshold:
        adaptive_threshold = float(np.nextafter(previous_threshold, -np.inf))

    return adaptive_threshold


def in_prior_support(parameters: np.ndarray) -> bool:
    """Return True when parameters lie inside the rectangular prior support."""
    parameters = np.asarray(parameters, dtype=np.float64)
    return bool(np.all(parameters >= PRIOR_LOWER) and np.all(parameters <= PRIOR_UPPER))


def prior_density(parameters: np.ndarray) -> float:
    """Uniform prior density on the rectangular support."""
    if not in_prior_support(parameters):
        return 0.0
    return 1.0 / PRIOR_VOLUME


def sample_prior_parameters(rng: np.random.Generator) -> np.ndarray:
    """Sample one parameter vector from the uniform prior."""
    return rng.uniform(PRIOR_LOWER, PRIOR_UPPER).astype(np.float64)


def simulate_summary_statistics(parameters: np.ndarray,
                                rng: np.random.Generator,
                                simulation_context: dict) -> np.ndarray:
    """
    Simulate the model at a fixed parameter vector and compute summary statistics.
    """
    beta, gamma, rho = np.asarray(parameters, dtype=np.float64)

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
        infected_fraction[early_time_window_min:early_time_window_max] + log_offset
    )
    slope_early_infection = compute_linear_slope(
        early_log_infection_fraction,
        early_time_points_centered,
        early_time_points_denom,
    )

    early_log_rewire_counts = np.log(
        rewire_counts[early_time_window_min:early_time_window_max] + log_offset
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
        infected_fraction[late_time_window_min:late_time_window_max] + log_offset
    )
    slope_late_infection = compute_linear_slope(
        late_log_infection_fraction,
        late_time_points_centered,
        late_time_points_denom,
    )

    rewiring_per_infection = (
        np.sum(rewire_counts, dtype=np.float64) /
        (np.sum(infected_fraction, dtype=np.float64) + log_offset)
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
    """Standardize summaries using the saved rejection-reference scaling."""
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


def systematic_resample(weights: np.ndarray,
                        rng: np.random.Generator,
                        n_samples: int | None = None) -> np.ndarray:
    """Systematic resampling for normalized importance weights."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    n_particles_local = len(weights) if n_samples is None else int(n_samples)

    positions = (rng.random() + np.arange(n_particles_local)) / n_particles_local
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0

    indices = np.zeros(n_particles_local, dtype=np.int64)
    particle_idx = 0
    cumulative_idx = 0

    while particle_idx < n_particles_local:
        if positions[particle_idx] < cumulative_sum[cumulative_idx]:
            indices[particle_idx] = cumulative_idx
            particle_idx += 1
        else:
            cumulative_idx += 1

    return indices


def effective_sample_size_from_weights(weights: np.ndarray) -> float:
    """Return the particle effective sample size for normalized weights."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights ** 2))


def empirical_covariance(particles: np.ndarray,
                         weights: np.ndarray | None = None) -> np.ndarray:
    """Return the empirical covariance of the particle population."""
    particles = np.asarray(particles, dtype=np.float64)
    n_params = particles.shape[1]

    if particles.shape[0] <= 1:
        return np.eye(n_params, dtype=np.float64) * 1e-8

    if weights is None:
        covariance = np.cov(particles, rowvar=False, ddof=1)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        weighted_mean = np.sum(particles * weights[:, None], axis=0)
        centered = particles - weighted_mean
        denominator = 1.0 - np.sum(weights ** 2)

        if denominator <= 1e-12:
            covariance = np.diag(np.maximum(np.var(particles, axis=0, ddof=1), 1e-8))
        else:
            covariance = (centered * weights[:, None]).T @ centered / denominator

    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = np.atleast_2d(covariance)
    return 0.5 * (covariance + covariance.T)


def build_beaumont_kernel_covariance(particles: np.ndarray,
                                     weights: np.ndarray | None = None) -> np.ndarray:
    """Beaumont PMC-ABC kernel covariance: twice the empirical covariance."""
    return 2.0 * empirical_covariance(particles, weights)


def stabilize_covariance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add diagonal jitter until the covariance matrix is positive definite."""
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    dim = covariance.shape[0]

    mean_variance = np.trace(covariance) / max(dim, 1)
    base_jitter = max(1e-10, 1e-8 * mean_variance) if np.isfinite(mean_variance) else 1e-8
    identity = np.eye(dim, dtype=np.float64)

    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        candidate_covariance = covariance + (base_jitter * multiplier) * identity
        try:
            candidate_cholesky = np.linalg.cholesky(candidate_covariance)
            return candidate_covariance, candidate_cholesky
        except np.linalg.LinAlgError:
            continue

    raise np.linalg.LinAlgError("Unable to stabilize the SMC proposal covariance.")


def log_multivariate_normal_density(x: np.ndarray,
                                    means: np.ndarray,
                                    cholesky_factor: np.ndarray) -> np.ndarray:
    """Evaluate log N(x | mean_j, covariance) for every row mean_j in means."""
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    diff = means - x
    solved = np.linalg.solve(cholesky_factor, diff.T)
    quadratic_form = np.sum(solved ** 2, axis=0)
    dim = means.shape[1]
    log_det = 2.0 * np.sum(np.log(np.diag(cholesky_factor)))
    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + quadratic_form)


def compute_beaumont_weights(current_particles: np.ndarray,
                             previous_particles: np.ndarray,
                             previous_weights: np.ndarray,
                             proposal_cholesky: np.ndarray) -> np.ndarray:
    """
    Compute the Beaumont et al. PMC-ABC importance weights for one population.
    """
    current_particles = np.asarray(current_particles, dtype=np.float64)
    previous_particles = np.asarray(previous_particles, dtype=np.float64)
    previous_weights = np.asarray(previous_weights, dtype=np.float64)
    previous_weights = previous_weights / previous_weights.sum()

    log_weights = np.full(current_particles.shape[0], -np.inf, dtype=np.float64)

    for particle_idx, particle in enumerate(current_particles):
        particle_prior_density = prior_density(particle)
        if particle_prior_density <= 0.0:
            continue

        log_kernel_densities = log_multivariate_normal_density(
            particle,
            previous_particles,
            proposal_cholesky,
        )
        max_log_density = np.max(log_kernel_densities)
        mixture_density = np.sum(
            previous_weights * np.exp(log_kernel_densities - max_log_density)
        )
        if not np.isfinite(mixture_density) or mixture_density <= 0.0:
            continue

        log_denominator = max_log_density + np.log(mixture_density)
        log_weights[particle_idx] = np.log(particle_prior_density) - log_denominator

    finite_mask = np.isfinite(log_weights)
    if not finite_mask.any():
        raise ValueError("All SMC-ABC importance weights became non-finite.")

    log_weights[~finite_mask] = -np.inf
    max_log_weight = np.max(log_weights[finite_mask])
    weights = np.exp(log_weights - max_log_weight)
    weights[~finite_mask] = 0.0

    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("SMC-ABC importance weights failed to normalize.")

    weights /= weight_sum
    return weights


def posterior_summary(x: np.ndarray) -> dict:
    """Return a compact posterior summary."""
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }


def print_posterior_comparison(reference_posterior: np.ndarray,
                               smc_posterior: np.ndarray) -> None:
    """Print a compact rejection-vs-SMC posterior comparison."""
    for param_idx, name in enumerate(PARAMETER_NAMES):
        rejection_stats = posterior_summary(reference_posterior[:, param_idx])
        smc_stats = posterior_summary(smc_posterior[:, param_idx])

        print(f"\n{name.upper()}")
        print("Rejection ABC:")
        print(rejection_stats)
        print("SMC-ABC:")
        print(smc_stats)


def draw_matched_comparison_samples(reference_posterior: np.ndarray,
                                    smc_posterior: np.ndarray,
                                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw equal-sized posterior samples for visual comparison plots.

    The larger posterior sample is downsampled without replacement to match the
    smaller one. This keeps the histogram comparison apples-to-apples.
    """
    reference_posterior = np.asarray(reference_posterior, dtype=np.float64)
    smc_posterior = np.asarray(smc_posterior, dtype=np.float64)

    matched_size = min(reference_posterior.shape[0], smc_posterior.shape[0])
    if matched_size <= 0:
        raise ValueError("Cannot build comparison plots from empty posterior samples.")

    if reference_posterior.shape[0] > matched_size:
        reference_indices = rng.choice(reference_posterior.shape[0], size=matched_size, replace=False)
        reference_plot_samples = reference_posterior[reference_indices]
    else:
        reference_plot_samples = reference_posterior.copy()

    if smc_posterior.shape[0] > matched_size:
        smc_indices = rng.choice(smc_posterior.shape[0], size=matched_size, replace=False)
        smc_plot_samples = smc_posterior[smc_indices]
    else:
        smc_plot_samples = smc_posterior.copy()

    return reference_plot_samples, smc_plot_samples


def sample_initial_population(distance_threshold: float,
                              tolerance_quantile: float,
                              rng: np.random.Generator,
                              simulation_context: dict,
                              reference_results: dict,
                              n_stages: int) -> dict:
    """
    Generate the stage-1 population by prior sampling until the distance
    falls below the stage-1 distance threshold.
    """
    particles = np.zeros((N_particles, len(PARAMETER_NAMES)), dtype=np.float64)
    summaries = np.zeros((N_particles, len(SUMMARY_INDICES)), dtype=np.float64)
    distances = np.zeros(N_particles, dtype=np.float64)
    total_attempts = 0

    for particle_idx in tqdm(
        range(N_particles),
        desc=f"Running SMC-ABC stage 1/{n_stages} (distance={distance_threshold:.6f})",
    ):
        while True:
            total_attempts += 1
            candidate_parameters = sample_prior_parameters(rng)
            candidate_summary = simulate_summary_statistics(
                candidate_parameters,
                rng,
                simulation_context,
            )
            candidate_distance = compute_distance(
                candidate_summary,
                reference_results["standardized_observed"],
                reference_results["summary_mu"],
                reference_results["summary_sigma"],
                reference_results["zero_sigma_mask"],
            )

            if np.isfinite(candidate_distance) and candidate_distance <= distance_threshold:
                particles[particle_idx] = candidate_parameters
                summaries[particle_idx] = candidate_summary
                distances[particle_idx] = candidate_distance
                break

    weights = np.full(N_particles, 1.0 / N_particles, dtype=np.float64)
    kernel_covariance = build_beaumont_kernel_covariance(particles)
    kernel_covariance, kernel_cholesky = stabilize_covariance(kernel_covariance)

    return {
        "stage": 1,
        "tolerance_quantile": float(tolerance_quantile),
        "distance_threshold": float(distance_threshold),
        "particles": particles,
        "summaries": summaries,
        "distances": distances,
        "weights": weights,
        "kernel_covariance": kernel_covariance,
        "kernel_cholesky": kernel_cholesky,
        "proposal_attempts": int(total_attempts),
        "acceptance_rate": float(N_particles / total_attempts),
        "ess": float(N_particles),
    }


def sample_mutated_population(previous_population: dict,
                              distance_threshold: float,
                              tolerance_quantile: float,
                              stage_number: int,
                              n_stages: int,
                              rng: np.random.Generator,
                              simulation_context: dict,
                              reference_results: dict) -> dict:
    """
    Generate one later SMC population using weighted resampling and mutation.
    """
    previous_particles = np.asarray(previous_population["particles"], dtype=np.float64)
    previous_weights = np.asarray(previous_population["weights"], dtype=np.float64)
    proposal_covariance = np.asarray(previous_population["kernel_covariance"], dtype=np.float64)
    proposal_cholesky = np.asarray(previous_population["kernel_cholesky"], dtype=np.float64)

    particles = np.zeros((N_particles, len(PARAMETER_NAMES)), dtype=np.float64)
    summaries = np.zeros((N_particles, len(SUMMARY_INDICES)), dtype=np.float64)
    distances = np.zeros(N_particles, dtype=np.float64)
    total_attempts = 0

    for particle_idx in tqdm(
        range(N_particles),
        desc=(
            f"Running SMC-ABC stage {stage_number}/{n_stages} "
            f"(distance={distance_threshold:.6f})"
        ),
    ):
        while True:
            total_attempts += 1
            ancestor_idx = rng.choice(N_particles, p=previous_weights)
            ancestor = previous_particles[ancestor_idx]
            candidate_parameters = rng.multivariate_normal(
                mean=ancestor,
                cov=proposal_covariance,
            )

            if not in_prior_support(candidate_parameters):
                continue

            candidate_summary = simulate_summary_statistics(
                candidate_parameters,
                rng,
                simulation_context,
            )
            candidate_distance = compute_distance(
                candidate_summary,
                reference_results["standardized_observed"],
                reference_results["summary_mu"],
                reference_results["summary_sigma"],
                reference_results["zero_sigma_mask"],
            )

            if np.isfinite(candidate_distance) and candidate_distance <= distance_threshold:
                particles[particle_idx] = candidate_parameters
                summaries[particle_idx] = candidate_summary
                distances[particle_idx] = candidate_distance
                break

    weights = compute_beaumont_weights(
        particles,
        previous_particles,
        previous_weights,
        proposal_cholesky,
    )
    kernel_covariance = build_beaumont_kernel_covariance(particles, weights)
    kernel_covariance, kernel_cholesky = stabilize_covariance(kernel_covariance)

    return {
        "stage": int(stage_number),
        "tolerance_quantile": float(tolerance_quantile),
        "distance_threshold": float(distance_threshold),
        "particles": particles,
        "summaries": summaries,
        "distances": distances,
        "weights": weights,
        "kernel_covariance": kernel_covariance,
        "kernel_cholesky": kernel_cholesky,
        "proposal_attempts": int(total_attempts),
        "acceptance_rate": float(N_particles / total_attempts),
        "ess": effective_sample_size_from_weights(weights),
    }


def run_smc_abc(rng: np.random.Generator,
                simulation_context: dict,
                reference_results: dict) -> dict:
    """
    Run Sequential Monte Carlo ABC following the Beaumont PMC-ABC algorithm.
    """
    quantile_schedule = build_tolerance_quantile_schedule(reference_results)
    n_stages = len(quantile_schedule)
    stage_populations = []
    distance_thresholds = [
        compute_initial_distance_threshold(reference_results, quantile_schedule[0])
    ]

    current_population = sample_initial_population(
        distance_threshold=distance_thresholds[0],
        tolerance_quantile=quantile_schedule[0],
        rng=rng,
        simulation_context=simulation_context,
        reference_results=reference_results,
        n_stages=n_stages,
    )
    stage_populations.append(current_population)

    for stage_idx in range(1, n_stages):
        next_threshold = compute_adaptive_distance_threshold(
            previous_population=current_population,
            previous_quantile=quantile_schedule[stage_idx - 1],
            current_quantile=quantile_schedule[stage_idx],
        )
        distance_thresholds.append(next_threshold)

        current_population = sample_mutated_population(
            previous_population=current_population,
            distance_threshold=next_threshold,
            tolerance_quantile=quantile_schedule[stage_idx],
            stage_number=stage_idx + 1,
            n_stages=n_stages,
            rng=rng,
            simulation_context=simulation_context,
            reference_results=reference_results,
        )
        stage_populations.append(current_population)

    final_population = stage_populations[-1]
    posterior_indices = systematic_resample(
        final_population["weights"],
        rng,
        n_samples=FINAL_RESAMPLE_SIZE,
    )
    posterior_samples = final_population["particles"][posterior_indices]

    stage_records = []
    for population in stage_populations:
        kernel_sd = np.sqrt(np.clip(np.diag(population["kernel_covariance"]), 0.0, None))
        stage_records.append(
            {
                "stage": population["stage"],
                "tolerance_quantile": population["tolerance_quantile"],
                "distance_threshold": population["distance_threshold"],
                "population_size": population["particles"].shape[0],
                "proposal_attempts": population["proposal_attempts"],
                "acceptance_rate": population["acceptance_rate"],
                "ess": population["ess"],
                "kernel_sd_beta": float(kernel_sd[0]),
                "kernel_sd_gamma": float(kernel_sd[1]),
                "kernel_sd_rho": float(kernel_sd[2]),
            }
        )

    return {
        "quantile_schedule": quantile_schedule,
        "distance_thresholds": np.asarray(distance_thresholds, dtype=np.float64),
        "stage_populations": stage_populations,
        "stage_records": stage_records,
        "final_population": final_population,
        "posterior_samples": posterior_samples,
    }


def plot_posterior_comparison(reference_posterior: np.ndarray,
                              smc_posterior: np.ndarray,
                              output_dir: Path,
                              final_threshold: float) -> None:
    """Save rejection-vs-SMC posterior comparison histograms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_size = reference_posterior.shape[0]

    for param_idx, param_name in enumerate(PARAMETER_NAMES):
        plt.figure(figsize=(7, 4))
        plt.hist(
            reference_posterior[:, param_idx],
            bins=17,
            density=True,
            alpha=0.5,
            label=f"Rejection ABC (n={matched_size})",
        )
        plt.hist(
            smc_posterior[:, param_idx],
            bins=17,
            density=True,
            alpha=0.5,
            label=f"SMC-ABC (n={matched_size})",
        )
        plt.xlabel(param_name)
        plt.ylabel("Density")
        plt.title(
            f"Posterior comparison: {param_name} "
            f"(final distance={final_threshold:.6f})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{param_name}_rejection_vs_smc_abc.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def plot_stage_progression(stage_populations: list[dict],
                           output_dir: Path) -> None:
    """Save posterior density overlays across the SMC tolerance schedule."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for param_idx, param_name in enumerate(PARAMETER_NAMES):
        plt.figure(figsize=(7, 4))
        for population in stage_populations:
            plt.hist(
                population["particles"][:, param_idx],
                bins=17,
                density=True,
                weights=population["weights"],
                alpha=0.35,
                label=(
                    f"Stage {population['stage']}: "
                    f"distance={population['distance_threshold']:.6f}"
                ),
            )
        plt.xlabel(param_name)
        plt.ylabel("Density")
        plt.title(f"SMC-ABC stage progression: {param_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{param_name}_smc_abc_stage_progression.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def save_smc_outputs(smc_results: dict,
                     reference_results: dict) -> None:
    """Save posterior samples, weighted particles, diagnostics, and plots."""
    SMC_ABC_DIR.mkdir(parents=True, exist_ok=True)
    SMC_ABC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SMC_ABC_PARAM_DIR.mkdir(parents=True, exist_ok=True)

    reference_posterior = np.asarray(reference_results["accepted_parameters"], dtype=np.float64)
    posterior_samples = np.asarray(smc_results["posterior_samples"], dtype=np.float64)
    final_population = smc_results["final_population"]
    final_particles = np.asarray(final_population["particles"], dtype=np.float64)
    final_weights = np.asarray(final_population["weights"], dtype=np.float64)
    comparison_rng = np.random.default_rng(seed)
    reference_plot_samples, smc_plot_samples = draw_matched_comparison_samples(
        reference_posterior,
        posterior_samples,
        comparison_rng,
    )

    pd.DataFrame(reference_posterior, columns=PARAMETER_NAMES).to_csv(
        SMC_ABC_PARAM_DIR / "abc_rejection_reference_output.csv",
        index=False,
    )
    pd.DataFrame(posterior_samples, columns=PARAMETER_NAMES).to_csv(
        SMC_ABC_PARAM_DIR / "smc_abc_output.csv",
        index=False,
    )

    weighted_particles_df = pd.DataFrame(final_particles, columns=PARAMETER_NAMES)
    weighted_particles_df["weight"] = final_weights
    weighted_particles_df.to_csv(
        SMC_ABC_PARAM_DIR / "smc_abc_weighted_particles.csv",
        index=False,
    )
    pd.DataFrame(reference_plot_samples, columns=PARAMETER_NAMES).to_csv(
        SMC_ABC_PARAM_DIR / "abc_rejection_reference_output_matched_for_plot.csv",
        index=False,
    )
    pd.DataFrame(smc_plot_samples, columns=PARAMETER_NAMES).to_csv(
        SMC_ABC_PARAM_DIR / "smc_abc_output_matched_for_plot.csv",
        index=False,
    )

    diagnostics_df = pd.DataFrame(smc_results["stage_records"])
    diagnostics_df.to_csv(SMC_ABC_DIR / "smc_abc_diagnostics.csv", index=False)

    save_payload = {
        "posterior_samples": posterior_samples,
        "final_particles": final_particles,
        "final_summaries": np.asarray(final_population["summaries"], dtype=np.float64),
        "final_distances": np.asarray(final_population["distances"], dtype=np.float64),
        "final_weights": final_weights,
        "reference_parameters": reference_results["reference_parameters"],
        "reference_summaries": reference_results["reference_summaries"],
        "rejection_accepted_parameters": reference_posterior,
        "rejection_accepted_summaries": reference_results["accepted_summaries"],
        "rejection_accepted_distances": reference_results["accepted_distances"],
        "observed_summary": reference_results["observed_summary_subset"],
        "standardized_observed": reference_results["standardized_observed"],
        "summary_mu": reference_results["summary_mu"],
        "summary_sigma": reference_results["summary_sigma"],
        "zero_sigma_mask": reference_results["zero_sigma_mask"],
        "reference_distance_threshold": float(reference_results["distance_threshold"]),
        "summary_indices": np.asarray(SUMMARY_INDICES, dtype=np.int64),
        "summary_names": np.asarray(SUMMARY_NAMES, dtype=object),
        "tolerance_quantiles": np.asarray(smc_results["quantile_schedule"], dtype=np.float64),
        "distance_thresholds": np.asarray(smc_results["distance_thresholds"], dtype=np.float64),
    }

    for population in smc_results["stage_populations"]:
        stage = population["stage"]
        save_payload[f"stage_{stage}_particles"] = np.asarray(population["particles"], dtype=np.float64)
        save_payload[f"stage_{stage}_summaries"] = np.asarray(population["summaries"], dtype=np.float64)
        save_payload[f"stage_{stage}_distances"] = np.asarray(population["distances"], dtype=np.float64)
        save_payload[f"stage_{stage}_weights"] = np.asarray(population["weights"], dtype=np.float64)
        save_payload[f"stage_{stage}_kernel_covariance"] = np.asarray(
            population["kernel_covariance"],
            dtype=np.float64,
        )

    np.savez(SMC_ABC_DIR / "smc_abc_population.npz", **save_payload)

    plot_posterior_comparison(
        reference_plot_samples,
        smc_plot_samples,
        SMC_ABC_PLOTS_DIR,
        smc_results["distance_thresholds"][-1],
    )
    plot_stage_progression(
        smc_results["stage_populations"],
        SMC_ABC_PLOTS_DIR,
    )


###############
#
#
# 4. MAIN FUNCTION
#
#
###############
def main() -> None:
    """
    Execute the SMC-ABC workflow using the chosen reference summary statistics.
    """
    rng = np.random.default_rng(seed)
    simulation_context = build_simulation_context(N)
    reference_results = load_reference_rejection_results()

    smc_results = run_smc_abc(
        rng=rng,
        simulation_context=simulation_context,
        reference_results=reference_results,
    )

    print_posterior_comparison(
        reference_results["accepted_parameters"],
        smc_results["posterior_samples"],
    )
    save_smc_outputs(smc_results, reference_results)

    print(f"\nSummary set: {SUMMARY_SET_NAME}")
    print(f"Loaded reference file: {REFERENCE_RESULTS_PATH}")
    print(f"Reference rejection simulations: {reference_results['reference_parameters'].shape[0]}")
    print(f"Reference accepted samples: {reference_results['accepted_parameters'].shape[0]}")
    print(f"SMC population size: {N_particles}")
    print(f"Reference rejection final threshold: {reference_results['distance_threshold']:.6f}")
    print(
        "Target quantile schedule: "
        + ", ".join(f"{quantile:.3f}" for quantile in smc_results["quantile_schedule"])
    )
    print(
        "Adaptive distance thresholds: "
        + ", ".join(
            f"{threshold:.6f}" for threshold in smc_results["distance_thresholds"]
        )
    )
    print(f"Saved final SMC posterior samples: {smc_results['posterior_samples'].shape[0]}")
    print("Saved weighted particles, diagnostics, and posterior comparison plots.")


###############
#
#
# 5. RUN MAIN FUNCTION
#
#
###############
if __name__ == "__main__":
    main()
