"""
Approximate Bayesian Computation (ABC) for SIR Epidemic Model with Adaptive Network

Date: 2026-04-07

Description
-----------
This script implements the ABC rejection algorithm to infer the posterior 
distribution of parameters (beta, gamma, rho) in an adaptive SIR epidemic model.

The workflow:
1. Sample parameters from prior distributions
2. Simulate epidemic and network dynamics
3. Compute summary statistics
4. Standardize summaries
5. Compute distance between simulated and observed summaries
6. Accept simulations within a tolerance threshold (ε)
7. Analyze posterior samples and perform diagnostic checks

Key Design Choices
------------------
- Summary statistics:
    * Max infection fraction
    * Time to peak
    * Early infection growth rate
    * Early rewiring growth rate
    * Degree variance
    * Late infection decay rate
    * Rewiring per infection
    * Infection peak width at half maximum

- Distance function:
    Euclidean distance on standardized summaries

- Normalization:
    Simulation-based mean and standard deviation

- Tolerance:
    Top 1% closest simulations (ε = 1% quantile)

Outputs
-------
- Posterior samples (beta, gamma, rho)
- Diagnostic plots (posterior predictive checks)

Notes
-----
- The posterior is approximate due to non-sufficient summaries and non-zero ε.
- Results depend on choice of summaries and prior ranges.
"""

#####################
#
#
# 1. IMPORT LIBRARIES
#
#
#####################
import numpy as np
import pandas as pd
from simulator import simulate
from observed_summaries import get_obs_summaries
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re


##########################
#
#
# 2. GLOBAL VARIABLES
#
#
###########################
N = 200
N_sim = 100_000 # with threshold of 1%, we will have 300 accepted samples for posterior analysis
degree_counts_max = 30 + 1
epsilon = 1e-8
seed = 2026
# SUMMARY STATISTIC 1: Max infection fraction INFORMS BETA/GAMMA
# SUMMARY STATISTIC 2: Time to peak INFORMS BETA/GAMMA
# SUMMARY STATISTIC 3: Early growth rate INFORMS BETA/GAMMA
# SUMMARY STATISTIC 4: Mean rewire counts during early infection
# SUMMARY STATISTIC 5: Variance structure of degree counts INFORMS RHO
#                      ↑rho = distortion from Erdos-Renyi random graph's binomial distribution
#                      ↓rho = closer to binomial distribution
# SUMMARY STATISTIC 6: Decay structure of infection INFORMS RHO
# SUMMARY STATISTIC 7: Rewiring per infection INFORMS RHO
# SUMMARY STATISTIC 8: Infection peak width at half maximum INFORMS GAMMA

summary_statistics_name = [
    "Max infection fraction",
    "Time to peak",
    "Early growth rate of infection",
    "Early growth rate of rewiring",
    "Variance structure of degree counts",
    "Late decay rate of infection",
    "Rewiring per infection",
    "Infection peak width at half maximum",
]
MAX_INFECTION_IDX = 0
TIME_TO_PEAK_IDX = 1
EARLY_INFECTION_GROWTH_IDX = 2
EARLY_REWIRE_COUNT_IDX = 3
DEGREE_VARIANCE_IDX = 4
LATE_INFECTION_DECAY_IDX = 5
REWIRING_PER_INFECTION_IDX = 6
PEAK_WIDTH_HALF_MAX_IDX = 7

SUMMARY_SET_INDICES = {
    "Rich set": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_INFECTION_GROWTH_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
        LATE_INFECTION_DECAY_IDX,
        REWIRING_PER_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX,
    ),

    # lack of peak width statistic leads to worse inference of beta/gamma, which in turn leads to worse inference of rho since the network dynamics are driven by the epidemic dynamics
    "Reduced set A": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_INFECTION_GROWTH_IDX,
        LATE_INFECTION_DECAY_IDX,
    ),
    "Reduced set B": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_INFECTION_GROWTH_IDX,
        LATE_INFECTION_DECAY_IDX,
        EARLY_REWIRE_COUNT_IDX,
    ),
    "Reduced set C": (
        EARLY_INFECTION_GROWTH_IDX,
        LATE_INFECTION_DECAY_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
    ),
    "Reduced set D": (
        MAX_INFECTION_IDX,
        LATE_INFECTION_DECAY_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
    ),
    "Reduced set E": (
        MAX_INFECTION_IDX,
        EARLY_INFECTION_GROWTH_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
    ),
    "Reduced set F": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        LATE_INFECTION_DECAY_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
    ),
    "Reduced set G": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        LATE_INFECTION_DECAY_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ),
    "Reduced set H": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ),
    "Reduced set I": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX
    ),

}

REFERENCE_SUMMARY_SET_NAME = "Reduced set I"
REFERENCE_SUMMARY_SET_INDICES = SUMMARY_SET_INDICES[REFERENCE_SUMMARY_SET_NAME]
REFERENCE_SUMMARY_SET_SLUG = REFERENCE_SUMMARY_SET_NAME.lower().replace(" ", "_")
PARAMETER_NAMES = ("beta", "gamma", "rho")
PARAMETER_PRIOR_BOUNDS = (
    (0.05, 0.5),
    (0.02, 0.2),
    (0.0, 0.8),
)

MAX_INFECTION_IDX = 0
TIME_TO_PEAK_IDX = 1
EARLY_INFECTION_GROWTH_IDX = 2
EARLY_REWIRE_COUNT_IDX = 3
DEGREE_VARIANCE_IDX = 4
LATE_INFECTION_DECAY_IDX = 5
REWIRING_PER_INFECTION_IDX = 6
PEAK_WIDTH_HALF_MAX_IDX = 7

# GAMMA (weak) -> A1, A2
# GAMMA (strong) -> A3
# GAMMA (stable) -> A5

# RHO (weak) -> B1,B2, B3
# RHO (strongest) -> B4

# BETA (weak) -> A1, A2
# BETA (decent) -> A3
# BETA (stable) -> A5

# BETA, GAMMA, RHO (weak) -> C1
# BETA, GAMMA, RHO (strong) -> C2 (tighter posteriors)
# BETA, GAMMA, RHO (stable) -> C3, FULL RICH SET
IDENTIFIABILITY_SUMMARY_SETS = {


    #################################
    #
    # A: Beta, Gamma identifiability
    #
    # NOTE: BETA GAMMA weak interaction from individual 1.1, 1.2
    #       BETA GAMMA best interaction from 1.3
    #       BETA GAMMA stabilized interaction from 1.5 with additional summary stats LATE_INFECTION_DECAY_IDX 
    #
    #################################

    # rejected lower beta, weak gamma
    "A1 {Beta, Gamma}": (
        MAX_INFECTION_IDX,
    ),
    
    # rejected higher beta, weak gamma
    "A2 {Beta, Gamma}": (
        PEAK_WIDTH_HALF_MAX_IDX,
    ),

    # concentrated beta to 0.2, gamma BEST 0.75
    "A3 {Beta, Gamma}": (
        MAX_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX,
    ),

    # both beta and gamma stabilised from A3 with late infection decay
    "A4 {Beta, Gamma}": (
        MAX_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX,
        LATE_INFECTION_DECAY_IDX,
    ),

    # NOTE: suggests that Late Infection Decay can possibly stabilize gamma
    #       not using this, using A3 instead
    
    # rejected higher beta, mid gamma 
    # "A5 {Beta, Gamma}": (
    #     PEAK_WIDTH_HALF_MAX_IDX,
    #     LATE_INFECTION_DECAY_IDX,
    # ),
    

    # stabilized (time to peak ALONE little effect to existing summary stats)
    # "A6 {Beta, Gamma}": (
    #     MAX_INFECTION_IDX,
    #     PEAK_WIDTH_HALF_MAX_IDX,
    #     LATE_INFECTION_DECAY_IDX,
    #     TIME_TO_PEAK_IDX, # beta
    # ),
    
    # both beta and gamma stabilised from A3 with early infection growth + rejected high rho
    # "A7 {Beta, Gamma}": (
    #     MAX_INFECTION_IDX,
    #     PEAK_WIDTH_HALF_MAX_IDX,
    #     LATE_INFECTION_DECAY_IDX,
    #     EARLY_INFECTION_GROWTH_IDX,
    # ),
    
    ######################
    #
    # B: RHO IDENTIFIABILITY
    #
    ######################
    # rho mean around 0.4
    "B1 {Gamma, Rho}": (
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ),

    # time to peak concentrates rho to 0.25 (better)
    "B2 {Beta, Gamma, Rho}": (
        TIME_TO_PEAK_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ), 

    # rewiring count concentrates to 0.3 
    "B3 {Beta, Gamma, Rho}": (
        EARLY_INFECTION_GROWTH_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ),
    # BEST rho concentrated to 0.32
    "B4 {Gamma, Rho}": (
        MAX_INFECTION_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ),

    ############################
    #
    # C: Combined identifiability
    #
    #############################

    # Balanced for beta gamma rho
    "C1 {Beta, Gamma, Rho}": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
    ),
    
    # EXPECTED TO BE BEST FOR COMBINED 1.3 and 9.6 (Gamma, Rho lower variance)
    # NOTE: Posteriors tighter for beta and gamma
    "C2 {Beta, Gamma, Rho}": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX,

        # does not have Early rewire count from reduced set I
        # additional peak width
    ),
    # "C3: {Beta, Gamma, Rho} worse than 9.4, but more balanced overall, mid rho to 0.32": (
    #     MAX_INFECTION_IDX,
    #     PEAK_WIDTH_HALF_MAX_IDX,
    #     DEGREE_VARIANCE_IDX,
    #     REWIRING_PER_INFECTION_IDX,
    # ),
    

    # worse than B4 and C1, mid rho with mean 0.3
    # "C4: {Beta, Gamma, Rho}": (
    #     MAX_INFECTION_IDX,
    #     EARLY_INFECTION_GROWTH_IDX,
    #     DEGREE_VARIANCE_IDX,
    #     REWIRING_PER_INFECTION_IDX,
    # ),

    #  stabalising version of C1
    # "C5 {Beta, Gamma, Rho}": (
    #     MAX_INFECTION_IDX,
    #     TIME_TO_PEAK_IDX,
    #     EARLY_INFECTION_GROWTH_IDX,
    #     DEGREE_VARIANCE_IDX,
    #     REWIRING_PER_INFECTION_IDX,
    # ),


    # Stablilised of C2
    "C3 {Beta, Gamma, Rho}": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
        REWIRING_PER_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX
    ),

    # Stablilised of C2
    "FULL RICH SET {Beta, Gamma, Rho}": (
        MAX_INFECTION_IDX,
        TIME_TO_PEAK_IDX,
        EARLY_INFECTION_GROWTH_IDX,
        EARLY_REWIRE_COUNT_IDX,
        DEGREE_VARIANCE_IDX,
        LATE_INFECTION_DECAY_IDX,
        REWIRING_PER_INFECTION_IDX,
        PEAK_WIDTH_HALF_MAX_IDX,
    ),
}
IDENTIFIABILITY_COMPARISON_PAIRS = (
    ("B4 {Gamma, Rho}", "FULL RICH SET {Beta, Gamma, Rho}"),
    ("A4 {Beta, Gamma}", "FULL RICH SET {Beta, Gamma, Rho}"),
    ("C2 {Beta, Gamma, Rho}", "C3 {Beta, Gamma, Rho}"),
    ("C2 {Beta, Gamma, Rho}", "FULL RICH SET {Beta, Gamma, Rho}"),
)
IDENTIFIABILITY_SPREAD_HEATMAP_PREFIXES = (
    "A1",
    "A2",
    "A3",
    "A4",
    "B1",
    "B2",
    "B3",
    "B4",
    "C1",
    "C2",
    "C3",
    "FULL"
)
acceptance_epsilon_list = [0.005, 0.01, 0.03]
POSTERIOR_COMPARISON_EPSILON = 0.01
BASE_DIR = Path(__file__).resolve().parent
BASIC_ABC_DIR = BASE_DIR / "outputs" / "basic_abc"
SANITY_CHECK_DIR = BASIC_ABC_DIR / "sanity_check"
PARAM_ESTIMATES_DIR = BASIC_ABC_DIR / "param_estimates"
SUMMARY_SET_STUDY_DIR = BASIC_ABC_DIR / "summary_set_study"
IDENTIFIABILITY_POSTERIOR_DIR = SUMMARY_SET_STUDY_DIR / "identifiability"
REGRESSION_ADJUSTMENT_DIR = BASE_DIR / "data" / "intermediate"
REFERENCE_RESULTS_PATH = REGRESSION_ADJUSTMENT_DIR / "abc_rejection_output.npz"
PPC_DIR = BASIC_ABC_DIR / "posterior_predictive_checks"        # ADDED
JOINT_POSTERIOR_DIR = BASIC_ABC_DIR / "joint_posteriors"       # ADDED
MARGINAL_POSTERIOR_DIR = BASIC_ABC_DIR / "marginal_posteriors" # ADDED

early_time_window_min = 2
early_time_window_max = 6 + 1
early_time_points = np.arange(early_time_window_min, early_time_window_max, dtype=np.float64)
early_time_points_centered = early_time_points - early_time_points.mean()
early_time_points_denom = np.dot(early_time_points_centered, early_time_points_centered)

late_time_window_min = 13
late_time_window_max = 20 + 1
late_time_points = np.arange(late_time_window_min, late_time_window_max, dtype=np.float64)
late_time_points_centered = late_time_points - late_time_points.mean()
late_time_points_denom = np.dot(late_time_points_centered, late_time_points_centered)

degrees = np.arange(degree_counts_max, dtype=np.float64)

_WORKER_SIMULATION_CONTEXT = None

###############
#
#
# 3. HELPER FUNCTIONS
#
#
###############
def build_simulation_context(N: int) -> dict:
    """
    Build simulation-invariant structures once and reuse across all runs.

    Parameters
    ----------
    N : int
        Number of nodes in the network.

    Returns
    -------
    dict
        Dictionary containing precomputed structures for `simulate()`.
    """
    upper_rows, upper_cols = np.triu_indices(N, k=1)
    return {
        "upper_rows": upper_rows,
        "upper_cols": upper_cols,
    }

def compute_linear_slope(y_values: np.ndarray, x_centered: np.ndarray, x_denom: float) -> float:
    """
    Compute slope of simple linear regression y = a + b*x using precomputed x terms.

    This is algebraically equivalent to np.polyfit(x, y, 1)[0] but faster for
    fixed x-grids repeatedly reused across simulations.
    """
    return float(np.dot(x_centered, y_values) / x_denom)

def _initialize_worker(simulation_context: dict) -> None:
    """Store shared simulation context in each worker process."""
    global _WORKER_SIMULATION_CONTEXT
    _WORKER_SIMULATION_CONTEXT = simulation_context

def _one_simulation_from_seed(seed: int) -> list:
    """Worker entrypoint for one simulation with an independent RNG seed."""
    rng = np.random.default_rng(int(seed))
    return one_simulation(rng=rng, simulation_context=_WORKER_SIMULATION_CONTEXT)

def one_simulation(rng: np.random.Generator, simulation_context: dict) -> list: 
    """
    Run a single simulation of the adaptive SIR model and compute summary statistics.

    This function:
    1. Samples parameters (beta, gamma, rho) from their prior distributions.
    2. Simulates the epidemic and network dynamics.
    3. Computes a set of summary statistics used for ABC inference.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used for parameter sampling and simulation dynamics.
    simulation_context : dict
        Precomputed simulation-invariant structures passed through to `simulate()`.

    Returns
    -------
    list
        A list containing two elements:
        
        1. tuple of summary statistics:
            - max_infection_frac (float):
                Maximum fraction of infected individuals.
                Informs the balance between infection (beta) and recovery (gamma).
            
            - time_to_peak (int):
                Time step at which infection reaches its max_infection_frac.
                Informs infection speed (beta) and recovery dynamics (gamma).
            
            - slope_early_infection (float):
                Slope of log infection fraction during early time window.
                Approximates early exponential growth rate → strongly informs beta.
            
            - slope_rewire (float):
                Slope of log rewiring counts during early time window.
                Captures how quickly network adaptation occurs → informs rho.
            
            - var_degree (float):
                Variance of node degree distribution.
                Captures network heterogeneity → informs rho.
            
            - slope_late_infection (float):
                Slope of log infection fraction during late time window.
                Captures decay dynamics → informs gamma (and indirectly rho).

            - rewiring_per_infection (float):
                Total rewiring activity divided by total infection burden
                across the trajectory.
                Captures network-adaptation intensity relative to epidemic size
                → informs rho.

            - peak_width_half_max (float):
                Duration for which infection remains above half of its peak value.
                Captures how broad the epidemic peak is → informs beta/gamma.

        2. tuple of sampled parameters:
            - beta (float): infection rate
            - gamma (float): recovery rate
            - rho (float): rewiring rate

    Notes
    -----
    - Log transformations are used to approximate exponential growth/decay.
    - A small epsilon is added before taking logs to avoid numerical issues with zero values.
    """

    # Sample parameters from prior distributions
    beta = rng.uniform(0.05, 0.5)  # Infection rate
    gamma = rng.uniform(0.02, 0.2) # Recovery rate
    rho = rng.uniform(0, 0.8)  # Rewiring rate

    # Simulate data using the sampled parameters
    infected_fraction, rewire_counts, degree_histogram = simulate(beta=beta, gamma=gamma, rho=rho,
                                                                  rng=rng,
                                                                  simulation_context=simulation_context)
    # print(infected_fraction)

    # SUMMARY STATISTIC 1: Max infection fraction INFORMS BETA/GAMMA
    max_infection_frac = np.max(infected_fraction)
    # print("Max infection fraction:", max_infection_frac)

    # SUMMARY STATISTIC 2: Time to peak INFORMS BETA/GAMMA
    time_to_peak = np.argmax(infected_fraction)
    # print("Time to peak:", time_to_peak)



    # SUMMARY STATISTIC 3: Early growth rate INFORMS BETA/GAMMA
    early_log_infection_fraction = np.log(infected_fraction[early_time_window_min:early_time_window_max] + epsilon)  # Add epsilon to avoid log(0)
    slope_early_infection = compute_linear_slope(early_log_infection_fraction, early_time_points_centered, early_time_points_denom)
    # print("Early infection growth rate (slope):", slope_early_infection)

    # SUMMARY STATISTIC 4: Mean rewire counts during early infection
    early_log_rewire_counts = np.log(rewire_counts[early_time_window_min:early_time_window_max] + epsilon)  # Add epsilon to avoid log(0)
    slope_rewire = compute_linear_slope(early_log_rewire_counts, early_time_points_centered, early_time_points_denom)
    # print("Early rewire count slope:", slope_rewire)

    # SUMMARY STATISTIC 5: Variance structure of degree counts INFORMS RHO
    mean_degree = np.sum(degrees * degree_histogram) / N
    mean_degree_sq = np.sum((degrees**2) * degree_histogram) / N

    var_degree = mean_degree_sq - mean_degree**2
    # print("Variance of degree counts:", var_degree)

    # SUMMARY STATISTIC 6: Decay structure of infection INFORMS RHO
    late_log_infection_fraction = np.log(infected_fraction[late_time_window_min:late_time_window_max] + epsilon)  # Add epsilon to avoid log(0)
    slope_late_infection = compute_linear_slope(late_log_infection_fraction, late_time_points_centered, late_time_points_denom)
    # print("Late infection decay rate (slope):", slope_late_infection)

    # SUMMARY STATISTIC 7: Rewiring per infection INFORMS RHO
    rewiring_per_infection = (
        np.sum(rewire_counts, dtype=np.float64) /
        (np.sum(infected_fraction, dtype=np.float64) + epsilon)
    )

    # SUMMARY STATISTIC 8: Infection peak width at half maximum INFORMS BETA/GAMMA
    peak_val = np.max(infected_fraction)
    half_peak = 0.5 * peak_val
    above_half = np.where(infected_fraction >= half_peak)[0]
    peak_width_half_max = float(above_half[-1] - above_half[0]) if above_half.size >= 2 else 0.0

    return [
        (max_infection_frac, time_to_peak, slope_early_infection, 
            slope_rewire, var_degree, slope_late_infection, rewiring_per_infection, peak_width_half_max),
        (beta, gamma, rho)
    ]

def scale_summary_statistics(simulated_summary_statistics, observed_summary_statistics):
    """
    Standardize simulated and observed summary statistics using simulation-based scaling.

    This function computes the mean and standard deviation of the simulated 
    summary statistics and uses them to standardize both simulated and observed 
    summaries. This ensures that all summary statistics contribute comparably 
    to the distance metric in ABC.

    Parameters
    ----------
    simulated_summary_statistics : array-like of shape (N, d)
        Summary statistics computed from N simulated datasets, where d is the 
        number of summary statistics.

    observed_summary_statistics : array-like of shape (d,)
        Summary statistics computed from the observed dataset.

    Returns
    -------
    standardized_simulated : ndarray of shape (N, d)
        Standardized simulated summary statistics.

    standardized_observed : ndarray of shape (d,)
        Standardized observed summary statistics.

    Notes
    -----
    - Standardization is performed as:
          (S - mu) / sigma
      where mu and sigma are computed from simulated summaries.

    - Simulation-based scaling is used because only one observed dataset is 
      available, making it impossible to estimate variability from observed data alone.

    - This scaling approximates a diagonal Mahalanobis distance when used 
      with Euclidean distance in ABC.

    - If any summary statistic has zero variance (sigma = 0), this may lead 
      to division by zero and should be handled externally.
    """
    simulated_summary_statistics = np.array(simulated_summary_statistics, dtype=np.float64)
    observed_summary_statistics = np.array(observed_summary_statistics, dtype=np.float64)
    simulated_mu = simulated_summary_statistics.mean(axis=0)
    simulated_sigma = simulated_summary_statistics.std(axis=0)

    # If a summary has zero simulation variance, it carries no discrimination
    # power in distance calculations, so we set its standardized contribution to 0.
    zero_sigma_mask = simulated_sigma == 0
    safe_sigma = simulated_sigma.copy()
    safe_sigma[zero_sigma_mask] = 1.0

    standardized_simulated = (simulated_summary_statistics - simulated_mu) / safe_sigma
    standardized_observed = (observed_summary_statistics - simulated_mu) / safe_sigma
    standardized_simulated[:, zero_sigma_mask] = 0.0
    standardized_observed[zero_sigma_mask] = 0.0

    return standardized_simulated, standardized_observed

def select_summary_set(summary_statistics, summary_indices):
    """Return a subset of summary statistics using the provided summary indices."""
    summary_statistics = np.asarray(summary_statistics, dtype=np.float64)
    return np.take(summary_statistics, summary_indices, axis=-1)

def get_finite_distance_support(distances):
    """Validate ABC distances and return the finite subset used for thresholding."""
    distances = np.asarray(distances, dtype=np.float64)
    finite_mask = np.isfinite(distances)
    finite_distances = distances[finite_mask]
    if finite_distances.size == 0:
        raise ValueError("All ABC distances are non-finite; cannot compute acceptance threshold.")
    return distances, finite_mask, finite_distances

def compute_distances_for_summary_set(simulated_summary_statistics,
                                      observed_summary_statistics,
                                      summary_indices):
    """Compute ABC distances for one chosen summary-statistic set."""
    subset_simulated = select_summary_set(simulated_summary_statistics, summary_indices)
    subset_observed = select_summary_set(observed_summary_statistics, summary_indices)
    standardized_simulated, standardized_observed = scale_summary_statistics(
        subset_simulated,
        subset_observed
    )
    return np.linalg.norm(standardized_simulated - standardized_observed, axis=1)

def get_accepted_indices_by_epsilon(distances, acceptance_epsilon_list):
    """Return accepted simulation masks for each ABC tolerance in the list."""
    distances, finite_mask, finite_distances = get_finite_distance_support(distances)
    accepted_idx_by_epsilon = {}
    for acceptance_epsilon in acceptance_epsilon_list:
        acceptance_threshold = np.quantile(finite_distances, acceptance_epsilon)
        accepted_idx_by_epsilon[acceptance_epsilon] = finite_mask & (distances <= acceptance_threshold)
    return accepted_idx_by_epsilon

def obtain_accepted_summaries(standardized_simulated, 
                              standardized_observed, 
                              simulated_summary_statistics,
                              acceptance_epsilon=0.01):
    """
    Select accepted summary statistics based on ABC rejection criterion.

    This function computes distances between standardized simulated and observed 
    summary statistics, determines a tolerance threshold (ε) based on a specified 
    quantile, and returns the subset of simulated summaries that are sufficiently 
    close to the observed summaries.

    Parameters
    ----------
    standardized_simulated : ndarray of shape (N, d)
        Standardized summary statistics from N simulations.

    standardized_observed : ndarray of shape (d,)
        Standardized summary statistics from observed data.

    simulated_summary_statistics : array-like of shape (N, d)
        Original (unstandardized) simulated summary statistics.

    acceptance_epsilon : float, optional (default=0.01)
        Quantile used to define the acceptance threshold ε.
        For example, 0.01 corresponds to accepting the closest 1% of simulations.

    Returns
    -------
    accepted_summaries : ndarray of shape (N_accepted, d)
        Subset of simulated summary statistics whose distance to the observed 
        summaries is less than or equal to ε.

    distances : ndarray of shape (N,)
        Euclidean distances between each simulated summary and the observed summary.

    Notes
    -----
    - Distances are computed using Euclidean norm:
          d_i = || S_sim_scaled - S_obs_scaled ||

    - The tolerance ε is determined as:
          ε = quantile(distances, acceptance_epsilon)

    - Accepted summaries correspond to simulations that best match the observed 
      data in terms of summary statistics.

    - The returned summaries are unstandardized to preserve interpretability 
      for diagnostic plots and analysis.
    """
    # Compute distances between standardized simulated and observed summary statistics
    distances = np.linalg.norm(standardized_simulated - standardized_observed, axis=1)
    distances, finite_mask, finite_distances = get_finite_distance_support(distances)

    acceptance_epsilon = np.quantile(finite_distances, acceptance_epsilon)  # 1% acceptance threshold
    accepted_idx = finite_mask & (distances <= acceptance_epsilon)
    # accepted_summaries = np.array(simulated_summary_statistics)[accepted_idx]
    return (accepted_idx, distances)

def save_samples_and_plots(simulated_summary_statistics, 
                           observed_summary_statistics,
                           distances,
                           simulated_parameters,
                           acceptance_epsilon_list=None):
    """
    Generate diagnostic plots for accepted summary statistics and save posterior samples.

    This function:
    1. Plots histograms of accepted summary statistics.
    2. Overlays the observed summary statistic as a vertical reference line.
    3. Extracts accepted parameter samples based on the ABC tolerance criterion.
    4. Saves the accepted parameter samples as a CSV file (optional, currently commented).

    Parameters
    ----------
    simulated_summary_statistics : ndarray of shape (N, d)
        Unstandardized summary statistics from all simulations.

    observed_summary_statistics : array-like of shape (d,)
        Summary statistics computed from the observed dataset.

    distances : ndarray of shape (N,)
        Euclidean distances between each simulated summary and the observed summary.

    simulated_parameters : list of tuples of length N
        Parameter samples corresponding to each simulation, where each tuple is 
        (beta, gamma, rho).

    acceptance_epsilon_list : list of float or None, optional (default=None)
        Quantile thresholds used to determine accepted samples.
        If None, defaults to [0.01, 0.05, 0.10, 0.20].

    Behavior
    --------
    - For each summary statistic:
        * Four overlaid histograms are plotted, one for each acceptance ε.
        * The observed value is shown as a red vertical line.
    - Accepted parameter samples are filtered and saved separately for each ε threshold.
    - A timestamp is generated to uniquely label outputs.

    Outputs
    -------
    - Diagnostic plots (displayed via matplotlib; saving is currently commented out).
    - CSV file of accepted parameter samples (currently commented out):
        Format: [beta, gamma, rho]

    Notes
    -----
    - Plots are generated using unstandardized summaries for interpretability.
    - Histogram binning (bins=30) may appear sparse for discrete summaries 
      (e.g., time to peak).
    - The ABC posterior is approximate due to:
        * Non-sufficient summary statistics
        * Non-zero tolerance ε
    - The quantile threshold for each ε is computed directly from `distances`.

    Returns
    -------
    None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    simulated_summary_statistics = np.array(simulated_summary_statistics)
    if acceptance_epsilon_list is None:
        acceptance_epsilon_list = [0.005, 0.01, 0.02]

    # Ensure output directories exist regardless of launch directory.
    SANITY_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    PARAM_ESTIMATES_DIR.mkdir(parents=True, exist_ok=True)

    # Precompute accepted sets for each epsilon value
    accepted_idx_by_epsilon = get_accepted_indices_by_epsilon(distances, acceptance_epsilon_list)
    accepted_summaries_by_epsilon = {}

    for acceptance_epsilon in acceptance_epsilon_list:
        accepted_idx = accepted_idx_by_epsilon[acceptance_epsilon]
        accepted_idx_by_epsilon[acceptance_epsilon] = accepted_idx
        accepted_summaries_by_epsilon[acceptance_epsilon] = simulated_summary_statistics[accepted_idx]

    for i in range(simulated_summary_statistics.shape[1]):
        plt.figure()
        for acceptance_epsilon in acceptance_epsilon_list:
            accepted_summaries = accepted_summaries_by_epsilon[acceptance_epsilon]
            plt.hist(
                accepted_summaries[:, i],
                bins=30,
                alpha=0.35,
                density=True,
                label=f"ε={acceptance_epsilon:.3f} (n={accepted_summaries.shape[0]})"
            )
        plt.axvline(observed_summary_statistics[i], color='red', linewidth=2)
        plt.title(f"Summary {summary_statistics_name[i]} (Accepted Simulations by ε)")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plot_path = SANITY_CHECK_DIR / f"summary_{summary_statistics_name[i]}_overlay_eps_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    # Save accepted parameters to CSV for each epsilon threshold
    for acceptance_epsilon in acceptance_epsilon_list:
        accepted_idx = accepted_idx_by_epsilon[acceptance_epsilon]
        accepted_parameters = [params for params, keep in zip(simulated_parameters, accepted_idx) if keep]
        print(f"[ABC] ε={acceptance_epsilon:.4f}: accepted {len(accepted_parameters)} / {len(simulated_parameters)} simulations")
        final_chosen_posteriors = pd.DataFrame(accepted_parameters, columns=['beta', 'gamma', 'rho'])
        filename = f"abc-basic_eps-{acceptance_epsilon:.4f}_{timestamp}.csv"
        csv_path = PARAM_ESTIMATES_DIR / filename
        final_chosen_posteriors.to_csv(csv_path, index=False)

def plot_posterior_comparison_plots(simulated_summary_statistics,
                                    observed_summary_statistics,
                                    simulated_parameters,
                                    comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """Plot the requested identifiability posterior comparisons."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    simulated_summary_statistics = np.asarray(simulated_summary_statistics, dtype=np.float64)
    observed_summary_statistics = np.asarray(observed_summary_statistics, dtype=np.float64)
    simulated_parameters = np.asarray(simulated_parameters, dtype=np.float64)

    comparison_dir = IDENTIFIABILITY_POSTERIOR_DIR / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    accepted_idx_by_set = {}
    for summary_set_name, summary_indices in IDENTIFIABILITY_SUMMARY_SETS.items():
        distances = compute_distances_for_summary_set(
            simulated_summary_statistics,
            observed_summary_statistics,
            summary_indices
        )
        accepted_idx_by_set[summary_set_name] = get_accepted_indices_by_epsilon(
            distances,
            [comparison_epsilon]
        )[comparison_epsilon]

    for left_set_name, right_set_name in IDENTIFIABILITY_COMPARISON_PAIRS:
        left_parameters = simulated_parameters[accepted_idx_by_set[left_set_name]]
        right_parameters = simulated_parameters[accepted_idx_by_set[right_set_name]]

        fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(15, 4.5))
        for param_idx, (ax, param_name) in enumerate(zip(axes, PARAMETER_NAMES)):
            ax.hist(
                left_parameters[:, param_idx],
                bins=30,
                alpha=0.45,
                density=True,
                label=f"{left_set_name} (n={left_parameters.shape[0]})"
            )
            ax.hist(
                right_parameters[:, param_idx],
                bins=30,
                alpha=0.45,
                density=True,
                label=f"{right_set_name} (n={right_parameters.shape[0]})"
            )
            ax.set_title(f"Posterior of {param_name}")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Density")
            ax.legend()

        fig.suptitle(
            f"Posterior comparison at ε={comparison_epsilon:.3f}: "
            f"{left_set_name} vs {right_set_name}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        comparison_slug = (
            f"{make_safe_slug(left_set_name)}_vs_"
            f"{make_safe_slug(right_set_name)}"
        )
        plot_path = (
            comparison_dir
            / f"posterior_{comparison_slug}_eps-{comparison_epsilon:.4f}_{timestamp}.png"
        )

        plt.show()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def save_summary_set_outputs(summary_set_name,
                             summary_indices,
                             simulated_summary_statistics,
                             observed_summary_statistics,
                             distances,
                             simulated_parameters,
                             acceptance_epsilon_list=None):
    """Save accepted-summary plots and posterior samples for one chosen summary set."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    simulated_summary_statistics = np.asarray(simulated_summary_statistics, dtype=np.float64)
    observed_summary_statistics = np.asarray(observed_summary_statistics, dtype=np.float64)
    simulated_parameters = np.asarray(simulated_parameters, dtype=np.float64)
    summary_indices = tuple(summary_indices)

    if acceptance_epsilon_list is None:
        acceptance_epsilon_list = [0.005, 0.01, 0.02]

    summary_set_slug = summary_set_name.lower().replace(" ", "_")
    summary_set_dir = SUMMARY_SET_STUDY_DIR / summary_set_slug
    sanity_check_dir = summary_set_dir / "sanity_check"
    param_estimates_dir = summary_set_dir / "param_estimates"
    regression_adjustment_dir = summary_set_dir / "regression_adjustment"

    sanity_check_dir.mkdir(parents=True, exist_ok=True)
    param_estimates_dir.mkdir(parents=True, exist_ok=True)
    regression_adjustment_dir.mkdir(parents=True, exist_ok=True)

    accepted_idx_by_epsilon = get_accepted_indices_by_epsilon(distances, acceptance_epsilon_list)
    accepted_summaries_by_epsilon = {}

    for acceptance_epsilon in acceptance_epsilon_list:
        accepted_idx = accepted_idx_by_epsilon[acceptance_epsilon]
        accepted_summaries_by_epsilon[acceptance_epsilon] = simulated_summary_statistics[accepted_idx]

    for summary_idx in summary_indices:
        plt.figure()
        for acceptance_epsilon in acceptance_epsilon_list:
            accepted_summaries = accepted_summaries_by_epsilon[acceptance_epsilon]
            plt.hist(
                accepted_summaries[:, summary_idx],
                bins=30,
                alpha=0.35,
                density=True,
                label=f"ε={acceptance_epsilon:.3f} (n={accepted_summaries.shape[0]})"
            )
        plt.axvline(observed_summary_statistics[summary_idx], color='red', linewidth=2)
        plt.title(
            f"Summary {summary_statistics_name[summary_idx]} "
            f"(Accepted Simulations by ε, {summary_set_name})"
        )
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plot_path = (
            sanity_check_dir
            / f"{summary_set_slug}_summary_{summary_statistics_name[summary_idx]}_overlay_eps_{timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    for acceptance_epsilon in acceptance_epsilon_list:
        accepted_idx = accepted_idx_by_epsilon[acceptance_epsilon]
        accepted_parameters = simulated_parameters[accepted_idx]
        print(
            f"[ABC:{summary_set_name}] ε={acceptance_epsilon:.4f}: "
            f"accepted {accepted_parameters.shape[0]} / {simulated_parameters.shape[0]} simulations"
        )
        final_chosen_posteriors = pd.DataFrame(accepted_parameters, columns=['beta', 'gamma', 'rho'])
        filename = f"{summary_set_slug}_abc-basic_eps-{acceptance_epsilon:.4f}_{timestamp}.csv"
        csv_path = param_estimates_dir / filename
        final_chosen_posteriors.to_csv(csv_path, index=False)

    main_acceptance_idx = accepted_idx_by_epsilon[POSTERIOR_COMPARISON_EPSILON]
    np.savez(
        regression_adjustment_dir / f"{summary_set_slug}_abc_rejection_output_eps-{POSTERIOR_COMPARISON_EPSILON:.4f}.npz",
        accepted_parameters=simulated_parameters[main_acceptance_idx],
        accepted_summaries=simulated_summary_statistics[main_acceptance_idx][:, summary_indices],
        observed_summary=observed_summary_statistics[list(summary_indices)],
        accepted_distances=np.asarray(distances, dtype=np.float64)[main_acceptance_idx],
        summary_indices=np.asarray(summary_indices, dtype=np.int64),
        summary_names=np.asarray([summary_statistics_name[idx] for idx in summary_indices], dtype=object),
    )

####################################################################### Added to see which summary sets are most informative #########################################################
def compute_posterior_spread_table(simulated_parameters,
                                   simulated_summary_statistics,
                                   observed_summary_statistics,
                                   summary_sets=None,
                                   comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """
    For each summary set, compute normalized posterior std per parameter.
    Lower = tighter posterior = more informative summary set.
    """
    simulated_parameters = np.asarray(simulated_parameters, dtype=np.float64)
    if summary_sets is None:
        summary_sets = SUMMARY_SET_INDICES
    prior_widths = np.array(
        [upper - lower for lower, upper in PARAMETER_PRIOR_BOUNDS],
        dtype=np.float64
    )
    
    rows = []
    for set_name, indices in summary_sets.items():
        distances = compute_distances_for_summary_set(
            simulated_summary_statistics, observed_summary_statistics, indices
        )
        accepted_idx = get_accepted_indices_by_epsilon(
            distances, [comparison_epsilon]
        )[comparison_epsilon]
        posterior = simulated_parameters[accepted_idx]
        
        if posterior.shape[0] < 10:
            continue
        
        normalized_std = np.std(posterior, axis=0) / prior_widths
        rows.append({
            "Summary set": set_name,
            "n_dims": len(indices),
            "β std (norm)": round(normalized_std[0], 3),
            "γ std (norm)": round(normalized_std[1], 3),
            "ρ std (norm)": round(normalized_std[2], 3),
            "mean std": round(normalized_std.mean(), 3),
        })
    
    return pd.DataFrame(rows).sort_values("mean std")

def plot_posterior_spread_heatmap(spread_df, filename):
    """Heatmap: rows=summary sets, cols=parameters, values=normalized posterior std."""
    SUMMARY_SET_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    matrix = spread_df.set_index("Summary set")[["β std (norm)", "γ std (norm)", "ρ std (norm)"]].values
    set_names = spread_df["Summary set"].tolist()
    matrix_min = float(np.min(matrix))
    matrix_max = float(np.max(matrix))
    color_padding = max(0.01, 0.05 * (matrix_max - matrix_min))

    if abs(matrix_max - matrix_min) <= 1e-12:
        vmin = matrix_min - color_padding
        vmax = matrix_max + color_padding
    else:
        vmin = max(0.0, matrix_min - color_padding)
        vmax = matrix_max + color_padding
    
    fig, ax = plt.subplots(figsize=(7, len(set_names) * 0.7 + 1))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["β", "γ", "ρ"])
    ax.set_yticks(range(len(set_names)))
    ax.set_yticklabels(set_names)
    
    for i in range(len(set_names)):
        for j in range(3):
            normalized_value = (matrix[i, j] - vmin) / max(vmax - vmin, 1e-12)
            text_color = "white" if normalized_value >= 0.6 else "black"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha='center', va='center', fontsize=9, color=text_color)
    
    plt.colorbar(im, ax=ax, label="Normalized posterior std (lower = tighter)")
    ax.set_title(f"Posterior spread by summary set (ε={POSTERIOR_COMPARISON_EPSILON})")
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)

def plot_posterior_predictive_checks(simulated_summary_statistics,
                                     observed_summary_statistics,
                                     simulated_parameters,
                                     simulation_context,
                                     n_ppc_samples=200,
                                     comparison_epsilon=POSTERIOR_COMPARISON_EPSILON,
                                     seed=seed):
    """
    Posterior predictive check: overlay simulated trajectories from accepted
    parameters against observed data for all three observables:
    - Infected fraction time series
    - Rewiring counts time series  
    - Final degree histogram
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    PPC_DIR.mkdir(parents=True, exist_ok=True)

    # Get accepted parameters from reference set
    reference_distances = compute_distances_for_summary_set(
        simulated_summary_statistics, observed_summary_statistics,
        REFERENCE_SUMMARY_SET_INDICES
    )
    accepted_idx = get_accepted_indices_by_epsilon(
        reference_distances, [comparison_epsilon]
    )[comparison_epsilon]
    accepted_params = np.asarray(simulated_parameters)[accepted_idx]

    # Subsample for PPC if too many
    rng = np.random.default_rng(seed)
    n_accepted = accepted_params.shape[0]
    sample_idx = rng.choice(n_accepted, size=min(n_ppc_samples, n_accepted), replace=False)
    ppc_params = accepted_params[sample_idx]

    # Run new simulations from accepted parameters to get full trajectories
    ppc_infected   = []
    ppc_rewire     = []
    ppc_degree     = []

    for beta, gamma, rho in tqdm(ppc_params, desc="Running PPC simulations"):
        inf_ts, rew_ts, deg_hist = simulate(
            beta=beta, gamma=gamma, rho=rho,
            rng=np.random.default_rng(int(rng.integers(1e9))),
            simulation_context=simulation_context
        )
        ppc_infected.append(inf_ts)
        ppc_rewire.append(rew_ts)
        ppc_degree.append(deg_hist)

    ppc_infected = np.array(ppc_infected)   # (n_samples, T+1)
    ppc_rewire   = np.array(ppc_rewire)     # (n_samples, T+1)
    ppc_degree   = np.array(ppc_degree)     # (n_samples, 31)

    # Load observed data for full trajectories
    infected_df    = pd.read_csv(BASE_DIR / "data" / "infected_timeseries.csv")
    rewire_df      = pd.read_csv(BASE_DIR / "data" / "rewiring_timeseries.csv")
    degree_df      = pd.read_csv(BASE_DIR / "data" / "final_degree_histograms.csv")

    obs_inf_mean   = infected_df.groupby("time")["infected_fraction"].mean().values
    obs_inf_lower  = infected_df.groupby("time")["infected_fraction"].quantile(0.1).values
    obs_inf_upper  = infected_df.groupby("time")["infected_fraction"].quantile(0.9).values

    obs_rew_mean   = rewire_df.groupby("time")["rewire_count"].mean().values
    obs_rew_lower  = rewire_df.groupby("time")["rewire_count"].quantile(0.1).values
    obs_rew_upper  = rewire_df.groupby("time")["rewire_count"].quantile(0.9).values

    obs_deg_mean   = degree_df.groupby("degree")["count"].mean().values
    obs_deg_lower  = degree_df.groupby("degree")["count"].quantile(0.1).values
    obs_deg_upper  = degree_df.groupby("degree")["count"].quantile(0.9).values

    T        = ppc_infected.shape[1]
    t_axis   = np.arange(T)
    deg_axis = np.arange(31)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Infected fraction ---
    ax = axes[0]
    for row in ppc_infected:
        ax.plot(t_axis, row, color='steelblue', alpha=0.08, linewidth=0.7)
    ax.plot(t_axis, np.median(ppc_infected, axis=0), color='steelblue',
            linewidth=2, label="PPC median")
    ax.fill_between(t_axis,
                    np.percentile(ppc_infected, 10, axis=0),
                    np.percentile(ppc_infected, 90, axis=0),
                    color='steelblue', alpha=0.2, label="PPC 10–90%")
    ax.plot(t_axis, obs_inf_mean, color='red', linewidth=2, label="Observed mean")
    ax.fill_between(t_axis, obs_inf_lower, obs_inf_upper,
                    color='red', alpha=0.15, label="Observed 10–90%")
    ax.set_title("Infected fraction")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Fraction infected")
    ax.legend(fontsize=8)

    # --- Panel 2: Rewiring counts ---
    ax = axes[1]
    for row in ppc_rewire:
        ax.plot(t_axis, row, color='darkorange', alpha=0.08, linewidth=0.7)
    ax.plot(t_axis, np.median(ppc_rewire, axis=0), color='darkorange',
            linewidth=2, label="PPC median")
    ax.fill_between(t_axis,
                    np.percentile(ppc_rewire, 10, axis=0),
                    np.percentile(ppc_rewire, 90, axis=0),
                    color='darkorange', alpha=0.2, label="PPC 10–90%")
    ax.plot(t_axis, obs_rew_mean, color='red', linewidth=2, label="Observed mean")
    ax.fill_between(t_axis, obs_rew_lower, obs_rew_upper,
                    color='red', alpha=0.15, label="Observed 10–90%")
    ax.set_title("Rewiring counts")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Rewire count")
    ax.legend(fontsize=8)

    # --- Panel 3: Degree histogram ---
    ax = axes[2]
    ax.fill_between(deg_axis,
                    np.percentile(ppc_degree, 10, axis=0),
                    np.percentile(ppc_degree, 90, axis=0),
                    color='forestgreen', alpha=0.3, label="PPC 10–90%")
    ax.plot(deg_axis, np.median(ppc_degree, axis=0), color='forestgreen',
            linewidth=2, label="PPC median")
    ax.plot(deg_axis, obs_deg_mean, color='red', linewidth=2,
            label="Observed mean")
    ax.fill_between(deg_axis, obs_deg_lower, obs_deg_upper,
                    color='red', alpha=0.15, label="Observed 10–90%")
    ax.set_title("Final degree histogram")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Posterior predictive check — {REFERENCE_SUMMARY_SET_NAME} "
        f"(ε={comparison_epsilon}, n={len(ppc_params)} samples)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = PPC_DIR / f"ppc_all_observables_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def get_accepted_posterior_for_summary_set(simulated_parameters,
                                           simulated_summary_statistics,
                                           observed_summary_statistics,
                                           summary_set_name,
                                           summary_indices,
                                           comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """Return accepted posterior samples for one chosen summary-statistic set."""
    simulated_parameters = np.asarray(simulated_parameters, dtype=np.float64)
    distances = compute_distances_for_summary_set(
        simulated_summary_statistics,
        observed_summary_statistics,
        summary_indices
    )
    accepted_idx = get_accepted_indices_by_epsilon(
        distances,
        [comparison_epsilon]
    )[comparison_epsilon]
    posterior = simulated_parameters[accepted_idx]
    if posterior.shape[0] == 0:
        raise ValueError(
            f"{summary_set_name} produced no accepted samples at ε={comparison_epsilon:.3f}."
        )
    return posterior


def make_safe_slug(value: str) -> str:
    """Convert labels into Windows-safe filesystem slugs."""
    slug = value.lower()
    slug = re.sub(r'[<>:"/\\\\|?*]+', "", slug)
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "plot"


def get_selected_summary_sets(summary_sets: dict,
                              name_prefixes: tuple[str, ...]) -> dict:
    """Return ordered summary sets matching the requested name prefixes."""
    selected_summary_sets = {}
    for prefix in name_prefixes:
        matched_name = next(
            (name for name in summary_sets if name.startswith(prefix)),
            None
        )
        if matched_name is None:
            raise KeyError(f"Could not find summary set matching prefix '{prefix}'.")
        selected_summary_sets[matched_name] = summary_sets[matched_name]
    return selected_summary_sets


def print_joint_posterior_correlations(posterior,
                                       summary_set_name,
                                       comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """Print Pearson correlations for the three parameter-pair marginals."""
    correlation_pairs = (
        (0, 1, "beta", "gamma"),
        (0, 2, "beta", "rho"),
        (1, 2, "gamma", "rho"),
    )
    print(f"\n[Joint posterior Pearson correlations: {summary_set_name}] ε={comparison_epsilon:.3f}")
    for i, j, label_i, label_j in correlation_pairs:
        x = posterior[:, i]
        y = posterior[:, j]

        if posterior.shape[0] < 2:
            print(f"{label_i} vs {label_j}: undefined (fewer than 2 accepted samples)")
            continue
        if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            print(f"{label_i} vs {label_j}: undefined (zero posterior variance)")
            continue

        correlation = np.corrcoef(x, y)[0, 1]
        print(f"{label_i} vs {label_j}: {correlation:.4f}")


def plot_joint_posteriors(simulated_parameters,
                          simulated_summary_statistics,
                          observed_summary_statistics,
                          summary_set_name=REFERENCE_SUMMARY_SET_NAME,
                          summary_indices=REFERENCE_SUMMARY_SET_INDICES,
                          comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """
    Joint posterior scatter plots for parameter pairs.
    Most useful pairs for this model:
    - beta vs rho: both drive S-I edge dynamics, partially aliased
    - gamma vs rho: main identifiability problem
    - beta vs gamma: jointly determine epidemic size/speed
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    JOINT_POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)

    posterior = get_accepted_posterior_for_summary_set(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
        summary_set_name,
        summary_indices,
        comparison_epsilon=comparison_epsilon,
    )

    pairs = [
        (0, 2, "beta", "rho",   "β vs ρ — both suppress via S-I edges"),
        (1, 2, "gamma", "rho",  "γ vs ρ — main aliasing pair"),
        (0, 1, "beta", "gamma", "β vs γ — jointly determine epidemic size"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (i, j, xlabel, ylabel, title) in zip(axes, pairs):
        # Scatter
        ax.scatter(posterior[:, i], posterior[:, j],
                   alpha=0.35, s=12, color='steelblue', label="Accepted samples")

        # Add 2D KDE contours
        try:
            from scipy.stats import gaussian_kde
            xy  = np.vstack([posterior[:, i], posterior[:, j]])
            kde = gaussian_kde(xy)
            xi  = np.linspace(posterior[:, i].min(), posterior[:, i].max(), 80)
            yi  = np.linspace(posterior[:, j].min(), posterior[:, j].max(), 80)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi  = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            ax.contour(Xi, Yi, Zi, levels=5, colors='navy', linewidths=0.8, alpha=0.6)
        except Exception:
            pass  # skip contours if scipy unavailable or KDE fails

        # Posterior means
        ax.axvline(posterior[:, i].mean(), color='red', linestyle='--',
                   linewidth=1.2, label=f"mean {xlabel}={posterior[:,i].mean():.3f}")
        ax.axhline(posterior[:, j].mean(), color='red', linestyle=':',
                   linewidth=1.2, label=f"mean {ylabel}={posterior[:,j].mean():.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Joint posteriors — {summary_set_name} "
        f"(ε={comparison_epsilon}, n={posterior.shape[0]})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    summary_set_slug = make_safe_slug(summary_set_name)
    plot_path = JOINT_POSTERIOR_DIR / f"joint_posteriors_{summary_set_slug}_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    print_joint_posterior_correlations(
        posterior,
        summary_set_name,
        comparison_epsilon=comparison_epsilon,
    )


def plot_marginal_posteriors(simulated_parameters,
                             simulated_summary_statistics,
                             observed_summary_statistics,
                             acceptance_epsilon_list=None,
                             comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """
    Marginal posterior histograms for all three parameters,
    overlaid across epsilon values to show sensitivity to tolerance.
    """
    if acceptance_epsilon_list is None:
        acceptance_epsilon_list = [0.005, 0.01, 0.03]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    MARGINAL_POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)

    reference_distances = compute_distances_for_summary_set(
        simulated_summary_statistics, observed_summary_statistics,
        REFERENCE_SUMMARY_SET_INDICES
    )
    accepted_by_eps = get_accepted_indices_by_epsilon(
        reference_distances, acceptance_epsilon_list
    )
    simulated_parameters = np.asarray(simulated_parameters)

    colors = ['steelblue', 'darkorange', 'forestgreen']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for param_idx, (ax, pname, bounds) in enumerate(zip(axes, PARAMETER_NAMES, PARAMETER_PRIOR_BOUNDS)):
        for eps, color in zip(sorted(acceptance_epsilon_list, reverse=True), colors):
            accepted = simulated_parameters[accepted_by_eps[eps]]
            ax.hist(accepted[:, param_idx], bins=30, density=True,
                    alpha=0.45, color=color,
                    label=f"ε={eps} (n={accepted.shape[0]})")

        # Prior as flat reference line
        prior_density = 1.0 / (bounds[1] - bounds[0])
        ax.axhline(prior_density, color='black', linestyle='--',
                   linewidth=1.2, label="Prior (uniform)")

        ax.set_title(f"Posterior of {pname}")
        ax.set_xlabel(pname)
        ax.set_ylabel("Density")
        ax.set_xlim(bounds)
        ax.legend(fontsize=8)

    fig.suptitle(f"Marginal posteriors — {REFERENCE_SUMMARY_SET_NAME}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = MARGINAL_POSTERIOR_DIR / f"marginal_posteriors_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_identifiability_posteriors(simulated_parameters,
                                    simulated_summary_statistics,
                                    observed_summary_statistics,
                                    comparison_epsilon=POSTERIOR_COMPARISON_EPSILON):
    """
    Plot marginal posteriors for targeted summary subsets to make identifiability explicit.

    Each requested summary family gets its own 3-panel figure and includes the
    uniform-prior density as a horizontal reference line for comparison.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    IDENTIFIABILITY_POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)

    simulated_parameters = np.asarray(simulated_parameters, dtype=np.float64)
    simulated_summary_statistics = np.asarray(simulated_summary_statistics, dtype=np.float64)
    observed_summary_statistics = np.asarray(observed_summary_statistics, dtype=np.float64)
    colors = plt.cm.tab10.colors

    for summary_set_idx, (summary_set_name, summary_indices) in enumerate(IDENTIFIABILITY_SUMMARY_SETS.items()):
        plot_color = colors[summary_set_idx % len(colors)]
        distances = compute_distances_for_summary_set(
            simulated_summary_statistics,
            observed_summary_statistics,
            summary_indices
        )
        accepted_idx = get_accepted_indices_by_epsilon(
            distances,
            [comparison_epsilon]
        )[comparison_epsilon]
        posterior = simulated_parameters[accepted_idx]
        if posterior.shape[0] == 0:
            raise ValueError(
                f"{summary_set_name} produced no accepted samples at ε={comparison_epsilon:.3f}."
            )

        fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(15, 4.5))
        for param_idx, (ax, pname, bounds) in enumerate(zip(axes, PARAMETER_NAMES, PARAMETER_PRIOR_BOUNDS)):
            ax.hist(
                posterior[:, param_idx],
                bins=30,
                density=True,
                alpha=0.55,
                color=plot_color,
                label=f"Posterior (n={posterior.shape[0]})"
            )

            prior_density = 1.0 / (bounds[1] - bounds[0])
            ax.axhline(
                prior_density,
                color='black',
                linestyle='--',
                linewidth=1.2,
                label="Prior (uniform)"
            )

            ax.set_title(f"Posterior of {pname}")
            ax.set_xlabel(pname)
            ax.set_ylabel("Density")
            ax.set_xlim(bounds)
            ax.legend(fontsize=8)

        summary_names = ", ".join(summary_statistics_name[idx] for idx in summary_indices)
        fig.suptitle(
            f"Identifiability posterior — {summary_set_name}\n"
            f"Summaries: {summary_names}\n"
            f"ε={comparison_epsilon:.3f}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))

        summary_set_slug = make_safe_slug(summary_set_name)
        plot_path = (
            IDENTIFIABILITY_POSTERIOR_DIR
            / f"identifiability_{summary_set_slug}_eps-{comparison_epsilon:.4f}_{timestamp}.png"
        )
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

###############
#
#
# 4. MAIN FUNCTION
#
#
###############
def main() -> None:
    """
    Execute the full ABC rejection pipeline for parameter inference in the SIR model.

    This function orchestrates the end-to-end workflow:
    1. Generates simulated datasets by repeatedly sampling parameters and computing 
       summary statistics.
    2. Standardizes simulated and observed summary statistics using simulation-based scaling.
    3. Applies the ABC rejection algorithm to select simulations whose summaries are 
       closest to the observed data.
    4. Produces diagnostic plots and saves accepted posterior samples.

    Workflow
    --------
    - Simulation:
        Runs N_sim independent simulations using `one_simulation()`, which returns:
        * Summary statistics (used for matching)
        * Corresponding parameters (beta, gamma, rho)

    - Normalization:
        Uses `scale_summary_statistics()` to standardize both simulated and observed 
        summaries based on simulation-derived mean and standard deviation.

    - ABC Rejection:
        Uses `obtain_accepted_summaries()` to:
        * Compute Euclidean distances between standardized summaries
        * Select the closest simulations based on a quantile threshold (default: 1%)

    - Output:
        Uses `save_samples_and_plots()` to:
        * Visualize posterior predictive checks (accepted summaries vs observed)
        * Save accepted parameter samples (approximate posterior)

    Notes
    -----
    - A fixed random seed is set for reproducibility.
    - The resulting posterior is approximate due to:
        * Use of non-sufficient summary statistics
        * Finite tolerance (ε)
    - The quality of inference depends heavily on:
        * Choice of summary statistics
        * Distance metric and normalization
        * Acceptance threshold

    Returns
    -------
    None
    """
    # set seed for reproducibility during parameter sampling per simulation 
    seed = 2026  # Set seed for reproducibility
    seed_sequence = np.random.SeedSequence(seed)
    simulation_seeds = seed_sequence.generate_state(N_sim, dtype=np.uint64).tolist()
    simulation_context = build_simulation_context(N)
    n_workers = max(1, (os.cpu_count() or 1) - 1)

    # simulate 30_000 times
    if n_workers == 1:
        simulation_results = [
            one_simulation(np.random.default_rng(int(sim_seed)), simulation_context)
            for sim_seed in tqdm(simulation_seeds, desc="Running ABC")
        ]
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=spawn_context,
            initializer=_initialize_worker,
            initargs=(simulation_context,)
        ) as executor:
            simulation_results = list(
                tqdm(
                    executor.map(_one_simulation_from_seed, simulation_seeds, chunksize=64),
                    total=N_sim,
                    desc="Running ABC"
                )
            )

    simulated_summary_statistics, simulated_parameters = zip(*simulation_results)

    # scale simulated data and observed data
    observed_summary_statistics = get_obs_summaries()
    standardized_simulated, standardized_observed = scale_summary_statistics(simulated_summary_statistics, 
                                                                             observed_summary_statistics)
    
    # obtain accepted summaries from ABC rejection algo
    accepted_idx, distances = obtain_accepted_summaries(standardized_simulated, 
                                             standardized_observed,
                                             simulated_summary_statistics)
    
    # save parameters and plots of accepted summaries
    save_samples_and_plots(simulated_summary_statistics,
                           observed_summary_statistics,
                           distances,
                           simulated_parameters,
                           acceptance_epsilon_list=acceptance_epsilon_list)
    reference_set_distances = compute_distances_for_summary_set(
        simulated_summary_statistics,
        observed_summary_statistics,
        REFERENCE_SUMMARY_SET_INDICES
    )
    plot_posterior_comparison_plots(
        simulated_summary_statistics,
        observed_summary_statistics,
        simulated_parameters,
        comparison_epsilon=POSTERIOR_COMPARISON_EPSILON
    )
    save_summary_set_outputs(
        REFERENCE_SUMMARY_SET_NAME,
        REFERENCE_SUMMARY_SET_INDICES,
        simulated_summary_statistics,
        observed_summary_statistics,
        reference_set_distances,
        simulated_parameters,
        acceptance_epsilon_list=acceptance_epsilon_list
    )

    # save the chosen reference-summary calibration for downstream methods
    reference_indices = REFERENCE_SUMMARY_SET_INDICES
    reference_simulated_summaries = np.array(simulated_summary_statistics)[:, reference_indices]
    reference_observed_summary = np.array(observed_summary_statistics)[list(reference_indices)]
    reference_summary_mu = reference_simulated_summaries.mean(axis=0)
    reference_summary_sigma = reference_simulated_summaries.std(axis=0)
    reference_zero_sigma_mask = reference_summary_sigma == 0
    reference_safe_sigma = reference_summary_sigma.copy()
    reference_safe_sigma[reference_zero_sigma_mask] = 1.0
    reference_standardized_observed = (
        reference_observed_summary - reference_summary_mu
    ) / reference_safe_sigma
    reference_standardized_observed[reference_zero_sigma_mask] = 0.0
    reference_distances, reference_finite_mask, reference_finite_distances = get_finite_distance_support(
        reference_set_distances
    )
    reference_distance_threshold = np.quantile(
        reference_finite_distances,
        POSTERIOR_COMPARISON_EPSILON
    )
    reference_accepted_idx = reference_finite_mask & (
        reference_distances <= reference_distance_threshold
    )
    accepted_summaries = reference_simulated_summaries[reference_accepted_idx]
    accepted_parameters = np.array(simulated_parameters)[reference_accepted_idx]
    accepted_distances = reference_distances[reference_accepted_idx]
    if accepted_parameters.shape[0] == 0:
        raise ValueError(
            f"{REFERENCE_SUMMARY_SET_NAME} rejection run produced no accepted samples "
            f"at ε={POSTERIOR_COMPARISON_EPSILON:.2f}."
        )
    np.savez(
        REFERENCE_RESULTS_PATH,
        reference_parameters=np.array(simulated_parameters),
        reference_summaries=reference_simulated_summaries,
        distances=reference_distances,
        accepted_parameters=accepted_parameters,
        accepted_summaries=accepted_summaries,
        observed_summary=reference_observed_summary,
        standardized_observed=reference_standardized_observed,
        summary_mu=reference_summary_mu,
        summary_sigma=reference_safe_sigma,
        zero_sigma_mask=reference_zero_sigma_mask,
        accepted_distances=accepted_distances,
        acceptance_epsilon=POSTERIOR_COMPARISON_EPSILON,
        distance_threshold=reference_distance_threshold,
        initial_parameters=accepted_parameters[0],
        initial_summary=accepted_summaries[0],
        initial_distance=accepted_distances[0],
        summary_indices=np.array(reference_indices),
        summary_names=np.array([summary_statistics_name[idx] for idx in reference_indices], dtype=object),
        summary_set_name=np.array(REFERENCE_SUMMARY_SET_NAME, dtype=object),
    )
    
    spread_df = compute_posterior_spread_table(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
    )
    print("\n[Summary set spread table]")
    print(spread_df.to_string(index=False))
    
    plot_posterior_spread_heatmap(
        spread_df,
        SUMMARY_SET_STUDY_DIR / f"posterior_spread_heatmap_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"
    )
    identifiability_summary_sets = get_selected_summary_sets(
        IDENTIFIABILITY_SUMMARY_SETS,
        IDENTIFIABILITY_SPREAD_HEATMAP_PREFIXES,
    )
    identifiability_spread_df = compute_posterior_spread_table(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
        summary_sets=identifiability_summary_sets,
    )
    print("\n[Identifiability spread table]")
    print(identifiability_spread_df.to_string(index=False))
    plot_posterior_spread_heatmap(
        identifiability_spread_df,
        IDENTIFIABILITY_POSTERIOR_DIR / (
            f"identifiability_posterior_spread_heatmap_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"
        )
    )

    plot_posterior_predictive_checks(
        simulated_summary_statistics,
        observed_summary_statistics,
        simulated_parameters,
        simulation_context,
        n_ppc_samples=200,
        comparison_epsilon=POSTERIOR_COMPARISON_EPSILON,
    )

    plot_joint_posteriors(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
        comparison_epsilon=POSTERIOR_COMPARISON_EPSILON,
    )
    rich_set_posterior = get_accepted_posterior_for_summary_set(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
        "Rich set",
        SUMMARY_SET_INDICES["Rich set"],
        comparison_epsilon=POSTERIOR_COMPARISON_EPSILON,
    )
    print_joint_posterior_correlations(
        rich_set_posterior,
        "Rich set",
        comparison_epsilon=POSTERIOR_COMPARISON_EPSILON,
    )

    plot_marginal_posteriors(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
        acceptance_epsilon_list=acceptance_epsilon_list,
    )

    plot_identifiability_posteriors(
        simulated_parameters,
        simulated_summary_statistics,
        observed_summary_statistics,
        comparison_epsilon=POSTERIOR_COMPARISON_EPSILON,
    )

###############
#
#
# 5. RUN MAIN FUNCTION
#
#
###############
if __name__ == "__main__":
    main()
