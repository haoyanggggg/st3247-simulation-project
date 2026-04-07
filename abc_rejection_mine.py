"""
Approximate Bayesian Computation (ABC) for SIR Epidemic Model with Adaptive Network

Author: Chris 
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

###############
#
#
# 1. IMPORT LIBRARIES
#
#
###############
import numpy as np
import pandas as pd
from simulator import simulate
from observed_summaries import get_obs_summaries
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


###############
#
#
# 2. GLOBAL VARIABLES
#
#
###############
N = 200
N_sim = 30_000 # with threshold of 1%, we will have 300 accepted samples for posterior analysis
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

summary_statistics_name = [
    "Max infection fraction",
    "Time to peak",
    "Early growth rate of infection",
    "Early growth rate of rewiring",
    "Variance structure of degree counts",
    "Late decay rate of infection"
]

###############
#
#
# 3. HELPER FUNCTIONS
#
#
###############
def one_simulation()-> list: 
    """
    Run a single simulation of the adaptive SIR model and compute summary statistics.

    This function:
    1. Samples parameters (beta, gamma, rho) from their prior distributions.
    2. Simulates the epidemic and network dynamics.
    3. Computes a set of summary statistics used for ABC inference.

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
    beta = np.random.uniform(0.05, 0.5)  # Infection rate
    gamma = np.random.uniform(0.02, 0.2) # Recovery rate
    rho = np.random.uniform(0, 0.8)  # Rewiring rate

    # Simulate data using the sampled parameters
    infected_fraction, rewire_counts, degree_histogram = simulate(beta=beta, gamma=gamma, rho=rho)
    # print(infected_fraction)

    # SUMMARY STATISTIC 1: Max infection fraction INFORMS BETA/GAMMA
    max_infection_frac = np.max(infected_fraction)
    # print("Max infection fraction:", max_infection_frac)

    # SUMMARY STATISTIC 2: Time to peak INFORMS BETA/GAMMA
    time_to_peak = np.argmax(infected_fraction)
    # print("Time to peak:", time_to_peak)



    early_time_window_min = 2
    early_time_window_max = 6+1
    early_time_points = np.arange(early_time_window_min, early_time_window_max)

    # SUMMARY STATISTIC 3: Early growth rate INFORMS BETA/GAMMA
    early_log_infection_fraction = np.log(infected_fraction[early_time_window_min:early_time_window_max] + epsilon)  # Add epsilon to avoid log(0)
    slope_early_infection, _ = np.polyfit(early_time_points, early_log_infection_fraction, 1)
    # print("Early infection growth rate (slope):", slope_early_infection)

    # SUMMARY STATISTIC 4: Mean rewire counts during early infection
    early_log_rewire_counts = np.log(rewire_counts[early_time_window_min:early_time_window_max] + epsilon)  # Add epsilon to avoid log(0)
    slope_rewire, _ = np.polyfit(early_time_points, early_log_rewire_counts, 1)
    # print("Early rewire count slope:", slope_rewire)

    # SUMMARY STATISTIC 5: Variance structure of degree counts INFORMS RHO
    degrees = np.arange(degree_counts_max)
    mean_degree = np.sum(degrees * degree_histogram) / N
    mean_degree_sq = np.sum((degrees**2) * degree_histogram) / N

    var_degree = mean_degree_sq - mean_degree**2
    # print("Variance of degree counts:", var_degree)


    late_time_window_min = 13
    late_time_window_max = 20+1
    late_time_points = np.arange(late_time_window_min, late_time_window_max)

    # SUMMARY STATISTIC 6: Decay structure of infection INFORMS RHO
    late_log_infection_fraction = np.log(infected_fraction[late_time_window_min:late_time_window_max] + epsilon)  # Add epsilon to avoid log(0)
    slope_late_infection, _ = np.polyfit(late_time_points, late_log_infection_fraction, 1)
    # print("Late infection decay rate (slope):", slope_late_infection)

    return [
        (max_infection_frac, time_to_peak, slope_early_infection, 
            slope_rewire, var_degree, slope_late_infection),
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
    simulated_mu = np.array(simulated_summary_statistics).mean(axis=0)
    simulated_sigma = np.array(simulated_summary_statistics).std(axis=0)

    standardized_simulated = (np.array(simulated_summary_statistics) - simulated_mu) / simulated_sigma
    standardized_observed = (np.array(observed_summary_statistics) - simulated_mu) / simulated_sigma

    return standardized_simulated, standardized_observed

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
    acceptance_epsilon = np.quantile(distances, acceptance_epsilon)  # 1% acceptance threshold
    accepted_idx = distances <= acceptance_epsilon
    accepted_summaries = np.array(simulated_summary_statistics)[accepted_idx]
    return (accepted_summaries, distances)

def save_samples_and_plots(accepted_summaries, 
                           observed_summary_statistics,
                           distances,
                           simulated_parameters,
                           acceptance_epsilon=0.01):
    """
    Generate diagnostic plots for accepted summary statistics and save posterior samples.

    This function:
    1. Plots histograms of accepted summary statistics.
    2. Overlays the observed summary statistic as a vertical reference line.
    3. Extracts accepted parameter samples based on the ABC tolerance criterion.
    4. Saves the accepted parameter samples as a CSV file (optional, currently commented).

    Parameters
    ----------
    accepted_summaries : ndarray of shape (N_accepted, d)
        Unstandardized summary statistics from accepted simulations.

    observed_summary_statistics : array-like of shape (d,)
        Summary statistics computed from the observed dataset.

    distances : ndarray of shape (N,)
        Euclidean distances between each simulated summary and the observed summary.

    simulated_parameters : list of tuples of length N
        Parameter samples corresponding to each simulation, where each tuple is 
        (beta, gamma, rho).

    acceptance_epsilon : float, optional (default=0.01)
        Quantile threshold used to determine accepted samples.
        Only simulations with distances ≤ quantile(distances, acceptance_epsilon)
        are retained.

    Behavior
    --------
    - For each summary statistic:
        * A histogram of accepted values is plotted.
        * The observed value is shown as a red vertical line.
    - Accepted parameter samples are filtered using the same ε threshold.
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
    - Ensure consistency: the same ε used here should match the one used during 
      acceptance in the ABC step.

    Returns
    -------
    None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    for i in range(accepted_summaries.shape[1]):
        plt.hist(accepted_summaries[:, i], bins=17, alpha=0.7)
        plt.axvline(observed_summary_statistics[i], color='red', linewidth=2)
        plt.title(f"Summary {summary_statistics_name[i]} (Accepted Simulations)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(f"outputs/chris/sanity_check/summary_{summary_statistics_name[i]}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    accepted_parameters = [params for params, dist in zip(simulated_parameters, distances) if dist <= acceptance_epsilon]

    # Save accepted parameters to CSV
    final_chosen_posteriors = pd.DataFrame(accepted_parameters, columns=['beta', 'gamma', 'rho'])
    filename = f"abc-basic_{timestamp}.csv"
    final_chosen_posteriors.to_csv(f"outputs/chris/param_estimates/{filename}", index=False)

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
    np.random.seed(2026)  # Set seed for reproducibility

    # simulate 30_000 times
    simulated_summary_statistics, simulated_parameters = zip(*[
        one_simulation() for _ in tqdm(range(N_sim), desc="Running ABC")
    ])

    # scale simulated data and observed data
    observed_summary_statistics = get_obs_summaries()
    standardized_simulated, standardized_observed = scale_summary_statistics(simulated_summary_statistics, 
                                                                             observed_summary_statistics)
    
    # obtain accepted summaries from ABC rejection algo
    accepted_summaries, distances = obtain_accepted_summaries(standardized_simulated, 
                                                            standardized_observed,
                                                            simulated_summary_statistics)
    
    # save parameters and plots of accepted summaries
    save_samples_and_plots(accepted_summaries, observed_summary_statistics, distances, simulated_parameters)

###############
#
#
# 5. RUN MAIN FUNCTION
#
#
###############
main()