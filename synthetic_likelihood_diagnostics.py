"""
Synthetic Likelihood Diagnostics for the adaptive-network SIR model.

This script performs the Assumption Check: 
Histograms of summaries at the posterior mean to check Normality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import multiprocessing as mp
from functools import partial

from simulator_optimised import simulate
from abc_rejection import (
    N, PARAMETER_NAMES, REFERENCE_RESULTS_PATH, build_simulation_context,
    compute_linear_slope, early_time_window_min, early_time_window_max,
    early_time_points_centered, early_time_points_denom, late_time_window_min,
    late_time_window_max, late_time_points_centered, late_time_points_denom,
    degrees, epsilon as log_offset, seed
)
from synthetic_likelihood_mcmc import (
    SUMMARY_INDICES, SL_MCMC_DIR, init_worker, simulate_summaries_worker
)

DIAG_PLOTS_DIR = SL_MCMC_DIR / "diagnostics"

SUMMARY_NAMES = [
    "Max Infection", "Time to Peak", "Early Inf Slope", "Early Rewire Slope",
    "Deg Variance", "Late Inf Slope", "Rewire/Inf", "Peak Width"
]
SUBSET_NAMES = [SUMMARY_NAMES[i] for i in SUMMARY_INDICES]

def run_assumption_check(post_mean, pool):
    """Generate histograms of summaries at a fixed point to check Normality."""
    print(f"Running Assumption Check: Simulating at posterior mean {post_mean}...")
    n_samples = 500 # Increased for better distribution check
    rng = np.random.default_rng(seed + 999)
    seeds = rng.integers(0, 2**31, size=n_samples)
    
    summaries = np.array(pool.map(partial(simulate_summaries_worker, post_mean), seeds))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(len(SUMMARY_INDICES)):
        data = summaries[:, i]
        axes[i].hist(data, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(min(data), max(data), 100)
        axes[i].plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2)
        
        axes[i].set_title(f"{SUBSET_NAMES[i]}\n(Shapiro p={stats.shapiro(data)[1]:.3f})")
        axes[i].set_xlabel("Standardized Summary")
    
    plt.tight_layout()
    plt.savefig(DIAG_PLOTS_DIR / "assumption_check_normality.png")
    plt.close()

def main():
    DIAG_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    post_path = SL_MCMC_DIR / "synthetic_likelihood_output.csv"
    if not post_path.exists():
        print("Error: Run synthetic_likelihood_mcmc.py first.")
        return
    
    post_df = pd.read_csv(post_path)
    post_mean = post_df.mean().values
    
    with np.load(REFERENCE_RESULTS_PATH, allow_pickle=True) as data:
        calibration = {"summary_mu": data["summary_mu"], "summary_sigma": data["summary_sigma"]}
    sim_context = build_simulation_context(N)
    
    print(f"Initializing process pool for assumption check...")
    with mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(sim_context, calibration)) as pool:
        run_assumption_check(post_mean, pool)

    print(f"\nAssumption Check Complete. Plots saved to: {DIAG_PLOTS_DIR}")

if __name__ == "__main__":
    main()
