"""
Synthetic-truth recovery run for Synthetic Likelihood MCMC (SL-MCMC).
Simulates data from known parameters and checks if SL-MCMC recovers them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from simulator_optimised import simulate
from abc_rejection import (
    N, PARAMETER_NAMES, REFERENCE_RESULTS_PATH, build_simulation_context,
    seed
)
from synthetic_likelihood_mcmc import (
    SUMMARY_INDICES, M_sims, PRIOR_LOWER, PRIOR_UPPER,
    init_worker, log_synthetic_likelihood, simulate_summaries_worker
)

# True parameters for recovery
TRUE_PARAMS = np.array([0.25, 0.08, 0.40])

BASE_DIR = Path(__file__).resolve().parent
RECOVERY_DIR = BASE_DIR / "outputs" / "sl_mcmc_recovery"
RECOVERY_PLOTS_DIR = RECOVERY_DIR / "plots"

def run_recovery_sl_mcmc():
    RECOVERY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed + 777)
    sim_context = build_simulation_context(N)
    
    # Load calibration manually
    with np.load(REFERENCE_RESULTS_PATH, allow_pickle=True) as data:
        calibration = {"summary_mu": data["summary_mu"], "summary_sigma": data["summary_sigma"]}
    
    # 1. Generate Synthetic "Observed" Data from Truth
    print(f"Generating synthetic observed data from truth: {dict(zip(PARAMETER_NAMES, TRUE_PARAMS))}")
    # We need to initialize the global variables for the worker to function in main thread
    init_worker(sim_context, calibration)
    obs_s = simulate_summaries_worker(TRUE_PARAMS, rng.integers(0, 2**31))
    
    # 2. Run SL-MCMC
    N_iter_recovery = 5000 # Shortened for verification
    BURN_IN_recovery = 1000
    
    n_workers = mp.cpu_count()
    print(f"Starting SL-MCMC Recovery Run ({n_workers} workers, {N_iter_recovery} steps)...")
    
    with mp.Pool(n_workers, initializer=init_worker, initargs=(sim_context, calibration)) as pool:
        curr_p = np.array([0.2, 0.1, 0.3]) # Starting point
        curr_L = log_synthetic_likelihood(curr_p, obs_s, pool, rng)
        
        chain = np.zeros((N_iter_recovery, 3))
        prop_sd = np.array([0.02, 0.01, 0.05])
        acc_count = 0
        
        for i in tqdm(range(N_iter_recovery), desc="Recovery Sampling"):
            if i > 0 and i % 100 == 0:
                prop_sd *= 1.02 if (acc_count / 100) > 0.234 else 0.98
                acc_count = 0
            
            prop_p = curr_p + rng.normal(0, prop_sd)
            if np.all(prop_p >= PRIOR_LOWER) and np.all(prop_p <= PRIOR_UPPER):
                prop_L = log_synthetic_likelihood(prop_p, obs_s, pool, rng)
                if rng.random() < np.exp(prop_L - curr_L):
                    curr_p, curr_L = prop_p, prop_L
                    acc_count += 1
            chain[i] = curr_p

    # 3. Save & Plot
    pd.DataFrame(chain, columns=PARAMETER_NAMES).to_csv(RECOVERY_DIR / "recovery_chain.csv", index=False)
    post = chain[BURN_IN_recovery:]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for j, p in enumerate(PARAMETER_NAMES):
        axes[j].hist(post[:, j], bins=30, alpha=0.6, color='skyblue', density=True, label="SL-MCMC Posterior")
        axes[j].axvline(TRUE_PARAMS[j], color='red', linestyle='--', lw=2, label="mean of accepted values")
        axes[j].set_xlabel(p)
        axes[j].set_title(f"Recovery: {p}")
        axes[j].legend()
    
    plt.tight_layout()
    plt.savefig(RECOVERY_PLOTS_DIR / "synthetic_truth_recovery_sl.png")
    plt.close()
    
    print(f"Recovery run complete. Plots saved to: {RECOVERY_PLOTS_DIR}")

if __name__ == "__main__":
    run_recovery_sl_mcmc()
