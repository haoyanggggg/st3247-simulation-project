"""
Optimized Synthetic Likelihood MCMC (SL-MCMC) for the adaptive-network SIR model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from numpy.linalg import slogdet
from scipy.linalg import solve
import multiprocessing as mp
from functools import partial

from simulator_optimised import simulate
from observed_summaries import get_obs_summaries
from abc_rejection import (
    N,
    PARAMETER_NAMES,
    REFERENCE_RESULTS_PATH,
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
    degrees,
    epsilon as log_offset,
    seed
)

# 1. CONFIGURATION
SUMMARY_INDICES = (0, 1, 3, 4, 6, 7)
M_sims = 30 # Optimized: 5x summary dimension
N_iter = 30000
BURN_IN = 5000

PRIOR_LOWER = np.array([0.05, 0.02, 0.0])
PRIOR_UPPER = np.array([0.50, 0.20, 0.80])

BASE_DIR = Path(__file__).resolve().parent
SL_MCMC_DIR = BASE_DIR / "outputs" / "synthetic_likelihood_mcmc"
SL_MCMC_PLOTS_DIR = SL_MCMC_DIR / "plots"

# 2. WORKER INITIALIZATION
_worker_sim_context = None
_worker_calibration = None

def init_worker(sim_context, calibration):
    global _worker_sim_context, _worker_calibration
    _worker_sim_context = sim_context
    _worker_calibration = calibration

def simulate_summaries_worker(params, seed):
    rng = np.random.default_rng(seed)
    beta, gamma, rho = params
    inf, rew, deg_hist = simulate(beta=beta, gamma=gamma, rho=rho, rng=rng, simulation_context=_worker_sim_context)
    s1 = np.max(inf)
    s2 = np.argmax(inf)
    s3 = compute_linear_slope(np.log(inf[early_time_window_min:early_time_window_max] + log_offset), early_time_points_centered, early_time_points_denom)
    s4 = compute_linear_slope(np.log(rew[early_time_window_min:early_time_window_max] + log_offset), early_time_points_centered, early_time_points_denom)
    m = np.sum(degrees * deg_hist) / N
    s5 = (np.sum((degrees**2) * deg_hist) / N) - m**2
    s6 = compute_linear_slope(np.log(inf[late_time_window_min:late_time_window_max] + log_offset), late_time_points_centered, late_time_points_denom)
    s7 = np.sum(rew) / (np.sum(inf) + log_offset)
    peak = s1
    half = 0.5 * peak
    idx = np.where(inf >= half)[0]
    s8 = float(idx[-1] - idx[0]) if len(idx) >= 2 else 0.0
    full_s = np.array([s1, s2, s3, s4, s5, s6, s7, s8])
    subset = full_s[list(SUMMARY_INDICES)]
    return (subset - _worker_calibration["summary_mu"]) / _worker_calibration["summary_sigma"]

def log_synthetic_likelihood(params, obs_s, pool, master_rng):
    seeds = master_rng.integers(0, 2**31, size=M_sims)
    summaries = np.array(pool.map(partial(simulate_summaries_worker, params), seeds))
    mu = np.mean(summaries, axis=0)
    S = np.cov(summaries, rowvar=False)
    d = mu.size
    Sigma = 0.9 * S + 0.1 * (np.eye(d) * np.trace(S) / d)
    diff = obs_s - mu
    sign, logdet = slogdet(Sigma)
    if sign <= 0: return -np.inf
    return -0.5 * (logdet + diff.T @ solve(Sigma, diff))

def run_sl_mcmc():
    SL_MCMC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with np.load(REFERENCE_RESULTS_PATH, allow_pickle=True) as data:
        calibration = {"summary_mu": data["summary_mu"], "summary_sigma": data["summary_sigma"]}
    sim_context = build_simulation_context(N)
    obs_s = (np.array(get_obs_summaries())[list(SUMMARY_INDICES)] - calibration["summary_mu"]) / calibration["summary_sigma"]

    print(f"Starting SL-MCMC (Procs: {mp.cpu_count()}, M: {M_sims})")
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(sim_context, calibration)) as pool:
        curr_p = np.array([0.2, 0.1, 0.3])
        curr_L = log_synthetic_likelihood(curr_p, obs_s, pool, rng)
        chain = np.zeros((N_iter, 3))
        prop_sd = np.array([0.02, 0.01, 0.05])
        acc_count = 0
        for i in tqdm(range(N_iter), desc="Sampling"):
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

    pd.DataFrame(chain, columns=PARAMETER_NAMES).to_csv(SL_MCMC_DIR / "synthetic_likelihood_chain.csv", index=False)
    post = chain[BURN_IN:]
    pd.DataFrame(post, columns=PARAMETER_NAMES).to_csv(SL_MCMC_DIR / "synthetic_likelihood_output.csv", index=False)
    
    # Diagnostics
    acc_rate = np.mean(np.diff(chain, axis=0).any(axis=1))
    print(f"Final Acceptance Rate: {acc_rate:.2%}")
    
    def ess(x):
        n = len(x)
        if n < 2: return 1
        x_c = x - np.mean(x)
        rho = np.correlate(x_c, x_c, mode='full')[n-1:]
        rho = rho / (rho[0] + 1e-12)
        neg = np.where(rho < 0)[0]
        max_lag = neg[0] if len(neg) > 0 else n
        return n / (1 + 2 * np.sum(rho[1:max_lag]))

    diag_df = pd.DataFrame({"metric": ["acceptance_rate"] + [f"ess_{p}" for p in PARAMETER_NAMES],
                            "value": [acc_rate] + [ess(post[:, j]) for j in range(3)]})
    diag_df.to_csv(SL_MCMC_DIR / "synthetic_likelihood_diagnostics.csv", index=False)

    # Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for j, p in enumerate(PARAMETER_NAMES):
        axes[j].plot(chain[:, j], lw=0.3)
        axes[j].axvline(BURN_IN, color='r', ls='--')
        axes[j].set_ylabel(p)
    plt.tight_layout()
    plt.savefig(SL_MCMC_PLOTS_DIR / "synthetic_likelihood_mcmc_diagnostics.png")
    
    abc_mcmc_path = BASE_DIR / "outputs" / "abc_mcmc" / "param_estimates" / "abc_mcmc_output.csv"
    if abc_mcmc_path.exists():
        abc = pd.read_csv(abc_mcmc_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for j, p in enumerate(PARAMETER_NAMES):
            axes[j].hist(abc[p], bins=30, alpha=0.5, label="ABC-MCMC", density=True)
            axes[j].hist(post[:, j], bins=30, alpha=0.5, label="SL-MCMC", density=True)
            axes[j].set_xlabel(p)
            axes[j].legend()
        plt.tight_layout()
        plt.savefig(SL_MCMC_PLOTS_DIR / "synthetic_likelihood_vs_abc_mcmc_comparison.png")
        plt.close()

    # Comparison with Basic ABC (eps=0.01)
    basic_abc_dir = BASE_DIR / "outputs" / "basic_abc" / "param_estimates"
    basic_abc_files = list(basic_abc_dir.glob("abc-basic_eps-0.0100_*.csv"))
    if basic_abc_files:
        basic_abc_path = basic_abc_files[0]
        basic = pd.read_csv(basic_abc_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for j, p in enumerate(PARAMETER_NAMES):
            axes[j].hist(basic[p], bins=30, alpha=0.5, label="Basic ABC (eps=0.01)", density=True)
            axes[j].hist(post[:, j], bins=30, alpha=0.5, label="SL-MCMC", density=True)
            axes[j].set_xlabel(p)
            axes[j].legend()
        plt.tight_layout()
        plt.savefig(SL_MCMC_PLOTS_DIR / "synthetic_likelihood_vs_basic_abc_comparison.png")
        plt.close()

if __name__ == "__main__":
    run_sl_mcmc()
