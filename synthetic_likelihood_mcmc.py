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
from time import perf_counter

from simulator_optimised import simulate
from observed_summaries import get_obs_summaries
from abc_rejection import (
    N,
    PARAMETER_NAMES,
    REFERENCE_RESULTS_PATH,
    REFERENCE_SUMMARY_SET_NAME,
    REFERENCE_SUMMARY_SET_INDICES,
    REFERENCE_SUMMARY_SET_SLUG,
    SUMMARY_SET_STUDY_DIR,
    POSTERIOR_COMPARISON_EPSILON,
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
from runtime_summary import write_runtime_summary

# 1. CONFIGURATION
SUMMARY_INDICES = tuple(REFERENCE_SUMMARY_SET_INDICES)
M_sims = 30
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
    summary_indices = _worker_calibration["summary_indices"]
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
    subset = full_s[list(summary_indices)]
    return (subset - _worker_calibration["summary_mu"]) / _worker_calibration["summary_sigma"]

def log_synthetic_likelihood(params, obs_s, pool, master_rng):
    seeds = master_rng.integers(0, 2**31, size=M_sims)
    simulation_runtime_start = perf_counter()
    summaries = np.array(pool.map(partial(simulate_summaries_worker, params), seeds))
    simulation_wall_clock_seconds = perf_counter() - simulation_runtime_start
    mu = np.mean(summaries, axis=0)
    S = np.cov(summaries, rowvar=False)
    d = mu.size
    Sigma = 0.9 * S + 0.1 * (np.eye(d) * np.trace(S) / d)
    diff = obs_s - mu
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        return -np.inf, float(simulation_wall_clock_seconds)
    return -0.5 * (logdet + diff.T @ solve(Sigma, diff)), float(simulation_wall_clock_seconds)

def run_sl_mcmc():
    SL_MCMC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with np.load(REFERENCE_RESULTS_PATH, allow_pickle=True) as data:
        calibration_summary_indices = tuple(
            np.asarray(
                data["summary_indices"] if "summary_indices" in data else SUMMARY_INDICES,
                dtype=np.int64,
            ).tolist()
        )
        calibration = {
            "summary_mu": np.asarray(data["summary_mu"], dtype=np.float64),
            "summary_sigma": np.asarray(data["summary_sigma"], dtype=np.float64),
            "summary_indices": calibration_summary_indices,
        }
    if calibration["summary_mu"].shape[0] != len(calibration["summary_indices"]):
        raise ValueError(
            "Saved rejection calibration has inconsistent summary dimensions. "
            f"summary_mu has length {calibration['summary_mu'].shape[0]} but "
            f"summary_indices has length {len(calibration['summary_indices'])}."
        )
    sim_context = build_simulation_context(N)
    obs_s = (
        np.array(get_obs_summaries())[list(calibration["summary_indices"])]
        - calibration["summary_mu"]
    ) / calibration["summary_sigma"]

    print(f"Starting SL-MCMC (Procs: {mp.cpu_count()}, M: {M_sims})")
    simulation_wall_clock_seconds = 0.0
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(sim_context, calibration)) as pool:
        curr_p = np.array([0.2, 0.1, 0.3])
        likelihood_evaluations = 1
        curr_L, current_simulation_seconds = log_synthetic_likelihood(curr_p, obs_s, pool, rng)
        simulation_wall_clock_seconds += current_simulation_seconds
        chain = np.zeros((N_iter, 3))
        prop_sd = np.array([0.02, 0.01, 0.05])
        acc_count = 0
        for i in tqdm(range(N_iter), desc="Sampling"):
            if i > 0 and i % 100 == 0:
                prop_sd *= 1.02 if (acc_count / 100) > 0.234 else 0.98
                acc_count = 0
            prop_p = curr_p + rng.normal(0, prop_sd)
            if np.all(prop_p >= PRIOR_LOWER) and np.all(prop_p <= PRIOR_UPPER):
                likelihood_evaluations += 1
                prop_L, proposal_simulation_seconds = log_synthetic_likelihood(prop_p, obs_s, pool, rng)
                simulation_wall_clock_seconds += proposal_simulation_seconds
                if rng.random() < np.exp(prop_L - curr_L):
                    curr_p, curr_L = prop_p, prop_L
                    acc_count += 1
            chain[i] = curr_p

    post = chain[BURN_IN:]
    pd.DataFrame(chain, columns=PARAMETER_NAMES).to_csv(SL_MCMC_DIR / "synthetic_likelihood_chain.csv", index=False)
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

    ess_by_parameter = {
        name: float(ess(post[:, j]))
        for j, name in enumerate(PARAMETER_NAMES)
    }
    diag_df = pd.DataFrame(
        {
            "metric": ["acceptance_rate"] + [f"ess_{p}" for p in PARAMETER_NAMES],
            "value": [acc_rate] + [ess_by_parameter[name] for name in PARAMETER_NAMES],
        }
    )
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

    # Comparison with the current reference-set rejection ABC posterior
    basic_abc_dir = SUMMARY_SET_STUDY_DIR / REFERENCE_SUMMARY_SET_SLUG / "param_estimates"
    basic_abc_files = sorted(
        basic_abc_dir.glob(
            f"{REFERENCE_SUMMARY_SET_SLUG}_abc-basic_eps-{POSTERIOR_COMPARISON_EPSILON:.4f}_*.csv"
        )
    )
    if basic_abc_files:
        basic_abc_path = basic_abc_files[-1]
        basic = pd.read_csv(basic_abc_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for j, p in enumerate(PARAMETER_NAMES):
            axes[j].hist(
                basic[p],
                bins=30,
                alpha=0.5,
                label=(
                    f"Basic ABC ({REFERENCE_SUMMARY_SET_NAME}, "
                    f"eps={POSTERIOR_COMPARISON_EPSILON:.2f})"
                ),
                density=True,
            )
            axes[j].hist(post[:, j], bins=30, alpha=0.5, label="SL-MCMC", density=True)
            axes[j].set_xlabel(p)
            axes[j].legend()
        plt.tight_layout()
        plt.savefig(SL_MCMC_PLOTS_DIR / "synthetic_likelihood_vs_basic_abc_comparison.png")
        plt.close()

    write_runtime_summary(
        method_name="synthetic_likelihood_mcmc",
        total_simulator_calls=int(likelihood_evaluations * M_sims),
        wall_clock_seconds=float(simulation_wall_clock_seconds),
        posterior_sample_size=int(post.shape[0]),
        acceptance_rate=float(acc_rate),
        ess=ess_by_parameter,
    )

if __name__ == "__main__":
    run_sl_mcmc()
