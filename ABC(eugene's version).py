import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulator_optimised import simulate

from numpy.linalg import inv, LinAlgError


# ==============================
# PRIOR
# ==============================
def sample_prior(rng):
    return np.array([
        rng.uniform(0.05, 0.5),
        rng.uniform(0.02, 0.2),
        rng.uniform(0.0, 0.8)
    ])


# ==============================
# SUMMARY STATISTICS
# ==============================

def compute_summaries_basic(infected_ts, rewire_ts, degree_hist):
    peak        = np.max(infected_ts)
    t_peak      = np.argmax(infected_ts)
    total       = np.sum(infected_ts)
    early_growth = infected_ts[5] - infected_ts[0]
    return np.array([peak, t_peak, total, early_growth])


def compute_summaries_extended(infected_ts, rewire_ts, degree_hist):
    peak        = np.max(infected_ts)
    t_peak      = np.argmax(infected_ts)
    total       = np.sum(infected_ts)
    early_growth = infected_ts[5] - infected_ts[0]
    total_rewire = np.sum(rewire_ts)

    degrees     = np.arange(len(degree_hist))
    total_nodes = np.sum(degree_hist)
    mean_deg    = np.sum(degrees * degree_hist) / total_nodes
    var_deg     = np.sum((degrees - mean_deg) ** 2 * degree_hist) / total_nodes

    return np.array([peak, t_peak, total, early_growth,
                     total_rewire, mean_deg, var_deg])


# ==============================
# OBSERVED SUMMARIES
# ==============================
def compute_observed_summaries(infected_df, rewire_df, degree_df, summary_func):
    summaries = []
    for rep_id in infected_df['replicate_id'].unique():
        inf_ts   = infected_df[infected_df.replicate_id == rep_id]['infected_fraction'].values
        rew_ts   = rewire_df[rewire_df.replicate_id == rep_id]['rewire_count'].values
        deg_hist = degree_df[degree_df.replicate_id == rep_id]['count'].values
        summaries.append(summary_func(inf_ts, rew_ts, deg_hist))
    return np.mean(summaries, axis=0)


# ==============================
# ABC
# ==============================
def abc_rejection(simulate, observed_summaries, summary_func,
                  n_samples=5000, accept_frac=1.0, seed=42):
    rng = np.random.default_rng(seed)
    thetas, summaries = [], []

    for i in range(n_samples):
        if i % 500 == 0:
            print(f"{i}/{n_samples}")
        theta = sample_prior(rng)
        inf_ts, rew_ts, deg_hist = simulate(*theta, rng=rng)
        thetas.append(theta)
        summaries.append(summary_func(inf_ts, rew_ts, deg_hist))

    thetas    = np.array(thetas)
    summaries = np.array(summaries)
    scale     = np.std(summaries, axis=0) + 1e-8
    distances = np.sum(((summaries - observed_summaries) / scale) ** 2, axis=1)

    n_accept = int(accept_frac * n_samples)
    idx      = np.argsort(distances)[:n_accept]

    return thetas[idx], thetas, summaries, distances

# ==============================
# MAHALANOBIS DISTANCE (can consider using this instead of scaled Euclidean)
# ==============================
def run_acceptance_experiment(accept_fracs, thetas, distances):
    results = {}
    for frac in accept_fracs:
        n_accept     = int(frac * len(thetas))
        idx          = np.argsort(distances)[:n_accept]
        results[frac] = thetas[idx]
    return results


def compute_scale_mahalanobis(summaries):
    cov = np.cov(summaries.T)
    try:
        cov_inv = inv(cov)
    except LinAlgError:
        # fallback to diagonal if singular
        cov_inv = np.diag(1 / (np.var(summaries, axis=0) + 1e-8))
    return cov_inv

def compute_distance_mahalanobis(s_sim, s_obs, cov_inv):
    diff = s_sim - s_obs
    return diff @ cov_inv @ diff

# ==============================
# PLOTTING
# ==============================

def plot_posterior_comparison(posterior_basic, posterior_extended, filename):
    labels = ['beta', 'gamma', 'rho']
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.hist(posterior_basic[:, i], bins=30, density=True)
        plt.title(f"Basic: {labels[i]}")
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.hist(posterior_extended[:, i], bins=30, density=True)
        plt.title(f"Extended: {labels[i]}")
    plt.suptitle("Posterior Comparison: Basic vs Extended")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_acceptance_overlay(results, filename):
    labels = ['beta', 'gamma', 'rho']
    fracs  = sorted(results.keys())
    plt.figure(figsize=(15, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        for frac in fracs:
            plt.hist(results[frac][:, i], bins=30, density=True, alpha=0.4, label=f"{frac}")
        plt.title(labels[i])
        plt.xlabel(labels[i])
        plt.ylabel("Density")
        plt.legend(title="accept_frac")
    plt.suptitle("Posterior Comparison Across Acceptance Fractions")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_pairwise(samples, filename):
    labels = ['beta', 'gamma', 'rho']
    pairs  = [(0,1), (0,2), (1,2)]
    plt.figure(figsize=(12, 4))
    for idx, (i, j) in enumerate(pairs):
        plt.subplot(1, 3, idx+1)
        plt.scatter(samples[:, i], samples[:, j], alpha=0.3, s=10)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
    plt.suptitle("Pairwise Posterior Scatter")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ==============================
# LOAD DATA
# ==============================
infected_df = pd.read_csv("infected_timeseries.csv")
rewire_df   = pd.read_csv("rewiring_timeseries.csv")
degree_df   = pd.read_csv("final_degree_histograms.csv")


# ==============================
# RUN
# ==============================
fixed_frac   = 0.02
accept_fracs = [0.1, 0.05, 0.02, 0.01]

# --- Basic ---
obs_basic = compute_observed_summaries(infected_df, rewire_df, degree_df, compute_summaries_basic)
posterior_basic, _, _, _ = abc_rejection(
    simulate, obs_basic, compute_summaries_basic, n_samples=5000, accept_frac=fixed_frac
)

# --- Extended: simulate once, reuse for all epsilon experiments ---
obs_extended = compute_observed_summaries(infected_df, rewire_df, degree_df, compute_summaries_extended)
posterior_extended, all_thetas, all_summaries, all_distances = abc_rejection(
    simulate, obs_extended, compute_summaries_extended, n_samples=5000, accept_frac=fixed_frac
)

# --- Plots ---
plot_posterior_comparison(posterior_basic, posterior_extended, "posterior_basic_vs_extended.png")

results = run_acceptance_experiment(accept_fracs, all_thetas, all_distances)
plot_acceptance_overlay(results, "epsilon_comparison.png")

plot_pairwise(results[fixed_frac], "pairwise_scatter.png")