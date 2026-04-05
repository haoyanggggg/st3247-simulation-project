import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulator import simulate


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

# BASIC
def compute_summaries_basic(infected_ts, rewire_ts, degree_hist):
    peak = np.max(infected_ts)
    t_peak = np.argmax(infected_ts)
    final = infected_ts[-1]
    total = np.sum(infected_ts)
    duration = np.sum(infected_ts > 0.01)
    early_growth = infected_ts[5] - infected_ts[0]

    return np.array([
        peak, t_peak, final, total,
        duration, early_growth
    ])


# EXTENDED
def compute_summaries_extended(infected_ts, rewire_ts, degree_hist):
    # infected
    peak = np.max(infected_ts)
    t_peak = np.argmax(infected_ts)
    final = infected_ts[-1]
    total = np.sum(infected_ts)
    duration = np.sum(infected_ts > 0.01)
    early_growth = infected_ts[5] - infected_ts[0]

    # rewiring
    total_rewire = np.sum(rewire_ts)
    peak_rewire = np.max(rewire_ts)

    # degree
    degrees = np.arange(len(degree_hist))
    total_nodes = np.sum(degree_hist)

    mean_deg = np.sum(degrees * degree_hist) / total_nodes
    var_deg = np.sum((degrees - mean_deg) ** 2 * degree_hist) / total_nodes

    return np.array([
        peak, t_peak, final, total,
        duration, early_growth,
        total_rewire, peak_rewire,
        mean_deg, var_deg
    ])


# ==============================
# DISTANCE
# ==============================
def compute_distance(s_sim, s_obs, scale):
    return np.sum(((s_sim - s_obs) / scale) ** 2)


# ==============================
# OBSERVED SUMMARIES
# ==============================
def compute_observed_summaries(infected_df, rewire_df, degree_df, summary_func):
    summaries = []

    for rep_id in infected_df['replicate_id'].unique():
        inf_ts = infected_df[infected_df.replicate_id == rep_id]['infected_fraction'].values
        rew_ts = rewire_df[rewire_df.replicate_id == rep_id]['rewire_count'].values
        deg_hist = degree_df[degree_df.replicate_id == rep_id]['count'].values

        summaries.append(summary_func(inf_ts, rew_ts, deg_hist))

    return np.mean(summaries, axis=0)


# ==============================
# ABC
# ==============================
def abc_rejection(simulate, observed_summaries, summary_func,
                  n_samples=5000, accept_frac=0.02, seed=42):

    rng = np.random.default_rng(seed)

    thetas = []
    summaries = []

    for i in range(n_samples):
        if i % 500 == 0:
            print(f"{i}/{n_samples}")

        theta = sample_prior(rng)
        beta, gamma, rho = theta

        inf_ts, rew_ts, deg_hist = simulate(beta, gamma, rho, rng=rng)
        s_sim = summary_func(inf_ts, rew_ts, deg_hist)

        thetas.append(theta)
        summaries.append(s_sim)

    thetas = np.array(thetas)
    summaries = np.array(summaries)

    scale = np.std(summaries, axis=0) + 1e-8

    distances = np.array([
        compute_distance(summaries[i], observed_summaries, scale)
        for i in range(n_samples)
    ])

    n_accept = int(accept_frac * n_samples)
    idx = np.argsort(distances)[:n_accept]

    return thetas[idx]


# ==============================
# ACCEPT_FRAC EXPERIMENT
# ==============================
def run_acceptance_experiment(accept_fracs, simulate, observed_summaries, summary_func):
    results = {}

    for frac in accept_fracs:
        print(f"\nRunning ABC with accept_frac = {frac}")
        posterior = abc_rejection(
            simulate,
            observed_summaries,
            summary_func,
            n_samples=5000,
            accept_frac=frac
        )
        results[frac] = posterior

    return results


# ==============================
# PLOTTING
# ==============================

# Posterior comparison (basic vs extended)
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


# Epsilon overlay (extended to compare different accept_fracs)
def plot_acceptance_overlay(results, filename):
    labels = ['beta', 'gamma', 'rho']
    fracs = sorted(results.keys())

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


# Pairwise scatter
def plot_pairwise(samples, filename):
    plt.figure(figsize=(8, 8))

    plt.subplot(2,2,1)
    plt.scatter(samples[:,0], samples[:,1], alpha=0.3)
    plt.xlabel("beta"); plt.ylabel("gamma")

    plt.subplot(2,2,2)
    plt.scatter(samples[:,0], samples[:,2], alpha=0.3)
    plt.xlabel("beta"); plt.ylabel("rho")

    plt.subplot(2,2,3)
    plt.scatter(samples[:,1], samples[:,2], alpha=0.3)
    plt.xlabel("gamma"); plt.ylabel("rho")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ==============================
# LOAD DATA
# ==============================
infected_df = pd.read_csv("infected_timeseries.csv")
rewire_df = pd.read_csv("rewiring_timeseries.csv")
degree_df = pd.read_csv("final_degree_histograms.csv")


# ==============================
# RUN
# ==============================

# Posterior comparison (fixed epsilon after choosing from experiment)
fixed_frac = 0.02

obs_basic = compute_observed_summaries(
    infected_df, rewire_df, degree_df,
    compute_summaries_basic
)

posterior_basic = abc_rejection(
    simulate,
    obs_basic,
    compute_summaries_basic,
    accept_frac=fixed_frac
)

obs_extended = compute_observed_summaries(
    infected_df, rewire_df, degree_df,
    compute_summaries_extended
)

posterior_extended = abc_rejection(
    simulate,
    obs_extended,
    compute_summaries_extended,
    accept_frac=fixed_frac
)

plot_posterior_comparison(
    posterior_basic,
    posterior_extended,
    "posterior_basic_vs_extended.png"
)


# Epsilon experiment (extended only)
accept_fracs = [0.1, 0.05, 0.02, 0.01]

results = run_acceptance_experiment(
    accept_fracs,
    simulate,
    obs_extended,
    compute_summaries_extended
)

plot_acceptance_overlay(results, "epsilon_comparison.png")


# Choose best epsilon after inspecting plot
best_frac = 0.02  # selected after inspection

plot_pairwise(results[best_frac], "pairwise_scatter.png")