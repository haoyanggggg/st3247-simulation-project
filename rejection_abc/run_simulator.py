import numpy as np
import pandas as pd
from simulator import simulate

def calculate_observed_summaries():
    # 1. Load the datasets [cite: 445]
    inf_df = pd.read_csv('infected_timeseries.csv')
    rew_df = pd.read_csv('rewiring_timeseries.csv')
    deg_df = pd.read_csv('final_degree_histograms.csv')

    # --- Statistic 1: Mean Peak Infected Fraction ---
    # Find the maximum infected fraction for each replicate, then average them
    peak_infections = inf_df.groupby('replicate_id')['infected_fraction'].max()
    obs_peak = peak_infections.mean()

    # --- Statistic 2: Mean Total Rewiring Events ---
    # Sum all rewiring events for each replicate, then average them
    total_rewires = rew_df.groupby('replicate_id')['rewire_count'].sum()
    obs_rewire = total_rewires.mean()

    # --- Statistic 3: Mean Final Degree ---
    # For each replicate, calculate mean degree: sum(degree * count) / total_nodes
    # Total nodes N = 200 [cite: 423]
    def get_mean_degree(group):
        return (group['degree'] * group['count']).sum() / 200

    mean_degrees = deg_df.groupby('replicate_id').apply(get_mean_degree)
    obs_mean_deg = mean_degrees.mean()

    # Combine into the observed summary vector
    s_obs = np.array([obs_peak, obs_rewire, obs_mean_deg])
    
    print("Observed Summary Statistics (S_obs):")
    print(f"  Peak Infected Fraction: {obs_peak:.4f}")
    print(f"  Total Rewiring Events:  {obs_rewire:.2f}")
    print(f"  Mean Final Degree:      {obs_mean_deg:.4f}")
    
    return s_obs

# 1. Define Observed Summaries 
s_obs = calculate_observed_summaries()

# 2. Setup ABC parameters
N_samples = 10000 
pilot_params = []
pilot_summaries = []

# Initialize a Random Number Generator for the prior sampling
rng_master = np.random.default_rng(seed=42) 

print(f"Starting Pilot Run of {N_samples} simulations...")

for i in range(N_samples):
    # Sample from Priors
    beta = rng_master.uniform(0.05, 0.50)
    gamma = rng_master.uniform(0.02, 0.20)
    rho = rng_master.uniform(0.0, 0.8)
    
    # Run simulation with a unique seed for the simulator itself
    # This ensures the internal stochasticity is controlled
    sim_rng = np.random.default_rng(seed=i) 
    inf_frac, rew_counts, deg_hist = simulate(beta, gamma, rho, rng=sim_rng)
    
    # Calculate Summaries
    s_sim = np.array([
        np.max(inf_frac),
        np.sum(rew_counts),
        np.sum(np.arange(31) * deg_hist) / 200
    ])
    
    pilot_params.append([beta, gamma, rho])
    pilot_summaries.append(s_sim)
    
    if (i + 1) % 100 == 0:
        print(f"  Completed {i+1}/{N_samples} simulations")

# Convert to arrays
pilot_params = np.array(pilot_params)
pilot_summaries = np.array(pilot_summaries)

# 3. Save the Pilot Data
# This is the "set" of outputs you should keep
pilot_data = pd.DataFrame(
    np.hstack([pilot_params, pilot_summaries]),
    columns=['beta', 'gamma', 'rho', 'sim_peak', 'sim_rewire', 'sim_mean_deg']
)

# Assuming pilot_data is your DataFrame with the 1,000 runs
# Calculate standard deviations of the simulated summaries
sigma_peak = pilot_data['sim_peak'].std()
sigma_rewire = pilot_data['sim_rewire'].std()
sigma_deg = pilot_data['sim_mean_deg'].std()

print(f"Scaling Factors:\n Peak: {sigma_peak:.4f}, Rewire: {sigma_rewire:.2f}, Degree: {sigma_deg:.4f}")

sigmas = np.array([sigma_peak, sigma_rewire, sigma_deg])

# 2. Calculate distances for all rows
# We divide the difference by sigma to normalize
diffs = (pilot_data[['sim_peak', 'sim_rewire', 'sim_mean_deg']].values - s_obs) / sigmas
distances = np.sqrt(np.sum(diffs**2, axis=1))

# 3. Add distances to your dataframe
pilot_data['distance'] = distances

# 4. Accept the top 1% (e.g., 10 best samples out of 1000)
epsilon = np.percentile(distances, 1)  # 1st percentile
accepted_samples = pilot_data[pilot_data['distance'] <= epsilon]

print(f"Accepted {len(accepted_samples)} samples with tolerance epsilon = {epsilon:.4f}")

# Save to your folder
accepted_samples.to_csv('pilot_ABC_data.csv', index=False)
print("Success! Data saved to 'pilot_ABC_data.csv' in your current folder.")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set the visual style
sns.set_theme(style="white")

# 2. Create the PairPlot
# accepted_samples contains the top 1% parameters from your rejection step
g = sns.pairplot(
    accepted_samples[['beta', 'gamma', 'rho']], 
    diag_kind="hist", 
    kind="scatter",
    plot_kws={'alpha': 0.5, 's': 40, 'color': '#2c3e50'},
    diag_kws={'bins': 15, 'color': '#3498db', 'edgecolor': 'white'}
)

# 3. Add Title using the non-deprecated .figure attribute
g.figure.suptitle("Approximate Posterior Distributions (Basic ABC)", y=1.05, fontsize=14)

# 4. Refine axis labels for clarity in the report
# beta: Infection, gamma: Recovery, rho: Rewiring [cite: 430]
axes = g.axes
axes[2,0].set_xlabel(r"$\beta$ (Infection)")
axes[2,1].set_xlabel(r"$\gamma$ (Recovery)")
axes[2,2].set_xlabel(r"$\rho$ (Rewiring)")
axes[0,0].set_ylabel(r"$\beta$")
axes[1,0].set_ylabel(r"$\gamma$")
axes[2,0].set_ylabel(r"$\rho$")

plt.tight_layout()

# Save for Section 2 of your report 
g.figure.savefig("basic_rejection_posterior.png", dpi=300, bbox_inches='tight')
