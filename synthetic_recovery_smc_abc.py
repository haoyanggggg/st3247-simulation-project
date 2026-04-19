"""
Synthetic-truth recovery run for SMC-ABC.

This script simulates "observed" data from known parameters and checks
whether the SMC-ABC pipeline can recover the true parameters.
"""

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
    build_simulation_context,
    seed,
)
from smc_abc import (
    SUMMARY_INDICES,
    SUMMARY_NAMES,
    N_particles,
    FINAL_RESAMPLE_SIZE,
    PRIOR_LOWER,
    PRIOR_UPPER,
    load_reference_rejection_results,
    simulate_summary_statistics,
    compute_distance,
    run_smc_abc,
    systematic_resample,
)

# True parameters for recovery
TRUE_PARAMETERS = np.array([0.25, 0.08, 0.4], dtype=np.float64)

BASE_DIR = Path(__file__).resolve().parent
RECOVERY_DIR = BASE_DIR / "outputs" / "smc_abc_recovery"
RECOVERY_PLOTS_DIR = RECOVERY_DIR / "plots"
RECOVERY_PARAM_DIR = RECOVERY_DIR / "param_estimates"

def run_recovery_smc_abc(rng: np.random.Generator,
                         simulation_context: dict,
                         reference_results: dict,
                         true_parameters: np.ndarray) -> dict:
    """
    Run SMC-ABC using synthetic data generated from true_parameters.
    """
    # 1. Generate synthetic observed summaries
    print(f"Generating synthetic observed data from true parameters: {dict(zip(PARAMETER_NAMES, true_parameters))}")
    N_RECOVERY_REPLICATES = 40
    replicate_summaries = []
    for _ in range(N_RECOVERY_REPLICATES):
        s = simulate_summary_statistics(true_parameters, rng, simulation_context)
        replicate_summaries.append(s)
    true_summaries = np.mean(replicate_summaries, axis=0)
    
    # 2. Standardize these true summaries using reference scaling
    standardized_true_observed = (true_summaries - reference_results["summary_mu"]) / reference_results["summary_sigma"]
    standardized_true_observed[reference_results["zero_sigma_mask"]] = 0.0
    
    # 3. Create a modified reference_results for the recovery run
    recovery_reference_results = reference_results.copy()
    recovery_reference_results["observed_summary_subset"] = true_summaries
    recovery_reference_results["standardized_observed"] = standardized_true_observed
    
    # We also need to recompute the distances for the stage-1 calibration if we want to be perfectly consistent,
    # but smc_abc.py uses the distances from the rejection reference to pick the initial threshold.
    # If the synthetic truth is far from the actual observed data, the initial threshold from rejection
    # might be too tight or too loose. 
    # Let's recompute the distances for the reference parameters relative to our new synthetic truth.
    
    print("Re-calibrating initial threshold for synthetic truth...")
    ref_summaries = reference_results["reference_summaries"]
    # Standardize all reference summaries against our NEW synthetic truth is not needed, 
    # we just need distances to the NEW synthetic truth.
    
    # Standardize reference summaries
    std_ref_summaries = (ref_summaries - reference_results["summary_mu"]) / reference_results["summary_sigma"]
    std_ref_summaries[:, reference_results["zero_sigma_mask"]] = 0.0
    
    # Compute distances to the synthetic truth
    new_distances = np.linalg.norm(std_ref_summaries - standardized_true_observed, axis=1)
    recovery_reference_results["distances"] = new_distances
    
    # Update finite mask just in case
    recovery_reference_results["finite_mask"] = np.isfinite(new_distances)
    
    # Re-calculate distance threshold based on the same target quantile
    target_quantile = reference_results["target_quantile"]
    recovery_reference_results["distance_threshold"] = np.quantile(new_distances[recovery_reference_results["finite_mask"]], target_quantile)

    # 4. Run SMC-ABC
    return run_smc_abc(
        rng=rng,
        simulation_context=simulation_context,
        reference_results=recovery_reference_results,
    )

def plot_recovery_results(smc_results: dict,
                          true_parameters: np.ndarray,
                          output_dir: Path) -> None:
    """Plot recovery results in a 1x3 grid with truth labeled as 'mean of accepted values'."""
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior_samples = smc_results["posterior_samples"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for param_idx, param_name in enumerate(PARAMETER_NAMES):
        axes[param_idx].hist(
            posterior_samples[:, param_idx],
            bins=20,
            density=True,
            alpha=0.6,
            color='skyblue',
            label="SMC-ABC Posterior",
        )
        axes[param_idx].axvline(
            true_parameters[param_idx],
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"True value = {true_parameters[param_idx]:.3f}",
        )
        axes[param_idx].set_xlabel(param_name)
        axes[param_idx].set_ylabel("Density")
        axes[param_idx].set_title(f"Recovery: {param_name}")
        axes[param_idx].legend()
    
    plt.tight_layout()
    plt.savefig(
        output_dir / "synthetic_truth_recovery_smc_abc.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

def main() -> None:
    rng = np.random.default_rng(seed + 1) # Different seed for recovery run
    simulation_context = build_simulation_context(N)
    
    print(f"Loading reference results from {REFERENCE_RESULTS_PATH}...")
    reference_results = load_reference_rejection_results()
    
    smc_results = run_recovery_smc_abc(
        rng=rng,
        simulation_context=simulation_context,
        reference_results=reference_results,
        true_parameters=TRUE_PARAMETERS,
    )
    
    # Save results
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    RECOVERY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RECOVERY_PARAM_DIR.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(smc_results["posterior_samples"], columns=PARAMETER_NAMES).to_csv(
        RECOVERY_PARAM_DIR / "recovery_posterior_samples.csv",
        index=False,
    )
    
    diagnostics_df = pd.DataFrame(smc_results["stage_records"])
    diagnostics_df.to_csv(RECOVERY_DIR / "recovery_diagnostics.csv", index=False)
    
    plot_recovery_results(smc_results, TRUE_PARAMETERS, RECOVERY_PLOTS_DIR)
    
    print("\nRecovery Run Complete.")
    print(f"True Parameters: {dict(zip(PARAMETER_NAMES, TRUE_PARAMETERS))}")
    print("Posterior Summaries:")
    for i, name in enumerate(PARAMETER_NAMES):
        samples = smc_results["posterior_samples"][:, i]
        print(f"  {name}: mean={np.mean(samples):.4f}, std={np.std(samples):.4f}, median={np.median(samples):.4f}")

if __name__ == "__main__":
    main()
