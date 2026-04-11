# SBI Infection

Approximate Bayesian Computation workflow for an adaptive-network SIR epidemic model.  
The project estimates the posterior of `(beta, gamma, rho)` using:

- basic ABC rejection
- regression-adjusted ABC
- ABC-MCMC calibrated from the saved rejection run

## Project Layout

```text
SBI_infection/
├─ data/
│  ├─ infected_timeseries.csv
│  ├─ rewiring_timeseries.csv
│  ├─ final_degree_histograms.csv
│  └─ intermediate/
│     └─ abc_rejection_output.npz
├─ outputs/
│  ├─ basic_abc/
│  │  ├─ param_estimates/
│  │  ├─ sanity_check/
│  │  └─ summary_set_study/
│  ├─ abc_mcmc/
│  └─ regression_adjustment/
├─ simulator.py
├─ observed_summaries.py
├─ abc_rejection.py
├─ abc_rejection_regression.py
├─ abc_mcmc.py
└─ approximate_posterior_exploration.py
```

## Requirements

Run the scripts from the project root (`SBI_infection/`). Several files load data using relative paths.

Required Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`
- `scikit-learn`
- `seaborn`

## Main Scripts

`simulator.py`  
Simulates one adaptive-network SIR epidemic. Optimized from original simulator.py script obtained from 
"https://github.com/alexxthiery/SBI_infection"

`observed_summaries.py`  
Computes the observed summary statistics from the provided data. Function is used in `abc_rejection.py`

`abc_rejection.py`  
Runs the baseline basic ABC rejection pipeline. It:

- simulates `N_sim = 30,000` prior draws
- computes six summary statistics
- standardizes summaries using simulation-based scaling
- accepts posterior samples using euclidean distance

Additionally
- performs posterior sample distribution comparison for multiple `ε` values
- performs the summary subset study to obtain minimal summary statistics
- writes the saved reference calibration used later by `abc_rejection_regression.py` and `abc_mcmc.py`

`abc_rejection_regression.py`  
Applies Beaumont-style local linear regression adjustment to the accepted rejection samples.

`abc_mcmc.py`  
Runs ABC-MCMC using chosen `Reduced set E` summary statistics and loads the calibration file produced by `abc_rejection.py`.  
The current setup uses:

- `N_proposals = 30,000`
- `N_mcmc = 30,001` stored chain states
- `burn_in = 3,000`

`approximate_posterior_exploration.py`  
Loads the latest `ε = 0.01` posterior CSVs for the rich summary set and Reduced
set E, then produces separate color-coded seaborn pairplots for visual comparison.

## Summary Statistics

The full model uses six summaries:

1. Max infection fraction
2. Time to peak
3. Early growth rate of infection
4. Early growth rate of rewiring
5. Variance structure of degree counts
6. Late decay rate of infection

The summary-set comparison study in `abc_rejection.py` compares the full rich set with reduced sets A-F.  
Subsequent algorithms uses `Reduced set E`.

## Recommended Run Order

1. Run basic rejection ABC:

```powershell
py abc_rejection.py
```

This produces:

- `outputs/basic_abc/param_estimates/`
- `outputs/basic_abc/sanity_check/`
- `outputs/basic_abc/summary_set_study/`
- `data/intermediate/abc_rejection_output.npz`

2. Run regression adjustment:

```powershell
py abc_rejection_regression.py
```

This reads `data/intermediate/abc_rejection_output.npz` and writes to:

- `outputs/regression_adjustment/plots/`
- `outputs/regression_adjustment/param_estimates/`

3. Run ABC-MCMC:

```powershell
py abc_mcmc.py
```

This requires `data/intermediate/abc_rejection_output.npz` from step 1 and writes to:

- `outputs/abc_mcmc/plots/`
- `outputs/abc_mcmc/param_estimates/`
- `outputs/abc_mcmc/abc_mcmc_chain.npz`
- `outputs/abc_mcmc/abc_mcmc_diagnostics.csv`

NOTE: longer runtime due to dependent markov chain updates, unable to parallelize

4. Optional: explore the latest `ε = 0.01` rich-set and Reduced set E posteriors visually:

```powershell
py approximate_posterior_exploration.py
```

## Additional Notes

- The rejection and MCMC comparison is designed to be fair: `abc_mcmc.py` loads the same `Reduced set E` scaling and `ε` threshold saved by `abc_rejection.py`.
- If you change the rejection setup in `abc_rejection.py`, rerun it before running extension algorithms like `abc_rejection_regression.py` or `abc_mcmc.py`.
- `simulator.py` is the base simulator and should remain stable across inference scripts.
