"""
Pairplot for Rich Set vs Reduced Set E

Date: 2026-04-11

Description
-----------
This script loads the latest `ε = 0.01` posterior samples from the rich summary
set and from Reduced set E, then visualizes their posterior structure using
separate seaborn pairplots.

The workflow:
1. Locate the latest rich-set posterior CSV at `ε = 0.01`
2. Locate the latest Reduced set E posterior CSV at `ε = 0.01`
3. Load both posterior sample tables
4. Produce one pairplot for the rich set
5. Produce one pairplot for Reduced set E

Key Design Choices
------------------
- Comparison target:
    Rich summary set versus Reduced set E

- Epsilon selection:
    Fixed at `ε = 0.01`

- Plotting:
    Separate pairplots with distinct colors for each summary set

Outputs
-------
- One pairplot for the rich-set posterior samples
- One pairplot for the Reduced set E posterior samples

Notes
-----
- The script always selects the latest matching timestamped CSV for each source.
- It assumes `abc_rejection.py` has already been run and the expected output
  files are present.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abc_rejection import PARAM_ESTIMATES_DIR, SUMMARY_SET_STUDY_DIR


RICH_SET_PARAM_DIR = PARAM_ESTIMATES_DIR
REDUCED_SET_E_PARAM_DIR = SUMMARY_SET_STUDY_DIR / "reduced_set_e" / "param_estimates"
TARGET_EPSILON_SLUG = "0.0100"


def get_latest_matching_csv(param_estimates_dir: Path, pattern: str, label: str) -> Path:
    csv_candidates = sorted(param_estimates_dir.glob(pattern))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No {label} posterior CSVs found in {param_estimates_dir}. "
            "Run abc_rejection.py first."
        )
    return csv_candidates[-1]


def plot_posterior_pairplot(df: pd.DataFrame, title: str, color: str) -> None:
    grid = sns.pairplot(
        df,
        diag_kind="hist",
        corner=True,
        plot_kws={"color": color, "alpha": 0.75},
        diag_kws={"color": color, "alpha": 0.75},
    )
    grid.fig.suptitle(title, y=1.02)


def main() -> None:
    rich_set_csv_path = get_latest_matching_csv(
        RICH_SET_PARAM_DIR,
        f"abc-basic_eps-{TARGET_EPSILON_SLUG}_*.csv",
        "rich set",
    )
    reduced_set_e_csv_path = get_latest_matching_csv(
        REDUCED_SET_E_PARAM_DIR,
        f"reduced_set_e_abc-basic_eps-{TARGET_EPSILON_SLUG}_*.csv",
        "Reduced set E",
    )

    rich_set_df = pd.read_csv(rich_set_csv_path)
    reduced_set_e_df = pd.read_csv(reduced_set_e_csv_path)

    print(f"Loaded rich-set posterior samples from: {rich_set_csv_path}")
    print(f"Loaded Reduced set E posterior samples from: {reduced_set_e_csv_path}")
    plot_posterior_pairplot(rich_set_df, "Rich set posterior samples", "#1f77b4")
    plot_posterior_pairplot(reduced_set_e_df, "Reduced set E posterior samples", "#d62728")
    plt.show()


if __name__ == "__main__":
    main()
