"""
Observed Summary Statistics for the Adaptive-Network SIR Model

Date: 2026-04-10

Description
-----------
This module computes the observed summary statistics used by the ABC inference
scripts from the empirical epidemic and network datasets stored in `data/`.

The workflow:
1. Load the observed infection, rewiring, and degree-histogram datasets
2. Clean and reshape the raw tables when needed
3. Compute six observed summary statistics
4. Return them in the same order used by the simulation-based ABC scripts

Key Design Choices
------------------
- Data sources:
    `infected_timeseries.csv`, `rewiring_timeseries.csv`,
    `final_degree_histograms.csv`

- Summary statistics:
    * Max infection fraction
    * Time to peak
    * Early infection growth rate
    * Early rewiring growth rate
    * Degree variance
    * Late infection decay rate

- Main interface:
    `get_obs_summaries()`

Outputs
-------
- Ordered list of observed summary statistics for ABC matching

Notes
-----
- The ordering of summaries must remain consistent with `abc_rejection.py`
  and `abc_mcmc.py`.
- This module provides the observed-data counterpart to the simulated summaries
  produced during inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
REPLICATE_ID = 'replicate_id'
TIME = 'time'
INFECTED_FRAC = 'infected_fraction'
REWIRE_COUNT = 'rewire_count'
DEGREE = 'degree'
COUNT = 'count'
R = 40
SUMMARY_STATISTICS = [
    'mean_max_infection_frac',
    'mean_time_infection_frac',
    'mean_late_infection_frac_slope',
    'mean_infection_frac_slope',
    'mean_rewire_count_slope',
    'degree_var'
]
# HELPER FUNCTIONS
def get_data():
    
    infected_ts = pd.read_csv('data/infected_timeseries.csv')
    rewire_ts = pd.read_csv('data/rewiring_timeseries.csv')
    degree_count_ts = pd.read_csv('data/final_degree_histograms.csv')

    return infected_ts, rewire_ts, degree_count_ts

def pivot_df(df:pd.DataFrame, values:str, columns:str=TIME) -> pd.DataFrame:
    return df.pivot(index=REPLICATE_ID, columns=columns, values=values).reset_index()

def clean_data(infected_ts:pd.DataFrame, rewire_ts:pd.DataFrame, degree_count_ts:pd.DataFrame):
    # pivot wider to ensure each time [1 to 200] belongs to it's own row
    infected_wide_ts = pivot_df(infected_ts,INFECTED_FRAC)
    rewire_wide_ts = pivot_df(rewire_ts, REWIRE_COUNT)
    degree_count_wide_ts = pivot_df(degree_count_ts, COUNT, DEGREE)
    return infected_wide_ts, rewire_wide_ts, degree_count_wide_ts


def summary_statistic_3_early_growth_rate_exploration(
        infected_ts: pd.DataFrame,
        min_t:int=2,
        max_t:int=6) -> None:
    """
    1.  Comparing between max_t in [6,7,8] we can see that growth rate starts to decrease at t=6, so we choose max_t as 5
    2.  Despite belief that there is exponential growth at the early stages of the epidemic, the graph appears relatively 
        linear at early epidemic stage. Therefore, we will use linear slope as summary statistic during early growth
    3.  same observation for rewire counts
    4. same ovservation 
    """

    proxy = np.array(infected_ts.iloc[:, min_t:max_t])
        
    for row in proxy:
        plt.plot(np.log(row))
    
    plt.xlabel(TIME)
    plt.ylabel(INFECTED_FRAC)
    plt.show()

    proxy_avg = infected_ts.mean()
    plt.plot(np.log(proxy_avg[min_t:max_t]))
    plt.show()

def summary_statistic_4_rewire_count_exploration(
        rewire_wide_ts: pd.DataFrame,
        max_t:int=30
) -> None:
    proxy = np.array(rewire_wide_ts.iloc[max_t])
    
    for row in proxy:
        plt.plot(row)
    
    plt.xlabel(TIME)
    plt.ylabel(REWIRE_COUNT)
    plt.show()

    proxy_avg = rewire_wide_ts.mean()
    print(f'{proxy_avg.idxmax() =: }')
    plt.plot(proxy_avg[1:max_t])
    plt.show()

def summary_statistic_5_degree_count_exploration(
        degree_count_wide_ts: pd.DataFrame,
        max_t:int=5
) -> None:
    proxy = np.array(degree_count_wide_ts.iloc[:, 1:30])
    count = 0
    for row in proxy:
        if count < 15:
            count += 1
            print(count)
            continue
        plt.plot(row)
        count += 1
       
    plt.xlabel(DEGREE)
    plt.ylabel(COUNT)
    plt.show()

    proxy_avg = degree_count_wide_ts.mean()
    print(f'{proxy_avg.idxmax() =: }')
    plt.plot(proxy_avg[1:30])
    plt.show()

# SUMMARY STATISTIC 1: Max infection fraction INFORMS BETA/GAMMA
def get_summary_statistic_1(infected_ts:pd.DataFrame) -> float:
    max_infection_frac_per_replicate_id = infected_ts.groupby('replicate_id')[INFECTED_FRAC].max()
    mean_max_infection_frac = max_infection_frac_per_replicate_id.mean() 
    return mean_max_infection_frac

# SUMMARY STATISTIC 2: Time to peak INFORMS BETA/GAMMA
def get_summary_statistic_2(infected_ts:pd.DataFrame) -> float:
    
    time_infection_frac_per_replicate_id = infected_ts.set_index(TIME).groupby('replicate_id')[INFECTED_FRAC].idxmax()
    mean_time_infection_frac = time_infection_frac_per_replicate_id.mean()
    return mean_time_infection_frac

# SUMMARY STATISTIC 3: Early growth rate INFORMS BETA/GAMMA
# this window gives linear log infected_fraction
def get_summary_statistic_3(infected_ts:pd.DataFrame) -> float:
    summary_stat_3_condition = (
        (infected_ts[TIME] >= 2) &
        (infected_ts[TIME] <= 6)
    )
    early_infection_infection_df = infected_ts[summary_stat_3_condition]
    infection_frac_slopes = (
        early_infection_infection_df
            .groupby(REPLICATE_ID)
            .apply(lambda g: np.polyfit(g[TIME], np.log(g[INFECTED_FRAC]), 1)[0])
    )
    mean_infection_frac_slope = np.mean(infection_frac_slopes)
    return mean_infection_frac_slope


# SUMMARY STATISTIC 4: Mean rewire counts during early infection
def get_summary_statistic_4(rewire_ts:pd.DataFrame) -> float:
    summary_stat_4_condition = (
        (rewire_ts[TIME] >= 2) &
        (rewire_ts[TIME] <= 6)
    )
    early_infection_rewire_df = rewire_ts[summary_stat_4_condition]
    rewire_slopes = (
        early_infection_rewire_df.groupby(REPLICATE_ID)
            .apply(lambda g: np.polyfit(g[TIME], np.log(g[REWIRE_COUNT]), 1)[0])
    )
    mean_rewire_count_slope = np.mean(rewire_slopes)
    return mean_rewire_count_slope

# SUMMARY STATISTIC 5: Variance structure of degree counts INFORMS RHO
#                      ↑rho = distortion from Erdos-Renyi random graph's binomial distribution
#                      ↓rho = closer to binomial distribution
def get_summary_statistic_5(df):
    """"""
    def per_rep(g):
        k = g["degree"].values
        c = g["count"].values
        
        N = c.sum()
        
        mean = np.sum(k * c) / N
        mean_sq = np.sum((k**2) * c) / N
        
        var = mean_sq - mean**2
        
        return var
    
    return (
        df.groupby("replicate_id")
            .apply(per_rep)
            .reset_index(name="degree_var")
            ['degree_var'].mean()
    )

    # degree_var = get_summary_statistic_5(degree_count_ts)
    # print(f'{degree_var =: }')

# SUMMARY STATISTIC 6: Decay structure of infection INFORMS RHO 
def get_summary_statistic_6(infected_ts:pd.DataFrame) -> float:
    summary_stat_6_condition = (
        (infected_ts[TIME] >= 13) &
        (infected_ts[TIME] <= 20)
    )
    late_infection_infection_df = infected_ts[summary_stat_6_condition]
    late_infection_frac_slopes = (
        late_infection_infection_df.groupby(REPLICATE_ID)
            .apply(lambda g: np.polyfit(g[TIME], np.log(g[INFECTED_FRAC]), 1)[0])
    )
    mean_late_infection_frac_slope = np.mean(late_infection_frac_slopes)
    return mean_late_infection_frac_slope

# EXPLORATION SECION (OPTIONAL)

# summary_statistic_3_early_growth_rate_exploration(infected_wide_ts)
# summary_statistic_3_early_growth_rate_exploration(infected_wide_ts, 13, 20)
# summary_statistic_3_early_growth_rate_exploration(rewire_wide_ts)


# main
def get_obs_summaries(to_print:bool=False) -> list:
    infected_ts, rewire_ts, degree_count_ts = get_data()
    infected_wide_ts, rewire_wide_ts, degree_count_wide_ts = clean_data(infected_ts, rewire_ts, degree_count_ts)

    summary_statistics_lst = [
        get_summary_statistic_1(infected_ts),
        get_summary_statistic_2(infected_ts),
        get_summary_statistic_3(infected_ts),
        get_summary_statistic_4(rewire_ts),
        get_summary_statistic_5(degree_count_ts),
        get_summary_statistic_6(infected_ts)
    ]
    max_length = max(len(name) for name in SUMMARY_STATISTICS)
    if to_print:        
        print(f"\nObserved Summary Statistics:")
        print("-" * 30)
        # ensure that printed '=' is aligned for better readability
        for stat_name, stat_value in zip(SUMMARY_STATISTICS, summary_statistics_lst):    
            print(f'{stat_name:<{max_length}} =  {stat_value}')
        print('\n')
    return summary_statistics_lst

# get_obs_summaries()
    

