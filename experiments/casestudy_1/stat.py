#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Sample data: a list of dictionaries with evaluation scores.
data = [
  {"grammar": 2, "consistency": 1, "coherence": 1, "deviates_baseline": True},
  {"grammar": 2, "consistency": 1, "coherence": 1, "deviates_baseline": True},
  {"grammar": 1, "consistency": 0, "coherence": 0, "deviates_baseline": True},
  {"grammar": 7, "consistency": 8, "coherence": 6, "deviates_baseline": False},
  {"grammar": 2, "consistency": 1, "coherence": 1, "deviates_baseline": True},
  {"grammar": 2, "consistency": 0, "coherence": 0, "deviates_baseline": True},
  {"grammar": 6, "consistency": 8, "coherence": 7, "deviates_baseline": False},
  {"grammar": 1, "consistency": 0, "coherence": 0, "deviates_baseline": True},
  {"grammar": 6, "consistency": 7, "coherence": 6, "deviates_baseline": False},
  {"grammar": 3, "consistency": 2, "coherence": 2, "deviates_baseline": True}
]

def perform_welch_tests(data):
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)
    
    # Split data into two groups based on deviates_baseline flag.
    group_deviate = df[df["deviates_baseline"] == True]
    group_non_deviate = df[df["deviates_baseline"] == False]
    
    metrics = ["grammar", "consistency", "coherence"]
    
    print("Group Means:")
    for metric in metrics:
        mean_deviate = group_deviate[metric].mean()
        mean_non_deviate = group_non_deviate[metric].mean()
        print(f"  {metric.capitalize()}: Deviates = {mean_deviate}, Non-deviates = {mean_non_deviate}")
    
    print("\nWelch's t-test results (assuming unequal variances):")
    for metric in metrics:
        # Use ttest_ind with equal_var set to False (Welch's t-test)
        t_stat, p_val = ttest_ind(group_deviate[metric], group_non_deviate[metric], equal_var=False)
        print(f"  {metric.capitalize()}: t-statistic = {t_stat}, p-value = {p_val}")

def main():
    perform_welch_tests(data)

if __name__ == "__main__":
    main()
