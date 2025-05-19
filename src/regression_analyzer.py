import pandas as pd
import numpy as np
import statsmodels.api as sm
from src import config

def perform_simple_linear_regression(df: pd.DataFrame, dependent_var: str, independent_var: str) -> str:
    """
    Performs simple linear regression and returns the model summary as a string.
    Handles NaNs by pair-wise deletion for the specific regression.
    """
    if dependent_var not in df.columns or independent_var not in df.columns:
        return f"Error: Dependent ('{dependent_var}') or Independent ('{independent_var}') variable not found in DataFrame.\n"

    # Prepare data: ensure numeric and drop NaNs for the pair
    Y_series = pd.to_numeric(df[dependent_var], errors='coerce')
    X_series = pd.to_numeric(df[independent_var], errors='coerce')

    valid_indices = Y_series.notna() & X_series.notna()
    Y = Y_series[valid_indices]
    X = X_series[valid_indices]

    if len(Y) < 10: # Arbitrary minimum for a somewhat meaningful regression
        return f"Skipping regression for {dependent_var} ~ {independent_var}: Insufficient data points ({len(Y)}) after NaN removal or non-numeric conversion.\n"

    X_const = sm.add_constant(X)  # Adds a constant term to the predictor

    try:
        model = sm.OLS(Y, X_const)
        results = model.fit()
        summary_str = f"Regression Analysis: {dependent_var} ~ {independent_var}\n"
        summary_str += results.summary().as_text()
        summary_str += "\n\n"
    except Exception as e:
        summary_str = f"Error during regression for {dependent_var} ~ {independent_var}: {e}\n\n"
    return summary_str

def perform_regression_analyses(df: pd.DataFrame, regression_pairs: list, report_file: config.Path):
    """Performs simple linear regression for specified pairs and saves summaries."""
    print("\n--- Performing Regression Analysis ---")
    all_summaries = "Simple Linear Regression Summaries:\n\n"

    for dep_var, ind_var in regression_pairs:
        print(f"Running regression: {dep_var} predicted by {ind_var}")
        summary = perform_simple_linear_regression(df, dep_var, ind_var)
        all_summaries += summary

    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(all_summaries)
        print(f"Regression summaries saved to {report_file}")
    except Exception as e:
        print(f"Error saving regression report: {e}")