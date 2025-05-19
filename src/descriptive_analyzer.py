import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from src import config # mode from scipy.stats can be problematic if multiple modes or non-numeric

def calculate_descriptive_stats(series: pd.Series, series_name: str) -> pd.Series:
    """Calculates a comprehensive set of descriptive statistics for a series."""
    if not pd.api.types.is_numeric_dtype(series):
        print(f"Warning: Series '{series_name}' is not numeric (Dtype: {series.dtype}). Cannot calculate all descriptive stats.")
        # Return what we can for non-numeric or partially numeric
        stats = {
            "Count": series.count(),
            "Missing": series.isnull().sum(),
            "Unique": series.nunique(),
            "Mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        }
        return pd.Series(stats, name=series_name)

    # For numeric series
    mode_val = series.mode()
    mode_display = mode_val.iloc[0] if not mode_val.empty else np.nan
    if len(mode_val) > 1:
        mode_display = f"{mode_val.iloc[0]} (and others)"


    stats = {
        "Mean": series.mean(),
        "Median": series.median(),
        "Mode": mode_display,
        "StdDev": series.std(),
        "Variance": series.var(),
        "Min": series.min(),
        "Max": series.max(),
        "Range": series.max() - series.min() if series.notna().any() else np.nan,
        "IQR": series.quantile(0.75) - series.quantile(0.25),
        "Skewness": skew(series.dropna()),
        "Kurtosis (Fisher)": kurtosis(series.dropna(), fisher=True), # Fisher's (normal ==> 0.0)
        "Pctl_1st": series.quantile(0.01),
        "Pctl_25th (Q1)": series.quantile(0.25),
        "Pctl_50th (Median)": series.quantile(0.50),
        "Pctl_75th (Q3)": series.quantile(0.75),
        "Pctl_99th": series.quantile(0.99),
        "Count (Non-NA)": series.count(),
        "Missing": series.isnull().sum()
    }
    return pd.Series(stats, name=series_name)

def plot_histogram(series: pd.Series, series_name: str, output_file: config.Path):
    """Generates and saves a histogram for the series."""
    if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
        print(f"Skipping histogram for non-numeric or empty series: {series_name}")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(series.dropna(), kde=True, bins='auto')
    plt.title(f"Histogram of {series_name}")
    plt.xlabel(series_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    try:
        plt.savefig(output_file)
        print(f"Histogram saved to {output_file}")
    except Exception as e:
        print(f"Error saving histogram {output_file}: {e}")
    plt.close()


def plot_boxplot(series: pd.Series, series_name: str, output_file: config.Path):
    """Generates and saves a boxplot for the series."""
    if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
        print(f"Skipping boxplot for non-numeric or empty series: {series_name}")
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=series.dropna())
    plt.title(f"Boxplot of {series_name}")
    plt.ylabel(series_name)
    plt.tight_layout()
    try:
        plt.savefig(output_file)
        print(f"Boxplot saved to {output_file}")
    except Exception as e:
        print(f"Error saving boxplot {output_file}: {e}")
    plt.close()


def perform_descriptive_analysis(df: pd.DataFrame, columns_to_analyze: list, report_file: config.Path, viz_dir: config.Path):
    """Performs descriptive analysis for specified columns and saves results."""
    print("\n--- Performing Descriptive Analysis ---")
    all_stats_list = []

    for col_name in columns_to_analyze:
        if col_name in df.columns:
            series = df[col_name]
            stats_series = calculate_descriptive_stats(series, col_name)
            all_stats_list.append(stats_series)

            # Visualizations
            hist_file = viz_dir / f"histogram_{col_name}.png"
            plot_histogram(series, col_name, hist_file)

            box_file = viz_dir / f"boxplot_{col_name}.png"
            plot_boxplot(series, col_name, box_file)
        else:
            print(f"Warning: Column '{col_name}' not found in DataFrame for descriptive analysis.")

    if all_stats_list:
        all_stats_df = pd.concat(all_stats_list, axis=1)
        print("\nDescriptive Statistics Summary:")
        print(all_stats_df)
        try:
            # Save as CSV for easier machine readability
            all_stats_df.to_csv(report_file.with_suffix(".csv"))
            # Save as TXT for human readability
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("Descriptive Statistics Summary:\n\n")
                f.write(all_stats_df.to_string())
            print(f"Descriptive statistics report saved to {report_file} and {report_file.with_suffix('.csv')}")
        except Exception as e:
            print(f"Error saving descriptive stats report: {e}")
    else:
        print("No descriptive statistics were generated.")