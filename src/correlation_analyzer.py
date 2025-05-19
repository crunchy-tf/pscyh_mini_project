import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from src import config

def calculate_spearman_correlation_pair(df: pd.DataFrame, col1: str, col2: str) -> tuple:
    """Calculates Spearman's rho and p-value for a pair of columns."""
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Warning: One or both columns ('{col1}', '{col2}') not found for correlation.")
        return np.nan, np.nan
    
    series1 = pd.to_numeric(df[col1], errors='coerce')
    series2 = pd.to_numeric(df[col2], errors='coerce')
    
    # Drop rows with NaNs in either of the two columns for this specific pair-wise correlation
    valid_indices = series1.notna() & series2.notna()
    if valid_indices.sum() < 3: # Spearman correlation needs at least 3 non-NaN pairs
        print(f"Warning: Not enough non-NaN data points for correlation between '{col1}' and '{col2}'. Found {valid_indices.sum()} pairs.")
        return np.nan, np.nan

    rho, p_value = spearmanr(series1[valid_indices], series2[valid_indices])
    return rho, p_value

def plot_scatter(df: pd.DataFrame, col1: str, col2: str, rho: float, p_value: float, output_file: config.Path):
    """Generates and saves a scatter plot for two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Skipping scatter plot: One or both columns ('{col1}', '{col2}') not found.")
        return
    
    series1 = pd.to_numeric(df[col1], errors='coerce')
    series2 = pd.to_numeric(df[col2], errors='coerce')
    valid_indices = series1.notna() & series2.notna()

    if valid_indices.sum() < 2:
        print(f"Skipping scatter plot for '{col1}' vs '{col2}': Not enough data points ({valid_indices.sum()}).")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=series1[valid_indices], y=series2[valid_indices], alpha=0.7)
    # Add a regression line for visual trend
    sns.regplot(x=series1[valid_indices], y=series2[valid_indices], scatter=False, color='red', line_kws={'linewidth':1})
    
    title = f"Scatter Plot: {col1} vs {col2}\nSpearman ρ = {rho:.2f}, p = {p_value:.3f}"
    if p_value < 0.001: title = f"Scatter Plot: {col1} vs {col2}\nSpearman ρ = {rho:.2f}, p < 0.001"

    plt.title(title)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    try:
        plt.savefig(output_file)
        print(f"Scatter plot saved to {output_file}")
    except Exception as e:
        print(f"Error saving scatter plot {output_file}: {e}")
    plt.close()


def generate_correlation_matrix(df: pd.DataFrame, columns_for_matrix: list, matrix_output_file: config.Path, heatmap_output_file: config.Path):
    """Generates Spearman correlation matrix, saves it, and plots a heatmap."""
    valid_columns = [col for col in columns_for_matrix if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(valid_columns) < 2:
        print("Not enough valid numeric columns to generate a correlation matrix.")
        return None

    corr_matrix = df[valid_columns].corr(method='spearman')

    print("\nSpearman Correlation Matrix:")
    print(corr_matrix)
    try:
        corr_matrix.to_csv(matrix_output_file)
        print(f"Correlation matrix saved to {matrix_output_file}")
    except Exception as e:
        print(f"Error saving correlation matrix {matrix_output_file}: {e}")


    plt.figure(figsize=(max(10, len(valid_columns)*0.8), max(8, len(valid_columns)*0.7))) # Adjust size
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title("Spearman Correlation Heatmap")
    plt.tight_layout()
    try:
        plt.savefig(heatmap_output_file)
        print(f"Correlation heatmap saved to {heatmap_output_file}")
    except Exception as e:
        print(f"Error saving heatmap {heatmap_output_file}: {e}")
    plt.close()

    return corr_matrix

def perform_correlation_analysis(df: pd.DataFrame,
                                 correlation_pairs: list,
                                 matrix_columns: list,
                                 ders_subscale_cols: list, # Pass explicitly the DERS subscale clean names
                                 report_file: config.Path,
                                 viz_dir: config.Path):
    """Performs correlation analysis for specified pairs and matrix columns."""
    print("\n--- Performing Correlation Analysis ---")
    report_content = "Spearman Correlation Analysis Results:\n\n"

    # Specific pairs
    report_content += "Pair-wise Correlations:\n"
    for col1, col2 in correlation_pairs:
        rho, p_value = calculate_spearman_correlation_pair(df, col1, col2)
        if not (np.isnan(rho) or np.isnan(p_value)):
            result_str = f"- {col1} vs {col2}: Spearman ρ = {rho:.3f}, p-value = {p_value:.4f}"
            if p_value < 0.001: result_str = f"- {col1} vs {col2}: Spearman ρ = {rho:.3f}, p-value < 0.001"
            print(result_str)
            report_content += result_str + "\n"
            scatter_file = viz_dir / f"scatter_{col1}_vs_{col2}.png"
            plot_scatter(df, col1, col2, rho, p_value, scatter_file)
        else:
            msg = f"Could not calculate correlation for {col1} vs {col2} (insufficient data or columns not numeric/found)."
            print(msg)
            report_content += f"- {col1} vs {col2}: {msg}\n"
    report_content += "\n"

    # DERS Subscales correlations
    ders_sub_cols_present = [col for col in ders_subscale_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(ders_sub_cols_present) > 1:
        report_content += "Correlations among DERS Subscales:\n"
        ders_corr_matrix = df[ders_sub_cols_present].corr(method='spearman')
        print("\nDERS Subscales Correlation Matrix:")
        print(ders_corr_matrix)
        report_content += ders_corr_matrix.to_string() + "\n\n"
        ders_heatmap_file = viz_dir / "heatmap_ders_subscales.png"
        plt.figure(figsize=(max(8, len(ders_sub_cols_present)*0.8), max(6, len(ders_sub_cols_present)*0.7)))
        sns.heatmap(ders_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
        plt.title("Spearman Correlation Heatmap of DERS Subscales")
        plt.tight_layout()
        try:
            plt.savefig(ders_heatmap_file)
            print(f"DERS subscales heatmap saved to {ders_heatmap_file}")
        except Exception as e:
            print(f"Error saving DERS heatmap {ders_heatmap_file}: {e}")
        plt.close()

    else:
        msg = "Not enough numeric DERS subscale columns present for DERS subscale correlation matrix."
        print(msg)
        report_content += f"{msg}\n\n"

    # Full matrix for specified columns
    matrix_csv_file = config.REPORTS_DIR / "correlation_matrix_full.csv"
    heatmap_file = viz_dir / "heatmap_full_matrix.png"
    full_corr_matrix = generate_correlation_matrix(df, matrix_columns, matrix_csv_file, heatmap_file)
    if full_corr_matrix is not None:
        report_content += "Full Correlation Matrix (Spearman) for selected Totals and DERS Subscales is available.\n(See 'correlation_matrix_full.csv' and 'heatmap_full_matrix.png')\n"
    else:
        report_content += "Full correlation matrix could not be generated.\n"

    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Correlation analysis report saved to {report_file}")
    except Exception as e:
        print(f"Error saving correlation report: {e}")