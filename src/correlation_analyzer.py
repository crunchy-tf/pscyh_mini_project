import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from src import config
# Import specific table generators
from src.table_generator import save_correlation_matrix_table_image, save_single_correlation_table_image # <--- MODIFIED

# calculate_spearman_correlation_pair and plot_scatter remain the same as previous version

def calculate_spearman_correlation_pair(df: pd.DataFrame, col1: str, col2: str) -> tuple:
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Warning: One or both columns ('{col1}', '{col2}') not found for correlation.")
        return np.nan, np.nan, 0 # Added count of pairs
    
    series1 = pd.to_numeric(df[col1], errors='coerce')
    series2 = pd.to_numeric(df[col2], errors='coerce')
    
    valid_indices = series1.notna() & series2.notna()
    num_pairs = valid_indices.sum()
    if num_pairs < 3:
        print(f"Warning: Not enough non-NaN data points for correlation between '{col1}' and '{col2}'. Found {num_pairs} pairs.")
        return np.nan, np.nan, num_pairs
    rho, p_value = spearmanr(series1[valid_indices], series2[valid_indices])
    return rho, p_value, num_pairs

def plot_scatter(df: pd.DataFrame, col1: str, col2: str, rho: float, p_value: float, output_file: config.Path):
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
    sns.regplot(x=series1[valid_indices], y=series2[valid_indices], scatter=False, color='red', line_kws={'linewidth':1})
    title_str = f"Scatter: {col1} vs {col2}\nSpearman ρ = {rho:.2f}, p "
    title_str += "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    plt.title(title_str)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    try:
        plt.savefig(output_file)
        print(f"Scatter plot saved to {output_file}")
    except Exception as e: print(f"Error saving scatter plot {output_file}: {e}")
    plt.close()

# generate_correlation_matrix now calls the improved save_correlation_matrix_table_image
def generate_correlation_matrix_and_images(df: pd.DataFrame, columns_for_matrix: list,
                                 matrix_csv_output_file: config.Path,
                                 heatmap_output_file: config.Path,
                                 table_image_suffix: str):
    valid_columns = [col for col in columns_for_matrix if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(valid_columns) < 2:
        print(f"Not enough valid numeric columns for correlation matrix ('{table_image_suffix}').")
        return None
    corr_matrix = df[valid_columns].corr(method='spearman')
    print(f"\nSpearman Correlation Matrix ({table_image_suffix}):\n{corr_matrix}")
    try:
        corr_matrix.to_csv(matrix_csv_output_file)
        print(f"Correlation matrix ({table_image_suffix}) CSV saved to {matrix_csv_output_file}")
    except Exception as e: print(f"Error saving correlation matrix CSV {matrix_csv_output_file}: {e}")

    plt.figure(figsize=(max(10, len(valid_columns)*0.9), max(8, len(valid_columns)*0.8))) # Adjusted size
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, annot_kws={"size": 8})
    plt.title(f"Spearman Correlation Heatmap ({table_image_suffix.replace('_', ' ').title()})", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    try:
        plt.savefig(heatmap_output_file)
        print(f"Correlation heatmap ({table_image_suffix}) saved to {heatmap_output_file}")
    except Exception as e: print(f"Error saving heatmap {heatmap_output_file}: {e}")
    plt.close()

    save_correlation_matrix_table_image(corr_matrix, filename_suffix=table_image_suffix)
    return corr_matrix

def perform_correlation_analysis(df: pd.DataFrame,
                                 correlation_pairs_config: list, # Renamed from correlation_pairs
                                 matrix_columns: list,
                                 ders_subscale_cols: list,
                                 report_file: config.Path,
                                 viz_dir: config.Path):
    print("\n--- Performing Correlation Analysis ---")
    report_content = "Spearman Correlation Analysis Results:\n\n"
    report_content += "Pair-wise Correlations:\n"

    for col1, col2 in correlation_pairs_config:
        rho, p_value, num_pairs = calculate_spearman_correlation_pair(df, col1, col2)
        if not (np.isnan(rho) or np.isnan(p_value)):
            p_value_str = "< 0.001" if p_value < 0.001 else f"{p_value:.4f}"
            result_str = f"- {col1} vs {col2}: Spearman ρ = {rho:.3f}, p-value = {p_value_str} (N={num_pairs})"
            print(result_str)
            report_content += result_str + "\n"

            scatter_file = viz_dir / f"scatter_{col1}_vs_{col2}.png"
            plot_scatter(df, col1, col2, rho, p_value, scatter_file)

            # --- CREATE AND SAVE TABLE FOR THIS PAIR ---
            pair_data = {
                'Statistic': ['Spearman ρ', 'p-value', 'N (pairs)'],
                'Value': [f"{rho:.3f}", p_value_str, num_pairs]
            }
            pair_df = pd.DataFrame(pair_data).set_index('Statistic')
            table_title = f"Spearman Correlation: {col1} vs {col2}"
            table_filename = f"table_corr_{col1}_vs_{col2}.png"
            save_single_correlation_table_image(pair_df, table_title, table_filename)
            # --------------------------------------------
        else:
            msg = f"Could not calculate correlation for {col1} vs {col2}."
            print(msg)
            report_content += f"- {col1} vs {col2}: {msg}\n"
    report_content += "\n"

    # DERS Subscales correlation matrix
    ders_sub_cols_present = [col for col in ders_subscale_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(ders_sub_cols_present) > 1:
        report_content += "Correlations among DERS Subscales:\n"
        ders_matrix_csv_file = config.REPORTS_DIR / "correlation_matrix_ders_subscales.csv"
        ders_heatmap_file = viz_dir / "heatmap_ders_subscales.png"
        # Note: generate_correlation_matrix_and_images now also saves the table image
        ders_corr_matrix = generate_correlation_matrix_and_images(
            df, ders_sub_cols_present,
            ders_matrix_csv_file, ders_heatmap_file,
            table_image_suffix="ders_subscales"
        )
        if ders_corr_matrix is not None:
            report_content += "(See CSV and image files for DERS Subscales matrix)\n\n"
        else:
            report_content += "DERS Subscale correlation matrix could not be generated.\n\n"
    else:
        msg = "Not enough numeric DERS subscale columns for DERS subscale correlation matrix."
        print(msg)
        report_content += f"{msg}\n\n"

    # Full correlation matrix (totals + DERS subscales)
    full_matrix_csv_file = config.REPORTS_DIR / "correlation_matrix_full.csv"
    full_heatmap_file = viz_dir / "heatmap_full_matrix.png"
    # Note: generate_correlation_matrix_and_images now also saves the table image
    full_corr_matrix = generate_correlation_matrix_and_images(
        df, matrix_columns,
        full_matrix_csv_file, full_heatmap_file,
        table_image_suffix="full_totals_and_subscales" # More descriptive suffix
    )
    if full_corr_matrix is not None:
        report_content += "Full Correlation Matrix for Totals and DERS Subscales is available.\n(See CSV and image files)\n"
    else:
        report_content += "Full correlation matrix could not be generated.\n"

    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Correlation analysis report saved to {report_file}")
    except Exception as e:
        print(f"Error saving correlation report: {e}")