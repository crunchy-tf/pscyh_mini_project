import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from src import config
# Import the specific table generators
from src.table_generator import save_correlation_matrix_table_image, save_single_correlation_table_image
# We can actually reuse save_correlation_matrix_table_image for the 2x2 matrices
# by passing the 2x2 df and a specific suffix.
# Let's rename save_single_correlation_table_image to be more generic or remove if not needed.
# For clarity, let's ensure save_correlation_matrix_table_image handles small matrices well.

def calculate_spearman_correlation_pair_values(df: pd.DataFrame, col1: str, col2: str) -> tuple: # Renamed for clarity
    """Calculates Spearman rho, p-value, and N for a single pair of columns."""
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Warning: Columns for correlation ('{col1}', '{col2}') not found."); return np.nan, np.nan, 0
    s1 = pd.to_numeric(df[col1], errors='coerce'); s2 = pd.to_numeric(df[col2], errors='coerce')
    valid = s1.notna() & s2.notna(); num_pairs = valid.sum()
    if num_pairs < 3:
        print(f"Warning: Not enough data for correlation '{col1}' vs '{col2}' ({num_pairs} pairs)."); return np.nan, np.nan, num_pairs
    rho, p = spearmanr(s1[valid], s2[valid]); return rho, p, num_pairs

def plot_scatter(df: pd.DataFrame, col1: str, col2: str, rho: float, p_value: float, output_file: config.Path):
    # ... (This function remains unchanged from the previous version)
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Skipping scatter: Columns ('{col1}', '{col2}') not found."); return
    s1 = pd.to_numeric(df[col1], errors='coerce'); s2 = pd.to_numeric(df[col2], errors='coerce')
    valid = s1.notna() & s2.notna()
    if valid.sum() < 2:
        print(f"Skipping scatter for '{col1}' vs '{col2}': Not enough data ({valid.sum()})."); return
    plt.figure(figsize=(10,6)); sns.scatterplot(x=s1[valid], y=s2[valid], alpha=0.7)
    sns.regplot(x=s1[valid], y=s2[valid], scatter=False, color='red', line_kws={'linewidth':1})
    title = f"Scatter: {col1} vs {col2}\nSpearman ρ = {rho:.2f}, p {'< 0.001' if p_value < 0.001 else f'= {p_value:.3f}'}"
    plt.title(title); plt.xlabel(col1); plt.ylabel(col2); plt.tight_layout()
    try: plt.savefig(output_file); print(f"Scatter plot saved: {output_file}")
    except Exception as e: print(f"Error saving scatter {output_file}: {e}")
    plt.close()


def generate_correlation_matrix_and_images(df: pd.DataFrame, cols: list, csv_path: config.Path, heatmap_path: config.Path, suffix: str):
    # ... (This function remains unchanged from the previous version,
    # it already calls save_correlation_matrix_table_image from table_generator.py)
    valid_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2: print(f"Not enough numeric columns for corr matrix ('{suffix}')."); return None
    matrix = df[valid_cols].corr(method='spearman')
    print(f"\nSpearman Correlation Matrix ({suffix}):\n{matrix}")
    try: matrix.to_csv(csv_path); print(f"Corr matrix CSV ({suffix}) saved: {csv_path}")
    except Exception as e: print(f"Error saving corr matrix CSV {csv_path}: {e}")
    plt.figure(figsize=(max(10,len(valid_cols)*0.9), max(8,len(valid_cols)*0.8)))
    sns.heatmap(matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=.5,vmin=-1,vmax=1,annot_kws={"size":8})
    plt.title(f"Spearman Heatmap ({suffix.replace('_',' ').title()})",fontsize=14)
    plt.xticks(rotation=45,ha="right",fontsize=9); plt.yticks(fontsize=9); plt.tight_layout()
    try: plt.savefig(heatmap_path); print(f"Corr heatmap ({suffix}) saved: {heatmap_path}")
    except Exception as e: print(f"Error saving heatmap {heatmap_path}: {e}")
    plt.close()
    save_correlation_matrix_table_image(matrix, filename_suffix=suffix)
    return matrix


def perform_correlation_analysis(df: pd.DataFrame,
                                 correlation_pairs_config: list,
                                 matrix_columns: list, # For the full matrix
                                 ders_subscale_cols: list,
                                 report_file: config.Path,
                                 viz_dir: config.Path):
    print("\n--- Performing Correlation Analysis ---")
    report_content = "Spearman Correlation Analysis Results:\n\n"
    report_content += "Pair-wise Correlations (Presented as 2x2 Matrices and Scatter Plots):\n"

    for col1, col2 in correlation_pairs_config:
        # Calculate rho and p-value for reporting and scatter plot title
        rho, p_value, num_pairs = calculate_spearman_correlation_pair_values(df, col1, col2)

        if not (np.isnan(rho) or np.isnan(p_value)):
            p_value_str = "< 0.001" if p_value < 0.001 else f"{p_value:.4f}"
            result_str = f"- {col1} vs {col2}: Spearman ρ = {rho:.3f}, p-value = {p_value_str} (N={num_pairs})"
            print(result_str)
            report_content += result_str + "\n"

            # Generate scatter plot
            scatter_file = viz_dir / f"scatter_{col1}_vs_{col2}.png"
            plot_scatter(df, col1, col2, rho, p_value, scatter_file)

            # --- CREATE AND SAVE 2x2 CORRELATION MATRIX TABLE FOR THIS PAIR ---
            # Ensure columns exist and are numeric for the sub-matrix calculation
            cols_for_pair_matrix = []
            if col1 in df.columns and pd.api.types.is_numeric_dtype(df[col1]):
                cols_for_pair_matrix.append(col1)
            if col2 in df.columns and pd.api.types.is_numeric_dtype(df[col2]):
                cols_for_pair_matrix.append(col2)
            
            # Ensure no duplicate columns if col1 can be same as col2 (not in this config)
            cols_for_pair_matrix = sorted(list(set(cols_for_pair_matrix)))


            if len(cols_for_pair_matrix) == 2: # We need exactly two distinct columns
                pair_corr_matrix = df[cols_for_pair_matrix].corr(method='spearman')
                
                # Sanitize col names for filename (simple replacement)
                c1_fn = col1.replace(':', '_').replace(' ', '_').replace('(','').replace(')','')
                c2_fn = col2.replace(':', '_').replace(' ', '_').replace('(','').replace(')','')

                table_title = f"Spearman Correlation: {col1} & {col2}"
                # Use a more descriptive suffix for the table image function
                table_suffix = f"pair_{c1_fn}_vs_{c2_fn}"
                # The save_correlation_matrix_table_image function is suitable here
                save_correlation_matrix_table_image(pair_corr_matrix, filename_suffix=table_suffix)
            else:
                print(f"  Skipping 2x2 correlation table for '{col1}' vs '{col2}' due to missing or non-numeric columns.")
            # --------------------------------------------------------------------
        else:
            msg = f"Could not calculate correlation values for {col1} vs {col2}."
            print(msg)
            report_content += f"- {col1} vs {col2}: {msg}\n"
    report_content += "\n"

    # DERS Subscales correlation matrix (uses generate_correlation_matrix_and_images)
    ders_sub_cols_present = [col for col in ders_subscale_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(ders_sub_cols_present) > 1:
        report_content += "Correlations among DERS Subscales:\n"
        ders_matrix_csv_file = config.REPORTS_DIR / "correlation_matrix_ders_subscales.csv"
        ders_heatmap_file = viz_dir / "heatmap_ders_subscales.png"
        ders_corr_matrix = generate_correlation_matrix_and_images(
            df, ders_sub_cols_present,
            ders_matrix_csv_file, ders_heatmap_file,
            suffix="ders_subscales" # Suffix for generate_correlation_matrix_and_images
        )
        if ders_corr_matrix is not None:
            report_content += "(See CSV and image files for DERS Subscales matrix)\n\n"
        else:
            report_content += "DERS Subscale correlation matrix could not be generated.\n\n"
    else:
        msg = "Not enough numeric DERS subscale columns present for DERS subscale correlation matrix."
        print(msg)
        report_content += f"{msg}\n\n"

    # Full correlation matrix (totals + DERS subscales)
    full_matrix_csv_file = config.REPORTS_DIR / "correlation_matrix_full.csv"
    full_heatmap_file = viz_dir / "heatmap_full_matrix.png"
    full_corr_matrix = generate_correlation_matrix_and_images(
        df, matrix_columns, # These are config.COLS_FOR_CORR_MATRIX
        full_matrix_csv_file, full_heatmap_file,
        suffix="full_totals_and_subscales"
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