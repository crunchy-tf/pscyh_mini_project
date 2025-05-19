import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from src import config
from src.table_generator import save_correlation_matrix_table_image, save_single_correlation_table_image

def calculate_spearman_correlation_pair(df: pd.DataFrame, col1: str, col2: str) -> tuple:
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Warning: Columns for correlation ('{col1}', '{col2}') not found."); return np.nan, np.nan, 0
    s1 = pd.to_numeric(df[col1], errors='coerce'); s2 = pd.to_numeric(df[col2], errors='coerce')
    valid = s1.notna() & s2.notna(); num_pairs = valid.sum()
    if num_pairs < 3:
        print(f"Warning: Not enough data for correlation '{col1}' vs '{col2}' ({num_pairs} pairs)."); return np.nan, np.nan, num_pairs
    rho, p = spearmanr(s1[valid], s2[valid]); return rho, p, num_pairs

def plot_scatter(df: pd.DataFrame, col1: str, col2: str, rho: float, p_value: float, output_file: config.Path):
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
    save_correlation_matrix_table_image(matrix, filename_suffix=suffix) # Call to updated table generator
    return matrix

def perform_correlation_analysis(df: pd.DataFrame, pairs_cfg: list, matrix_cols: list, ders_cols: list, report_path: config.Path, viz_dir: config.Path):
    print("\n--- Performing Correlation Analysis ---")
    report = "Spearman Correlation Analysis Results:\n\nPair-wise Correlations:\n"
    for c1, c2 in pairs_cfg:
        rho, p, n = calculate_spearman_correlation_pair(df, c1, c2)
        if not (np.isnan(rho) or np.isnan(p)):
            p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
            res_str = f"- {c1} vs {c2}: Spearman ρ = {rho:.3f}, p-value = {p_str} (N={n})"
            print(res_str); report += res_str + "\n"
            plot_scatter(df,c1,c2,rho,p, viz_dir / f"scatter_{c1}_vs_{c2}.png")
            pair_data = {'Statistic':['Spearman ρ','p-value','N (pairs)'],'Value':[f"{rho:.3f}",p_str,n]}
            pair_df = pd.DataFrame(pair_data).set_index('Statistic')
            save_single_correlation_table_image(pair_df, f"Spearman: {c1} vs {c2}", f"table_corr_{c1}_vs_{c2}.png")
        else: msg=f"Could not calc corr for {c1} vs {c2}."; print(msg); report += f"- {c1} vs {c2}: {msg}\n"
    report += "\n"

    ders_present = [c for c in ders_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(ders_present) > 1:
        report += "Correlations among DERS Subscales:\n"
        ders_csv = config.REPORTS_DIR/"correlation_matrix_ders_subscales.csv"
        ders_heatmap = viz_dir/"heatmap_ders_subscales.png"
        ders_matrix = generate_correlation_matrix_and_images(df,ders_present,ders_csv,ders_heatmap,"ders_subscales")
        if ders_matrix is not None: report += "(See CSV and image files for DERS Subscales matrix)\n\n"
        else: report += "DERS Subscale correlation matrix could not be generated.\n\n"
    else: msg="Not enough DERS subscales for matrix."; print(msg); report+=f"{msg}\n\n"

    full_csv = config.REPORTS_DIR/"correlation_matrix_full.csv"
    full_heatmap = viz_dir/"heatmap_full_matrix.png"
    full_matrix = generate_correlation_matrix_and_images(df,matrix_cols,full_csv,full_heatmap,"full_totals_and_subscales")
    if full_matrix is not None: report += "Full Correlation Matrix available (see CSV and image files).\n"
    else: report += "Full correlation matrix could not be generated.\n"
    try:
        with open(report_path,"w",encoding="utf-8") as f: f.write(report)
        print(f"Correlation report saved: {report_path}")
    except Exception as e: print(f"Error saving correlation report: {e}")