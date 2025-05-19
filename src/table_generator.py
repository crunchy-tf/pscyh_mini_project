import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src import config
import dataframe_image as dfi

def save_dataframe_as_image_dfi(df: pd.DataFrame, output_path: config.Path, title: str = "Table",
                                table_styler=None, max_rows=None, max_cols=None, dpi=200,
                                table_conversion_backend='playwright'):
    # ... (This function remains unchanged from the previous version)
    if df.empty:
        print(f"DataFrame for '{title}' is empty. Skipping image generation for {output_path}.")
        return
    print(f"Attempting to save table '{title}' to {output_path} using dataframe_image (backend: {table_conversion_backend})...")
    try:
        obj_to_export = df 
        if table_styler:
            obj_to_export = table_styler
        else:
            obj_to_export = df.style.set_caption(title)\
                                .set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#004C99'), ('color', 'white'), ('text-align', 'center'), ('font-weight', 'bold'), ('padding', '5px')]},
                                    {'selector': 'td', 'props': [('text-align', 'left'), ('border', '1px solid #dddddd'), ('padding', '5px')]},
                                    {'selector': 'tr:nth-child(even) td', 'props': [('background-color', '#f8f8f8')]},
                                    {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '1.3em'), ('font-weight', 'bold'), ('margin-bottom', '10px'), ('color', '#333333'), ('white-space', 'pre-line')]} # Allow newlines in caption
                                ])\
                                .format(precision=3, na_rep="-")
        dfi.export(
            obj=obj_to_export, filename=output_path, dpi=dpi,
            table_conversion=table_conversion_backend,
        )
        print(f"Table image for '{title}' saved successfully to: {output_path}")
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR saving table image '{title}' using dataframe_image to {output_path}: {e}")
        # ... (rest of error message)
        print(f"  Backend used: {table_conversion_backend}")
        print(f"  Ensure a headless browser environment is correctly set up.")
        print(f"  If using Playwright (recommended):")
        print(f"    1. `pip install playwright`")
        print(f"    2. `python -m playwright install --with-deps chromium` (or other browser like firefox, webkit)")
        print(f"  If issues persist, check dataframe_image documentation for troubleshooting.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def save_descriptive_stats_table_image(stats_df: pd.DataFrame):
    # ... (This function remains unchanged from the previous version)
    if stats_df.empty:
        print("Descriptive stats DataFrame is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "descriptive" / "table_descriptive_stats.png"
    df_for_table = stats_df.transpose() 
    styled_df = df_for_table.style.set_caption("Descriptive Statistics Summary")\
        .set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center; background-color: #003366; color: white; font-weight: bold; padding: 8px;'},
            {'selector': 'th.row_heading', 'props': 'text-align: left; font-weight: bold; background-color: #e0e0e0; padding: 8px;'},
            {'selector': 'td', 'props': 'text-align: right; border: 1px solid #bfbfbf; padding: 6px;'},
            {'selector': 'tr:nth-child(even) td', 'props': 'background-color: #f5f5f5;'},
            {'selector': 'tr:nth-child(odd) td', 'props': 'background-color: #ffffff;'},
            {'selector': 'caption', 'props': 'caption-side: top; font-size: 1.4em; font-weight: bold; margin: 20px 0; color: #222; white-space: pre-line;'}
        ])\
        .format(precision=3, na_rep="-")
    save_dataframe_as_image_dfi(df_for_table, output_path, title="Descriptive Statistics Summary", table_styler=styled_df, dpi=180)


def save_correlation_matrix_table_image(corr_df: pd.DataFrame, filename_suffix: str = "full", title_override: str = None): # <--- Added title_override
    if corr_df.empty:
        print(f"Correlation matrix '{filename_suffix}' is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "correlations" / f"table_correlation_matrix_{filename_suffix}.png"
    
    # Use title_override if provided, otherwise construct default
    display_title = title_override if title_override else f"Spearman Correlation Matrix ({filename_suffix.replace('_', ' ').title()})"

    styled_corr_matrix = corr_df.style.set_caption(display_title)\
        .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)\
        .set_properties(**{'font-size': '10pt' if corr_df.shape[0] <= 3 else '8pt', # Adjust font for small 2x2
                           'border': '1px solid #aaa', 
                           'width': '100px' if corr_df.shape[0] <=3 else '80px', # Adjust width for 2x2
                           'text-align': 'center'})\
        .format("{:.3f}")\
        .set_table_styles([
            {'selector': 'th', 'props': f"font-size: {'10pt' if corr_df.shape[0] <=3 else '9pt'}; font-weight: bold; text-align: center; padding: 5px; background-color: #003366; color:white;"},
            {'selector': 'caption', 'props': 'caption-side: top; font-size: 1.2em; font-weight: bold; margin: 15px 0; color: #222; white-space: pre-line; text-align: center;'} # Allow newlines in caption and center it
        ])
    # Pass the display_title to the underlying dfi function as it's used in error messages etc.
    save_dataframe_as_image_dfi(corr_df, output_path, title=display_title, table_styler=styled_corr_matrix, dpi=180)


def save_regression_summary_as_image(summary_text: str, model_name: str, output_path: config.Path):
    # ... (This function remains unchanged from the previous version)
    if not summary_text.strip():
        print(f"Regression summary text for '{model_name}' is empty. Skipping image generation for {output_path}.")
        return
    lines = summary_text.splitlines()
    line_height_inches = 0.18; padding_inches_vertical = 2.5; padding_inches_horizontal = 1.5
    max_line_len = 0
    if lines: max_line_len = max(len(line) for line in lines)
    fig_width = min(22, max(12, max_line_len * 0.095 + padding_inches_horizontal))
    fig_height = len(lines) * line_height_inches + padding_inches_vertical
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    fig.text(0.03, 0.97, summary_text, fontfamily='monospace', fontsize=10,
             va='top', ha='left', wrap=False, linespacing=1.5)
    ax = plt.gca(); ax.axis('off')
    plt.suptitle(f"Regression Summary: {model_name}", fontsize=16, y=0.985, weight='bold', x=0.03, ha='left')
    try:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Regression summary image saved to: {output_path}")
    except Exception as e: print(f"Error saving regression summary image {output_path}: {e}")
    plt.close(fig)