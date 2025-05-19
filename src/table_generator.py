import pandas as pd
import matplotlib.pyplot as plt # Still needed for text-to-image regression summary
import numpy as np
# No longer need from pandas.plotting import table
from src import config
import dataframe_image as dfi # <--- NEW IMPORT
# import textwrap # textwrap is now handled by dataframe_image's HTML rendering mostly

# You can set global options for dataframe_image if needed, e.g.:
# dfi.options['chrome_path'] = '/path/to/your/chrome' # If auto-detection fails
# dfi.options['dpi'] = 200

def save_dataframe_as_image_dfi(df: pd.DataFrame, output_path: config.Path, title: str = "Table",
                                table_styler=None, max_rows=None, max_cols=None, dpi=200,
                                table_conversion_backend='playwright'):
    """
    Saves a Pandas DataFrame as a PNG image using dataframe_image.
    'table_styler' can be a Styler object from pandas for custom CSS-like styling.
    """
    if df.empty:
        print(f"DataFrame for '{title}' is empty. Skipping image generation for {output_path}.")
        return

    print(f"Attempting to save table '{title}' to {output_path} using dataframe_image (backend: {table_conversion_backend})...")

    try:
        obj_to_export = df # Default to exporting the raw DataFrame

        if table_styler:
            obj_to_export = table_styler # Use the pre-styled object
        else:
            # Apply some very basic default styling if no styler is provided
            # This is optional; dfi can render unstyled DataFrames too.
            # For more complex tables, it's better to create and pass a Styler object.
            obj_to_export = df.style.set_caption(title)\
                                .set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#004C99'), ('color', 'white'), ('text-align', 'center'), ('font-weight', 'bold'), ('padding', '5px')]},
                                    {'selector': 'td', 'props': [('text-align', 'left'), ('border', '1px solid #dddddd'), ('padding', '5px')]},
                                    {'selector': 'tr:nth-child(even) td', 'props': [('background-color', '#f8f8f8')]}, # Slightly different even row color
                                    {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '1.3em'), ('font-weight', 'bold'), ('margin-bottom', '10px'), ('color', '#333333')]}
                                ])\
                                .format(precision=3, na_rep="-") # Format floats, show NaNs as "-"

        dfi.export(
            obj=obj_to_export,
            filename=output_path,
            dpi=dpi,
            table_conversion=table_conversion_backend,
            # max_rows=max_rows, # For very large tables
            # max_cols=max_cols
        )
        print(f"Table image for '{title}' saved successfully to: {output_path}")
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR saving table image '{title}' using dataframe_image to {output_path}: {e}")
        print(f"  Backend used: {table_conversion_backend}")
        print(f"  Ensure a headless browser environment is correctly set up.")
        print(f"  If using Playwright (recommended):")
        print(f"    1. `pip install playwright`")
        print(f"    2. `python -m playwright install --with-deps chromium` (or other browser like firefox, webkit)")
        print(f"  If issues persist, check dataframe_image documentation for troubleshooting.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Specific functions for different tables (now using the dfi version) ---

def save_descriptive_stats_table_image(stats_df: pd.DataFrame):
    if stats_df.empty:
        print("Descriptive stats DataFrame is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "descriptive" / "table_descriptive_stats.png"
    df_for_table = stats_df.transpose() # Variables as rows, stats as columns

    styled_df = df_for_table.style.set_caption("Descriptive Statistics Summary")\
        .set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center; background-color: #003366; color: white; font-weight: bold; padding: 8px;'}, # Darker header
            {'selector': 'th.row_heading', 'props': 'text-align: left; font-weight: bold; background-color: #e0e0e0; padding: 8px;'}, # Index/Variable names
            {'selector': 'td', 'props': 'text-align: right; border: 1px solid #bfbfbf; padding: 6px;'},
            {'selector': 'tr:nth-child(even) td', 'props': 'background-color: #f5f5f5;'},
            {'selector': 'tr:nth-child(odd) td', 'props': 'background-color: #ffffff;'},
            {'selector': 'caption', 'props': 'caption-side: top; font-size: 1.4em; font-weight: bold; margin: 20px 0; color: #222;'}
        ])\
        .format(precision=3, na_rep="-")\
        # Example: highlight min/max in a column if desired
        # .highlight_max(subset=['Mean', 'Median'], color='lightgreen', axis=0)\
        # .highlight_min(subset=['Mean', 'Median'], color='pink', axis=0)

    save_dataframe_as_image_dfi(df_for_table, output_path, title="Descriptive Statistics Summary", table_styler=styled_df, dpi=180)


def save_single_correlation_table_image(df_corr_pair: pd.DataFrame, title: str, filename: str):
    if df_corr_pair.empty:
        print(f"Correlation data for '{title}' is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "correlations" / filename

    styled_df = df_corr_pair.style.set_caption(title)\
        .set_table_styles([
            {'selector': 'th.col_heading', 'props': 'text-align: center; background-color: #003366; color: white; font-weight: bold; padding: 6px;'},
            {'selector': 'th.row_heading', 'props': 'text-align: left; font-weight: bold; background-color: #e0e0e0; padding: 6px;'},
            {'selector': 'td', 'props': 'text-align: center; border: 1px solid #bfbfbf; padding: 6px;'}, # Center align values
            {'selector': 'caption', 'props': 'caption-side: top; font-size: 1.2em; font-weight: bold; margin: 15px 0; color: #222;'}
        ])\
        .format(precision=3, na_rep="-") # Value column might have mixed types, format handles numerics

    save_dataframe_as_image_dfi(df_corr_pair, output_path, title=title, table_styler=styled_df, dpi=180)


def save_correlation_matrix_table_image(corr_df: pd.DataFrame, filename_suffix: str = "full"):
    if corr_df.empty:
        print(f"Correlation matrix '{filename_suffix}' is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "correlations" / f"table_correlation_matrix_{filename_suffix}.png"
    title_str = f"Spearman Correlation Matrix ({filename_suffix.replace('_', ' ').title()})"

    styled_corr_matrix = corr_df.style.set_caption(title_str)\
        .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)\
        .set_properties(**{'font-size': '10pt', 'border': '1px solid #aaa', 'width': '80px', 'text-align': 'center'})\
        .format("{:.3f}")\
        .set_table_styles([
            {'selector': 'th', 'props': 'font-size: 10pt; font-weight: bold; text-align: center; padding: 5px; background-color: #003366; color:white;'},
            {'selector': 'caption', 'props': 'caption-side: top; font-size: 1.4em; font-weight: bold; margin: 20px 0; color: #222;'}
        ])
    save_dataframe_as_image_dfi(corr_df, output_path, title=title_str, table_styler=styled_corr_matrix, dpi=180)


def save_regression_summary_as_image(summary_text: str, model_name: str, output_path: config.Path):
    """Saves a text summary (like regression output) as a PNG image using matplotlib."""
    if not summary_text.strip():
        print(f"Regression summary text for '{model_name}' is empty. Skipping image generation for {output_path}.")
        return

    lines = summary_text.splitlines()
    line_height_inches = 0.18
    padding_inches_vertical = 2.5 # Increased padding for suptitle
    padding_inches_horizontal = 1.5
    max_line_len = 0
    if lines: max_line_len = max(len(line) for line in lines)

    fig_width = min(22, max(12, max_line_len * 0.095 + padding_inches_horizontal)) # Adjusted char width heuristic
    fig_height = len(lines) * line_height_inches + padding_inches_vertical

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white') # Ensure white background
    # Use fig.text for better control over multi-line text block positioning relative to figure
    fig.text(0.03, 0.97, summary_text, fontfamily='monospace', fontsize=10, # Slightly larger font
             va='top', ha='left', wrap=False, linespacing=1.5)

    # Add a main title to the figure using suptitle
    plt.suptitle(f"Regression Summary: {model_name}", fontsize=16, y=0.985, weight='bold', x=0.03, ha='left')

    # Turn off axis for the text plot
    ax = plt.gca()
    ax.axis('off')

    # plt.tight_layout(rect=[0, 0, 1, 0.95]) # rect can clip suptitle if not careful

    try:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Regression summary image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving regression summary image {output_path}: {e}")
    plt.close(fig)