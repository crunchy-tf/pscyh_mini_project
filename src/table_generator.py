import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import table as pd_table_plot # Renamed to avoid conflict
from src import config
import textwrap # For wrapping long text in cells

def _get_column_widths(df, max_char_width=25):
    """Helper to estimate column widths based on content, with a max width."""
    # Estimate width based on max length of content in each column
    # and header length. Max char width helps prevent extremely wide columns.
    col_widths = []
    for col in df.columns:
        header_len = len(str(col))
        # Ensure data is string for len calculation, handle NaN
        max_content_len = df[col].astype(str).map(len).max()
        # If max_content_len is NaN (e.g., all NaNs in column), use header_len
        if pd.isna(max_content_len):
            max_content_len = header_len

        # Heuristic: 0.015 units per character, adjust as needed
        # Cap the width to avoid extremely wide columns for long strings
        # Effective width is a balance between header and content, capped
        effective_len = min(max(header_len, int(max_content_len)), max_char_width)
        col_widths.append(effective_len * 0.02) # Adjusted multiplier for width

    # Normalize if total width is too small or too large (optional, complex)
    # For now, let this basic estimation work.
    return col_widths


def save_dataframe_as_image(df: pd.DataFrame, output_path: config.Path, title: str = "Table",
                            font_size=10, title_font_size=16,
                            header_color='#004C99', header_font_color='white',
                            row_colors=None, edge_color='grey',
                            dpi=200, cell_height_scale=1.8, first_col_bold=True,
                            max_col_width_chars=30):
    """
    Saves a Pandas DataFrame as a PNG image using matplotlib with enhanced styling.
    """
    if df.empty:
        print(f"DataFrame for '{title}' is empty. Skipping image generation for {output_path}.")
        return

    if row_colors is None:
        row_colors = ['#E6E6E6', '#FFFFFF'] # Light grey, white

    df_to_plot = df.copy()
    # Reset index to make the current index a column, for consistent plotting
    if df_to_plot.index.name is None:
        idx_name_base = "Index"
        idx_name = idx_name_base
        i = 0
        while idx_name in df_to_plot.columns:
            i += 1
            idx_name = f"{idx_name_base}_{i}"
        df_to_plot.index.name = idx_name
    df_to_plot = df_to_plot.reset_index()

    # Wrap text in cells to prevent overly wide tables
    for col in df_to_plot.columns:
        if df_to_plot[col].dtype == 'object': # Only wrap string columns
            df_to_plot[col] = df_to_plot[col].astype(str).apply(
                lambda x: textwrap.fill(x, width=max_col_width_chars) if isinstance(x, str) else x
            )

    num_rows, num_cols = df_to_plot.shape

    # Estimate column widths based on content
    col_widths_est = _get_column_widths(df_to_plot, max_char_width=max_col_width_chars)

    # Figure size estimation
    # Base cell width and height (in inches)
    # These are heuristics and might need significant tuning depending on font & content
    base_cell_width_in = sum(col_widths_est) * 0.8 # Overall width factor
    base_cell_height_in = 0.03 * font_size * cell_height_scale # Based on font size

    fig_width = max(8, base_cell_width_in * num_cols * 0.2 + 2) # Add padding
    # Estimate number of lines per cell for height calculation after text wrapping
    avg_lines_per_cell = df_to_plot.map(lambda x: len(str(x).split('\n'))).max().max()
    fig_height = max(4, (num_rows + 1) * base_cell_height_in * avg_lines_per_cell * 0.6 + 2) # +1 for header, add padding

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Create the table using pandas.plotting.table for better initial layout
    the_table = pd_table_plot(ax, df_to_plot, loc='center', cellLoc='left', colWidths=col_widths_est)

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(font_size)

    # Iterate through cells for custom styling
    cells = the_table.get_celld()
    for i in range(num_rows + 1): # +1 for header
        for j in range(num_cols):
            cell = cells[(i, j)]
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.5)
            cell.set_height(base_cell_height_in * avg_lines_per_cell * 0.025 * cell_height_scale) # Adjust cell height

            if i == 0:  # Header row
                cell.set_text_props(weight='bold', color=header_font_color, ha='center')
                cell.set_facecolor(header_color)
            else:  # Data rows
                cell.set_facecolor(row_colors[(i - 1) % len(row_colors)])
                cell.set_text_props(ha='left', va='center') # Align text left and center vertically
                if j == 0 and first_col_bold: # Style the first data column (usually the former index)
                    cell.set_text_props(weight='bold', ha='left', va='center')
            # Add padding within cells
            cell.PAD = 0.05


    plt.suptitle(title, fontsize=title_font_size, y=0.98, weight='bold')
    plt.tight_layout(pad=1.5) # Adjust padding as needed

    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Table image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving table image {output_path}: {e}")
    plt.close(fig)


# --- Specific functions for different tables ---

def save_descriptive_stats_table_image(stats_df: pd.DataFrame):
    if stats_df.empty:
        print("Descriptive stats DataFrame is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "descriptive" / "table_descriptive_stats.png"
    # stats_df often has stats as rows, variables as columns. Transpose for typical table view.
    df_for_table = stats_df.transpose()
    df_for_table = df_for_table.round(3) # Round numeric values
    save_dataframe_as_image(df_for_table, output_path, title="Descriptive Statistics Summary",
                            font_size=9, title_font_size=14, cell_height_scale=1.5,
                            max_col_width_chars=20) # Adjust max_col_width_chars

def save_single_correlation_table_image(df_corr_pair: pd.DataFrame, title: str, filename: str):
    """Saves a small correlation table (e.g., for a single pair) as an image."""
    if df_corr_pair.empty:
        print(f"Correlation data for '{title}' is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "correlations" / filename
    df_corr_pair_rounded = df_corr_pair.round(3)
    save_dataframe_as_image(df_corr_pair_rounded, output_path, title=title,
                            font_size=10, title_font_size=14, first_col_bold=False,
                            max_col_width_chars=25)

def save_correlation_matrix_table_image(corr_df: pd.DataFrame, filename_suffix: str = "full"):
    if corr_df.empty:
        print(f"Correlation matrix '{filename_suffix}' is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "correlations" / f"table_correlation_matrix_{filename_suffix}.png"
    corr_df_rounded = corr_df.round(3)
    title_str = f"Spearman Correlation Matrix ({filename_suffix.replace('_', ' ').title()})"
    save_dataframe_as_image(corr_df_rounded, output_path, title=title_str,
                            font_size=8, title_font_size=12,
                            max_col_width_chars=15) # Smaller font for matrix

def save_regression_summary_as_image(summary_text: str, model_name: str, output_path: config.Path):
    """Saves a text summary (like regression output) as a PNG image."""
    if not summary_text.strip():
        print(f"Regression summary text for '{model_name}' is empty. Skipping image generation for {output_path}.")
        return

    lines = summary_text.splitlines()
    line_height_inches = 0.18
    padding_inches_vertical = 2.0
    padding_inches_horizontal = 1.0
    # Estimate width based on longest line, up to a max.
    max_line_len = 0
    if lines: # Check if lines list is not empty
        max_line_len = max(len(line) for line in lines)

    fig_width = min(20, max(10, max_line_len * 0.09 + padding_inches_horizontal)) # Char width heuristic for monospace
    fig_height = len(lines) * line_height_inches + padding_inches_vertical

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Use fig.text for better control over multi-line text block positioning
    fig.text(0.02, 0.98, summary_text, fontfamily='monospace', fontsize=9,
             va='top', ha='left', wrap=False)
    ax.axis('off')
    plt.suptitle(f"Regression Summary: {model_name}", fontsize=14, y=0.99, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Regression summary image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving regression summary image {output_path}: {e}")
    plt.close(fig)