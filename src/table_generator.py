import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import table # For DataFrame table plotting
from src import config # To use config.VISUALIZATIONS_DIR etc.

def save_dataframe_as_image(df: pd.DataFrame, output_path: config.Path, title: str = "Table", col_widths=None, font_size=10, header_color='#40466e', row_colors=None, edge_color='w', dpi=200):
    """
    Saves a Pandas DataFrame as a PNG image using matplotlib.
    """
    if df.empty:
        print(f"DataFrame for '{title}' is empty. Skipping image generation for {output_path}.")
        return

    if row_colors is None:
        row_colors = ['#f1f1f2', 'w'] # Default row colors

    # If index is a simple range, we might not want to display it or make it less prominent
    df_to_plot = df.copy()
    # Reset index to make the current index a column, useful for plotting
    # Give index a name if it doesn't have one, to ensure it becomes a column with a header
    if df_to_plot.index.name is None:
        idx_name_base = "Index"
        idx_name = idx_name_base
        i = 0
        while idx_name in df_to_plot.columns: # Ensure unique name if "Index" already a column
            i += 1
            idx_name = f"{idx_name_base}_{i}"
        df_to_plot.index.name = idx_name
    df_to_plot = df_to_plot.reset_index()


    num_rows, num_cols = df_to_plot.shape

    # Create a figure and an axes
    # Adjust figure size based on DataFrame size
    fig_width = max(8, num_cols * 1.8) # Increased multiplier for column width
    fig_height = max(4, num_rows * 0.4 + 1.5) # Increased base height and row multiplier
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off') # Turn off the axis borders and ticks

    # Create the table
    tab = table(ax, df_to_plot, loc='center', cellLoc='left', colWidths=col_widths)

    # Style the table
    tab.auto_set_font_size(False)
    tab.set_fontsize(font_size)
    # Try to make columns fit content. This can be tricky with matplotlib table.
    # The 'auto_set_column_width' might not always give optimal results for very varied content.
    try:
        tab.auto_set_column_width(col=list(range(len(df_to_plot.columns))))
    except Exception as e:
        print(f"Note: auto_set_column_width failed for '{title}', using default widths. Error: {e}")


    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0: # Header row
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            cell.set_height(0.05) # Slightly taller header
        else: # Data rows
            cell.set_facecolor(row_colors[(row -1) % len(row_colors)]) # (row-1) because header is row 0
        if col == 0 and row !=0 : # Style the first column (former index) differently if needed
            cell.set_text_props(weight='bold')


    # Add title to the figure
    plt.suptitle(title, fontsize=font_size + 6, y=0.98, weight='bold') # Adjust y, make title bigger and bold

    plt.tight_layout(pad=2.0) # Add some padding

    try:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Table image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving table image {output_path}: {e}")
    plt.close(fig)


# --- Specific functions for different tables ---

def save_descriptive_stats_table_image(stats_df: pd.DataFrame):
    """Saves the descriptive statistics DataFrame as an image."""
    if stats_df.empty:
        print("Descriptive stats DataFrame is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "descriptive" / "descriptive_stats_table.png"
    # stats_df usually has stats as rows and variables as columns.
    # For better image layout, we might transpose it if there are few variables.
    # If stats_df.shape[1] (num variables) < stats_df.shape[0] (num stats), transposing might be better.
    # Let's keep original orientation for now, as transposing could make variable names too long for cells.
    save_dataframe_as_image(stats_df, output_path, title="Descriptive Statistics Summary", font_size=9)

def save_correlation_matrix_table_image(corr_df: pd.DataFrame, filename_suffix: str = "full"):
    """Saves a correlation matrix DataFrame as an image."""
    if corr_df.empty:
        print(f"Correlation matrix '{filename_suffix}' is empty. Skipping table image generation.")
        return
    output_path = config.VISUALIZATIONS_DIR / "correlations" / f"correlation_matrix_table_{filename_suffix}.png"
    # Round the correlation matrix for display
    corr_df_rounded = corr_df.round(3) # Rounded to 3 decimal places
    title_str = f"Spearman Correlation Matrix ({filename_suffix.replace('_', ' ').title()})"
    save_dataframe_as_image(corr_df_rounded, output_path, title=title_str, font_size=8)


def save_regression_summary_as_image(summary_text: str, model_name: str, output_path: config.Path):
    """
    Saves a (potentially long) text summary as a PNG image.
    Adjusts figure height based on the number of lines in the text.
    """
    if not summary_text.strip():
        print(f"Regression summary text for '{model_name}' is empty. Skipping image generation for {output_path}.")
        return

    lines = summary_text.splitlines()
    # Estimate figure height: 0.15 inches per line + some padding
    # Estimate figure width: fixed for monospace font or based on max line length
    # These are heuristics and may need adjustment
    line_height_inches = 0.18
    padding_inches = 2.0
    fig_height = len(lines) * line_height_inches + padding_inches
    fig_width = 12  # A reasonable fixed width for summaries

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.text(0.01, 0.99, summary_text, fontfamily='monospace', fontsize=9,
            va='top', ha='left', wrap=False) # wrap=False for summaries, rely on pre-formatting
    ax.axis('off')
    plt.suptitle(f"Regression Summary: {model_name}", fontsize=14, y=0.99, weight='bold') # Title for the whole figure

    plt.tight_layout(pad=1.0)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Regression summary image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving regression summary image {output_path}: {e}")
    plt.close(fig)