import pandas as pd
import numpy as np
import statsmodels.api as sm
from src import config
# Import the new table generator
from src.table_generator import save_regression_summary_as_image # <--- ADD THIS IMPORT

def perform_simple_linear_regression(df: pd.DataFrame, dependent_var: str, independent_var: str) -> tuple[str, str, str]:
    model_name = f"{dependent_var}_vs_{independent_var}" # Used for error messages, not filename here
    if dependent_var not in df.columns or independent_var not in df.columns:
        error_msg = f"Error: Dependent ('{dependent_var}') or Independent ('{independent_var}') variable not found in DataFrame.\n"
        return error_msg, dependent_var, independent_var # Return original var names for context

    Y_series = pd.to_numeric(df[dependent_var], errors='coerce')
    X_series = pd.to_numeric(df[independent_var], errors='coerce')

    valid_indices = Y_series.notna() & X_series.notna()
    Y = Y_series[valid_indices]
    X = X_series[valid_indices]

    if len(Y) < 10:
        error_msg = f"Skipping regression for {dependent_var} ~ {independent_var}: Insufficient data points ({len(Y)}) after NaN removal or non-numeric conversion.\n"
        return error_msg, dependent_var, independent_var

    X_const = sm.add_constant(X)

    try:
        model = sm.OLS(Y, X_const)
        results = model.fit()
        summary_str = f"Regression Analysis: {dependent_var} ~ {independent_var}\n"
        summary_str += results.summary().as_text()
        summary_str += "\n\n"
    except Exception as e:
        summary_str = f"Error during regression for {dependent_var} ~ {independent_var}: {e}\n\n"
    return summary_str, dependent_var, independent_var

def perform_regression_analyses(df: pd.DataFrame, regression_pairs: list, report_file: config.Path):
    print("\n--- Performing Regression Analysis ---")
    all_summaries_text = "Simple Linear Regression Summaries:\n\n"
    
    # Ensure the 'regressions' subdirectory exists for images
    regressions_viz_dir = config.VISUALIZATIONS_DIR / "regressions"
    regressions_viz_dir.mkdir(parents=True, exist_ok=True)


    for dep_var, ind_var in regression_pairs:
        print(f"Running regression: {dep_var} predicted by {ind_var}")
        summary_text, dv_name, iv_name = perform_simple_linear_regression(df, dep_var, ind_var)
        all_summaries_text += summary_text

        # Sanitize variable names for filename (simple replacement)
        dv_clean_filename = dv_name.replace(':', '_').replace(' ', '_')
        iv_clean_filename = iv_name.replace(':', '_').replace(' ', '_')
        
        img_filename = f"regression_{dv_clean_filename}_on_{iv_clean_filename}.png"
        output_path_img = regressions_viz_dir / img_filename
        
        model_display_name = f"{dv_name} ~ {iv_name}" # For the image title
        save_regression_summary_as_image(summary_text, model_display_name, output_path_img)
        # -------------------------------------------------

    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(all_summaries_text)
        print(f"Regression summaries text report saved to {report_file}")
    except Exception as e:
        print(f"Error saving regression text report: {e}")