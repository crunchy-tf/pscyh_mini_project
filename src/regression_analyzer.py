import pandas as pd
import numpy as np
import statsmodels.api as sm
from src import config
from src.table_generator import save_regression_summary_as_image # This function is now the matplotlib text-to-image one

def perform_simple_linear_regression(df: pd.DataFrame, dep_var: str, ind_var: str) -> tuple[str, str, str]:
    if dep_var not in df.columns or ind_var not in df.columns:
        err = f"Error: Vars ('{dep_var}', '{ind_var}') not found.\n"; return err, dep_var, ind_var
    Y_s = pd.to_numeric(df[dep_var],errors='coerce'); X_s = pd.to_numeric(df[ind_var],errors='coerce')
    valid = Y_s.notna() & X_s.notna(); Y = Y_s[valid]; X = X_s[valid]
    if len(Y) < 10:
        err = f"Skipping regression {dep_var}~{ind_var}: Insufficient data ({len(Y)}).\n"; return err, dep_var, ind_var
    X_c = sm.add_constant(X)
    try:
        model = sm.OLS(Y,X_c); results = model.fit()
        summary = f"Regression: {dep_var} ~ {ind_var}\n{results.summary().as_text()}\n\n"
    except Exception as e: summary = f"Error in regression {dep_var}~{ind_var}: {e}\n\n"
    return summary, dep_var, ind_var

def perform_regression_analyses(df: pd.DataFrame, pairs: list, report_file: config.Path):
    print("\n--- Performing Regression Analysis ---")
    summaries_txt = "Simple Linear Regression Summaries:\n\n"
    viz_dir = config.VISUALIZATIONS_DIR/"regressions"; viz_dir.mkdir(parents=True,exist_ok=True)
    for dep, ind in pairs:
        print(f"Running regression: {dep} predicted by {ind}")
        summary, dv_n, iv_n = perform_simple_linear_regression(df, dep, ind)
        summaries_txt += summary
        dv_fn = dv_n.replace(':','_').replace(' ','_').replace('(','').replace(')','')
        iv_fn = iv_n.replace(':','_').replace(' ','_').replace('(','').replace(')','')
        img_fn = f"table_regression_{dv_fn}_on_{iv_fn}.png"
        img_path = viz_dir/img_fn
        model_disp_name = f"{dv_n} ~ {iv_n}"
        save_regression_summary_as_image(summary, model_disp_name, img_path)
    try:
        with open(report_file,"w",encoding="utf-8") as f: f.write(summaries_txt)
        print(f"Regression summaries text report saved: {report_file}")
    except Exception as e: print(f"Error saving regression text report: {e}")