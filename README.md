# Student Data Analysis Project

Automated data cleaning, statistical analysis, and report generation for student survey data, focusing on Resilience, ASIR, and DERS scales. Generates PNG tables and plots.


## Core Features

*   Data cleaning (filtering, imputation, recalculations).
*   Descriptive statistics with visualizations.
*   Spearman correlation analysis (pairs, DERS subscales, full matrix) with plots and tables.
*   Simple linear regression analysis.
*   Outputs PNG tables for all key results using `dataframe_image`.
*   Clears previous outputs on each run.

## Quick Start

### 1. Setup Environment

*   Python 3.8+ installed.
*   Place your CSV as `study_project/data/original_dataset.csv`. **Backup this file first!**
*   Navigate to `study_project/` in your terminal.
*   Install dependencies:
    ```bash
    pip install -r requirements.txt
    python -m playwright install --with-deps chromium
    ```
    (`requirements.txt` should contain: `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `dataframe_image`, `playwright`)

### 2. Verify Configuration

*   Open `study_project/src/config.py`.
*   **Crucially, ensure `DERS_RAW_ITEMS_PREFIX` and keys in `DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN` exactly match your CSV headers.** Also check other column name constants.

### 3. Run Analysis

*   From the `study_project/` directory:
    ```bash
    python src/main.py
    ```
*   Type `YES_OVERWRITE` when prompted (after backing up data).

## Output

*   **`data/original_dataset.csv`**: Overwritten with cleaned data.
*   **`outputs/reports/`**: `.txt` and `.csv` summaries.
*   **`outputs/visualizations/`**: `.png` plots and tables for descriptive stats, correlations, and regression summaries.

## Troubleshooting

*   **`ModuleNotFoundError: No module named 'src'`**: Run `python src/main.py` from the `study_project/` root.
*   **`KeyError`**: Check column name configurations in `src/config.py` against your CSV headers.
*   **`dataframe_image` errors**: Ensure Playwright and its browser (`chromium`) are correctly installed. See detailed README or `dataframe_image` docs for browser issues.
