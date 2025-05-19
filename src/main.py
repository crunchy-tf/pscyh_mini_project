import sys
import os
import shutil # <--- Import shutil for rmtree

# Add the project root directory (study_project) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) # directory of main.py (src)
project_root = os.path.dirname(current_dir) # directory of src (study_project)
if project_root not in sys.path: # Add only if not already present
    sys.path.insert(0, project_root)

from src import config
from src.data_io import load_data, save_data_overwrite
from src.data_cleaner import clean_data
from src.descriptive_analyzer import perform_descriptive_analysis
from src.correlation_analyzer import perform_correlation_analysis
from src.regression_analyzer import perform_regression_analyses

# --- NEW FUNCTION TO CLEAN OUTPUTS ---
def clean_output_directories():
    """
    Deletes and recreates the main output directories (reports and visualizations)
    to ensure a clean slate for new outputs.
    """
    print("Cleaning previous output directories...")

    # Paths from config
    reports_dir = config.REPORTS_DIR
    viz_dir = config.VISUALIZATIONS_DIR

    # Directories to clean and recreate under visualizations
    viz_subdirs_to_recreate = ["descriptive", "correlations", "regressions"]

    # Clean reports directory
    if reports_dir.exists():
        try:
            shutil.rmtree(reports_dir) # Deletes the directory and all its contents
            print(f"  Removed directory: {reports_dir}")
        except Exception as e:
            print(f"  Error removing reports directory {reports_dir}: {e}")
    try:
        reports_dir.mkdir(parents=True, exist_ok=True) # Recreate it
        print(f"  Recreated directory: {reports_dir}")
    except Exception as e:
        print(f"  Error recreating reports directory {reports_dir}: {e}")

    # Clean visualizations directory (and its expected structure)
    if viz_dir.exists():
        try:
            shutil.rmtree(viz_dir)
            print(f"  Removed directory: {viz_dir}")
        except Exception as e:
            print(f"  Error removing visualizations directory {viz_dir}: {e}")
    try:
        # Recreate the main viz_dir first
        viz_dir.mkdir(parents=True, exist_ok=True)
        # Then recreate specific subdirectories
        for subdir_name in viz_subdirs_to_recreate:
            (viz_dir / subdir_name).mkdir(parents=True, exist_ok=True)
        print(f"  Recreated visualizations directory structure under: {viz_dir}")
    except Exception as e:
        print(f"  Error recreating visualization subdirectories under {viz_dir}: {e}")
    print("Output directories cleaned.")
# --- END OF NEW FUNCTION ---

def main():
    # --- CALL THE CLEANING FUNCTION AT THE BEGINNING ---
    clean_output_directories()
    # ---------------------------------------------------

    print("\nStarting data analysis pipeline...") # Added a newline for better separation
    print("="*50)
    print("CRITICAL WARNING:")
    print(f"This script is configured to OVERWRITE your original data file:")
    print(f"  '{config.ORIGINAL_DATA_FILE}'")
    print("It is STRONGLY recommended to make a backup of this file before proceeding.")
    print("="*50)

    confirmation = input("Type 'YES_OVERWRITE' to confirm and proceed with modifying the original file: ")
    if confirmation != "YES_OVERWRITE":
        print("Confirmation not received. Exiting script to prevent data modification.")
        sys.exit()

    print("\nConfirmation received. Proceeding with data operations...\n")

    # --- Stage 1: Load, Clean, and Overwrite Original File ---
    try:
        print(f"Attempting to load raw data from: {config.ORIGINAL_DATA_FILE}")
        raw_df = load_data(config.ORIGINAL_DATA_FILE)

        cleaned_df_in_memory = clean_data(raw_df)
        if cleaned_df_in_memory.empty:
            print("Error: Data cleaning resulted in an empty DataFrame. Cannot proceed.")
            sys.exit()

        print(f"Attempting to save cleaned data and OVERWRITE: {config.ORIGINAL_DATA_FILE}")
        save_data_overwrite(cleaned_df_in_memory, config.ORIGINAL_DATA_FILE)
        print("Original file has been overwritten with cleaned data.")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure your 'original_dataset.csv' is correctly placed and named.")
        sys.exit()
    except IOError as e:
        print(f"An IO error occurred during file operations: {e}")
        sys.exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading/cleaning/saving: {e}")
        sys.exit()


    # --- Stage 2: Load the (Now Modified) Data and Perform Analysis ---
    try:
        print(f"\nRe-loading the modified data from: {config.ORIGINAL_DATA_FILE} for analysis.")
        df_for_analysis = load_data(config.ORIGINAL_DATA_FILE)
        if df_for_analysis.empty:
            print("Error: Failed to load the modified data for analysis. DataFrame is empty.")
            sys.exit()

        # Descriptive Analysis
        descriptive_viz_dir = config.VISUALIZATIONS_DIR / "descriptive" # Path used by analyzer
        descriptive_report_file = config.REPORTS_DIR / "descriptive_stats.txt"
        perform_descriptive_analysis(df_for_analysis,
                                     config.COLS_FOR_DESCRIPTIVE,
                                     descriptive_report_file,
                                     descriptive_viz_dir)

        # Correlation Analysis
        correlation_viz_dir = config.VISUALIZATIONS_DIR / "correlations" # Path used by analyzer
        correlation_report_file = config.REPORTS_DIR / "spearman_correlations.txt"
        perform_correlation_analysis(df_for_analysis,
                                     config.CORRELATION_PAIRS_SPEARMAN,
                                     config.COLS_FOR_CORR_MATRIX,
                                     config.DERS_SUBSCALE_COLS_CLEAN,
                                     correlation_report_file,
                                     correlation_viz_dir)

        # Regression Analysis
        # regression_viz_dir is handled inside perform_regression_analyses based on config.VISUALIZATIONS_DIR
        regression_report_file = config.REPORTS_DIR / "regression_summaries.txt"
        perform_regression_analyses(df_for_analysis,
                                    config.REGRESSION_PAIRS,
                                    regression_report_file)

    except FileNotFoundError as e:
        print(f"Critical Error: The modified data file seems to be missing: {e}")
        sys.exit()
    except IOError as e:
        print(f"An IO error occurred during analysis phase file operations: {e}")
        sys.exit()
    except Exception as e:
        print(f"An unexpected error occurred during the analysis phase: {e}")
        sys.exit()


    print("\nData analysis pipeline finished successfully.")
    print(f"Outputs (reports and visualizations) can be found in: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()