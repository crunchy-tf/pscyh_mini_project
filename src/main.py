import sys
import os

# Add the project root directory (study_project) to the Python path
# This ensures that 'src' can be found as a top-level package
# when main.py is run directly.
current_dir = os.path.dirname(os.path.abspath(__file__)) # directory of main.py (src)
project_root = os.path.dirname(current_dir) # directory of src (study_project)
sys.path.insert(0, project_root)

from src import config
from src.data_io import load_data, save_data_overwrite
from src.data_cleaner import clean_data
from src.descriptive_analyzer import perform_descriptive_analysis
from src.correlation_analyzer import perform_correlation_analysis
from src.regression_analyzer import perform_regression_analyses
# import sys # No need to import sys again here

def main():
    # ... rest of your main function
    # ... (no changes needed to the rest of the main function)
    print("Starting data analysis pipeline...")
    print("="*50)
    print("CRITICAL WARNING:")
    print(f"This script is configured to OVERWRITE your original data file:")
    print(f"  '{config.ORIGINAL_DATA_FILE}'")
    print("It is STRONGLY recommended to make a backup of this file before proceeding.")
    print("="*50)

    confirmation = input("Type 'YES_OVERWRITE' to confirm and proceed with modifying the original file: ")
    if confirmation != "YES_OVERWRITE":
        print("Confirmation not received. Exiting script to prevent data modification.")
        sys.exit() # sys is already imported at the top

    print("\nConfirmation received. Proceeding with data operations...\n")

    # --- Stage 1: Load, Clean, and Overwrite Original File ---
    try:
        # 1. Load Raw Data
        print(f"Attempting to load raw data from: {config.ORIGINAL_DATA_FILE}")
        raw_df = load_data(config.ORIGINAL_DATA_FILE)

        # 2. Clean Data (in memory)
        cleaned_df_in_memory = clean_data(raw_df)
        if cleaned_df_in_memory.empty:
            print("Error: Data cleaning resulted in an empty DataFrame. Cannot proceed.")
            sys.exit()

        # 3. Save Cleaned Data (Overwrite Original File)
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

        # 4. Descriptive Analysis
        descriptive_viz_dir = config.VISUALIZATIONS_DIR / "descriptive"
        descriptive_report_file = config.REPORTS_DIR / "descriptive_stats.txt"
        perform_descriptive_analysis(df_for_analysis,
                                     config.COLS_FOR_DESCRIPTIVE,
                                     descriptive_report_file,
                                     descriptive_viz_dir)

        # 5. Correlation Analysis
        correlation_viz_dir = config.VISUALIZATIONS_DIR / "correlations"
        correlation_report_file = config.REPORTS_DIR / "spearman_correlations.txt"
        perform_correlation_analysis(df_for_analysis,
                                     config.CORRELATION_PAIRS_SPEARMAN,
                                     config.COLS_FOR_CORR_MATRIX,
                                     config.DERS_SUBSCALE_COLS_CLEAN, # Pass clean DERS subscale names
                                     correlation_report_file,
                                     correlation_viz_dir)

        # 6. Regression Analysis
        regression_report_file = config.REPORTS_DIR / "regression_summaries.txt"
        perform_regression_analyses(df_for_analysis,
                                    config.REGRESSION_PAIRS,
                                    regression_report_file)

    except FileNotFoundError as e: # Should not happen if previous stage succeeded
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