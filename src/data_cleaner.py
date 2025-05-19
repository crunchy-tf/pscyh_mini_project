import pandas as pd
import numpy as np
from src import config

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names: lowercase, replaces spaces and special chars with underscores."""
    original_cols = df.columns.tolist()
    new_cols = []
    col_map_debug = {} # For debugging name changes

    for col_idx, col_original_case in enumerate(original_cols):
        col_str = str(col_original_case).strip() # Original name from CSV
        new_col_name = col_str.lower()
        new_col_name = new_col_name.replace(' ', '_').replace(':', '_').replace("'", "")
        new_col_name = new_col_name.replace('é', 'e').replace('è', 'e') # Normalize accents

        # Specific handling for DERS subscales based on exact original CSV header
        if col_str in config.DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN:
            new_col_name = config.DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN[col_str]
        # Specific handling for scale items
        elif col_str.startswith(config.RESILIENCE_ITEMS_ORIGINAL_PREFIX):
            item_num = col_str[len(config.RESILIENCE_ITEMS_ORIGINAL_PREFIX):]
            new_col_name = f"{config.RESILIENCE_PREFIX_CLEAN}{item_num}"
        elif col_str.startswith(config.ASIR_ITEMS_ORIGINAL_PREFIX):
            item_num = col_str[len(config.ASIR_ITEMS_ORIGINAL_PREFIX):]
            new_col_name = f"{config.ASIR_PREFIX_CLEAN}{item_num}"
        elif col_str.startswith(config.DERS_RAW_ITEMS_PREFIX): # e.g., DEERS1
            item_num = col_str[len(config.DERS_RAW_ITEMS_PREFIX):]
            new_col_name = f"{config.DERS_PREFIX_CLEAN}{item_num}"
        elif col_str.lower().startswith(config.DERS_PREFIX_CLEAN) and col_str[len(config.DERS_PREFIX_CLEAN):].isdigit(): # e.g. DERS2 (if DERS_RAW_ITEMS_PREFIX was different)
            # This case might be tricky if DERS_RAW_ITEMS_PREFIX is "DERS" as well
            # The goal is to ensure DERS items become "ders1", "ders2", etc.
            item_num = col_str[len(config.DERS_PREFIX_CLEAN):]
            new_col_name = f"{config.DERS_PREFIX_CLEAN}{item_num}"


        # Standardize key demographic/ID columns to known clean names
        if col_str == config.COL_SEX: new_col_name = "sexe" # ensure it's 'sexe'
        elif col_str == config.COL_AGE: new_col_name = "age"
        elif col_str == config.COL_UNIVERSITY: new_col_name = "universite"
        elif col_str == config.COL_LEVEL: new_col_name = "niveau"
        elif col_str == config.COL_GOVERNORATE: new_col_name = "gouvernorat_dorigine"

        new_cols.append(new_col_name)
        col_map_debug[col_str] = new_col_name

    # print("Column name mapping (debug):", col_map_debug) # Uncomment for debugging
    df.columns = new_cols
    return df

def filter_by_sex(df: pd.DataFrame, sex_col_cleaned: str = "sexe", sex_to_keep: str = 'F') -> pd.DataFrame:
    """Filters the DataFrame to keep only specified sex."""
    if sex_col_cleaned not in df.columns:
        print(f"Warning: Sex column '{sex_col_cleaned}' not found. Skipping sex filter.")
        return df
    original_rows = len(df)
    df = df[df[sex_col_cleaned].astype(str).str.upper() == sex_to_keep.upper()]
    rows_after_filter = len(df)
    print(f"Filtered by sex '{sex_to_keep}'. Kept {rows_after_filter} of {original_rows} rows.")
    return df

def impute_categorical_missing(df: pd.DataFrame, cols_to_impute_original: list, fill_value: str = "Not Specified") -> pd.DataFrame:
    """Imputes missing values in specified categorical columns (uses original names to find them, then cleaned names)."""
    # Map original config names to current (cleaned) column names
    # This is a bit complex because cleaning happens first. We need to find the *cleaned* version of these original columns.
    current_cols_map = {original_name: cleaned_name for original_name, cleaned_name in zip(df.columns, df.columns)} # Placeholder, real mapping is trickier
                                                                                                                    # This needs to be improved. Let's assume clean_column_names correctly maps them.
    
    cleaned_cols_to_impute = []
    # Find the *current* (cleaned) names for the original categorical columns specified
    if "universite" in df.columns: cleaned_cols_to_impute.append("universite")
    if "niveau" in df.columns: cleaned_cols_to_impute.append("niveau")
    if "age" in df.columns: cleaned_cols_to_impute.append("age")
    if "gouvernorat_dorigine" in df.columns: cleaned_cols_to_impute.append("gouvernorat_dorigine")
    # Note: 'sexe' is handled by filtering, but if imputation was needed, it would be 'sexe'.

    for col_cleaned_name in cleaned_cols_to_impute:
        if col_cleaned_name in df.columns:
            df[col_cleaned_name].fillna(fill_value, inplace=True)
            if col_cleaned_name == 'age' and fill_value == "Not Specified":
                 df[col_cleaned_name] = df[col_cleaned_name].astype(str)
        else:
            print(f"Warning: Column '{col_cleaned_name}' (intended for imputation) not found after name cleaning.")
    print(f"Imputed missing values in identified categorical columns with '{fill_value}'.")
    return df

def convert_items_to_numeric(df: pd.DataFrame, item_cols_cleaned: list) -> pd.DataFrame:
    """Converts specified item columns (cleaned names) to numeric, coercing errors to NaN."""
    for col in item_cols_cleaned:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Item column '{col}' not found for numeric conversion.")
    return df

def get_scale_item_column_names(df_columns, prefix_clean, num_items):
    """Helper to get actual item column names present in the DataFrame using cleaned prefix."""
    return [col for col in df_columns if col.startswith(prefix_clean) and
            col[len(prefix_clean):].isdigit() and
            1 <= int(col[len(prefix_clean):]) <= num_items]

def handle_missing_numerical_for_scales(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows if any item essential for any of the main scale totals is missing."""
    original_rows = len(df)

    resilience_items = get_scale_item_column_names(df.columns, config.RESILIENCE_PREFIX_CLEAN, config.N_RESILIENCE_ITEMS)
    asir_items = get_scale_item_column_names(df.columns, config.ASIR_PREFIX_CLEAN, config.N_ASIR_ITEMS)
    ders_items = get_scale_item_column_names(df.columns, config.DERS_PREFIX_CLEAN, config.N_DERS_ITEMS)

    all_essential_items = resilience_items + asir_items + ders_items
    # Remove duplicates if any item set overlaps (should not happen with good prefixes)
    all_essential_items = list(set(all_essential_items))
    
    # Ensure all listed essential items actually exist in df before trying to check for nulls
    existing_essential_items = [item for item in all_essential_items if item in df.columns]
    if not existing_essential_items:
        print("Warning: No essential scale item columns found. Skipping row deletion for missing numericals.")
        return df

    df_cleaned = df.dropna(subset=existing_essential_items, how='any')

    rows_dropped = original_rows - len(df_cleaned)
    print(f"Removed {rows_dropped} rows due to missing values in essential scale items for total score calculation.")
    if original_rows > 0 and (rows_dropped / original_rows) > config.ROW_DELETION_SIGNIFICANCE_THRESHOLD:
        print(f"Warning: A significant number of rows ({rows_dropped}, {rows_dropped/original_rows:.1%}) were dropped. This may affect data integrity.")
    return df_cleaned

def recalculate_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculates total scores for Resilience, ASIR, and DERS."""
    res_items = get_scale_item_column_names(df.columns, config.RESILIENCE_PREFIX_CLEAN, config.N_RESILIENCE_ITEMS)
    asir_items = get_scale_item_column_names(df.columns, config.ASIR_PREFIX_CLEAN, config.N_ASIR_ITEMS)
    ders_items = get_scale_item_column_names(df.columns, config.DERS_PREFIX_CLEAN, config.N_DERS_ITEMS)

    if res_items: df[config.TOTAL_RESILIENCE_CALC] = df[res_items].sum(axis=1, skipna=False)
    else: print("Warning: No resilience items found for total calculation.")
    if asir_items: df[config.TOTAL_ASIR_CALC] = df[asir_items].sum(axis=1, skipna=False)
    else: print("Warning: No ASIR items found for total calculation.")
    if ders_items: df[config.TOTAL_DERS_CALC] = df[ders_items].sum(axis=1, skipna=False)
    else: print("Warning: No DERS items found for total calculation.")
    print("Recalculated total scores.")
    return df

def verify_data_types(df: pd.DataFrame):
    """Prints DataFrame info and attempts to convert key analysis columns to numeric if they aren't."""
    print("\nVerifying data types (DataFrame.info()):")
    df.info()

    numeric_cols_to_check = config.DERS_SUBSCALE_COLS_CLEAN + \
                            [config.TOTAL_RESILIENCE_CALC, config.TOTAL_ASIR_CALC, config.TOTAL_DERS_CALC]
    if 'age' in df.columns: # Age is also important if used numerically
        numeric_cols_to_check.append('age')

    for col in numeric_cols_to_check:
        if col in df.columns:
            # If 'age' was imputed with "Not Specified", it needs careful handling before numeric conversion
            if col == 'age' and "Not Specified" in df[col].unique():
                # Convert "Not Specified" to NaN for numeric conversion, then decide on imputation or leave as NaN
                df[col] = df[col].replace("Not Specified", np.nan)
                print(f"Column '{col}' contained 'Not Specified', converted to NaN before numeric check.")

            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: Column '{col}' is expected to be numeric but has Dtype {df[col].dtype}. Attempting conversion.")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any() and not df[col].isnull().all(): # Check if new NaNs were introduced
                    print(f"Warning: Coercion of '{col}' to numeric introduced NaNs. Original non-numeric values existed.")
            # Ensure 'age' is integer if possible, after numeric conversion
            if col == 'age' and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().all() and (df[col] % 1 == 0).all():
                df[col] = df[col].astype(int)

def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Main function to orchestrate all data cleaning steps."""
    print("Starting data cleaning process...")
    df = df_raw.copy()

    # 1. Clean column names
    df = clean_column_names(df)
    print("Standardized column names.")

    # 2. Filter by sex (female) - uses cleaned column name 'sexe'
    df = filter_by_sex(df, sex_col_cleaned='sexe', sex_to_keep='F')

    # 3. Impute missing categorical data (uses original names from config to identify columns)
    df = impute_categorical_missing(df, config.CATEGORICAL_COLS_TO_IMPUTE_ORIGINAL)

    # 4. Convert all scale items to numeric
    res_items_clean = get_scale_item_column_names(df.columns, config.RESILIENCE_PREFIX_CLEAN, config.N_RESILIENCE_ITEMS)
    asir_items_clean = get_scale_item_column_names(df.columns, config.ASIR_PREFIX_CLEAN, config.N_ASIR_ITEMS)
    ders_items_clean = get_scale_item_column_names(df.columns, config.DERS_PREFIX_CLEAN, config.N_DERS_ITEMS)
    all_scale_items_clean = res_items_clean + asir_items_clean + ders_items_clean
    df = convert_items_to_numeric(df, all_scale_items_clean)
    print(f"Converted {len(all_scale_items_clean)} identified scale item columns to numeric.")

    # 5. Handle missing numerical values essential for scale calculations (delete rows)
    df = handle_missing_numerical_for_scales(df)

    # 6. Recalculate total scores
    df = recalculate_totals(df)

    # 7. Verify data types (and convert DERS subscales, calculated totals, age if needed)
    verify_data_types(df) # This function now attempts conversion for key columns

    print(f"Data cleaning finished. Final shape: {df.shape}")
    return df