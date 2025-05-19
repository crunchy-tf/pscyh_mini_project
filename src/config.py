from pathlib import Path

# Project Root
# Assumes this config.py is in study_project/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
# THIS IS THE FILE THAT WILL BE OVERWRITTEN
ORIGINAL_DATA_FILE = DATA_DIR / "original_dataset.csv"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create output directories if they don't exist (safe operation)
(VISUALIZATIONS_DIR / "descriptive").mkdir(parents=True, exist_ok=True)
(VISUALIZATIONS_DIR / "correlations").mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Column Names (Match your CSV headers exactly for initial load) ---
COL_SEX = "sexe"
COL_AGE = "âge"
COL_UNIVERSITY = "unviersité" # Typo as in your example
COL_LEVEL = "niveau"
COL_GOVERNORATE = "gouvernorat d'origine"

# Categorical columns for imputation (using their original names before cleaning)
CATEGORICAL_COLS_TO_IMPUTE_ORIGINAL = [COL_UNIVERSITY, COL_LEVEL, COL_AGE, COL_GOVERNORATE]

# Scale Item Columns (Original Naming Pattern)
RESILIENCE_ITEMS_ORIGINAL_PREFIX = "Résilience"
ASIR_ITEMS_ORIGINAL_PREFIX = "ASIR"
# The provided data snippet had DEERS1 for DERS items, then DERS2.
# For consistency, you MUST verify what the actual prefix is for DERS items 1-36 in your CSV.
# If items are DERS1, DERS2,...DERS36, use "DERS".
# If items are DEERS1, DEERS2,...DEERS36, use "DEERS".
# If it's mixed, the cleaning logic for column names will need to be more complex.
# Assuming "DEERS" based on the first item shown in your sample data for the DERS scale.
DERS_RAW_ITEMS_PREFIX = "DEERS" # USER MUST VERIFY THIS PREFIX FOR DERS ITEMS IN THEIR CSV

# Target number of items for each scale
N_RESILIENCE_ITEMS = 12
N_ASIR_ITEMS = 15
N_DERS_ITEMS = 36

# Cleaned column name prefixes (after standardization)
RESILIENCE_PREFIX_CLEAN = "resilience"
ASIR_PREFIX_CLEAN = "asir"
DERS_PREFIX_CLEAN = "ders" # For DERS items, after cleaning names (e.g., ders1, ders2)

# DERS subscale column mapping from original CSV header to cleaned name
# Ensure these keys EXACTLY match the headers in your CSV file for these subscales.
DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN = {
    "N:non acceptation": "ders_n_non_acceptance",
    "G:goals": "ders_g_goals",
    "I:impulse": "ders_i_impulse",
    "A:awarness": "ders_a_awareness", # Corrected spelling from "awarness"
    "S:strategies": "ders_s_strategies",
    "C:clarity": "ders_c_clarity"
}
DERS_SUBSCALE_COLS_CLEAN = list(DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN.values())

# Calculated total score columns (cleaned names)
TOTAL_RESILIENCE_CALC = "total_resilience_calc"
TOTAL_ASIR_CALC = "total_asir_calc"
TOTAL_DERS_CALC = "total_ders_calc"

# Columns for descriptive analysis (using cleaned, calculated total names)
COLS_FOR_DESCRIPTIVE = [TOTAL_RESILIENCE_CALC, TOTAL_ASIR_CALC, TOTAL_DERS_CALC]

# Columns for correlation matrix (includes totals and DERS subscales)
COLS_FOR_CORR_MATRIX = COLS_FOR_DESCRIPTIVE + DERS_SUBSCALE_COLS_CLEAN

# Specific correlation pairs for scatter plots and detailed reporting
CORRELATION_PAIRS_SPEARMAN = [
    (TOTAL_RESILIENCE_CALC, TOTAL_ASIR_CALC),
    (TOTAL_RESILIENCE_CALC, TOTAL_DERS_CALC),
    (TOTAL_ASIR_CALC, TOTAL_DERS_CALC)
]

# Regression pairs (dependent_var, independent_var)
REGRESSION_PAIRS = [
    (TOTAL_DERS_CALC, TOTAL_RESILIENCE_CALC), (TOTAL_RESILIENCE_CALC, TOTAL_DERS_CALC),
    (TOTAL_DERS_CALC, TOTAL_ASIR_CALC), (TOTAL_ASIR_CALC, TOTAL_DERS_CALC),
    (TOTAL_ASIR_CALC, TOTAL_RESILIENCE_CALC), (TOTAL_RESILIENCE_CALC, TOTAL_ASIR_CALC)
]

# Significance threshold for row deletion warning
ROW_DELETION_SIGNIFICANCE_THRESHOLD = 0.10 # 10%