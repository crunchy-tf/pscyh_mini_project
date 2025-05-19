from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DATA_FILE = DATA_DIR / "original_dataset.csv"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"

(VISUALIZATIONS_DIR / "descriptive").mkdir(parents=True, exist_ok=True)
(VISUALIZATIONS_DIR / "correlations").mkdir(parents=True, exist_ok=True)
(VISUALIZATIONS_DIR / "regressions").mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Column Names
COL_SEX = "sexe"
COL_AGE = "âge"
COL_UNIVERSITY = "unviersité"
COL_LEVEL = "niveau"
COL_GOVERNORATE = "gouvernorat d'origine"
CATEGORICAL_COLS_TO_IMPUTE_ORIGINAL = [COL_UNIVERSITY, COL_LEVEL, COL_AGE, COL_GOVERNORATE]

RESILIENCE_ITEMS_ORIGINAL_PREFIX = "Résilience"
ASIR_ITEMS_ORIGINAL_PREFIX = "ASIR"
DERS_RAW_ITEMS_PREFIX = "DEERS" # USER MUST VERIFY

N_RESILIENCE_ITEMS = 12
N_ASIR_ITEMS = 15
N_DERS_ITEMS = 36

RESILIENCE_PREFIX_CLEAN = "resilience"
ASIR_PREFIX_CLEAN = "asir"
DERS_PREFIX_CLEAN = "ders"

DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN = {
    "N:non acceptation": "ders_n_non_acceptance",
    "G:goals": "ders_g_goals",
    "I:impulse": "ders_i_impulse",
    "A:awarness": "ders_a_awareness",
    "S:strategies": "ders_s_strategies",
    "C:clarity": "ders_c_clarity"
}
DERS_SUBSCALE_COLS_CLEAN = list(DERS_SUBSCALE_COL_MAPPING_ORIGINAL_TO_CLEAN.values())

TOTAL_RESILIENCE_CALC = "total_resilience_calc"
TOTAL_ASIR_CALC = "total_asir_calc"
TOTAL_DERS_CALC = "total_ders_calc"

COLS_FOR_DESCRIPTIVE = [TOTAL_RESILIENCE_CALC, TOTAL_ASIR_CALC, TOTAL_DERS_CALC]
COLS_FOR_CORR_MATRIX = COLS_FOR_DESCRIPTIVE + DERS_SUBSCALE_COLS_CLEAN

CORRELATION_PAIRS_SPEARMAN = [
    (TOTAL_RESILIENCE_CALC, TOTAL_ASIR_CALC),
    (TOTAL_RESILIENCE_CALC, TOTAL_DERS_CALC),
    (TOTAL_ASIR_CALC, TOTAL_DERS_CALC)
]
REGRESSION_PAIRS = [
    (TOTAL_DERS_CALC, TOTAL_RESILIENCE_CALC), (TOTAL_RESILIENCE_CALC, TOTAL_DERS_CALC),
    (TOTAL_DERS_CALC, TOTAL_ASIR_CALC), (TOTAL_ASIR_CALC, TOTAL_DERS_CALC),
    (TOTAL_ASIR_CALC, TOTAL_RESILIENCE_CALC), (TOTAL_RESILIENCE_CALC, TOTAL_ASIR_CALC)
]
ROW_DELETION_SIGNIFICANCE_THRESHOLD = 0.10