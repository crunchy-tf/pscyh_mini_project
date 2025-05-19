import pandas as pd
from pathlib import Path
from src import config

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    Removes trailing empty columns that might result from extra commas in CSV.
    Throws an error if the file is not found or cannot be loaded.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Error: Data file not found at {file_path}. "
                                f"Please ensure '{file_path.name}' is in the '{file_path.parent.name}' directory.")
    try:
        df = pd.read_csv(file_path)
        # Remove columns that are entirely NaN (often due to trailing commas)
        df.dropna(axis=1, how='all', inplace=True)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise IOError(f"Error loading data from {file_path}: {e}")

def save_data_overwrite(df: pd.DataFrame, file_path: Path):
    """
    Saves the DataFrame to a CSV file, OVERWRITING the existing file.
    Throws an error if saving fails.
    """
    try:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Data successfully saved and OVERWRITTEN to: {file_path}")
    except Exception as e:
        raise IOError(f"Error saving data to {file_path}: {e}")