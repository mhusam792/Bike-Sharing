import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.empty or df.columns.size == 0:
            raise ValueError("CSV file is empty or malformed")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {path}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file from {path}: {e}")
