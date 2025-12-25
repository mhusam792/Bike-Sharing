import pandas as pd

def load_dataframe(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
