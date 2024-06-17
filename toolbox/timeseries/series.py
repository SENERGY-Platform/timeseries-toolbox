import pandas as pd 

def convert_df_to_series(df: pd.DataFrame):
    df = df.set_index("time")
    return df["value"]