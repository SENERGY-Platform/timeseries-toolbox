import numpy as np
import pandas as pd 

def random_dates(start, end, n=10):
    start_u = start.value//10**9
    end_u = end.value//10**9

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

def random_series():
    start = pd.to_datetime('2015-01-01')
    end = pd.to_datetime('2015-01-03')
    n = 50 
    ts = random_dates(start, end, n)
    data = pd.Series(np.random.randint(0, 10, n), index=ts)
    return data

def random_df():
    start = pd.to_datetime('2015-01-01')
    end = pd.to_datetime('2015-01-03')
    n = 50 
    ts = random_dates(start, end, n)
    data = pd.DataFrame({"value": np.random.randint(0, 10, n), "time": ts})
    data = data.astype({'value': 'float64'})
    return data