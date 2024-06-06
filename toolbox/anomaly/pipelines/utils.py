import pandas as pd
from dataclasses import dataclass

@dataclass
class Reconstruction():
    timestamp: pd.Timestamp = None 
    reconstruction_error: int = None