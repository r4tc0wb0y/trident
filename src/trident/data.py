import pandas as pd
from pathlib import Path
from . import config

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    
    Args:
        file_path (str, optional): Custom path to the CSV. 
                                   Defaults to DATA_DIR/raw/network_data.csv.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Use the default path from config if none is provided
    if file_path is None:
        file_path = config.DATA_DIR / "raw" / "network_data.csv"
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
        
    # print(f"Loading data from {path}...") 
    # (Commented out print because we use logging in train.py)
    
    return pd.read_csv(path)