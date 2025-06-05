import os
import pandas as pd

def list_csv_files(folder="data"):
    return [f for f in os.listdir(folder) if f.endswith(".csv")]

def load_csv(file_path):
    return pd.read_csv(file_path)
