import zipfile
import os
import pandas as pd

def unzip_file(zip_file, extract_to="data"):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def list_csv_files(directory="data"):
    return [f for f in os.listdir(directory) if f.endswith(".csv")]

def load_csv(file_path):
    return pd.read_csv(file_path)