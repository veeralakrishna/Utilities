import pandas as pd
import os
import time
import sys
import psutil
import requests
from io import StringIO

def smart_load_data(file_path_or_dict):
    """
    Smartly loads data from various file formats including CSV, Parquet, Excel, TSV, JSON,
    and from dictionary variables.

    :param file_path_or_dict: Path to the data file or dictionary variable
    :return: Pandas DataFrame containing the loaded data
    """
    start_time = time.time()

    # Check if the input is a dictionary variable
    if isinstance(file_path_or_dict, dict):
        data = pd.DataFrame.from_dict(file_path_or_dict)
        file_size = sys.getsizeof(file_path_or_dict)
        print(f"Data loaded from dictionary variable (Size: {file_size} bytes)")
    else:
        # Check if the input is a URL
        if file_path_or_dict.startswith('http'):
            response = requests.get(file_path_or_dict)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch data from URL: {file_path_or_dict}")
            content = response.text
            data = pd.read_csv(StringIO(content))
            print(f"Data loaded from URL: {file_path_or_dict}")
        else:
            # Check if the input is a valid file path
            if not os.path.exists(file_path_or_dict):
                raise FileNotFoundError(f"File not found: {file_path_or_dict}")

            # Determine the file format based on the file extension
            _, file_extension = os.path.splitext(file_path_or_dict)
            file_extension = file_extension[1:].lower()  # Remove the leading dot

            # Load data based on the file format
            if file_extension == 'csv':
                data = pd.read_csv(file_path_or_dict)
            elif file_extension == 'parquet':
                data = pd.read_parquet(file_path_or_dict)
            elif file_extension in ['xls', 'xlsx']:
                data = pd.read_excel(file_path_or_dict)
            elif file_extension == 'tsv':
                data = pd.read_csv(file_path_or_dict, sep='\t')
            elif file_extension == 'json':
                data = pd.read_json(file_path_or_dict)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            print(f"Data loaded from file '{file_path_or_dict}'")

    end_time = time.time()
    print(f"Time taken to load data: {end_time - start_time:.2f} seconds")

    # Additional information
    total_memory_usage = psutil.virtual_memory().used
    print(f"Total memory usage: {total_memory_usage / (1024 * 1024)} MB")


    return data



# Example usage:
# file_path = 'https://raw.githubusercontent.com/veeralakrishna/DataCamp-Project-Solutions-Python/master/A%20Network%20analysis%20of%20Game%20of%20Thrones/datasets/book1.csv'
# data = smart_load_data(file_path)
# print(data.head())
