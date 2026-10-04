import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas


DATA_FOLDER = 'data'
FILE_NAME_1 = 'malicious_phish.csv'
FILE_NAME_2 = 'final_dataset.csv'

def download_data():
    # Creates  folder for the data, forces the download and creates a new CSV file with the data in it.
    os.makedirs(DATA_FOLDER, exist_ok=True)

    #Data Set 1
    download_1 = kagglehub.dataset_download('sid321axn/malicious-urls-dataset', force_download=True)
    file_1 = os.path.join(download_1, FILE_NAME_1)
    df_1 = pandas.read_csv(file_1, encoding="latin-1")
    df_1.to_csv(os.path.join(DATA_FOLDER, FILE_NAME_1), index=False)
    
    #Data Set 2
    download_2 = kagglehub.dataset_download("elifzelik/phishing-url-features-dataset", force_download=True)
    file_2 = os.path.join(download_2, FILE_NAME_2)
    df_2 = pandas.read_csv(file_2, encoding="latin-1")
    df_2.to_csv(os.path.join(DATA_FOLDER, FILE_NAME_2), index=False)
    return

