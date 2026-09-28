import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

DATA_FOLDER = 'data'
FILE_NAME_1 = 'malicious_phish.csv'
FILE_NAME_2 = 'final_dataset.csv'

def download_data():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    #Data Set 1
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 'sid321axn/malicious-urls-dataset', FILE_NAME_1,pandas_kwargs={'encoding': 'latin-1'})
    df.to_csv('data/malicious_phish.csv', index=False)
    
    #Data Set 2
    df_2 = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "elifzelik/phishing-url-features-dataset", FILE_NAME_2, pandas_kwargs={'encoding': 'latin-1'})
    df_2.to_csv('data/final_dataset.csv')
    return df 

