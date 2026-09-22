#lib
import pandas
#ours
import data_clean
import feature_extraction

DATA_PATH = r'malicious_phish.csv'
DATA_PATH_2 = r'final_dataset.csv'

## --------------------load, Clean and merge data--------------------
## More data https://www.kaggle.com/datasets/elifzelik/phishing-url-features-dataset/data?select=feature_description.csv
## Reference this - https://www.sciencedirect.com/science/article/pii/S2352340925008832#fig0005

#load Datas
df_1 = pandas.read_csv(DATA_PATH)
if df_1 is not None:
    print(f'Loaded Data 1: {len(df_1)} rows')

df_2 = pandas.read_csv(DATA_PATH_2)
if df_2 is not None:
    print(f'Loaded Data 2: {len(df_2)} rows')

#clean data
print('Cleaning dataset 1:')
df_1 = data_clean.clean_dataset_1(df_1)
print('Cleaning dataset 2:')
df_2 = data_clean.clean_dataset_2(df_2)

#Merge Data
df_final = data_clean.merge_datasets(df_1, df_2)

#Summarise data
counts = df_final['label'].value_counts()
percentages = df_final['label'].value_counts(normalize=True) * 100
summary = pandas.DataFrame({'count': counts, 'percentage': percentages.round(2)})
print(summary)
print('\n')

## --------------------Extract Features--------------------
df_final = feature_extraction.extract_features(df_final)