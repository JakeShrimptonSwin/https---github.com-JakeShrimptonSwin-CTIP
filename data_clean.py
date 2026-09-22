import pandas


def merge_datasets(df_1, df_2):
    df = pandas.concat([df_1, df_2], ignore_index=True)
    print('Cleaning Merged Dataset:')
    df = clean_dataset_2(df)
    print(f'Final dataset size: {len(df)} rows')

    return df

def clean_dataset_2(df):
    ## --------------------data transformations--------------------
    #This is specific to the new dataset
    # We want to remove all existing features as we want to use our own feature extraction
    df = df[['url', 'label']]
    
    # remove rows with missing url values
    rows_before = len(df)
    df = df.dropna(subset=['url'])
    rows_after = len(df)
    print(f'Removed {rows_before - rows_after} rows with missing url values')

    # remove rows with missing label values
    rows_before = len(df)
    df = df.dropna(subset=['label'])
    rows_after = len(df)

    print(f'Removed {rows_before - rows_after} rows with missing label values')

    # Remove duplicates
    rows_before = len(df)
    df = df.drop_duplicates(subset=['url'])
    rows_after = len(df)

    print(f'Removed {rows_before - rows_after} rows with duplicate URLs values')
    
    return df

def clean_dataset_1(df):
    ## --------------------data transformations--------------------
    ## This is specific cleaning actions for the original dataset
    # remove rows with missing url values
    rows_before = len(df)
    df = df.dropna(subset=['url'])
    rows_after = len(df)

    print(f'Removed {rows_before - rows_after} rows with missing url values')

    # remove rows with missing type values
    rows_before = len(df)
    df = df.dropna(subset=['type'])
    rows_after = len(df)

    print(f'Removed {rows_before - rows_after} rows with missing type values')

    # Remove duplicates
    rows_before = len(df)
    df = df.drop_duplicates(subset=['url'])
    rows_after = len(df)

    print(f'Removed {rows_before - rows_after} rows with duplicate URLs values')

    #Lets Assume that anything that is malware or defacement is not what we are detecting #NOTE ASK ABOUT THIS
    rows_before = len(df)
    df = df[~df['type'].isin(['defacement', 'malware'])]
    rows_after = len(df)

    print(f'Removed {rows_before - rows_after} defacement/malware rows')

    #NEW - Change phishing and benign to 1 or 0 to match other dataset
    df = df[['url', 'type']].rename(columns={'type': 'label'})
    df['label'] = df['label'].map({'phishing': 1, 'benign': 0})


    return df


