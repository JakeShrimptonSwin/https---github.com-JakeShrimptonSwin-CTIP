import os
import pandas
import numpy
import re
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

## More data https://www.kaggle.com/datasets/elifzelik/phishing-url-features-dataset/data?select=feature_description.csv

DATA_PATH = r'C:\Users\jakes\Desktop\CTIPTest\malicious_phish.csv'
df = pandas.read_csv(DATA_PATH)

print(df.head())
print(len(df))

## --------------------data transformations--------------------

# remove rows with missing values
rows_before = len(df)
df = df.dropna(subset=['url'])
rows_after = len(df)

print(f'Removed {rows_before - rows_after} rows with missing url values')

rows_before = len(df)
df = df.dropna(subset=['type'])
rows_after = len(df)

print(f'Removed {rows_before - rows_after} rows with missing type values')

# Remove duplicates
rows_before = len(df)
df = df.drop_duplicates(subset=['url'])
rows_after = len(df)

print(f'Removed {rows_before - rows_after} rows with duplicate URLs values')


## --------------------Feature Extraction--------------------
## Create collumns for extracting features of the URLs

# ----------does the url have https ? the :// is to check against things like 'httpsabd.com'----------
df['has_https'] = df['url'].str.startswith('https://').astype(int)

# This shows that HTTPS is not a good indicator of legitimacy.
# pandas.crosstab(df['has_https'], df['type']).plot(kind='bar')
# plot.title('HTTPS Usage by Label')
# plot.xlabel('Has HTTPS (0 = No, 1 = Yes)')
# plot.ylabel('Count')
# plot.legend(title='Label')
# plot.xticks(rotation=0)
# plot.show()


# ----------URL Length----------
df['url_length'] = df['url'].str.len().astype(int)


# ----------Does the Url contain a onedrive link?----------
df['has_one_drive'] = df['url'].str.contains('one.drive.live.com', case=False, na=False).astype(int)
df['has_one_drive_ext'] = df['url'].str.contains(r'onedrive\.live\.com|1drv\.ms|.*-my\.sharepoint\.com', case=False, na=False).astype(int)


# pandas.crosstab(df['has_one_drive'], df['type']).plot(kind='bar')
# plot.title('oneDrive use by Label')
# plot.xlabel('Has oneDrive (0 = No, 1 = Yes)')
# plot.ylabel('Count')
# plot.legend(title='Label')
# plot.xticks(rotation=0)
# plot.show()

# This was found to have no correlation at all
# pandas.crosstab(df['has_one_drive_ext'], df['type']).plot(kind='bar')
# plot.title('oneDrive use by Label')
# plot.xlabel('Has oneDrive (0 = No, 1 = Yes)')
# plot.ylabel('Count')
# plot.legend(title='Label')
# plot.xticks(rotation=0)
# plot.show()


# --------------------Write new file--------------------
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(DATA_PATH)), 'malicious_phish_updated.csv')
df.to_csv(OUTPUT_PATH, index=False)
df.to_csv('malicious_phish_updated.csv', index=False)