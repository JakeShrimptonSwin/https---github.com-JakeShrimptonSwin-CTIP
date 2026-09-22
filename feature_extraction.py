import os
import pandas
import numpy
import re
import matplotlib.pyplot as plot
import math
from collections import Counter
from urllib.parse import urlparse





def extract_features(df):

    ## --------------------Feature Extraction--------------------
    ## Create collumns for extracting features of the URLs

    # --------------------URL Length--------------------
    df['url_length'] = df['url'].str.len().astype(int)

    #NOTE No real correlation here.
    # avg_url_length = df[df['label'].isin(['phishing', 'benign'])].groupby('label')['url_length'].mean()

    # plot.figure()
    # avg_url_length.plot(kind='bar')
    # plot.title('Average URL Length: Phishing vs Benign')
    # plot.xlabel('Type')
    # plot.ylabel('Average URL Length')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------Domain Length--------------------
    def get_domain_length(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc
            return len(domain)
        except (ValueError, AttributeError):
            return 0

    df['domain_length'] = df['url'].apply(get_domain_length)

    #NOTE - anything with a domain length of 0 is 100% Phishing
    #NOTE - Phishing is averaged as a higher value here.
    # avg_domain_length = df[df['label'].isin(['phishing', 'benign'])].groupby('label')['domain_length'].mean()

    # plot.figure()
    # avg_domain_length.plot(kind='bar')
    # plot.title('Average Domain Length: Phishing vs Benign')
    # plot.xlabel('Type')
    # plot.ylabel('Average Domain Length')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------letter count--------------------
    df['letter_count'] = df['url'].str.count(r'[a-zA-Z]')

    # plot.figure(figsize=(10, 6))
    # types = df['label'].unique()
    # data = [df[df['label'] == t]['letter_count'] for t in types]
    # plot.hist(data, bins=20, label=types)
    # plot.title('Letter Count Distribution by Type')
    # plot.xlabel('Letter Count')
    # plot.ylabel('Count')
    # plot.legend(title='Type')
    # plot.show()


    # --------------------Number count--------------------
    df['number_count'] = df['url'].str.count(r'\d')

    # plot.figure(figsize=(10, 6))
    # types = df['label'].unique()
    # data = [df[df['label'] == t]['number_count'] for t in types]
    # plot.hist(data, bins=20, label=types)
    # plot.title('Number Count Distribution by Type')
    # plot.xlabel('Number Count')
    # plot.ylabel('Count')
    # plot.legend(title='Type')
    # plot.show()

    # --------------------Special Character count--------------------
    df['special_count'] = df['url'].str.count(r'[^a-zA-Z0-9]')

    # plot.figure(figsize=(10, 6))
    # types = df['label'].unique()
    # data = [df[df['label'] == t]['special_count'] for t in types]
    # plot.hist(data, bins=20, label=types)
    # plot.title('Special Character Count Distribution by Type')
    # plot.xlabel('Special Character Count')
    # plot.ylabel('Count')
    # plot.legend(title='Type')
    # plot.show()

    # --------------------does the url have https ? the :// is to check against things like 'www.httpsabd.com--------------------
    df['has_https'] = df['url'].str.startswith('https://').astype(int)

    # This shows that HTTPS is not a good indicator of legitimacy.
    # pandas.crosstab(df['has_https'], df['label']).plot(kind='bar')
    # plot.title('HTTPS Usage by Label')
    # plot.xlabel('Has HTTPS (0 = No, 1 = Yes)')
    # plot.ylabel('Count')
    # plot.legend(title='Label')
    # plot.xticks(rotation=0)
    # plot.show()


    # --------------------Does the Url contain a onedrive link?--------------------
    #This is a more advanced check for other potential links to oneDrive 
    df['has_one_drive_ext'] = df['url'].str.contains(r'onedrive\.live\.com|1drv\.ms|.*-my\.sharepoint\.com', case=False, na=False).astype(int)
    #NOTE This has a 100% detection rate?
    # onedrive_df = df[df['has_one_drive_ext'] == 1]
    # type_counts = onedrive_df['label'].value_counts()
    # plot.figure()
    # type_counts.plot(kind='bar')
    # plot.title('Type Breakdown for OneDrive URLs')
    # plot.xlabel('Type')
    # plot.ylabel('Count')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------Does the Url contain a Google Drive link?--------------------
    df['has_google_drive'] = df['url'].str.contains(
        r'drive\.google\.com|docs\.google\.com',
        case=False, na=False, regex=True
    ).astype(int)

    # NOTE Not as clearcut as onedrive, but still suggests phishing is hosted in GoogleDrives.
    googledrive_df = df[df['has_google_drive'] == 1]
    # type_counts = googledrive_df['label'].value_counts()
    # plot.figure()
    # type_counts.plot(kind='bar')
    # plot.title('Type Breakdown for Google Drive URLs')
    # plot.xlabel('Type')
    # plot.ylabel('Count')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------Is the domain an IP address?--------------------
    #Regex for an Ip adress, including Https.
    df['domain_is_ip'] = df['url'].str.contains(
        r'^(?:https?://)?(\d{1,3}\.){3}\d{1,3}',
        regex=True, na=False
    ).astype(int)

    #Visualise - NOTE this shows a high likelihood of IP adresses as domains as phishing.
    # ip_df = df[df['domain_is_ip'] == 1]
    # type_counts = ip_df['label'].value_counts()

    # plot.figure()
    # type_counts.plot(kind='bar')
    # plot.title('Type Breakdown for URLs with IP Address as Domain')
    # plot.xlabel('Type')
    # plot.ylabel('Count')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------Does the URL use @ Obfuscating--------------------

    df['has_at_symbol'] = df['url'].str.contains('@', na=False).astype(int)

    ip_df = df[df['has_at_symbol'] == 1]
    #type_counts = ip_df['label'].value_counts()

    # # NOTE Phishing is more likely to have an @, but not by much
    # plot.figure()
    # type_counts.plot(kind='bar')
    # plot.title('Type Breakdown for URLs with @ Symbol')
    # plot.xlabel('Type')
    # plot.ylabel('Count')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------Shannon Entropy--------------------
    def calculate_entropy(text):
        if len(text) == 0:
            return 0
        counts = Counter(text)
        length = len(text)
        probabilities = [count / length for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy

    df['url_entropy'] = df['url'].apply(calculate_entropy)

    # NOTE This may not be as helpful as it looks
    # avg_entropy = df[df['label'].isin(['phishing', 'benign'])].groupby('label')['url_entropy'].mean()
    # plot.figure()
    # avg_entropy.plot(kind='bar')
    # plot.title('Average URL Entropy: Phishing vs Benign')
    # plot.xlabel('Type')
    # plot.ylabel('Average Entropy')
    # plot.xticks(rotation=0)
    # plot.show()

    # --------------------Write new file--------------------
    print(df.head())
    df.to_csv('malicious_phish_updated.csv', index=False)
    return df