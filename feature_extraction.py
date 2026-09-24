import os
import pandas
import numpy
import re
import matplotlib.pyplot as plot
import math
from collections import Counter
from urllib.parse import urlparse
import base64

#NOTE - I removed all visualisations as i will create a new file that will handle all visualsations later.

def extract_features(df):

## --------------------Feature Extraction--------------------
    ## Create collumns for extracting features of the URLs

#create new collumn to house found tlds
    def get_tld(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc
    
            # Strip a port number if present, e.g. 'example.com:8080' -> 'example.com'
            domain = domain.split(':')[0]
    
            if '.' not in domain:
                return ''
    
            return domain.rsplit('.', 1)[-1].lower()
        except (ValueError, AttributeError):
            return ''
        
    def is_suspicious_tld(tld):
        return tld in SUSPICIOUS_TLDS
    
    df['tld'] = df['url'].apply(get_tld)
# --------------------URL Length--------------------
    #NOTE No real correlation here.
    df['url_length'] = df['url'].str.len().astype(int)

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
 

# --------------------letter count--------------------
    df['letter_count'] = df['url'].str.count(r'[a-zA-Z]')

# --------------------Number count--------------------
    df['number_count'] = df['url'].str.count(r'\d')

# --------------------Special Character count--------------------
    df['special_count'] = df['url'].str.count(r'[^a-zA-Z0-9]')

# --------------------Lowercase Character count--------------------
    df['lowercase_count'] = df['url'].str.count(r'[a-z]')

# --------------------Uppercase Character count--------------------
    df['uppercase_count'] = df['url'].str.count(r'[A-Z]')

# --------------------Dot count--------------------
    df['dot_count'] = df['url'].str.count(r'\.')

# --------------------Equals count--------------------
    df['equals_count'] = df['url'].str.count('=')

# --------------------Underscore count--------------------
    df['underscore_count'] = df['url'].str.count('_')

# --------------------Slash count--------------------
    df['slash_count'] = df['url'].str.count('/')

# --------------------domain Number count--------------------
    def get_domain_digit_count(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc.split(':')[0]
            return sum(c.isdigit() for c in domain)
        except (ValueError, AttributeError):
            return 0

    df['domain_digit_count'] = df['url'].apply(get_domain_digit_count)


# --------------------does the url have https ? the :// is to check against things like 'www.httpsabd.com--------------------
    df['has_https'] = df['url'].str.startswith('https://').astype(int)

# --------------------does the url have multiple https ? the :// is to check against things like 'www.httpsabd.com--------------------
    def has_multiple_https(url):
        return int(url.lower().count('https') > 1)

    df['has_multiple_https'] = df['url'].apply(has_multiple_https)
# --------------------http subtstring count--------------------
    df['http_substring_count'] = df['url'].str.lower().str.count('http')
    #print(df.groupby('label')[['http_substring_count']].mean())

# --------------------Does the Url contain a onedrive link?--------------------
    #This is a more advanced check for other potential links to oneDrive 
    df['has_one_drive_ext'] = df['url'].str.contains(r'onedrive\.live\.com|1drv\.ms|.*-my\.sharepoint\.com', case=False, na=False).astype(int)
    #NOTE This has a 100% detection rate?
   
# --------------------Does the Url contain a Google Drive link?--------------------
    # NOTE Not as clearcut as onedrive, but still suggests phishing is hosted in GoogleDrives.
    df['has_google_drive'] = df['url'].str.contains(
        r'drive\.google\.com|docs\.google\.com',
        case=False, na=False, regex=True
    ).astype(int)

# --------------------Is the domain an IP address?--------------------
    #Visualise - NOTE this shows a high likelihood of IP adresses as domains as phishing.
    #Regex for an Ip adress, including Https.
    df['domain_is_ip'] = df['url'].str.contains(
        r'^(?:https?://)?(\d{1,3}\.){3}\d{1,3}',
        regex=True, na=False
    ).astype(int)

# --------------------Does the URL use @ Obfuscating--------------------
    # # NOTE Phishing is more likely to have an @, but not by much

    df['has_at_symbol'] = df['url'].str.contains('@', na=False).astype(int)

    #ip_df = df[df['has_at_symbol'] == 1]
    #type_counts = ip_df['label'].value_counts()

# --------------------Shannon Entropy--------------------
    # NOTE This may not be as helpful as it looks
    def calculate_entropy(text):
        if len(text) == 0:
            return 0
        counts = Counter(text)
        length = len(text)
        probabilities = [count / length for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy

    df['url_entropy'] = df['url'].apply(calculate_entropy)

# --------------------Unusual Top level Domain--------------------
    #This is a list of typically suspicious tlds
    SUSPICIOUS_TLDS = {
        'tk', 'ml', 'ga', 'cf', 'gq',   #Freenom's free TLDs
        'xyz', 'top', 'club', 'info', 'work', 'click', 'link', 'loan',
        'download', 'review', 'country', 'stream', 'bid', 'win',}
    
    df['is_suspicious_tld'] = df['tld'].apply(is_suspicious_tld).astype(int)

    #NOTE This shows tlds are often used in suspicious links
    

# --------------------Suspiscious Keywords--------------------
    #This is a list of sus keywords that may be used to trick people
    SUSPICIOUS_KEYWORDS = [
        'login', 'verify', 'secure', 'account', 'update', 'confirm', 'banking'
    ]

    df['has_suspicious_keyword'] = df['url'].str.contains('|'.join(SUSPICIOUS_KEYWORDS),case=False, na=False, regex=True).astype(int)

    #Add count of suspicious keywords
    df['suspicious_keyword_count'] = df['url'].str.count('|'.join(SUSPICIOUS_KEYWORDS), flags=re.IGNORECASE)

# --------------------non-Ascii character in Domain--------------------
    #This checks for non ascii characters and returns 1 if there is a non ascii character
    def has_non_ascii_domain(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc.split(':')[0]  # strip port if present

            ascii_only = domain.encode('ascii', errors='ignore').decode('ascii')
            return int(len(ascii_only) != len(domain))
        except (ValueError, AttributeError):
            return 0

    df['has_non_ascii_domain'] = df['url'].apply(has_non_ascii_domain)

    #small amount have this, but almost all detected are phishing
    
# --------------------sub-domain Count--------------------
    def get_subdomain_count(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http//' + url
            domain = urlparse(url).netloc.split(':')[0]
            parts = domain.split('.')

            # the last 2 parts are treated as the main domain + TLD; everything before that is subdomains
            return max(len(parts) - 2, 0)
        except (ValueError, AttributeError):
            return 0

    df['subdomain_count'] = df['url'].apply(get_subdomain_count)

# --------------------Path depth--------------------
    def get_path_depth(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            path = urlparse(url).path

            # Split on '/' and drop empty strings (leading/trailing slashes create them)
            segments = [s for s in path.split('/') if s != '']
            return len(segments)
        except (ValueError, AttributeError):
            return 0

    df['path_depth'] = df['url'].apply(get_path_depth)

# --------------------Hyphen Count--------------------

    def get_domain_hyphen_count(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc.split(':')[0]  # strip port if present
            return domain.count('-')
        except (ValueError, AttributeError):
            return 0

    df['domain_hyphen_count'] = df['url'].apply(get_domain_hyphen_count)

# --------------------Query Parameter Count--------------------

    def get_query_param_count(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            query = urlparse(url).query

            if query == '':
                return 0

            return len(query.split('&'))
        except (ValueError, AttributeError):
            return 0

    df['query_param_count'] = df['url'].apply(get_query_param_count)

# --------------------Longest word in the domain--------------------
    def get_longest_domain_word(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc.split(':')[0]
            words = re.split(r'[^a-zA-Z]', domain)
            words = [w for w in words if w != '']
            return max((len(w) for w in words), default=0)
        except (ValueError, AttributeError):
            return 0
    df['longest_domain_word_length'] = df['url'].apply(get_longest_domain_word)
# --------------------Unusual // presence--------------------
    def has_extra_double_slash(url):
        #strip the URLs own '//' first, so we only check what comes after it
        stripped = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
        return int('//' in stripped)

    df['has_extra_double_slash'] = df['url'].apply(has_extra_double_slash)

# --------------------base-64 presence--------------------
    def has_base64_string(url):
        pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        matches = re.findall(pattern, url)

        for match in matches:
            try:
                decoded = base64.b64decode(match + '==')  # padding safety
                decoded.decode('utf-8')  # only counts if it decodes to readable text
                return 1
            except Exception:
                continue
        return 0

    df['has_base64_string'] = df['url'].apply(has_base64_string)

# --------------------Has www?--------------------
    def has_www(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            domain = urlparse(url).netloc.split(':')[0]
            return int(domain.lower().startswith('www.'))
        except (ValueError, AttributeError):
            return 0

    df['has_www'] = df['url'].apply(has_www)

# --------------------Does the URL have a port number?--------------------
    def has_port(url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            port = urlparse(url).port
            return int(port is not None)
        except (ValueError, AttributeError):
            return 0

    df['has_port'] = df['url'].apply(has_port)
   
# --------------------'%' Count--------------------
    df['percent_count'] = df['url'].str.count('%')

# --------------------Ratio of letter characters--------------------
    df['letter_ratio'] = df['letter_count'] / df['url_length']
     
# --------------------Ratio of number characters--------------------
    df['number_ratio'] = df['number_count'] / df['url_length']

# --------------------Ratio of special chracters--------------------
    df['special_ratio'] = df['special_count'] / df['url_length']

# --------------------Ratio of Lowercase--------------------
    df['lowercase_ratio'] = df['lowercase_count'] / df['url_length']

# --------------------Ratio of uppercase--------------------
    df['uppercase_ratio'] = df['uppercase_count'] / df['url_length']

# --------------------Domain to URL length ratio--------------------
    df['domain_url_ratio'] = df['domain_length'] / df['url_length'].replace(0, 1)
## --------------------Write new file--------------------
    print(df.head())
    df.to_csv('malicious_phish_updated.csv', index=False)
    return df