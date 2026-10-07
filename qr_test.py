# Jake Shrimpton
# This file is a test of the potential QR code to URL conversion that was planned to be inlcuded.
# images must be downloaded here https://www.kaggle.com/datasets/aaronloera/phishing-qr-codes/data?select=qr_images
# The folder must be extracted to 'data' such that this file can access it using 'data/qr_images'.
# The file size is too big for github or Canvas  

import cv2
import os
import pandas
import data_clean
DATA_FOLDER = 'data'
FILE_PATH = "data/qr_images"
OUTPUT_CSV = "data/qr_urls.csv"
LABEL_CSV = "data/qr_labels.csv"

qr_detector = cv2.QRCodeDetector()
results = []

def read_qr():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    for filename in os.listdir(FILE_PATH):
        file_path = os.path.join(FILE_PATH, filename)

            # Skip directories to remove error codes
        if not os.path.isfile(file_path):
            continue

        # Skip non-image files to remove error codes
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            continue
        # Read image
        image = cv2.imread(file_path)

        if image is None:
            print(f"{filename}: Could not read image")
            continue

        # Decode QR code
        url, points, _ = qr_detector.detectAndDecode(image)

        if url:
                #print to show it is workng
                print(url)
                results.append({
                    "url": url
                })
        else:
            results.append({
                "url": None
            })

    # Create DataFrame
    df = pandas.DataFrame(results)

    # Save to CSV
    #first pass
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} results to {OUTPUT_CSV}")
    #append labels which i found later
    urls = pandas.read_csv(OUTPUT_CSV)
    labels = pandas.read_csv(LABEL_CSV)
    urls["label"] = labels["label"].values
    #remove anything else 
    df = urls[["url", "label"]]
    #run a clean of the dataset
    df = data_clean.clean_dataset_2(df)
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    #this would then get joined to the malicious_phish_updated.csv but has been excluded as to not change results show in reports.
    print(df)

read_qr()