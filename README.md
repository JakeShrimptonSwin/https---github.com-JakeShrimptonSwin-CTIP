# Phishing Detection System Guide

This README provides a guide to using the Phishing detection system in it's current form.

## Install Requirements
__This list details the required installations for this project to work:__

* Python 3.14 or later (https://www.python.org/downloads/)
* Pandas ( pip install pandas)
* MatplotLib ( pip install matplotlib )
* Numpy ( pip install numpy )
* Tensorflow ( pip install tensorflow )
* Scikit-lean ( pip install scikit-learn )
* XGBoost ( pip install xgboost )
* Kagglehub ( pip install kagglehub )

## Running the code
__The following steps explain how to run this code once all required installs have been completed.__

1. Extract all files contained in the ZIP file to a folder on your computer.

2. Open the folder, locate the file 'main.py' and double click it. This will start the System.
    * Alternativley, use the CMD terminal and navigate to the new folder. Then, type __'python main.py'__.
    * If using a Powershell terminal, navigate to the new folder and type __'python main.py'__

3. The program will ask to download data from sources. This step downloads the data from Kaggle.com. if the require datasets exist already (final_dataset.csv & malicious_phish.csv), this step can be skipped by typing __'n'__. Otherwise, type __'y'__ to complete this step.

4. The next step is cleaning and merging the datasets. This step cleans both downloaded datasets, transforms them so that they are able to be merged, and then does so. After this, it then extracts the features needed for identifying Phishing. The full list of these features can be seen in the attatched report. Afterwards, the a new dataset callled 'malicious_phish_updated.csv' is created, which is used for the later steps. If this file exists already, this step can be skipped by typing __'n'__. Otherwise, type __'y'__ to complete this step.

5.  The next step uses visualisations to explore the importance of the extracted features. This step is not required for the models to work, but shows important insights into how important certain features are, and the correlation between them. Type __'y'__ or __'n'__ depending on your choice.

6. The next step asks to train and evaluate the models. This step runs through all models, and visualises the output and results for each of the models. Type __'y'__ or __'n'__ depending on your choice. Typing __'n'__ will end the program.


## Additional Dataset
And additional, non-standard dataset was included to act as a proof-of concept of converting alternative data types into usable data for this system. 

The dataset has to be downloaded from https://www.kaggle.com/datasets/aaronloera/phishing-qr-codes/data?select=qr_images 

This dataset is too big for GitHub and Canvas, so it needs to be downloaded manually and placed into the __'data'__ folder. 

The following steps outline how to run qr_test.py, which is separate from the main system due to time constraints.

1. Download the dataset containing QR codes from the link above. Place the folder __'qr_images'__ into the __'data'__ folder. 

2. Navigate to the folder containing this file using the file explorer. Then double click __'qr_test.py'__ to start the example.

3. Alternatively, navigate to the folder using the CMD terminal, and type __'python qr_test.py'.__

__NOTE__: This file is a standalone system and does not affect the main.py. It is a proof of concept that QR codes can be converted into URLs, which can have their features extracted and then used for training the model. This may provide some utility in the next phase of the project.