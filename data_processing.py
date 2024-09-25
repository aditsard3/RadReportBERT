"""
-------------------------------
PREPARE THE REPORT DATASET
-------------------------------

Data comes in as a large csv with a bunch of columns. All we care about
is the report, the selected conditions, and the labels.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split

CONDITIONS = ['Cardiomegaly', 'Edema', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']
LABELS = {0.0:'Negative', 1.0:'Positive', 2.0:'Absent'}
NUM_CONDITIONS = len(CONDITIONS)
NUM_LABELS = len(LABELS)

def clean_report(report):
    '''
    Strip whitespace and make the report lowercase.
    '''
    report = report.lower() # convert to lowercase
    report = report.strip() # remove whitespace
    report = re.sub('\n',' ', report) # replace new line char with space
    report = re.sub(r'\s+', ' ', report) # remove multiple spaces with a single space
    report = report.strip()
    return report

def prep_df(df):
    '''
    - Keep only the reports and the condtions we care about.
    - Convert NaN values to 2.0 (condition is absent from the report)
    '''

    df = df[['selection'] + CONDITIONS]
    df[CONDITIONS] = df[CONDITIONS].fillna(2.0)
    return df

def main():
    data = pd.read_csv('data/radreports.csv') # Read whole train csv

    # Process reports
    data['selection'] = data['selection'].apply(clean_report) # Clean reports
    data['selection'].to_csv('data/reports.csv', index=False) # Save reports

    # Prep df for analysis
    data = prep_df(data)
    data.to_csv('data/radreports_clean.csv', index=False)
    print(f'Our full dataset contains {data.shape[0]} reports.')
    print(data.head())

    # Split data into train/test set
    train, test = train_test_split(data, test_size=0.2, random_state=3)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

    # Let's take 5000 reports just to experiment with
    _, experiment_set = train_test_split(train, test_size=5000, random_state=3)
    experiment_set.to_csv('data/experiment_set.csv', index=False)

    print(f'Our train dataset contains {train.shape[0]} reports.')
    print(f'Our test dataset contains {test.shape[0]} reports.')
    print(f'Our experimental dataset contains {experiment_set.shape[0]} reports from the trainset.')


if __name__ == "__main__":
    main()