"""
-------------------------------
TOKENIZE RAD REPORTS
-------------------------------

Convert report text to tokenized vectors using BERT tokenizers
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from transformers import AutoTokenizer

def tokenize_reports(data, pretrained='bert-base-uncased', max_length=512):
    # Load pre-trained tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained)
    
    # Tokenize reports
    reports = list(data['selection'])
    tokenized = bert_tokenizer(reports, truncation=True, padding=True, max_length=max_length)
    return tokenized

def main():
    # Load data
    data = pd.read_csv('data/experiment_set.csv')

    # Tokenize reports
    t0 = time.time()
    tokenized = tokenize_reports(data)
    t1 = time.time()

    print(f'Took {(t1-t0):.2f} seconds to tokenize {data.shape[0]} reports.')

if __name__ == "__main__":
    main()