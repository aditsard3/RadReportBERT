"""
-------------------------------
TRAIN BERT MODEL
-------------------------------

Train and evaluate BERT model to classify radiology reports

Multi-class, multi-label classification using BERT architecture
"""

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, BertForSequenceClassification

from data_processing import CONDITIONS, LABELS, NUM_CONDITIONS, NUM_LABELS


# Init pretrained tokenizer and model
def init_model_tokenizer(pretrained='bert-base-uncased', num_labels=NUM_CONDITIONS*NUM_LABELS):
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=num_labels)
    print(f'Loaded {pretrained} pretrained model and tokenizer.')
    return tokenizer, model

# Custon Dataset class
class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

# Create dataloaders
def create_dataloaders(train_encodings, train_labels, test_encodings, test_labels, batch_size=16):
    train_data = BertDataset(train_encodings, train_labels)
    test_data = BertDataset(test_encodings, test_labels)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# Loss function
def calc_loss(logits, labels, criterion):
    logits = logits.view(-1, NUM_CONDITIONS, NUM_LABELS) # Resize preds to batch_size x 5 conditions x 3 labels
    labels = labels.view(-1, NUM_CONDITIONS) # Resize labels to batch_size x 5 conditions
    loss = 0.0

    # Loop through each condition
    for i in range(NUM_CONDITIONS):
        loss += criterion(logits[:, i, :], labels[:, i])

    # Average loss over the number conditions
    return loss / NUM_CONDITIONS

def train_model(model, dataloader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device) # batch_size x tokenized len
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) # batch_size x NUM_CONDITIONS

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits # batch_size x NUM_CONDITIONS * NUM_LABELS (15)

        loss = calc_loss(logits, labels, criterion)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def eval_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device) # batch_size x tokenized len
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # batch_size x NUM_CONDITIONS

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = calc_loss(logits, labels, criterion)
            total_loss += loss.item()
            
            logits = logits.view(-1, NUM_CONDITIONS, NUM_LABELS) # batch_size x NUM_CONDITIONS x NUM_LABELS

            preds = torch.argmax(logits, dim=2).cpu().numpy() # batch_size x NUM_CONDITIONS
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)

    labels = np.array(test_labels).flatten()
    preds = np.array(test_preds).flatten()
    return avg_loss, hamming_loss(labels, preds), test_preds, test_labels

def main():
    # Init model and tokenizer
    bert_tokenizer, bert_model = init_model_tokenizer()

    # Load and split data into train/test set
    data = pd.read_csv('data/experiment_set.csv')
    reports = list(data['selection'])
    labels = data[CONDITIONS].values.tolist()

    # Split data 
    train_rep, test_rep, train_lab, test_lab = train_test_split(reports, labels, test_size=0.2, random_state=3)

    # Tokenize reports
    train_encodings = bert_tokenizer(train_rep, truncation=True, padding=True, max_length=512)
    test_encodings = bert_tokenizer(test_rep, truncation=True, padding=True, max_length=512)

    # Create dataloader
    trainloader, testloader = create_dataloaders(train_encodings, train_lab, test_encodings, test_lab)

    print(f'Train data size: {len(trainloader.dataset)}, Test data size: {len(testloader.dataset)}')

    # Training setup
    device = torch.device("cpu")
    bert_model = bert_model.to(device)
    optimizer = AdamW(bert_model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 3
    min_hamming_loss = 1.0

    # Training loop
    print('Training bert model...')
    t0 = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}')
        train_loss = train_model(bert_model, trainloader, device, optimizer, criterion)
        print(f'Train Loss: {train_loss:.4f}')

        test_loss, test_hl, _, _ = eval_model(bert_model, testloader, device, criterion)
        print(f'Eval Loss: {test_loss:.4f}, Eval Hamming Loss: {test_hl:.4f}')
        
        print(f'Elapsed time: {(time.time() - t0)/60:.2f} minutes')
        print('='*80)

    # Save model and tokenizer
    bert_model.save_pretrained('models/bert_model_experimental')
    bert_tokenizer.save_pretrained('models/bert_tokenizer_experimental')
    print('Model and tokenizer trained and saved successfully.')

if __name__ == "__main__":
    main()