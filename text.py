#!/usr/bin/env python
# coding: utf-8

import pandas as pd
data = pd.read_csv('/Users/jiteshdewangan/Downloads/training.tsv', sep='\t')
df = data.sample(frac=.01, random_state=50)
df.shape
df.isna().sum()
df.dropna(inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df['text'] = df['title'] + " " + df['description']

df['tokens'] = df['text'].apply(lambda x: x.lower().split())
X = df['tokens']
y = df['category']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# STOP OWRLD removal from tkt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]
X_train = X_train.apply(remove_stopwords)
X_test = X_test.apply(remove_stopwords)


## text vectorization  
from gensim.models import Word2Vec
tokenizedtext = X_train.tolist()
word2vec_model = Word2Vec(sentences=tokenizedtext,
                           vector_size=100, window=5, min_count=1, workers=4)

# Function to convert tokens to vectors
import numpy as np
def tokens_to_vectors(tokens, model, vector_dim=100, max_length=20):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    # Pad or truncate to fixed length
    if len(vectors) < max_length:
        vectors += [np.zeros(vector_dim)] * (max_length - len(vectors))
    else:
        vectors = vectors[:max_length]
    return np.array(vectors)

max_length = 20  # Maximum sequence length
train_data = X_train.apply(lambda x: tokens_to_vectors(x,word2vec_model, max_length=max_length))
test_data = X_test.apply(lambda x: tokens_to_vectors(x,word2vec_model, max_length=max_length))

train_data.sample(5)


#Target encoding
from sklearn import preprocessing
l = preprocessing.LabelEncoder()
y_train = l.fit_transform(y_train)
y_test = l.transform(y_test)



## Pytorch implementation 
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

class TextDataset(Dataset):
    def __init__(self, vectors, targets):
        self.vectors = torch.tensor(np.stack(vectors), dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.vectors[idx], self.targets[idx]

# Create datasets and dataloaders
train_dataset = TextDataset(train_data, y_train)
test_dataset = TextDataset(test_data, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Model configuration
input_dim = 100  # Embedding dimension
hidden_dim = 64  # LSTM hidden layer size
output_dim = 2   # Number of classes
num_epochs = 1
learning_rate = 0.001

# Initialize model, loss, and optimizer
model = LSTMClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for vectors, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(vectors)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for vectors, targets in test_loader:
        outputs = model(vectors)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")





