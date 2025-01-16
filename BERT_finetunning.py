
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = [
    ("I love this product!", 1),
    ("This is terrible.", 0),
    ("Absolutely fantastic!", 1),
    ("Not worth the money.", 0),
    ("Really great experience.", 1),
    ("Awful, never again.", 0)
]

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(data, columns=["text", "label"])

data = pd.read_csv('/Users/jiteshdewangan/Downloads/training.tsv', sep='\t')
df = data.sample(frac=.01, random_state=50)

df['target'] = df['category'].apply(lambda x: 0 if x == 'S' else 1)
df['input'] = df['title'] + " " + df['description']

x_train = df['input']
y_train = df['target']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

x_train, x_test, y_train, y_test = train_test_split(x_train.tolist(), y_train.tolist(), test_size=0.1, random_state=42)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.texts)}")
        
        text = self.texts[idx]
        label = self.labels[idx]
        #print(f"Fetching index: {idx}")
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


train_dataset = TextDataset(x_train, y_train, tokenizer)
test_dataset = TextDataset(x_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = AdamW(model.parameters(), lr=0.0001)

def train(model, train_loader, optimizer,device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_losss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(input_ids, attention_mask=attention_mask, labels= labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_losss += loss.item()
        

    print(f"Epoch {epoch+1}, Loss: {total_losss/len(train_loader):.4f}")

train(model, train_loader, optimizer, "cpu", 2)

def evaluate_model(model, test_loader):
    model.eval()
    predictions , true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)

    return accuracy