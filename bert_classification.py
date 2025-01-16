from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # Or any other BERT variant
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # 2 for binary classification

# Sample data (replace with your data)
texts = ["This movie is great!", "This movie is terrible."]
labels = [1, 0]  # 1 for positive, 0 for negative

# Tokenize and encode input
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    encoded_texts['input_ids'], torch.tensor(labels), test_size=0.2
)
train_masks, val_masks, _, _ = train_test_split(
    encoded_texts['attention_mask'], torch.tensor(labels), test_size=0.2
)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in range(3):  # Example number of epochs
    optimizer.zero_grad()
    outputs = model(train_texts, attention_mask=train_masks, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


#Example of prediction
model.eval()
with torch.no_grad():
  outputs = model(val_texts, attention_mask=val_masks)
  predictions = torch.argmax(outputs.logits, dim=-1)
  print(predictions)