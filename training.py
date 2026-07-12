import pandas as pd
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("Dataset/Final_dataset/train.csv")

# Remove rows having missing text
data = data.dropna(subset=["text"])

# Reset indexing
data = data.reset_index(drop=True)

# Convert datatype
data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(int)

print("Dataset Loaded Successfully!")
print("Total Samples:", len(data))

# ==========================
# Tokenizer
# ==========================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Tokenizer Loaded Successfully!")

# ==========================
# Dataset Class
# ==========================

class SpamDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        text = self.data.iloc[index]["text"]
        label = self.data.iloc[index]["label"]

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ==========================
# DataLoader
# ==========================

dataset = SpamDataset(data)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("DataLoader Created!")

# ==========================
# Load Model
# ==========================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

print("Model Loaded!")

# ==========================
# Optimizer
# ==========================

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

# ==========================
# Training
# ==========================

model.train()

print("\nTraining Started...\n")

for batch in loader:

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )

    loss = outputs.loss

    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()

print("\nTraining Finished!")

# ==========================
# Save Model
# ==========================

model.save_pretrained("Models/SpamClassifier")
tokenizer.save_pretrained("Models/SpamClassifier")

print("\nModel Saved Successfully!")
