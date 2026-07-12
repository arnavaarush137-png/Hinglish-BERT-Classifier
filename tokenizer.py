import pandas as pd

from transformers import BertTokenizer

data = pd.read_csv(
    "Dataset/Final_dataset/train.csv"
)

print("Dataset Loaded Successfully!")

print(data.head())

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

print("\nTokenizer Loaded Successfully!")

sample = data['text'][0]

print("\nOriginal Sentence:\n")

print(sample)

encoded = tokenizer(
    sample,
    padding="max_length",
    truncation=True,
    max_length=32,
    return_tensors="pt"
)
print("\nInput IDs:\n")

print(encoded['input_ids'])

print("\nAttention Mask:\n") 

print(encoded['attention_mask'])
