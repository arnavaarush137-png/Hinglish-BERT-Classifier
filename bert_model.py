import torch

from transformers import BertTokenizer, BertForSequenceClassification

print("Loading BERT Tokenizer...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Tokenizer Loaded Successfully!\n")

print("Loading BERT Model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

#print("BERT Model Loaded Successfully!")
sample = "Congratulations! You have won Rs.5000. Click here to claim your prize."
inputs = tokenizer(
    sample,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
with torch.no_grad():

    outputs = model(**inputs)
    print("\nModel Output:\n")

print(outputs.logits)