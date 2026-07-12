import pandas as pd
import re

data = pd.read_csv(
    "Dataset/raw_data/SMS_Spam_Collection/spam.csv",
    encoding="latin-1"
)


data = data[['v1', 'v2']]


data.columns = ['label', 'text']

print("First Five Rows After Renaming:\n")
print(data.head())

data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nAfter Label Encoding:\n")
print(data.head())

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

data['text'] = data['text'].apply(clean_text)

print("\nAfter Text Cleaning:\n")
print(data.head())

data.to_csv(
    "Dataset/Processed_data/processed_dataset.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")