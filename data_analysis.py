import pandas as pd

# Read the dataset
data = pd.read_csv(
    "Dataset/raw_data/SMS_Spam_Collection/spam.csv",
    encoding="latin-1"
)

# Keep only the useful columns
data = data[['v1', 'v2']]

print("========== FIRST FIVE ROWS ==========")
print(data.head())

print("\n========== DATASET INFORMATION ==========")
data.info()

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== COLUMN NAMES ==========")
print(data.columns)

print("\n========== MESSAGE COUNT ==========")
print(data['v1'].value_counts())

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())