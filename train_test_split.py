import pandas as pd

from sklearn.model_selection import train_test_split


data = pd.read_csv(
    "Dataset/Processed_data/processed_dataset.csv"
)

print("Dataset Loaded Successfully!\n")

print(data.head())

X= data['text']
y= data['label']
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)
print("\nTraining Messages :", len(X_train))

print("Testing Messages :", len(X_test))

print("\nTraining Labels :", len(y_train))

print("Testing Labels :", len(y_test))

train_data = pd.DataFrame({

    'text': X_train,

    'label': y_train

})

test_data = pd.DataFrame({

    'text': X_test,

    'label': y_test

})

train_data.to_csv(

    "Dataset/Final_dataset/train.csv",

    index=False

)

test_data.to_csv(

    "Dataset/Final_dataset/test.csv",

    index=False

)

print("\nTraining and Testing datasets saved successfully!")