import torch
from transformers import BertTokenizer, BertForSequenceClassification


print("Loading Tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    "Models/SpamClassifier"
)

print("✅ Tokenizer Loaded Successfully!")

print("\nLoading Model...")

model = BertForSequenceClassification.from_pretrained(
    "Models/SpamClassifier"
)

print("✅ Model Loaded Successfully!")

model.eval()

print("\n==============================================")
print("     Hinglish BERT Spam Classifier")
print("==============================================")
print("Type your message below.")
print("Type 'exit' anytime to close.")
print("==============================================")

while True:

    print()

    message = input("📩 Enter Message : ")

    if message.lower() == "exit":
        print("\n👋 Thank you for using the Spam Classifier!")
        break

    if message.strip() == "":
        print("⚠ Please enter a valid message.")
        continue

    encoding = tokenizer(
        message,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][prediction].item() * 100

    print("\n================ Prediction ================")

    if prediction == 1:
        print("🚨 Prediction : SPAM")
    else:
        print("✅ Prediction : HAM")

    print(f"📊 Confidence : {confidence:.2f}%")
    
    print("============================================")
