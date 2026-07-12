from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)


tokenizer = BertTokenizer.from_pretrained("Models/SpamClassifier")


model = BertForSequenceClassification.from_pretrained("Models/SpamClassifier")
model.eval()


@app.route("/", methods=["GET", "POST"])
def home():

    message = ""
    prediction = None
    confidence = None
    color = None

    if request.method == "POST":

        
        message = request.form["message"]

    
        message = message.strip()

    
        if message == "":
            return render_template(
                "index.html",
                message="",
                prediction=None,
                confidence=None,
                color=None
            )

        
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

        pred = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0][pred].item() * 100

        # Prediction label
        if pred == 1:
            prediction = "SPAM"
            color = "red"
        else:
            prediction = "HAM"
            color = "green"

    return render_template(
        "index.html",
        message=message,
        prediction=prediction,
        confidence=confidence,
        color=color
    )


if __name__ == "__main__":
    app.run(debug=True)