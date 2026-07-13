# AI Based BERT Spam Detection System

## Overview
This project is an AI-powered spam detection system that classifies SMS messages as either Spam or Ham using a fine-tuned BERT model. The application is built with Python, Flask, and Hugging Face Transformers.

## Features
- Detects Spam and Ham messages
- Uses BERT for text classification
- Displays confidence score
- Simple and responsive web interface
- Built with Flask

## Technologies Used
- Python
- Flask
- BERT
- Hugging Face Transformers
- PyTorch
- HTML
- CSS

## Project Structure

```
Hinglish-BERT-Classifier/
│
├── App/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── Dataset/
├── Models/
├── Notebooks/
├── Screenshots/
│
├── preprocessing.py
├── bert_model.py
├── tokenizer.py
├── training.py
├── prediction.py
└── requirements.txt
```

## How to Run

1. Clone the repository

```
git clone <repository-url>
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the application

```
python App/app.py
```

4. Open

```
http://127.0.0.1:5000
```

## Results

The system predicts whether a message is:

- HAM
- SPAM

along with the confidence score.

## Future Improvements

- Email spam detection
- Multilingual spam detection
- Mobile application
- Cloud deployment

## Author

**Arnav Aarush**