from transformers import pipeline

# Load DistilBERT sentiment classifier
bert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_bert(text):
    result = bert_classifier(text)[0]
    label = result['label']  # 'POSITIVE' or 'NEGATIVE'
    score = result['score']  # confidence score (0 to 1)

    # We treat POSITIVE as "Real", NEGATIVE as "Fake"
    prediction = 1 if label == "POSITIVE" else 0
    return prediction, label, score
