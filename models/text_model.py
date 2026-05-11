from transformers import pipeline
from utils.config import TEXT_MODEL_NAME

class TextSentimentModel:
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model=TEXT_MODEL_NAME
        )

    def predict(self, text):
        result = self.classifier(text)[0]

        label = result["label"]
        score = float(result["score"])

        return {
            "label": label,
            "confidence": score,
            "sentiment_score": score if label == "POSITIVE" else -score
        }