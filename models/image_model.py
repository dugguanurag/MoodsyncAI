from transformers import pipeline
from utils.config import EMOTIONS, IMAGE_MODEL_NAME

class ImageEmotionModel:
    def __init__(self):

        self.classifier = pipeline(
            "image-classification",
            model=IMAGE_MODEL_NAME
        )

    def predict(self, image_path):

        predictions = self.classifier(image_path)

        result = {}

        for pred in predictions:
            label = pred["label"].lower()
            score = float(pred["score"])

            result[label] = score

        emotions = {emotion: result.get(emotion, 0.0) for emotion in EMOTIONS}

        top_emotion = max(emotions, key=emotions.get)
        sorted_emotions = sorted(
            emotions.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return {
            "emotion": top_emotion,
            "confidence": emotions[top_emotion],
            "all_emotions": emotions,
            "ranked_emotions": sorted_emotions
        }
