from utils.config import AMBIGUOUS_EMOTIONS, NEGATIVE_EMOTIONS, NEUTRAL_EMOTIONS, POSITIVE_EMOTIONS


class FusionModel:
    def __init__(self):
        self.positive_emotions = POSITIVE_EMOTIONS
        self.negative_emotions = NEGATIVE_EMOTIONS
        self.neutral_emotions = NEUTRAL_EMOTIONS
        self.ambiguous_emotions = AMBIGUOUS_EMOTIONS

    def fuse(self, text_result, image_result):
        text_label = text_result["label"]
        image_emotion = image_result["emotion"]
        text_confidence = float(text_result["confidence"])
        image_confidence = float(image_result["confidence"])

        if image_emotion in self.positive_emotions:
            image_valence = "POSITIVE"
        elif image_emotion in self.negative_emotions:
            image_valence = "NEGATIVE"
        elif image_emotion in self.neutral_emotions:
            image_valence = "NEUTRAL"
        else:
            image_valence = "AMBIGUOUS"

        if image_valence in ["NEUTRAL", "AMBIGUOUS"]:
            status = "UNCERTAIN"
            confidence = min(text_confidence, image_confidence)
            rationale = (
                "The visual channel is neutral or ambiguous, so the system "
                "does not have enough evidence to claim alignment or mismatch."
            )

        elif text_label == image_valence:
            status = "ALIGNED"
            confidence = (text_confidence + image_confidence) / 2
            rationale = (
                "The text sentiment and facial emotion point to the same "
                "overall emotional direction."
            )

        else:
            status = "MISMATCH"
            confidence = (text_confidence + image_confidence) / 2
            rationale = (
                "The text sentiment and facial emotion point in different "
                "emotional directions."
            )

        mismatch_score = confidence if status == "MISMATCH" else 1 - confidence

        return {
            "text_sentiment": text_label,
            "image_emotion": image_emotion,
            "image_valence": image_valence,
            "status": status,
            "confidence": confidence,
            "mismatch_score": mismatch_score,
            "rationale": rationale
        }
