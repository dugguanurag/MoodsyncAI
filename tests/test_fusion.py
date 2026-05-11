import unittest

from models.fusion import FusionModel


class FusionModelTest(unittest.TestCase):
    def setUp(self):
        self.model = FusionModel()

    def test_positive_text_and_happy_face_are_aligned(self):
        result = self.model.fuse(
            {"label": "POSITIVE", "confidence": 0.9},
            {"emotion": "happy", "confidence": 0.8},
        )

        self.assertEqual(result["status"], "ALIGNED")
        self.assertEqual(result["image_valence"], "POSITIVE")

    def test_positive_text_and_sad_face_are_mismatch(self):
        result = self.model.fuse(
            {"label": "POSITIVE", "confidence": 0.81},
            {"emotion": "sad", "confidence": 0.68},
        )

        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["image_valence"], "NEGATIVE")

    def test_neutral_face_is_uncertain(self):
        result = self.model.fuse(
            {"label": "NEGATIVE", "confidence": 0.73},
            {"emotion": "neutral", "confidence": 0.74},
        )

        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["image_valence"], "NEUTRAL")


if __name__ == "__main__":
    unittest.main()
