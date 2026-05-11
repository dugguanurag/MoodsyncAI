import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.config import GENERATION_MODEL_NAME


class ExplanationGenerator:
    def __init__(self):
        self.generator = None
        self.model_name = GENERATION_MODEL_NAME

        if os.getenv("MOODSYNC_USE_GPT2", "0") != "1":
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
        except Exception:
            self.generator = None

    def _template_explanation(self, fusion_result):
        text = fusion_result["text_sentiment"]
        emotion = fusion_result["image_emotion"]
        status = fusion_result["status"]

        if status == "ALIGNED":
            return (
                "The person's words and facial expression appear consistent, "
                f"both suggesting a {text.lower()} emotional state."
            )

        if status == "UNCERTAIN":
            return (
                f"The person expresses {text.lower()} sentiment verbally, while "
                f"the face appears {emotion}; the combined emotional state is "
                "not strong enough to mark a clear mismatch."
            )

        return (
            f"Despite expressing {text.lower()} sentiment verbally, "
            f"the facial expression suggests {emotion}. "
            "This mismatch may indicate hidden emotion, stress, or discomfort."
        )

    def _clean_generated_summary(self, summary):
        summary = re.split(r"\bSummary\s*:", summary, maxsplit=1)[0].strip()
        summary = re.sub(r"\s+", " ", summary)

        if not summary:
            return ""

        sentence_endings = [summary.find("."), summary.find("!"), summary.find("?")]
        sentence_endings = [idx for idx in sentence_endings if idx != -1]
        if sentence_endings:
            summary = summary[: min(sentence_endings) + 1]

        return summary

    def generate(self, fusion_result):
        # Deterministic explanations are used first to keep the deployed demo stable.
        # The GPT-2-family model remains loaded as an optional extension, but short
        # causal generation can repeat the prompt on Streamlit Cloud.
        return self._template_explanation(fusion_result)
