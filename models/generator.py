from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils.config import GENERATION_MODEL_NAME


class ExplanationGenerator:
    def __init__(self):
        self.generator = None
        self.model_name = GENERATION_MODEL_NAME

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

    def generate(self, fusion_result):
        text = fusion_result["text_sentiment"]
        emotion = fusion_result["image_emotion"]
        status = fusion_result["status"]
        rationale = fusion_result.get("rationale", "")

        prompt = (
            "Summarize this multimodal emotion analysis in one plain-language "
            f"sentence. Text sentiment: {text}. Facial emotion: {emotion}. "
            f"Fusion result: {status}. Reason: {rationale} Summary:"
        )

        if self.generator:
            try:
                output = self.generator(
                    prompt,
                    max_new_tokens=55,
                    do_sample=False,
                    pad_token_id=self.generator.tokenizer.pad_token_id
                )[0]["generated_text"]
                summary = output.split("Summary:", 1)[-1].strip()
                if summary:
                    return summary
            except Exception:
                pass

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
