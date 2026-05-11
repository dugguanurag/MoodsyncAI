import sys
import os
import tempfile

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import streamlit as st
from PIL import Image

from models.text_model import TextSentimentModel
from models.image_model import ImageEmotionModel
from models.fusion import FusionModel
from models.generator import ExplanationGenerator
from utils.config import GENERATION_MODEL_NAME, IMAGE_MODEL_NAME, TEXT_MODEL_NAME

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MoodSyncAI",
    page_icon="💭",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    return (
        TextSentimentModel(),
        ImageEmotionModel(),
        FusionModel(),
        ExplanationGenerator()
    )

text_model, image_model, fusion_model, generator = load_models()

# ---------------- UI ----------------

st.title("MoodSyncAI 💭")

st.info(
    "Upload a clear face image and enter what the person said to detect emotional alignment or mismatch."
)

st.caption("Multi-modal sentiment and facial emotion analyser for the DA3 final project.")

with st.sidebar:
    st.header("Model Stack")
    st.write("**Vision:** CNN/ViT facial emotion classifier")
    st.code(IMAGE_MODEL_NAME)
    st.write("**Text:** Transformer sentiment classifier")
    st.code(TEXT_MODEL_NAME)
    st.write("**Fusion:** rule-based multimodal alignment layer")
    st.write("**Summary:** generative language model with safe fallback")
    st.code(GENERATION_MODEL_NAME)

with st.expander("Architecture and assignment mapping"):
    st.write(
        "Image input is processed by a Vision Transformer face-expression model. "
        "Text input is processed by a DistilBERT sentiment model. The fusion layer "
        "maps both predictions into emotional valence and detects ALIGNED, MISMATCH, "
        "or UNCERTAIN states. The explanation component then generates a plain-language "
        "summary for the final demo output."
    )

# ---------------- INPUTS ----------------

input_mode = st.radio(
    "Choose face input",
    ["Upload image", "Use webcam"],
    horizontal=True
)

uploaded_file = None
camera_file = None

if input_mode == "Upload image":
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        key="face_upload"
    )
else:
    camera_file = st.camera_input(
        "Capture a face image",
        key="face_camera"
    )

image_source = uploaded_file or camera_file

text_input = st.text_input(
    "Enter what the person said",
    placeholder="Example: I think the project is going really well."
)

analysis_text = text_input.strip()

# ---------------- VALIDATION ----------------

if image_source and not analysis_text:
    st.warning("Please enter text input.")

# ---------------- PROCESS ----------------

if image_source and analysis_text:

    image_path = None

    try:

        # Open uploaded image
        image = Image.open(image_source)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )
        image_path = temp_file.name
        temp_file.close()
        image.convert("RGB").save(image_path)

        # ---------------- MODEL INFERENCE ----------------

        with st.spinner("Analyzing emotional signals..."):

            text_result = text_model.predict(analysis_text)

            image_result = image_model.predict(image_path)

            fusion_result = fusion_model.fuse(
                text_result,
                image_result
            )

            explanation = generator.generate(
                fusion_result
            )

        # ---------------- RESULTS ----------------

        st.subheader("Results")

        text_col, image_col, fusion_col = st.columns(3)

        with text_col:
            st.metric(
                "Textual Sentiment",
                text_result["label"],
                f"{text_result['confidence']:.0%} confidence"
            )

        with image_col:
            st.metric(
                "Visual Emotion",
                image_result["emotion"].title(),
                f"{image_result['confidence']:.0%} confidence"
            )

        with fusion_col:
            st.metric(
                "Fusion Result",
                fusion_result["status"],
                f"{fusion_result['confidence']:.0%} confidence"
            )

        # ---------------- EMOTION DISTRIBUTION ----------------

        st.subheader("Emotion Distribution")

        emotion_chart = pd.DataFrame(
            {
                "emotion": list(image_result["all_emotions"].keys()),
                "confidence": list(image_result["all_emotions"].values())
            }
        ).set_index("emotion")

        st.bar_chart(
            emotion_chart
        )

        # ---------------- CONFIDENCE LEVELS ----------------

        st.subheader("Confidence Levels")

        st.write("Text Confidence")

        st.progress(
            float(text_result["confidence"])
        )

        st.write("Image Confidence")

        st.progress(
            float(image_result["confidence"])
        )

        # ---------------- FUSION STATUS ----------------

        status = fusion_result["status"]

        if status == "MISMATCH":

            st.error(
                "⚠️ Emotional Inconsistency Detected"
            )

        elif status == "ALIGNED":

            st.success(
                "✅ Emotional Signals Are Consistent"
            )

        else:

            st.warning(
                "⚠️ Emotional State Uncertain"
            )

        st.write(f"**Fusion rationale:** {fusion_result['rationale']}")

        # ---------------- AI EXPLANATION ----------------

        st.subheader("AI Explanation")

        st.write(explanation)

    except Exception as e:

        st.error(
            "An error occurred while processing the inputs."
        )

        st.exception(e)

    finally:

        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
