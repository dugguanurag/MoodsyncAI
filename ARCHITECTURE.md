# Architecture Notes

## Goal

The system detects emotional consistency between a face image and a text statement. It is designed for the assignment scenario where a colleague says something positive while their facial expression may show sadness, fear, stress, or discomfort.

## Data Flow

1. User uploads a face image or captures one with the webcam.
2. User enters a sentence.
3. `ImageEmotionModel` sends the image to a Vision Transformer face-expression classifier.
4. `TextSentimentModel` sends the typed sentence to a DistilBERT sentiment classifier.
5. `FusionModel` maps facial emotion into broad emotional valence:

- `happy` -> positive
- `sad`, `angry`, `fear`, `disgust` -> negative
- `neutral` -> neutral
- `surprise` -> ambiguous

6. Fusion compares image valence with text sentiment:

- same direction -> `ALIGNED`
- opposite direction -> `MISMATCH`
- neutral or ambiguous visual result -> `UNCERTAIN`

7. `ExplanationGenerator` produces a plain-language summary for the demo output.

## Model Choices

| Component | Model / Method | Assignment Requirement |
| --- | --- | --- |
| Visual emotion | `trpakov/vit-face-expression` | CNN or ViT for facial emotion |
| Text sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | RNN/LSTM or Transformer for text sentiment |
| Fusion | Rule-based valence alignment | Multimodal fusion layer |
| Explanation | Configurable text-generation pipeline with fallback | Generative language summary |
## Optional Extended Features

| Feature | Status | Notes |
| --- | --- | --- |
| Webcam capture | Implemented | Captures a still face image directly in Streamlit |
| Audio input | Not implemented | Kept out to keep the final demo focused on text and visual modalities |
| Short video timeline | Not implemented | Requires frame sampling and temporal visualisation |
| Attention / Grad-CAM | Not implemented | Needs lower-level model access beyond the current Hugging Face pipeline wrapper |
| Learned fusion | Not implemented | Requires a labelled paired text-image training dataset |
| Cloud deployment | Documented as next step | Can be deployed to Streamlit Cloud or Hugging Face Spaces |

## Why Rule-Based Fusion?

The project does not include a labelled multimodal dataset for training a learned fusion network. A transparent rule-based layer is therefore used so the decision is easy to explain in the presentation. This also supports the assignment's example case: positive words with sad or fearful facial emotion produce a mismatch.

## Demo Talking Points

- The application uses two modalities, not just text sentiment.
- Confidence is shown separately for text and image to avoid hiding uncertainty.
- The fusion layer is intentionally interpretable.
- The summary turns model outputs into a human-readable explanation.
- Optional improvements would include audio transcription, attention visualisation, learned fusion, and cloud deployment.
