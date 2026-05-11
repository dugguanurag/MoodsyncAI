# MoodSyncAI Streamlit Cloud Deployment

## Deployment Target

Use Streamlit Community Cloud:

https://share.streamlit.io

## Repository Settings

After pushing this project to GitHub, create a new Streamlit app with:

- Repository: your GitHub repository for this project
- Branch: `main`
- Main file path: `app/main.py`
- Python version: Python 3.12

## Required Files

The deployment needs these files in the GitHub repository:

- `app/main.py`
- `models/`
- `utils/`
- `requirements.txt`
- `.streamlit/config.toml`

The `venv/` folder should not be uploaded to GitHub.

## Notes

- The first cloud run may take several minutes because Hugging Face models are downloaded.
- The webcam feature works only when the deployed site is opened with browser camera permission.
- No Streamlit secrets are required for the current version.

## Models Used

- Image emotion model: `trpakov/vit-face-expression`
- Text sentiment model: `distilbert-base-uncased-finetuned-sst-2-english`
- Explanation model: `distilgpt2` with fallback explanation
