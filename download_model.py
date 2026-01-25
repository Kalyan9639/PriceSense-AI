# download_model.py

import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"
MODEL_NAME = "rf_model.joblib"

# 🔁 Change these
REPO_ID = "your-username/your-model-repo"
FILENAME = "rf_model.joblib"

def rfmodel():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    if not os.path.exists(model_path):
        print("⬇️ Downloading model from Hugging Face...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists")

    return model_path
