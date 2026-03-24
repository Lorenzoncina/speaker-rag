"""
This script is runned when image building time to download all the necessary models and save them in the image.

The image will contain both code and models.
Author: Lorenzo Concina
Date: 10/12/2025
"""
import os
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Define paths matching your endpoints.py configuration
TITANET_MODEL_DIR = "./embedding_model"
TITANET_MODEL_NAME = "titanet_large"
WHISPER_MODEL_DIR = "./whisper_model"
WHISPER_MODEL_NAME = "openai/whisper-tiny"
GENDER_MODEL_DIR = "./gender_model"
GENDER_MODEL_NAME = "prithivMLmods/Common-Voice-Gender-Detection"
AGE_MODEL_DIR = "./age_model"

def download_and_save():
    print("--- Starting Model Download for Docker Image ---")

    # 1. Download & Save Titanet
    print(f"Downloading {TITANET_MODEL_NAME}...")
    os.makedirs(TITANET_MODEL_DIR, exist_ok=True)
    titanet_path = os.path.join(TITANET_MODEL_DIR, TITANET_MODEL_NAME + ".nemo")
    
    # Download from cloud
    model = EncDecSpeakerLabelModel.from_pretrained(model_name=TITANET_MODEL_NAME)
    # Save to local folder so endpoints.py finds it offline
    model.save_to(titanet_path)
    print(f" Titanet saved to {titanet_path}")

    # 2. Download & Save Whisper
    print(f"Downloading {WHISPER_MODEL_NAME}...")
    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
    processor.save_pretrained(WHISPER_MODEL_DIR)
    model.save_pretrained(WHISPER_MODEL_DIR)
    print(f"Whisper saved to {WHISPER_MODEL_DIR}")

    # 3. Download & Save Gender Model
    print(f"Downloading {GENDER_MODEL_NAME}...")
    os.makedirs(GENDER_MODEL_DIR, exist_ok=True)
    gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(GENDER_MODEL_NAME)
    gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(GENDER_MODEL_NAME)
    gender_processor.save_pretrained(GENDER_MODEL_DIR)
    gender_model.save_pretrained(GENDER_MODEL_DIR)
    print(f"Gender model saved to {GENDER_MODEL_DIR}")

    # --- Prepare Age Model Directory ---
    print(f"Preparing directory for custom Age Model at {AGE_MODEL_DIR}...")
    os.makedirs(AGE_MODEL_DIR, exist_ok=True)
    # NOTE: Since this is a custom model, the 'age_probe.pt' file must be present in the local folder during 'docker build'.

if __name__ == "__main__":
    download_and_save()