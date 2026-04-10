import os
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification 
from sentence_transformers import SentenceTransformer
from .identification_service import AgeProbe
from transformers import pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_audio_models(
    device: str,
    titanet_dir=os.path.join(CURRENT_DIR, "embedding_model"),
    titanet_name="titanet_large",
    whisper_dir=os.path.join(CURRENT_DIR, "whisper_model"),
    whisper_name="openai/whisper-tiny",
    gender_dir=os.path.join(CURRENT_DIR, "gender_model"),
    gender_name="prithivMLmods/Common-Voice-Gender-Detection",
    age_dir=os.path.join(CURRENT_DIR, "age_model"),
    age_name="age_probe.pt",
    semantic_name="all-MiniLM-L6-v2", 
    emotion_dir=os.path.join(CURRENT_DIR, "emotion_model"), # NEW
    emotion_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" # NEW
):
    """Loads and returns all audio processing models."""
    print(f"Loading audio models on device: {device}...")
    
    # 1. Titanet
    try:
        model_path = os.path.join(titanet_dir, titanet_name + ".nemo")
        titanet_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=device)
    except FileNotFoundError:
        titanet_model = EncDecSpeakerLabelModel.from_pretrained(model_name=titanet_name)
    titanet_model.to(device).freeze()

    # 2. Whisper
    try:
        whisper_processor = WhisperProcessor.from_pretrained(whisper_dir)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_dir).to(device)
    except OSError:
        whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_name).to(device)
    whisper_model.eval()

    # 3. Gender Model
    try:
        gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(gender_dir)
        gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(gender_dir).to(device)
    except OSError:
        gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(gender_name)
        gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(gender_name).to(device)
    gender_model.eval()

    # 4. Age Model
    age_model = AgeProbe(input_dim=384, num_classes=4)
    age_weights_path = os.path.join(age_dir, age_name)
    try:
        age_model.load_state_dict(torch.load(age_weights_path, map_location=device))
        age_model.to(device).eval()
    except Exception as e:
        print(f"Failed to load Age model: {e}")
        age_model = None
        
    # 5. Semantic Text Model
    print("Loading Semantic Text Model...")
    semantic_model = SentenceTransformer(semantic_name, device=device)

    # 6. Emotion Model 
    print("Loading Emotion Model...")
    try:
        # Explicitly check if the local folder exists to prevent HF Validation crashes
        if os.path.exists(emotion_dir):
            emotion_processor = AutoFeatureExtractor.from_pretrained(emotion_dir)
            emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_dir).to(device)
        else:
            raise FileNotFoundError("Local emotion directory not found. Falling back to Hub.")
    except Exception as e:
        print(f"Downloading Emotion Model from Hub: {emotion_name}")
        emotion_processor = AutoFeatureExtractor.from_pretrained(emotion_name)
        emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_name).to(device)
    emotion_model.eval()

    # 7. Semantic Text Emotion Model (NEW)
    print("Loading Semantic Text Emotion Model...")
    text_emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1, device=0 if device=="cuda" else -1)

    return {
        "embedding_model": titanet_model,
        "language_processor": whisper_processor,
        "language_model": whisper_model,
        "gender_processor": gender_processor,
        "gender_model": gender_model,
        "age_model": age_model,
        "semantic_model": semantic_model,
        "emotion_processor": emotion_processor, 
        "emotion_model": emotion_model,
        "text_emotion_pipeline": text_emotion_pipeline
    }