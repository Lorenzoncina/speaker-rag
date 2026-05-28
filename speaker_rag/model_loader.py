"""Startup model loader for SpeakerRAG.

Audio frontend (transcription + language + gender + age + emotion) is now
handled by a single Voxtral Mini 3B model running via vLLM. The remaining
two models that load eagerly here are the ones Voxtral can't replace:

  - Titanet (NeMo `EncDecSpeakerLabelModel`): speaker embeddings used for
    cross-call identity and diarization clustering.
  - sentence-transformers `all-MiniLM-L6-v2`: semantic search embeddings
    over utterance text in ChromaDB and the EntityResolver.
"""
import os

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sentence_transformers import SentenceTransformer

from .voxtral_extractor import VoxtralExtractor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_audio_models(
    device: str,
    titanet_dir: str = os.path.join(CURRENT_DIR, "embedding_model"),
    titanet_name: str = "titanet_large",
    semantic_name: str = "all-MiniLM-L6-v2",
    voxtral_name: str = VoxtralExtractor.DEFAULT_MODEL,
):
    """Load and return all models needed at startup.

    Returns a dict with three keys:
      - embedding_model:    NeMo Titanet for speaker embeddings.
      - semantic_model:     SentenceTransformer for text embeddings.
      - voxtral_extractor:  VoxtralExtractor singleton (replaces Whisper +
                            Wav2Vec2 gender + AgeProbe + Wav2Vec2 emotion
                            + RoBERTa text-emotion). One LLM call returns
                            all six attributes.
    """
    print(f"Loading audio models on device: {device}...")

    # 1. Titanet — speaker embeddings (Voxtral can't produce these).
    try:
        model_path = os.path.join(titanet_dir, titanet_name + ".nemo")
        titanet_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=device)
    except FileNotFoundError:
        titanet_model = EncDecSpeakerLabelModel.from_pretrained(model_name=titanet_name)
    titanet_model.to(device).freeze()

    # 2. Sentence-transformer — semantic text embeddings for retrieval + entity dedup.
    print("Loading Semantic Text Model...")
    semantic_model = SentenceTransformer(semantic_name, device=device)

    # 3. Voxtral — single speech LLM replacing Whisper / Wav2Vec2 / Age / Emotion.
    # Eagerly loaded so first enrollment is fast and load failures surface now.
    voxtral_extractor = VoxtralExtractor(model_name=voxtral_name, device=device)

    return {
        "embedding_model": titanet_model,
        "semantic_model": semantic_model,
        "voxtral_extractor": voxtral_extractor,
    }
