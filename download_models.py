"""Pre-download models at Docker build time so the image is self-contained.

Previously this script pulled Whisper-tiny, Wav2Vec2 gender, and the
Wav2Vec2 emotion model. Those were all retired when Voxtral Mini took
over audio attribute extraction. We now predownload:

  - Titanet (NeMo speaker-embedding backbone): unchanged, needed for
    cross-call identity and diarization clustering.
  - Voxtral Mini 3B (HuggingFace snapshot): the new speech LLM. Cached
    into HF_HOME so the vLLM `LLM()` constructor finds it offline.

The sentence-transformer (`all-MiniLM-L6-v2`) and the Qwen3-14B-AWQ used
by the KG extractor are downloaded on first call to keep the image
manageable; their cold-start cost is acceptable.
"""
import os

from huggingface_hub import snapshot_download
from nemo.collections.asr.models import EncDecSpeakerLabelModel

TITANET_MODEL_DIR = "./embedding_model"
TITANET_MODEL_NAME = "titanet_large"

VOXTRAL_MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"


def download_and_save():
    print("--- Starting Model Download for Docker Image ---")

    # 1. Titanet — saved as a single .nemo file so identification_service.py
    # finds it offline via restore_from().
    print(f"Downloading {TITANET_MODEL_NAME}...")
    os.makedirs(TITANET_MODEL_DIR, exist_ok=True)
    titanet_path = os.path.join(TITANET_MODEL_DIR, TITANET_MODEL_NAME + ".nemo")
    model = EncDecSpeakerLabelModel.from_pretrained(model_name=TITANET_MODEL_NAME)
    model.save_to(titanet_path)
    print(f"Titanet saved to {titanet_path}")

    # 2. Voxtral Mini 3B — populated into the HF cache so vLLM finds it on
    # disk at startup. snapshot_download is the right primitive for vLLM
    # (saves weights + tokenizer + processor config; vLLM resolves the
    # model id back to this path via HF_HOME).
    print(f"Downloading {VOXTRAL_MODEL_NAME}...")
    snapshot_download(
        repo_id=VOXTRAL_MODEL_NAME,
        # Pull everything the multimodal processor needs.
        # safetensors covers weights; the rest is config + tokenizer + processor.
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "preprocessor_config*",
            "processor_config*",
        ],
    )
    print(f"Voxtral cached for {VOXTRAL_MODEL_NAME}")


if __name__ == "__main__":
    download_and_save()
