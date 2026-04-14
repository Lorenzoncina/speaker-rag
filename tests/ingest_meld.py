#this script must be runned on the server since it uses directly the codebase and not the api.
import os
import uuid
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from chromadb import PersistentClient
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MELD_CSV_PATH = "../../corpora/MELD.Raw/train_sent_emo.csv"
MELD_AUDIO_DIR = "../../corpora/MELD.Raw/train_splits" # where .wav files are
CHROMA_DIR = "../enrolled_data/chroma"
SPEAKER_COLLECTION = "speaker_embeddings"
SEMANTIC_COLLECTION = "semantic_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ingest_meld():
    print("Loading AI Models...")
    embedding_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").to(DEVICE).freeze()
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    print("Connecting to ChromaDB...")
    chroma_client = PersistentClient(path=CHROMA_DIR)
    spk_col = chroma_client.get_or_create_collection(name=SPEAKER_COLLECTION, metadata={"hnsw:space": "cosine"})
    sem_col = chroma_client.get_or_create_collection(name=SEMANTIC_COLLECTION, metadata={"hnsw:space": "cosine"})

    print(f"Reading MELD Dataset from {MELD_CSV_PATH}...")
    df = pd.read_csv(MELD_CSV_PATH)
    
    # ---> LIMIT APPLIED HERE: Take the first 1000 utterances for testing
    print("Limiting ingestion to the first 1000 utterances...")
    df = df.head(1000) 

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Enrolling MELD"):
        try:
            dialogue_id = str(row['Dialogue_ID'])
            utterance_id = int(row['Utterance_ID'])
            speaker = str(row['Speaker']).lower().strip()
            text = str(row['Utterance'])
            emotion = str(row['Emotion'])
            
            # MELD audio files are usually named like dia0_utt0.wav or dia0_utt0.mp4
            audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav" 
            audio_path = os.path.join(MELD_AUDIO_DIR, audio_filename)
            
            if not os.path.exists(audio_path):
                # print(f"Audio missing, skipping: {audio_path}")
                continue

            # 1. Compute Acoustic Embedding (Titanet)
            x, sr = librosa.load(audio_path, sr=16000, mono=True)
            sig = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            audio_signal_len = torch.tensor([sig.shape[1]]).to(DEVICE)
            with torch.no_grad():
                _, spk_embedding = embedding_model.forward(input_signal=sig, input_signal_length=audio_signal_len)
            spk_emb_vector = spk_embedding.squeeze().cpu().tolist()

            # 2. Compute Semantic Embedding (SentenceTransformer)
            sem_emb_vector = semantic_model.encode(text).tolist()

            # 3. Create Shared ID and Metadata#this script must be runned on the server since it uses directly the codebase and not the api.
import os
import uuid
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from chromadb import PersistentClient
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MELD_CSV_PATH = "../../../datasets/MELD.Raw/train_sent_emo.csv"
MELD_AUDIO_DIR = "../../../MELD.Raw/train_splits" # where .wav files are
CHROMA_DIR = "../enrolled_data/chroma"
SPEAKER_COLLECTION = "speaker_embeddings"
SEMANTIC_COLLECTION = "semantic_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ingest_meld():
    print("Loading AI Models...")
    embedding_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").to(DEVICE).freeze()
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    print("Connecting to ChromaDB...")
    chroma_client = PersistentClient(path=CHROMA_DIR)
    spk_col = chroma_client.get_or_create_collection(name=SPEAKER_COLLECTION, metadata={"hnsw:space": "cosine"})
    sem_col = chroma_client.get_or_create_collection(name=SEMANTIC_COLLECTION, metadata={"hnsw:space": "cosine"})

    print(f"Reading MELD Dataset from {MELD_CSV_PATH}...")
    df = pd.read_csv(MELD_CSV_PATH)
    
    # Optional: Take a subset for testing so it doesn't take hours the first time
    # df = df.head(500) 

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Enrolling MELD"):
        try:
            dialogue_id = str(row['Dialogue_ID'])
            utterance_id = int(row['Utterance_ID'])
            speaker = str(row['Speaker']).lower().strip()
            text = str(row['Utterance'])
            emotion = str(row['Emotion'])
            
            # MELD audio files are usually named like dia0_utt0.wav or dia0_utt0.mp4
            audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav" 
            audio_path = os.path.join(MELD_AUDIO_DIR, audio_filename)
            
            if not os.path.exists(audio_path):
                # print(f"Audio missing, skipping: {audio_path}")
                continue

            # 1. Compute Acoustic Embedding (Titanet)
            x, sr = librosa.load(audio_path, sr=16000, mono=True)
            sig = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            audio_signal_len = torch.tensor([sig.shape[1]]).to(DEVICE)
            with torch.no_grad():
                _, spk_embedding = embedding_model.forward(input_signal=sig, input_signal_length=audio_signal_len)
            spk_emb_vector = spk_embedding.squeeze().cpu().tolist()

            # 2. Compute Semantic Embedding (SentenceTransformer)
            sem_emb_vector = semantic_model.encode(text).tolist()

            # 3. Create Shared ID and Metadata
            shared_uuid = str(uuid.uuid4())
            metadata = {
                "user_id": speaker,
                "audio_file": audio_path,
                "data_type": "enroll",
                "dialogue_id": dialogue_id,     # ---> THE CRITICAL NEW FIELDS
                "utterance_id": utterance_id,   # ---> THE CRITICAL NEW FIELDS
                "transcription": text,
                "text_emotion": emotion,        # MELD provides the ground truth!
                "acoustic_emotion": emotion, 
                "language": "en"                # MELD is English
            }

            # 4. Insert into both databases using the exact same UUID
            spk_col.add(
                ids=[shared_uuid],
                embeddings=[spk_emb_vector],
                metadatas=[metadata]
            )
            
            sem_col.add(
                ids=[shared_uuid],
                embeddings=[sem_emb_vector],
                metadatas=[metadata],
                documents=[text]
            )

        except Exception as e:
            print(f"\nFailed on Dialogue {row.get('Dialogue_ID')} Utt {row.get('Utterance_ID')}: {e}")

    print("MELD Ingestion Complete!")

if __name__ == "__main__":
    ingest_meld()
            shared_uuid = str(uuid.uuid4())
            metadata = {
                "user_id": speaker,
                "audio_file": audio_path,
                "data_type": "enroll",
                "dialogue_id": dialogue_id,     # ---> THE CRITICAL NEW FIELDS
                "utterance_id": utterance_id,   # ---> THE CRITICAL NEW FIELDS
                "transcription": text,
                "text_emotion": emotion,        # MELD provides the ground truth!
                "acoustic_emotion": emotion, 
                "language": "en"                # MELD is English
            }

            # 4. Insert into both databases using the exact same UUID
            spk_col.add(
                ids=[shared_uuid],
                embeddings=[spk_emb_vector],
                metadatas=[metadata]
            )
            
            sem_col.add(
                ids=[shared_uuid],
                embeddings=[sem_emb_vector],
                metadatas=[metadata],
                documents=[text]
            )

        except Exception as e:
            print(f"\nFailed on Dialogue {row.get('Dialogue_ID')} Utt {row.get('Utterance_ID')}: {e}")

    print("MELD Ingestion Complete!")

if __name__ == "__main__":
    ingest_meld()