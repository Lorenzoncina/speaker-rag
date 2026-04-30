import os
import json
import uuid
import shutil
import tempfile
import torch
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from speaker_rag.rag import SpeakerRAG

app = FastAPI(
    title="Speaker RAG API", 
    description="An LLM-powered RAG system for analyzing and querying speaker audio.",
    version="1.1.0"
)

rag_system = None

@app.on_event("startup")
def startup_event():
    global rag_system
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    hf_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    print(f"Starting Speaker RAG API on device: {device}")

    rag_system = SpeakerRAG(
        chroma_dir="./enrolled_data/chroma",
        collection_name="speaker_embeddings",
        device=device,
        hf_auth_token=hf_auth_token,
    )

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "Service is running", "models_loaded": rag_system is not None}

@app.get("/rag/enrolled", tags=["RAG"])
async def get_enrolled():
    """Returns a list of all enrolled users and their attributes."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    
    try:
        result = rag_system.get_enrolled_users()
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/rag/clear", tags=["RAG"])
async def clear_database():
    """Completely wipes the ChromaDB vector database of all enrolled speakers."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    
    try:
        result = rag_system.clear_database()
        if result["status"] == "success":
            return {"message": result["message"]}
        else:
            return JSONResponse(status_code=500, content={"error": result["message"]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/rag/index", tags=["RAG"])
async def rag_index_speaker(
    audios: List[UploadFile] = File(...),
    user_ids: Optional[List[str]] = Form(None, description="List of names. If omitted, every file gets a unique random UUID.")
):
    """Indexes one or multiple speakers' audio files into the database."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_paths = []
            
            # THE FIX: Apply safe filenames to enrollment so we don't lose training data!
            for i, audio in enumerate(audios):
                safe_filename = f"{i}_{audio.filename}"
                tmp_path = os.path.join(tmpdir, safe_filename)
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
                tmp_paths.append(tmp_path)

            # Pass the files and IDs directly to the indexer
            result = rag_system.index_audio(audio_paths=tmp_paths, user_ids=user_ids)
            return result
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/index_conversation", tags=["RAG"])
async def rag_index_conversation(
    audio: UploadFile = File(...),
    vtt_file: Optional[UploadFile] = File(
        None,
        description="Optional WebVTT (.vtt) transcript file (e.g. from Microsoft Teams). "
                    "When provided, pyannote diarization is skipped entirely — speaker names "
                    "and timestamps come from the VTT."
    ),
    speaker_map: Optional[str] = Form(
        None,
        description='Optional JSON dict mapping pyannote labels to names, e.g. {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}. Ignored if a VTT file is provided.'
    ),
    num_speakers: Optional[int] = Form(None, description="Optional hint for the number of speakers in the conversation. Ignored if a VTT file is provided."),
):
    """Ingests a full multi-speaker conversation (e.g. a Teams call or phone recording).

    Three modes:
      1. VTT file provided: timestamps + speaker names + transcriptions come from the VTT.
         Pyannote is skipped. This is the most accurate path for Teams/Meet recordings.
      2. Speaker map JSON provided: pyannote runs diarization, then the JSON maps cluster
         labels to real names (clustering is skipped).
      3. Neither: full pyannote diarization + agglomerative clustering.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    parsed_map = None
    if speaker_map:
        try:
            parsed_map = json.loads(speaker_map)
            if not isinstance(parsed_map, dict):
                parsed_map = None
        except Exception:
            parsed_map = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_filename = f"conversation_{audio.filename}"
            tmp_path = os.path.join(tmpdir, safe_filename)
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

            vtt_path = None
            if vtt_file:
                vtt_path = os.path.join(tmpdir, vtt_file.filename)
                with open(vtt_path, "wb") as f:
                    shutil.copyfileobj(vtt_file.file, f)

            result = rag_system.index_conversation(
                audio_path=tmp_path,
                speaker_map=parsed_map,
                num_speakers=num_speakers,
                vtt_path=vtt_path,
            )
            return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/retrieve", tags=["RAG"])
async def rag_retrieve_context(
    audios: Optional[List[UploadFile]] = File(None),
    question: str = Form(..., description="The user's query to guide the semantic search."),
    conversation_history: Optional[str] = Form(
        None,
        description='Optional JSON string: list of {"question": str, "answer": str} past turns.'
    ),
):
    """Builds a context Fact Sheet for an external LLM.

    Hybrid routing:
      - If audio files are provided: run full pipeline (identification + filtered semantic search).
      - Else: attempt speaker-name extraction from the question; fall back to unfiltered semantic search.

    The optional conversation_history is rendered into the recommended prompt as prior-turn context.
    The backend does not persist it — callers are responsible for maintaining and resetting memory.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    parsed_history = None
    if conversation_history:
        try:
            parsed_history = json.loads(conversation_history)
            if not isinstance(parsed_history, list):
                parsed_history = None
        except Exception:
            parsed_history = None

    try:
        if audios:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_paths = []
                for i, audio in enumerate(audios):
                    safe_filename = f"{i}_{audio.filename}"
                    tmp_path = os.path.join(tmpdir, safe_filename)
                    with open(tmp_path, "wb") as f:
                        shutil.copyfileobj(audio.file, f)
                    tmp_paths.append(tmp_path)

                result = rag_system.build_context(
                    audio_paths=tmp_paths,
                    question=question,
                    conversation_history=parsed_history,
                )
        else:
            result = rag_system.build_context(
                audio_paths=None,
                question=question,
                conversation_history=parsed_history,
            )

        return {
            "question": question,
            "fact_sheet": result["fact_sheet"],
            "recommended_prompt": result["recommended_prompt"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/admin/ingest_meld", tags=["Admin"])
def admin_ingest_meld():
    """Temporary backdoor endpoint to ingest MELD using pre-loaded GPU models."""
    import pandas as pd
    import librosa
    import torch
    import uuid
    import os

    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    csv_path = "/app/MELD.Raw/train_sent_emo.csv"
    audio_dir = "/app/MELD.Raw/train_splits"

    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": f"MELD dataset not found in container at {csv_path}"})

    print("Starting MELD Ingestion Bypass...")
    # Limit to 1000 for testing
    df = pd.read_csv(csv_path)#.head(1000)

    # Access the active services directly
    audio_svc = rag_system.audio_service
    sem_svc = rag_system.semantic_service
    device = audio_svc._device

    success_count = 0

    for index, row in df.iterrows():
        try:
            dialogue_id = str(row['Dialogue_ID'])
            utterance_id = int(row['Utterance_ID'])
            speaker = str(row['Speaker']).lower().strip()
            text = str(row['Utterance'])
            emotion = str(row['Emotion'])

            audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)

            if not os.path.exists(audio_path):
                continue

            # 1. Compute Acoustic Embedding (Using pre-loaded Titanet)
            x, sr = librosa.load(audio_path, sr=16000, mono=True)
            sig = torch.from_numpy(x).unsqueeze(0).to(device)
            audio_signal_len = torch.tensor([sig.shape[1]]).to(device)
            with torch.no_grad():
                _, spk_embedding = audio_svc._embedding_model.forward(input_signal=sig, input_signal_length=audio_signal_len)
            spk_emb_vector = spk_embedding.squeeze().cpu().tolist()

            # 2. Compute Semantic Embedding (Using pre-loaded SentenceTransformer)
            sem_emb_vector = sem_svc._model.encode(text).tolist()

            # 3. Create Metadata
            shared_uuid = str(uuid.uuid4())
            metadata = {
                "user_id": speaker,
                "audio_file": audio_filename, 
                "data_type": "enroll",
                "dialogue_id": dialogue_id,     
                "utterance_id": utterance_id,   
                "transcription": text,
                "text_emotion": emotion,        
                "acoustic_emotion": emotion, 
                "language": "en"                
            }

            # 4. Save safely to the active Chroma collections
            audio_svc._collection.add(
                ids=[shared_uuid],
                embeddings=[spk_emb_vector],
                metadatas=[metadata]
            )
            
            sem_svc._collection.add(
                ids=[shared_uuid],
                embeddings=[sem_emb_vector],
                metadatas=[metadata],
                documents=[text]
            )
            
            success_count += 1
            if success_count % 100 == 0:
                print(f"Ingested {success_count} / 1000 MELD files...")

        except Exception as e:
            print(f"Failed on Dialogue {row.get('Dialogue_ID')} Utt {row.get('Utterance_ID')}: {e}")

    print("MELD Ingestion Complete!")
    return {"status": "Complete", "enrolled_count": success_count}


@app.get("/admin/ingest_voxceleb", tags=["Admin"])
def admin_ingest_voxceleb(per_speaker: int = 10, seed: int = 42):
    """Ingest the 40 VoxCeleb1 test speakers (those referenced in Vox1-O.txt)
    into the RAG database. Picks `per_speaker` random utterances per speaker,
    EXCLUDING any file used as a probe in Vox1-O.txt to prevent train/test leakage.
    Each entry's user_id matches the VoxCeleb1 speaker ID (e.g. "id10270").
    """
    import os
    import csv
    import uuid
    import random
    import librosa
    import torch

    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    trials_path = "/app/VoxCeleb1/Vox1-O/Vox1-O.txt"
    wav_root = "/app/VoxCeleb1/wav/test"
    meta_path = "/app/VoxCeleb1/vox1_meta.csv"

    if not os.path.exists(trials_path):
        return JSONResponse(status_code=404, content={"error": f"Vox1-O trials not found at {trials_path}"})
    if not os.path.isdir(wav_root):
        return JSONResponse(status_code=404, content={"error": f"VoxCeleb1 test wavs not found at {wav_root}"})

    # 1. Parse trials -> probe set + speaker set
    probe_set = set()
    speakers = set()
    with open(trials_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, a1, a2 = parts
            probe_set.add(a1)
            probe_set.add(a2)
            speakers.add(a1.split("/", 1)[0])
            speakers.add(a2.split("/", 1)[0])
    speakers = sorted(speakers)
    print(f"VoxCeleb1: {len(speakers)} speakers, {len(probe_set)} probe files excluded.")

    # 2. Optional metadata (gender, nationality, real name)
    speaker_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 5:
                    spk_id, name, gender, nationality, _set = (c.strip() for c in row[:5])
                    speaker_meta[spk_id] = {
                        "real_name": name.replace("_", " "),
                        "gender_csv": gender,
                        "nationality": nationality,
                    }

    # 3. Select enrollment files per speaker
    rng = random.Random(seed)
    enrollment = {}
    for spk in speakers:
        spk_dir = os.path.join(wav_root, spk)
        if not os.path.isdir(spk_dir):
            continue
        all_files = []
        for video_dir in sorted(os.listdir(spk_dir)):
            vp = os.path.join(spk_dir, video_dir)
            if not os.path.isdir(vp):
                continue
            for wav in sorted(os.listdir(vp)):
                if wav.endswith(".wav"):
                    all_files.append(f"{spk}/{video_dir}/{wav}")
        candidates = [f for f in all_files if f not in probe_set]
        if not candidates:
            continue
        chosen = candidates if len(candidates) <= per_speaker else rng.sample(candidates, per_speaker)
        enrollment[spk] = chosen

    total_files = sum(len(v) for v in enrollment.values())
    print(f"VoxCeleb1: enrolling {total_files} files across {len(enrollment)} speakers...")

    # 4. Enroll into both collections
    audio_svc = rag_system.audio_service
    sem_svc = rag_system.semantic_service
    device = audio_svc._device

    success_count = 0
    for spk_id, rel_paths in enrollment.items():
        sm = speaker_meta.get(spk_id, {})
        for rel_path in rel_paths:
            audio_path = os.path.join(wav_root, rel_path)
            try:
                # Titanet acoustic embedding
                x, sr = librosa.load(audio_path, sr=16000, mono=True)
                sig = torch.from_numpy(x).unsqueeze(0).to(device)
                sig_len = torch.tensor([sig.shape[1]]).to(device)
                with torch.no_grad():
                    _, spk_embedding = audio_svc._embedding_model.forward(
                        input_signal=sig, input_signal_length=sig_len
                    )
                spk_emb_vector = spk_embedding.squeeze().cpu().tolist()

                # Placeholder text (VoxCeleb has no transcripts) -> semantic embedding
                text = f"VoxCeleb1 utterance from {sm.get('real_name', spk_id)}"
                sem_emb_vector = sem_svc._model.encode(text).tolist()

                shared_uuid = str(uuid.uuid4())
                metadata = {
                    "user_id": spk_id,
                    "audio_file": rel_path,
                    "data_type": "enroll",
                    "transcription": text,
                    "text_emotion": "Unknown",
                    "acoustic_emotion": "Unknown",
                    "language": "en",
                    "real_name": sm.get("real_name", ""),
                    "gender_csv": sm.get("gender_csv", ""),
                    "nationality": sm.get("nationality", ""),
                    "source": "voxceleb1_test",
                }

                audio_svc._collection.add(
                    ids=[shared_uuid],
                    embeddings=[spk_emb_vector],
                    metadatas=[metadata],
                )
                sem_svc._collection.add(
                    ids=[shared_uuid],
                    embeddings=[sem_emb_vector],
                    metadatas=[metadata],
                    documents=[text],
                )
                success_count += 1
                if success_count % 50 == 0:
                    print(f"  ...{success_count}/{total_files} enrolled")
            except Exception as e:
                print(f"Failed on {rel_path}: {e}")

    print("VoxCeleb1 Ingestion Complete!")
    return {
        "status": "Complete",
        "enrolled_count": success_count,
        "n_speakers": len(enrollment),
        "per_speaker": per_speaker,
    }