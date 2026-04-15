import os
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
    print(f"Starting Speaker RAG API on device: {device}")
    
    rag_system = SpeakerRAG(
        chroma_dir="./enrolled_data/chroma",
        collection_name="speaker_embeddings",
        device=device
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


@app.post("/rag/retrieve", tags=["RAG"])
async def rag_retrieve_context(
    audios: List[UploadFile] = File(...),
    question: str = Form(..., description="The user's query to guide the semantic search.")
):
    """Processes unknown audio files and returns the structured Fact Sheet for an external LLM to consume."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_paths = []
            for i, audio in enumerate(audios):
                safe_filename = f"{i}_{audio.filename}"
                tmp_path = os.path.join(tmpdir, safe_filename)
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
                tmp_paths.append(tmp_path)

            result = rag_system.build_context(audio_paths=tmp_paths, question=question)

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
    df = pd.read_csv(csv_path).head(1000)

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