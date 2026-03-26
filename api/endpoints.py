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
        llm_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device=device
    )

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "Service is running", "models_loaded": rag_system is not None}

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
            
            # Save files to disk preserving original filenames
            for audio in audios:
                tmp_path = os.path.join(tmpdir, audio.filename)
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
                tmp_paths.append(tmp_path)

            # Pass the files and IDs directly to the indexer
            result = rag_system.index_audio(audio_paths=tmp_paths, user_ids=user_ids)
            return result
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/rag/ask", tags=["RAG"])
async def rag_ask_question(
    audios: List[UploadFile] = File(...),
    question: str = Form(..., description="Ask the LLM about the speakers in the audio files.")
):
    """Processes unknown audio files and uses the LLM to answer your question."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_paths = []
            # Save all probe files preserving original filenames
            for audio in audios:
                tmp_path = os.path.join(tmpdir, audio.filename)
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
                tmp_paths.append(tmp_path)

            answer = rag_system.query(audio_paths=tmp_paths, question=question)
            
            return {
                "question": question, 
                "answer": answer,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})