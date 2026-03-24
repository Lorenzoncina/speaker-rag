import os
import shutil
import tempfile
import torch
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Import the fully encapsulated RAG class from our core module
from speaker_rag.rag import SpeakerRAG

app = FastAPI(
    title="Speaker RAG API", 
    description="An LLM-powered RAG system for analyzing and querying speaker audio.",
    version="1.0.0"
)

# Global instance
rag_system = None

@app.on_event("startup")
def startup_event():
    global rag_system
    # Initialize the entire system (Titanet, Whisper, LLM, ChromaDB)
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
    """Checks if the API is running and the models are loaded."""
    return {
        "status": "Service is running", 
        "models_loaded": rag_system is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/rag/index", tags=["RAG"])
async def rag_index_speaker(
    audio: UploadFile = File(...),
    user_id: str = Form(None, description="A human-readable name for the speaker (e.g., 'Mario Rossi')")
):
    """
    Indexes a known speaker's audio into the vector database.
    If no user_id is provided, a random UUID will be assigned.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        result = rag_system.index_audio(audio_path=tmp_path, user_id=user_id)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/rag/ask", tags=["RAG"])
async def rag_ask_question(
    audio: UploadFile = File(...),
    question: str = Form(..., description="Ask the LLM about the speaker in the audio.")
):
    """
    Processes an unknown audio file, retrieves speaker context from the database, 
    and uses the LLM to answer the provided question.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        answer = rag_system.query(audio_path=tmp_path, question=question)
        
        return {
            "question": question, 
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)