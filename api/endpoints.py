import os
import json
import uuid
import shutil
import tempfile
import torch
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from speaker_rag.rag import SpeakerRAG
from speaker_rag.llm_extractor import LLMExtractor
from speaker_rag.local_llm_extractor import LocalLLMExtractor

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


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None


class IndexChatSessionRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    user_id: Optional[str] = "user"
    assistant_id: Optional[str] = "assistant"
    enable_kg_extraction: bool = True


@app.post("/rag/index_chat_session", tags=["RAG"])
async def rag_index_chat_session(req: IndexChatSessionRequest):
    """Enroll a chat session (user/LLM exchanges) into the memory layer.

    Mirrors what audio enrollment does, but for text-only chat turns. Each
    message becomes one utterance under a shared dialogue_id. User messages
    get KG entity/relation extraction; assistant messages are stored for
    semantic retrieval only.

    Returns dialogue_id and per-role enrollment counts.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "messages is required"})

    try:
        result = rag_system.index_chat_session(
            messages=[m.dict() for m in req.messages],
            user_id=req.user_id or "user",
            assistant_id=req.assistant_id or "assistant",
            session_id=req.session_id,
            enable_kg_extraction=req.enable_kg_extraction,
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

@app.post("/admin/probe_similarity", tags=["Admin"])
async def probe_similarity(audio: UploadFile = File(...), top_k: int = 20):
    """Return cosine similarity of a probe audio against all enrolled speaker embeddings.

    Useful for diagnosing why certain speakers are not identified (scores below
    SIMILARITY_THRESHOLD=0.50 get labeled Unknown_Speaker in the FactSheet).
    Returns the top_k enrolled embeddings sorted by score descending.
    """
    import librosa

    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models loading"})

    audio_svc = rag_system.audio_service
    device = audio_svc._device
    tmp_path = f"/tmp/probe_sim_{uuid.uuid4()}.wav"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await audio.read())

        x, sr = librosa.load(tmp_path, sr=16000, mono=True)
        sig = torch.from_numpy(x).unsqueeze(0).to(device)
        sig_len = torch.tensor([sig.shape[1]]).to(device)
        with torch.no_grad():
            _, emb = audio_svc._embedding_model.forward(
                input_signal=sig, input_signal_length=sig_len
            )
        emb_list = emb.squeeze().cpu().tolist()

        results = audio_svc._collection.query(
            query_embeddings=[emb_list],
            n_results=min(top_k, 500),
            include=["metadatas", "distances"],
            where={"data_type": "enroll"},
        )

        matches = []
        if results["ids"] and results["ids"][0]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                score = round(1.0 - dist, 4)
                matches.append({
                    "speaker_id": meta.get("user_id", "?"),
                    "score": score,
                    "audio_file": os.path.basename(meta.get("audio_file", "")),
                })

        return {"filename": audio.filename, "n_enrolled_returned": len(matches), "matches": matches}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ----------------------------------------------------------------------
# Extractor mode management
#
# The KG layer can run in one of three modes:
#   - "anthropic": Claude Haiku (entities) + Sonnet (relations + judges)
#   - "local":     Qwen3-14B in 8-bit, all four LLM calls happen on-GPU
#   - "off":       no LLM-backed extraction; graph falls back to regex/spaCy
#
# Switching to "local" loads Qwen3-14B eagerly so the user sees the cost up
# front. Switching away from "local" unloads it to free VRAM (so the UI's
# Qwen2-7B answerer can fit alongside the audio stack again).
#
# `_extractor_mode` tracks the current mode for status reporting.
# `_local_extractor` is the live LocalLLMExtractor singleton — kept module-
# level so `/llm/generate` can call it without poking at GraphService internals.
# ----------------------------------------------------------------------
_extractor_mode: str = "off"
_local_extractor: Optional[LocalLLMExtractor] = None


def _build_mode_status(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render the current mode + underlying extractor status as one payload."""
    base = rag_system.graph_service.llm_status() if rag_system else {"enabled": False}
    out = {"mode": _extractor_mode, **base}
    if extra:
        out.update(extra)
    return out


def _switch_to_anthropic(api_key: str, model: Optional[str]) -> Dict[str, Any]:
    """Build an Anthropic extractor, swap it onto GraphService, drop any local model."""
    global _extractor_mode, _local_extractor

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if model:
        kwargs["model"] = model
    extractor = LLMExtractor(**kwargs)
    status = rag_system.graph_service.set_llm_extractor(extractor)

    if status.get("enabled"):
        if _local_extractor is not None:
            # Free the 8-bit Qwen3 weights now that we're back on the API path.
            _local_extractor.unload()
            _local_extractor = None
        _extractor_mode = "anthropic"
    return _build_mode_status()


def _switch_to_local(model_name: Optional[str]) -> Dict[str, Any]:
    """Eagerly load Qwen3-14B (or override) and route GraphService through it."""
    global _extractor_mode, _local_extractor

    if _local_extractor is not None:
        # Already loaded — reuse it. Avoid double-loading 16+ GB of weights.
        rag_system.graph_service.set_llm_extractor(_local_extractor)
        _extractor_mode = "local"
        return _build_mode_status()

    extractor = LocalLLMExtractor(model_name=model_name or LocalLLMExtractor.DEFAULT_MODEL)
    if not extractor.available:
        # Load failed — surface the reason and keep the previous mode untouched.
        return _build_mode_status(extra={
            "enabled": False,
            "reason": extractor.unavailable_reason,
        })

    _local_extractor = extractor
    rag_system.graph_service.set_llm_extractor(extractor)
    _extractor_mode = "local"
    return _build_mode_status()


def _switch_to_off() -> Dict[str, Any]:
    """Disable LLM-backed extraction entirely and free any local model."""
    global _extractor_mode, _local_extractor

    rag_system.graph_service.set_llm_extractor(None)
    if _local_extractor is not None:
        _local_extractor.unload()
        _local_extractor = None
    _extractor_mode = "off"
    return _build_mode_status()


class SetLLMKeyRequest(BaseModel):
    api_key: str
    model: Optional[str] = None  # defaults to LLMExtractor's default (Haiku 4.5)


class SetExtractorModeRequest(BaseModel):
    mode: str  # "anthropic" | "local" | "off"
    api_key: Optional[str] = None  # required when mode == "anthropic"
    model: Optional[str] = None  # optional override for either backend


@app.get("/admin/extractor_mode", tags=["Admin"])
async def extractor_mode_status():
    """Report the active extractor mode and underlying LLM status."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    return _build_mode_status()


@app.post("/admin/extractor_mode", tags=["Admin"])
async def set_extractor_mode(req: SetExtractorModeRequest):
    """Switch between Anthropic / local Qwen3-14B / off.

    Loading Qwen3-14B in 8-bit takes ~30 s once cached and ~10 min on the first
    request (HuggingFace download). The request blocks until the model is on
    GPU so the UI can render an honest "ready" status afterwards.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    mode = (req.mode or "").strip().lower()
    if mode == "anthropic":
        key = (req.api_key or "").strip()
        if not key:
            return JSONResponse(
                status_code=400,
                content={"error": "api_key is required for mode=anthropic"},
            )
        status = _switch_to_anthropic(api_key=key, model=req.model)
        if not status.get("enabled"):
            return JSONResponse(status_code=400, content=status)
        return status
    if mode == "local":
        status = _switch_to_local(model_name=req.model)
        if not status.get("enabled"):
            return JSONResponse(status_code=400, content=status)
        return status
    if mode == "off":
        return _switch_to_off()
    return JSONResponse(
        status_code=400,
        content={"error": f"unknown mode '{mode}'. Expected: anthropic | local | off"},
    )


# ----------------------------------------------------------------------
# Back-compat shims for the original /admin/llm_key endpoints.
# These were the only way to enable LLM extraction before /admin/extractor_mode
# existed. They keep working so older UI builds and curl scripts don't break.
# ----------------------------------------------------------------------
@app.get("/admin/llm_key", tags=["Admin"])
async def llm_key_status():
    """Report whether LLM-backed entity extraction is currently active.
    The key itself is never returned."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    return _build_mode_status()


@app.post("/admin/llm_key", tags=["Admin"])
async def set_llm_key(req: SetLLMKeyRequest):
    """Compatibility shim: equivalent to POST /admin/extractor_mode {mode='anthropic'}."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    key = (req.api_key or "").strip()
    if not key:
        return JSONResponse(status_code=400, content={"error": "api_key is required"})

    status = _switch_to_anthropic(api_key=key, model=req.model)
    if not status.get("enabled"):
        return JSONResponse(status_code=400, content=status)
    return status


@app.delete("/admin/llm_key", tags=["Admin"])
async def clear_llm_key():
    """Disable LLM-backed extraction; subsequent ingests fall back to regex/spaCy."""
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    return _switch_to_off()


# ----------------------------------------------------------------------
# Local-LLM generation pass-through used by the UI when in "local" mode.
# ----------------------------------------------------------------------
class LLMGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 300


@app.post("/llm/generate", tags=["LLM"])
async def llm_generate(req: LLMGenerateRequest):
    """Run the loaded local Qwen3-14B on a fact-sheet prompt and return the answer.

    Only works when mode == 'local'. In any other mode this returns 409 so the
    caller knows to fall back to its own (UI-side) Qwen2-7B pipeline.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    if _extractor_mode != "local" or _local_extractor is None or not _local_extractor.available:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Local LLM is not loaded. Switch /admin/extractor_mode to 'local' first.",
                "mode": _extractor_mode,
            },
        )
    if not (req.prompt or "").strip():
        return JSONResponse(status_code=400, content={"error": "prompt is required"})

    try:
        text = _local_extractor.generate(req.prompt, max_new_tokens=int(req.max_new_tokens))
        return {"model": _local_extractor.model_name, "text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/dedupe_entities", tags=["Admin"])
async def admin_dedupe_entities():
    """Run a retrospective entity-resolution pass over the existing graph.

    Walks every entity node from least- to most-mentioned and asks the resolver
    whether each could merge into another existing entity. Useful after a bulk
    ingest pre-dating the resolver, or for one-shot graph cleanup. Persists the
    graph after the pass.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    try:
        stats = rag_system.graph_service.dedupe_all_entities()
        rag_system.graph_service.persist()
        return {"status": "complete", **stats}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/rag/graph_viz", tags=["RAG"])
async def get_graph_viz(max_entities: int = 40):
    """Return a simplified speaker-entity bipartite graph for visualization.

    Utterance nodes are omitted. Entity nodes are limited to the top max_entities
    by total mention count. Useful for rendering in the Gradio UI.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})
    try:
        data = rag_system.graph_service.get_viz_data(max_entities=max_entities)
        return data
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

            rag_system.graph_service.index_utterance(
                speaker_id=speaker,
                chroma_id=shared_uuid,
                text=text,
                metadata=metadata,
            )

            success_count += 1
            if success_count % 100 == 0:
                print(f"Ingested {success_count} / 1000 MELD files...")

        except Exception as e:
            print(f"Failed on Dialogue {row.get('Dialogue_ID')} Utt {row.get('Utterance_ID')}: {e}")

    rag_system.graph_service.persist()
    print("MELD Ingestion Complete!")
    return {"status": "Complete", "enrolled_count": success_count}


class IngestVoxcelebRequest(BaseModel):
    per_speaker: int = 10
    seed: int = 42
    probe_rel_paths: Optional[List[str]] = None


@app.post("/admin/ingest_voxceleb", tags=["Admin"])
def admin_ingest_voxceleb(req: IngestVoxcelebRequest = IngestVoxcelebRequest()):
    """Ingest the 40 VoxCeleb1 test speakers into the RAG database.

    Picks req.per_speaker random utterances per speaker, EXCLUDING files that
    appear as probes in the evaluation set.

    req.probe_rel_paths: list of rel-paths (e.g. "id10270/.../00001.wav") to
      exclude from enrollment.  The UI reads vox1o_pairs.json locally and sends
      this list, so no shared filesystem between containers is needed.
      When omitted, ALL files in Vox1-O.txt are excluded (conservative fallback).
    """
    import csv
    import random
    import librosa

    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    trials_path = "/app/VoxCeleb1/Vox1-O/Vox1-O.txt"
    wav_root = "/app/VoxCeleb1/wav/test"
    meta_path = "/app/VoxCeleb1/vox1_meta.csv"

    if not os.path.exists(trials_path):
        return JSONResponse(status_code=404, content={"error": f"Vox1-O trials not found at {trials_path}"})
    if not os.path.isdir(wav_root):
        return JSONResponse(status_code=404, content={"error": f"VoxCeleb1 test wavs not found at {wav_root}"})

    # 1. Build speaker set from the full trials file
    speakers = set()
    with open(trials_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, a1, a2 = parts
            speakers.add(a1.split("/", 1)[0])
            speakers.add(a2.split("/", 1)[0])

    # 2. Build probe_set — use caller-supplied list when available
    if req.probe_rel_paths:
        probe_set = set(req.probe_rel_paths)
        print(f"VoxCeleb1: caller-supplied probe_set ({len(probe_set)} files)")
    else:
        probe_set = set()
        with open(trials_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                _, a1, a2 = parts
                probe_set.add(a1)
                probe_set.add(a2)
        print(f"VoxCeleb1: full Vox1-O probe_set ({len(probe_set)} files) — no list supplied")

    speakers = sorted(speakers)
    print(f"VoxCeleb1: {len(speakers)} speakers, {len(probe_set)} probe files excluded.")

    # 3. Optional metadata (gender, nationality, real name)
    speaker_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    spk_id, name, gender, nationality, _set = (c.strip() for c in row[:5])
                    speaker_meta[spk_id] = {
                        "real_name": name.replace("_", " "),
                        "gender_csv": gender,
                        "nationality": nationality,
                    }

    # 4. Select enrollment files per speaker
    rng = random.Random(req.seed)
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
            for wav_file in sorted(os.listdir(vp)):
                if wav_file.endswith(".wav"):
                    all_files.append(f"{spk}/{video_dir}/{wav_file}")
        candidates = [f for f in all_files if f not in probe_set]
        if not candidates:
            print(f"  [warn] {spk}: no enrollment candidates after probe exclusion — skipped")
            continue
        chosen = candidates if len(candidates) <= req.per_speaker else rng.sample(candidates, req.per_speaker)
        enrollment[spk] = chosen

    total_files = sum(len(v) for v in enrollment.values())
    print(f"VoxCeleb1: enrolling {total_files} files across {len(enrollment)} speakers...")

    # 5. Enroll into both collections
    audio_svc = rag_system.audio_service
    sem_svc = rag_system.semantic_service
    device = audio_svc._device

    success_count = 0
    for spk_id, rel_paths in enrollment.items():
        sm = speaker_meta.get(spk_id, {})
        for rel_path in rel_paths:
            audio_path = os.path.join(wav_root, rel_path)
            try:
                x, sr = librosa.load(audio_path, sr=16000, mono=True)
                sig = torch.from_numpy(x).unsqueeze(0).to(device)
                sig_len = torch.tensor([sig.shape[1]]).to(device)
                with torch.no_grad():
                    _, spk_embedding = audio_svc._embedding_model.forward(
                        input_signal=sig, input_signal_length=sig_len
                    )
                spk_emb_vector = spk_embedding.squeeze().cpu().tolist()

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
                audio_svc._collection.add(ids=[shared_uuid], embeddings=[spk_emb_vector], metadatas=[metadata])
                sem_svc._collection.add(ids=[shared_uuid], embeddings=[sem_emb_vector], metadatas=[metadata], documents=[text])
                rag_system.graph_service.index_utterance(
                    speaker_id=spk_id,
                    chroma_id=shared_uuid,
                    text=text,
                    metadata=metadata,
                )
                success_count += 1
                if success_count % 50 == 0:
                    print(f"  ...{success_count}/{total_files} enrolled")
            except Exception as e:
                print(f"Failed on {rel_path}: {e}")

    rag_system.graph_service.persist()
    print("VoxCeleb1 Ingestion Complete!")
    return {
        "status": "Complete",
        "enrolled_count": success_count,
        "n_speakers": len(enrollment),
        "per_speaker": req.per_speaker,
    }


# ==========================================
# LongMemEval benchmark
# ==========================================
from functools import lru_cache

_LONGMEMEVAL_DIR = "/app/LongMemEval_data"
_LONGMEMEVAL_SPLITS = {
    "oracle":    "longmemeval_oracle.json",
    "s_cleaned": "longmemeval_s_cleaned.json",
    "m_cleaned": "longmemeval_m_cleaned.json",
}


@lru_cache(maxsize=4)
def _load_longmemeval_split(split: str):
    """Lazy-load and cache a LongMemEval split file."""
    if split not in _LONGMEMEVAL_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {list(_LONGMEMEVAL_SPLITS)}")
    path = os.path.join(_LONGMEMEVAL_DIR, _LONGMEMEVAL_SPLITS[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found inside the container: {path}")
    with open(path) as f:
        return json.load(f)


@app.get("/admin/longmemeval/splits", tags=["LongMemEval"])
def admin_longmemeval_splits():
    """Report which LongMemEval split files are available inside the container."""
    out = []
    for key, fname in _LONGMEMEVAL_SPLITS.items():
        full = os.path.join(_LONGMEMEVAL_DIR, fname)
        out.append({
            "split": key,
            "file": fname,
            "exists": os.path.exists(full),
        })
    return {"dir": _LONGMEMEVAL_DIR, "splits": out}


@app.get("/admin/longmemeval/questions", tags=["LongMemEval"])
def admin_longmemeval_questions(split: str = "oracle", limit: int = -1):
    """Return a compact list of question metadata for one LongMemEval split.

    The benchmark runner in the UI iterates this list, calling /prepare and
    /retrieve (or /haystack for the LLM-only baseline) for each question.
    """
    try:
        data = _load_longmemeval_split(split)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    items = data if (limit is None or int(limit) <= 0) else data[: int(limit)]
    out = []
    for entry in items:
        out.append({
            "question_id":   entry.get("question_id", ""),
            "question":      entry.get("question", ""),
            "answer":        entry.get("answer", ""),
            "question_type": entry.get("question_type", ""),
            "question_date": entry.get("question_date", ""),
            "n_sessions":    len(entry.get("haystack_sessions", [])),
        })
    return {"split": split, "count": len(out), "questions": out}


class LongMemEvalPrepareRequest(BaseModel):
    split: str
    question_id: str
    enable_kg: bool = True


@app.post("/admin/longmemeval/prepare", tags=["LongMemEval"])
def admin_longmemeval_prepare(req: LongMemEvalPrepareRequest):
    """Clear the database and enroll the haystack of a single LongMemEval question.

    Each session in `haystack_sessions` becomes one chat dialogue, indexed via
    `index_chat_session`. The timestamp from `haystack_dates[i]` is attached
    to every message in that session so the retrieval/KG layers can use it.

    When `enable_kg=False`, the LLM entity/relation extraction is skipped
    (semantic retrieval still works) — useful for fast iteration on the
    larger splits.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        data = _load_longmemeval_split(req.split)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    entry = next((d for d in data if d.get("question_id") == req.question_id), None)
    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": f"question_id '{req.question_id}' not found in split '{req.split}'"},
        )

    # Fresh memory per instance — the benchmark requires per-question isolation.
    rag_system.clear_database()

    haystack_sessions    = entry.get("haystack_sessions", []) or []
    haystack_dates       = entry.get("haystack_dates", []) or []
    haystack_session_ids = entry.get("haystack_session_ids", []) or []

    total_turns = 0
    for i, session in enumerate(haystack_sessions):
        ts = haystack_dates[i] if i < len(haystack_dates) else None
        messages = []
        for msg in session:
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue
            messages.append({"role": role, "content": content, "timestamp": ts})

        if not messages:
            continue

        sess_id = haystack_session_ids[i] if i < len(haystack_session_ids) else f"sess_{i}"
        result = rag_system.index_chat_session(
            messages=messages,
            user_id="user",
            assistant_id="assistant",
            session_id=str(sess_id),
            enable_kg_extraction=req.enable_kg,
        )
        total_turns += result.get("enrolled_count", 0)

    return {
        "status": "success",
        "split": req.split,
        "question_id": req.question_id,
        "n_sessions_enrolled": len(haystack_sessions),
        "n_turns_enrolled": total_turns,
        "enable_kg": req.enable_kg,
    }


@app.get("/admin/longmemeval/haystack", tags=["LongMemEval"])
def admin_longmemeval_haystack(split: str, question_id: str):
    """Return the full haystack for one question — used by the LLM-only baseline.

    Returns the raw sessions and dates so the UI can flatten them into a
    single context blob and feed Qwen directly (no RAG retrieval involved).
    """
    try:
        data = _load_longmemeval_split(split)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    entry = next((d for d in data if d.get("question_id") == question_id), None)
    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": f"question_id '{question_id}' not found in split '{split}'"},
        )

    return {
        "split":             split,
        "question_id":       question_id,
        "question":          entry.get("question", ""),
        "answer":            entry.get("answer", ""),
        "question_type":     entry.get("question_type", ""),
        "question_date":     entry.get("question_date", ""),
        "haystack_dates":    entry.get("haystack_dates", []),
        "haystack_sessions": entry.get("haystack_sessions", []),
    }


# =============================================================================
# MLC-SLM Task 2 — Multilingual Conversational Speech Understanding benchmark.
#
# Same client/server split as LongMemEval: the CLI/UI runner is a thin HTTP
# client; the backend owns the dataset (multi-speaker phone-call .wav files +
# .txt reference transcripts + a QA JSONL with multi-choice questions).
#
# Path resolution: the QA JSONL paths look like
#   mlc-slm-2nd-dev/English_Australian/1127_015_phone.wav
# but on disk the file lives at
#   <audio_root>/English/Australian/1127_015_phone.wav
# We index every .wav under audio_root once (cached per root) and look up by
# basename — robust to any folder-naming drift in the JSONL.
# =============================================================================


from pathlib import Path as _PathML


@lru_cache(maxsize=4)
def _mlcslm_build_audio_index(audio_root: str) -> Dict[str, str]:
    """Walk audio_root and return {basename: absolute_path} for every *.wav.

    Cached per audio_root so repeated /prepare calls don't rewalk the tree.
    Cache invalidates only on process restart — the dev dataset is static, so
    that's fine.
    """
    root = _PathML(audio_root)
    if not root.is_dir():
        raise FileNotFoundError(f"audio_root not found: {audio_root}")
    return {p.name: str(p.resolve()) for p in root.rglob("*.wav")}


def _mlcslm_resolve_wav(jsonl_path: str, audio_root: str) -> str:
    """Resolve a QA-JSONL audio path to its actual on-disk path via basename."""
    index = _mlcslm_build_audio_index(audio_root)
    basename = os.path.basename(jsonl_path)
    real = index.get(basename)
    if not real:
        raise FileNotFoundError(
            f"audio {basename!r} (jsonl path {jsonl_path!r}) not found under {audio_root}"
        )
    return real


@app.get("/admin/mlcslm/qa", tags=["MLC-SLM"])
def admin_mlcslm_qa(qa_jsonl: str, limit: int = -1):
    """Load and return MLC-SLM Task 2 QA records from a JSONL on the BACKEND.

    The runner client uses this to fetch the question list without needing
    the dataset to be mounted in its own container.
    """
    if not os.path.isfile(qa_jsonl):
        return JSONResponse(status_code=404,
                            content={"error": f"QA file not found on backend: {qa_jsonl}"})
    try:
        records: List[dict] = []
        with open(qa_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"failed to parse JSONL: {e}"})

    if limit and int(limit) > 0:
        records = records[: int(limit)]
    n_questions = sum(len(r.get("questions", [])) for r in records)
    return {
        "qa_jsonl": qa_jsonl,
        "n_audios": len(records),
        "n_questions": n_questions,
        "records": records,
    }


@app.get("/admin/mlcslm/transcript", tags=["MLC-SLM"])
def admin_mlcslm_transcript(audio_path: str, audio_root: str, max_chars: int = 0):
    """Read the .txt reference transcript that ships alongside a .wav.

    audio_path: the JSONL-style path (or any path whose basename matches a
                .wav under audio_root).
    audio_root: backend-side root containing the dataset language folders.
    max_chars: optional truncation cap; <=0 means no truncation.
    """
    try:
        wav_real = _mlcslm_resolve_wav(audio_path, audio_root)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    txt_real = os.path.splitext(wav_real)[0] + ".txt"
    if not os.path.isfile(txt_real):
        return JSONResponse(status_code=404,
                            content={"error": f"transcript not found alongside wav: {txt_real}"})

    try:
        with open(txt_real, "r", errors="replace") as f:
            text = f.read().strip()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"failed to read transcript: {e}"})

    truncated = False
    if int(max_chars) > 0 and len(text) > int(max_chars):
        text = text[: int(max_chars)] + "\n\n[...truncated due to context limit...]"
        truncated = True

    return {
        "audio_path": audio_path,
        "wav_path":   wav_real,
        "txt_path":   txt_real,
        "transcript": text,
        "n_chars":    len(text),
        "truncated":  truncated,
    }


class SetEmotionModelRequest(BaseModel):
    enabled: bool


# Module-level stash for the emotion model + processor so we can restore them
# after a /admin/emotion_model {enabled:false} call.
_emotion_model_stash: Dict[str, Any] = {"model": None, "processor": None, "text_pipeline": None}


@app.post("/admin/emotion_model", tags=["Admin"])
def admin_set_emotion_model(req: SetEmotionModelRequest):
    """Enable or disable the speech-emotion model in the running worker.

    The English-only wav2vec2 emotion model triggers CUDA device-side asserts
    on non-English audio (Korean / Japanese / etc.) because some classifier
    weights are randomly initialized and produce out-of-range logits. For the
    MLC-SLM benchmark we don't need emotion labels in retrieval, so the
    cleanest fix is to skip the model entirely for this run.

    Toggling is in-process and survives until the worker restarts (or a
    matching call with enabled=true). After enabled=false:
      - `_emotion_model` and `_emotion_processor` on the Identification_Service
        singleton are nulled out; the emotion check at
        identification_service.py:219 then sets acoustic_emotion="Unknown".
      - The semantic text-emotion pipeline is also paused, since it depends
        on Whisper output and shares the same crash-prone code paths.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    svc = rag_system.audio_service

    if req.enabled:
        # Restore from the stash (if we ever disabled).
        if _emotion_model_stash["model"] is not None:
            svc._emotion_model = _emotion_model_stash["model"]
            svc._emotion_processor = _emotion_model_stash["processor"]
            svc._text_emotion_pipeline = _emotion_model_stash["text_pipeline"]
            _emotion_model_stash["model"] = None
            _emotion_model_stash["processor"] = None
            _emotion_model_stash["text_pipeline"] = None
            return {"status": "success", "enabled": True, "restored": True}
        # Nothing was disabled — report current state.
        return {
            "status": "success",
            "enabled": svc._emotion_model is not None,
            "restored": False,
        }

    # Disable: only stash if we haven't already.
    if _emotion_model_stash["model"] is None and svc._emotion_model is not None:
        _emotion_model_stash["model"] = svc._emotion_model
        _emotion_model_stash["processor"] = svc._emotion_processor
        _emotion_model_stash["text_pipeline"] = svc._text_emotion_pipeline
    svc._emotion_model = None
    svc._emotion_processor = None
    svc._text_emotion_pipeline = None
    return {"status": "success", "enabled": False}


class ReloadDiarizationRequest(BaseModel):
    hf_auth_token: str  # the fresh HF token, with access to pyannote/speaker-diarization-3.1


@app.post("/admin/reload_diarization", tags=["Admin"])
def admin_reload_diarization(req: ReloadDiarizationRequest):
    """Rebuild the pyannote diarization pipeline inside the running worker.

    The diarization service is a module-level singleton that's cached once at
    first use. If it was built with a missing/invalid HF token, the underlying
    pyannote pipeline ends up as None and every conversation enrollment dies
    with 'NoneType is not callable'. This endpoint lets you swap in a working
    token and force the singleton + the RAG-system-level cache to be rebuilt,
    without restarting the FastAPI process.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    token = (req.hf_auth_token or "").strip()
    if not token:
        return JSONResponse(status_code=400, content={"error": "hf_auth_token is required"})

    # Compatibility shims: pyannote.audio 4.x calls several torchaudio APIs
    # that were removed in torchaudio >= 2.5 / 2.7. Patch the module BEFORE
    # we trigger the pipeline build so pyannote finds the attributes it expects.
    try:
        import torchaudio

        # 1. torchaudio.AudioMetaData — removed in torchaudio >= 2.7.
        if not hasattr(torchaudio, "AudioMetaData"):
            _patched = False
            for modpath in ("torchaudio._backend.common", "torchaudio.backend.common"):
                try:
                    mod = __import__(modpath, fromlist=["AudioMetaData"])
                    torchaudio.AudioMetaData = mod.AudioMetaData
                    _patched = True
                    break
                except (ImportError, AttributeError):
                    continue
            if not _patched:
                class _AudioMetaDataStub:
                    pass
                torchaudio.AudioMetaData = _AudioMetaDataStub

        # 2. torchaudio.list_audio_backends — deprecated in 2.0, removed in 2.5+.
        # pyannote uses it to probe which backends are available; the new
        # backend layer always has soundfile, so reporting just that is safe.
        if not hasattr(torchaudio, "list_audio_backends"):
            def _list_audio_backends_compat():
                names = []
                for name in ("soundfile", "sox_io", "ffmpeg"):
                    try:
                        __import__(name)
                        names.append(name)
                    except ImportError:
                        continue
                return names or ["soundfile"]
            torchaudio.list_audio_backends = _list_audio_backends_compat

        # 3. torchaudio.get_audio_backend / set_audio_backend — same removal.
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *_args, **_kwargs: None

        # 4. torchaudio.load / info / save:
        #    - In torchaudio >= 2.7, when they exist, they default to a
        #      torchcodec backend; wrap to force backend="soundfile".
        #    - In some builds (e.g. torchaudio 2.8+), `info` is gone as a
        #      top-level attribute. Install a soundfile-based stub so
        #      pyannote's `torchaudio.info(path)` lookup keeps working.
        def _make_wrapper(orig_fn):
            def _wrapped(*args, **kwargs):
                if "backend" not in kwargs:
                    kwargs["backend"] = "soundfile"
                return orig_fn(*args, **kwargs)
            _wrapped._compat_wrapped = True
            return _wrapped

        def _info_stub(uri, *_args, **_kwargs):
            import soundfile as sf
            with sf.SoundFile(str(uri)) as f:
                meta = torchaudio.AudioMetaData()
                meta.sample_rate     = int(f.samplerate)
                meta.num_frames      = int(len(f))
                meta.num_channels    = int(f.channels)
                meta.bits_per_sample = 0
                meta.encoding        = str(f.subtype or "PCM_S")
                return meta
        _info_stub._compat_wrapped = True

        def _load_stub(uri, *_args, **kwargs):
            import soundfile as sf
            import torch as _torch
            data, sr = sf.read(str(uri), always_2d=True, dtype="float32")
            channels_first = kwargs.get("channels_first", True)
            arr = data.T if channels_first else data
            return _torch.from_numpy(arr).contiguous(), int(sr)
        _load_stub._compat_wrapped = True

        def _save_stub(uri, src, sample_rate, *_args, **kwargs):
            import soundfile as sf
            channels_first = kwargs.get("channels_first", True)
            t = src.cpu().detach().numpy()
            if channels_first:
                t = t.T
            sf.write(str(uri), t, int(sample_rate))
        _save_stub._compat_wrapped = True

        for _fn_name, _stub in (("load", _load_stub), ("info", _info_stub), ("save", _save_stub)):
            _fn = getattr(torchaudio, _fn_name, None)
            if _fn is None:
                # Attribute is gone — install a soundfile-based stub.
                setattr(torchaudio, _fn_name, _stub)
            elif not getattr(_fn, "_compat_wrapped", False):
                # Attribute exists — wrap it to default backend="soundfile".
                setattr(torchaudio, _fn_name, _make_wrapper(_fn))
    except Exception:
        pass

    # Wipe the module-level singleton so the factory rebuilds from scratch.
    from speaker_rag import diarization_service as diar_mod
    diar_mod._Diarization_Service._instance = None
    diar_mod._Diarization_Service._pipeline = None
    diar_mod._Diarization_Service._embedding_model = None
    diar_mod._Diarization_Service._device = None

    # Also clear the per-rag-system cache so the lazy property rebuilds.
    rag_system._diarization_service = None
    rag_system._hf_auth_token = token

    # Trigger a fresh build right now so the caller sees any failure here,
    # not on the first /admin/mlcslm/prepare call later.
    try:
        svc = rag_system.diarization_service  # property → factory call
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": f"diarization rebuild failed: {e}"})

    pipeline_ok = getattr(svc, "_pipeline", None) is not None
    if not pipeline_ok:
        return JSONResponse(status_code=500, content={
            "error": "diarization pipeline is still None after rebuild — token is likely "
                     "invalid, or the account has not accepted the gated model license at "
                     "https://huggingface.co/pyannote/speaker-diarization-3.1",
        })

    return {
        "status": "success",
        "message": "pyannote diarization pipeline rebuilt and ready.",
    }


class MLCSLMPrepareRequest(BaseModel):
    audio_path: str  # JSONL-style path, e.g. "mlc-slm-2nd-dev/English_Australian/1127.wav"
    audio_root: str  # backend-side absolute path to the dataset root
    clear_first: bool = True  # wipe the DB before enrolling (per-question isolation)


@app.post("/admin/mlcslm/prepare", tags=["MLC-SLM"])
def admin_mlcslm_prepare(req: MLCSLMPrepareRequest):
    """Resolve one MLC-SLM audio, clear the DB, and run the full enrollment.

    Equivalent to the LongMemEval /admin/longmemeval/prepare flow but for a
    single multi-speaker audio file. After this returns 200, the caller can
    immediately POST questions to /rag/retrieve + /llm/generate.
    """
    if not rag_system:
        return JSONResponse(status_code=500, content={"error": "Models are still loading."})

    try:
        wav_real = _mlcslm_resolve_wav(req.audio_path, req.audio_root)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    if req.clear_first:
        try:
            rag_system.clear_database()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"clear_database failed: {e}"})

    try:
        result = rag_system.index_conversation(
            audio_path=wav_real,
            speaker_map=None,
            num_speakers=None,
            vtt_path=None,
        )
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": f"index_conversation failed: {e}"})

    return {
        "status":      "success",
        "audio_path":  req.audio_path,
        "wav_path":    wav_real,
        "enrollment":  result,
    }