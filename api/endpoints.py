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