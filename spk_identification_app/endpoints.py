"""
Script Name: endpoints.py
Description: FastAPI endpoints for Speaker Identification (IAMI Compliant with Enroll/Probe logic).
Author: Lorenzo Concina
Date: 10/12/2025
"""
import os
import uuid
import json
import shutil
import tempfile
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from identification_service import Identification_Service, AgeProbe #for Docker
#from .identification_service import Identification_Service, AgeProbe #for local test

# ===================================================================
# IAMI Attribute Extraction endpoints Models  
# ===================================================================

class AttributeType(BaseModel):
    name: str
    entry_type: str = "natural_person"
    data_type: str = "string"

class IamiAttribute(BaseModel):
    type: AttributeType
    values: List[Any]
    confidences: List[float] = [1.0]
    source: Dict = {}
    notes: str = ""

class Relationship(BaseModel):
    source_entry_id: str
    target_entry_id: str
    relationship_type: str
    metadata: Dict = {}

class Entry(BaseModel):
    id: str
    collection: str = "natural_person"
    attributes: List[IamiAttribute]
    relationships: List[Relationship] = []
    collection_db_entry_id: Optional[str] = None
    enrollment_timestamp: str
    enrollment_operator: str = "system"
    session_id: Optional[str] = None
    case_id: Optional[str] = None
    group_id: Optional[str] = None

class AudioRequestData(BaseModel):
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None
    return_debug_info: bool = False

class IamiExtractionResponse(BaseModel):
    entries: List[Entry]
    debug_info: Dict = {}
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: str
    success: bool
    message: str

# ===================================================================
# IAMI Identify Endpoint Models
# ===================================================================

class ProbeEntity(BaseModel):
    id: str
    collection_db_entry_id: str 
    collection: Optional[str] = None
    attributes: Optional[List[Any]] = None
    
class ProbeSubgraph(BaseModel):
    anchor_entry_id: str
    entities: List[ProbeEntity]

class IdentifyRequest(BaseModel):
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None
    probe: ProbeSubgraph
    max_results: int = 5
    threshold: Optional[float] = None
    comparison_algorithms: Optional[Dict] = None

class MatchResult(BaseModel):
    final_score: float
    candidate_entry: Entry
    reasoning: List[str] = []

class IdentifyResponse(BaseModel):
    matches: List[MatchResult]
    debug_info: Dict = {}
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: str
    success: bool
    message: str

class UserIdPayload(BaseModel):
    user_id: str


# ===================================================================
# CONSTANTS & DEFAULTS
# ===================================================================

# This string will appear in the Swagger UI text box by default
DEFAULT_AUDIO_REQUEST_JSON = json.dumps({
    "case_id": "string",
    "session_id": "string",
    "request_id": "string",
    "timestamp": "2025-01-01T12:00:00Z",
    "metadata": {},
    "return_debug_info": False
}, indent=2)

# ===================================================================
# FastAPI App & Model Loading
# ===================================================================

app = FastAPI(title="Speaker Identification API", version="2.2.0")

TITANET_MODEL_DIR = "./embedding_model"
TITANET_MODEL_NAME = "titanet_large"
WHISPER_MODEL_DIR = "./whisper_model"
WHISPER_MODEL_NAME = "openai/whisper-tiny"
GENDER_MODEL_DIR = "./gender_model"
GENDER_MODEL_NAME = "prithivMLmods/Common-Voice-Gender-Detection"
AGE_MODEL_DIR = "./age_model"
AGE_MODEL_NAME = "age_probe.pt"

CHROMA_CLIENT_PATH = "./enrolled_data/chroma"
COLLECTION_NAME = "speaker_embeddings"

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Load Titanet
try:
    os.makedirs(TITANET_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(TITANET_MODEL_DIR, TITANET_MODEL_NAME + ".nemo")
    titanet_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=DEVICE)
except FileNotFoundError:
    titanet_model = EncDecSpeakerLabelModel.from_pretrained(model_name=TITANET_MODEL_NAME)
    titanet_model.save_to(model_path)
titanet_model.to(DEVICE)
titanet_model.freeze()

# 2. Load Whisper
try:
    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_DIR)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_DIR).to(DEVICE)
except OSError:
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
    whisper_processor.save_pretrained(WHISPER_MODEL_DIR)
    whisper_model.save_pretrained(WHISPER_MODEL_DIR)
    whisper_model.to(DEVICE)
whisper_model.eval()

# 3. Load Gender Model
try:
    os.makedirs(GENDER_MODEL_DIR, exist_ok=True)
    gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(GENDER_MODEL_DIR)
    gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(GENDER_MODEL_DIR).to(DEVICE)
except OSError:
    gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(GENDER_MODEL_NAME)
    gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(GENDER_MODEL_NAME)
    gender_processor.save_pretrained(GENDER_MODEL_DIR)
    gender_model.save_pretrained(GENDER_MODEL_DIR)
    gender_model.to(DEVICE)
gender_model.eval()

# 4. Load Age extimation Model
try:
    # input_dim=384 for Whisper-Tiny, num_classes=4 for the 4 age bins
    age_model = AgeProbe(input_dim=384, num_classes=4) 
    age_weights_path = os.path.join(AGE_MODEL_DIR, AGE_MODEL_NAME)
    age_model.load_state_dict(torch.load(age_weights_path, map_location=DEVICE))
    age_model.to(DEVICE).eval()
    print(f"Age model loaded successfully from {age_weights_path}")
except Exception as e:
    print(f"Failed to load Age model: {e}")
    age_model = None

# 5. Instantiate Service
service = Identification_Service(
    embedding_model=titanet_model,
    language_processor=whisper_processor,
    language_model=whisper_model,
    gender_processor=gender_processor,
    gender_model=gender_model,
    age_model=age_model,
    device=DEVICE,
    chroma_client=CHROMA_CLIENT_PATH,
    collection_name=COLLECTION_NAME
)

# ===================================================================
# Helper Functions
# ===================================================================

def map_attributes_to_iami_format(internal_attributes: Dict) -> List[IamiAttribute]:
    """
    Converts flat dictionary (from Chroma metadata) to IAMI format.
    """
    iami_attributes = []
    
    if "language" in internal_attributes:
        iami_attributes.append(IamiAttribute(
            type=AttributeType(name="language", data_type="string"),
            values=[internal_attributes["language"]],
            confidences=[internal_attributes.get("language_confidence", 1.0)]
        ))
    if "gender" in internal_attributes:
        iami_attributes.append(IamiAttribute(
            type=AttributeType(name="gender", data_type="string"),
            values=[internal_attributes["gender"]],
            confidences=[internal_attributes.get("gender_confidence", 1.0)]
        ))
    if "age" in internal_attributes:
        iami_attributes.append(IamiAttribute(
            type=AttributeType(name="age", data_type="string"),
            values=[internal_attributes["age"]],
            confidences=[internal_attributes.get("age_confidence", 1.0)]
        ))
    return iami_attributes


async def _process_audio_extraction(audios: List[UploadFile], request_data: str, data_type: str) -> IamiExtractionResponse:
    """
    Internal helper to process audio files for both Enroll and Probe endpoints.
    Args:
        data_type: 'enroll' or 'probe' - determines how data is tagged in ChromaDB.
    """
    if not service:
        return JSONResponse(status_code=500, content={"error": "Service not initialized."})
    
    try:
        req_data_dict = json.loads(request_data)
        req_model = AudioRequestData(**req_data_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in request_data: {str(e)}")


    entries_output = []
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            user_audio_path_map = {}
            
            for audio_file in audios:
                #1. Generate a unique "Speaker ID" for this specific file
                unique_speaker_id = str(uuid.uuid4())

                # 2. Save file to temp
                safe_filename = os.path.basename(audio_file.filename)
                temp_path = os.path.join(tmpdir, safe_filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(audio_file.file, buffer)

                # 3. Create the mapping: { UniqueID -> [File Path] }
                user_audio_path_map[unique_speaker_id] = [temp_path]
            
            # --- CALL SERVICE WITH data_type AND session_id ---
            extraction_results = service.extract_attributes(
                user_audios_map=user_audio_path_map,
                data_type=data_type,
                session_id=req_model.session_id,
                case_id=req_model.case_id
            )
            
            current_time_str = datetime.utcnow().isoformat()
            
            for assigned_user_id, results_list in extraction_results.items():
                for result_item in results_list:
                    converted_attrs = map_attributes_to_iami_format(result_item["attributes"])
                    
                    entry = Entry(
                        id=str(uuid.uuid4()),         # Unique ID for this API response object
                        collection="natural_person",
                        attributes=converted_attrs,
                        collection_db_entry_id=result_item["id"], # The ChromaDB Vector ID
                        enrollment_timestamp=current_time_str,
                        session_id=req_model.session_id,
                        case_id=req_model.case_id,
                        group_id=assigned_user_id 
                    )
                    entries_output.append(entry)

        return IamiExtractionResponse(
            entries=entries_output,
            debug_info={"processed_files": len(audios), "type": data_type},
            case_id=req_model.case_id,
            session_id=req_model.session_id,
            request_id=req_model.request_id,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
            message=f"Audio attributes extracted ({data_type}) successfully."
        )

    except Exception as e:
        return IamiExtractionResponse(
            entries=[],
            timestamp=datetime.utcnow().isoformat(),
            success=False,
            message=f"Processing failed: {str(e)}",
            request_id=req_model.request_id
        )

async def _process_identification_request(payload: IdentifyRequest, probe_only: bool) -> IdentifyResponse:
    """
    Internal helper to handle both Identification (Enroll+Probe) and Resolution (Probe only).
    """
    if not service:
        return JSONResponse(status_code=500, content={"error": "Service not initialized."})
    
    try:
        # 1. Find the Chroma ID for the anchor
        target_chroma_id = None
        anchor_id = payload.probe.anchor_entry_id
        
        for entity in payload.probe.entities:
            if entity.id == anchor_id:
                target_chroma_id = entity.collection_db_entry_id
                break
        
        if not target_chroma_id:
            return IdentifyResponse(
                matches=[], timestamp=datetime.utcnow().isoformat(),
                success=False, message="Anchor entity ID not found in entities list."
            )

        # 2. Call Service with the probe_only flag
        raw_results = service.identify_by_id(
            query_ids=[target_chroma_id],
            top_k=payload.max_results,
            session_id=payload.session_id,
            probe_only=probe_only 
        )
        
        # 3. Format Results to IAMI standard
        iami_matches = []
        current_time_str = datetime.utcnow().isoformat()

        if raw_results and len(raw_results) > 0:
            matches_data = raw_results[0]["matches"]
            
            for match in matches_data:
                chroma_meta = match.get("metadata", {})
                converted_attrs = map_attributes_to_iami_format(chroma_meta)
                
                candidate = Entry(
                    id=str(uuid.uuid4()), 
                    collection="natural_person",
                    attributes=converted_attrs,
                    relationships=[],
                    collection_db_entry_id=match["chroma_id"],
                    enrollment_timestamp=current_time_str,
                    session_id=payload.session_id,
                    case_id=payload.case_id,
                    group_id=chroma_meta.get("user_id", "unknown")
                )
                
                match_type = chroma_meta.get("data_type", "unknown")
                match_user = chroma_meta.get("user_id", "unknown")
                
                # Custom reasoning text based on mode
                mode_text = "Resolution" if probe_only else "Identification"
                reason_text = f"[{mode_text}] Matched {match_user} ({match_type}) with score {match['score']:.4f}"
                
                iami_matches.append(MatchResult(
                    final_score=match["score"],
                    candidate_entry=candidate,
                    reasoning=[reason_text]
                ))

        return IdentifyResponse(
            matches=iami_matches,
            debug_info={"queried_chroma_id": target_chroma_id, "mode": "resolution" if probe_only else "identification"},
            case_id=payload.case_id,
            session_id=payload.session_id,
            request_id=payload.request_id,
            timestamp=current_time_str,
            success=True,
            message=f"{'Resolution' if probe_only else 'Identification'} completed successfully."
        )

    except Exception as e:
        return IdentifyResponse(
            matches=[],
            timestamp=datetime.utcnow().isoformat(),
            success=False,
            message=f"Process failed: {str(e)}",
            request_id=payload.request_id
        )

# ===================================================================
# Endpoints
# ===================================================================

@app.get("/speaker-ID/", tags=["General"])
def read_root():
    return {"message": "Service is running"}

@app.get("/speaker-ID/users", tags=["User Management"])
async def list_users():
    try:
        results = service._collection.get(include=["metadatas"])
        user_ids = sorted(list(set(meta["user_id"] for meta in results["metadatas"])))
        return {"users": user_ids}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/speaker-ID/user/delete", tags=["User Management"])
async def delete_user_embeddings(payload: UserIdPayload):
    try:
        service._collection.delete(where={"user_id": payload.user_id})
        return {"status": "success", "message": f"Deleted user {payload.user_id}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 1. Endpoint for ENROLLMENT Data ---
@app.post("/speaker-ID/extract_attributes", response_model=IamiExtractionResponse, tags=["Core Functionality"])
async def extract_attributes(
    audios: List[UploadFile] = File(...),
    request_data: str = Form(
        DEFAULT_AUDIO_REQUEST_JSON, 
        description="JSON string of the AudioRequestData model."
    )
):
    """
    Extracts attributes for PERMANENT ENROLLMENT.
    Stored with metadata: data_type='enroll'.
    """
    return await _process_audio_extraction(audios, request_data, data_type="enroll")

# --- 2. Endpoint for PROBE Data ---
@app.post("/speaker-ID/extract_attributes_probe", response_model=IamiExtractionResponse, tags=["Core Functionality"])
async def extract_attributes_probe(
    audios: List[UploadFile] = File(...),
    request_data: str = Form(
        DEFAULT_AUDIO_REQUEST_JSON, 
        description="JSON string of the AudioRequestData model."
    )
):
    """
    Extracts attributes for SESSION PROBES.
    Stored with metadata: data_type='probe'.
    Only searchable within the same session_id.
    """
    return await _process_audio_extraction(audios, request_data, data_type="probe")

# --- 3. Identify Endpoint (Filtered Search) ---
@app.post("/speaker-ID/identify", response_model=IdentifyResponse, tags=["Core Functionality"])
async def identify_speakers(payload: IdentifyRequest):
    """
    Identifies speaker based on probe anchor ID.
    Search Scope: All 'enroll' data + 'probe' data from current session_id.
    """
    return await _process_identification_request(payload, probe_only=False)

# --- 4. Resolve Endpoint (Filtered Search: Probe Only) ---
@app.post("/speaker-ID/resolve", response_model=IdentifyResponse, tags=["Core Functionality"])
async def resolve_speakers(payload: IdentifyRequest):
    """
    Performs Speaker Resolution (Diarization/Clustering helper).
    Search Scope: STRICTLY 'probe' data from the current session_id.
    Used to find other segments in the same conversation belonging to this speaker.
    """
    return await _process_identification_request(payload, probe_only=True)