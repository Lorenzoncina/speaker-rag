"""
Script Name: identification_service.py

Description:
This service handles the core logic for Speaker Identification. It manages the extraction of 
speaker embeddings (using Nvidia Titanet), audio attribute extraction (using Whisper for language detection), 
and the storage/retrieval of these embeddings using a ChromaDB vector database. It supports 
different data types ('enroll' vs 'probe') to implement selective search logic where probes 
are only matched against enrollment data or other probes from the same session.

Key Components:
- _Identification_Service (Singleton): Main class wrapping all AI models and database interactions.
- Titanet (EncDecSpeakerLabelModel): Computes speaker embeddings from audio waveforms.
- Whisper (WhisperForConditionalGeneration): Used for language identification/transcription tasks.
- ChromaDB: Vector database for storing embeddings with metadata (user_id, session_id, data_type).
- extract_attributes: Processes audio files, extracts embeddings/attributes, and enrolls them in the DB.
- identify_by_id: Performs vector similarity search with specific filtering rules (Enroll vs Probe logic).

Usage:
This script is intended to be imported and instantiated by the FastAPI `endpoints.py` script.
It relies on pre-loaded models passed via the factory function `Identification_Service`.

Author: Lorenzo Concina
Date: 08/07/2025
"""
import uuid
import torch
import librosa
from typing import Dict, List, Any, Optional
from chromadb import Client, PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from nemo.collections.asr.models import EncDecSpeakerLabelModel


class AgeProbe(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int, pool_type: str = "mean", dropout: float = 0.2):
        super().__init__()
        self.pool_type = pool_type
        # Simplified for inference: focusing on 'mean' as it's common for Stage E
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x is (B, T, D) -> Mean pool over Time
        pooled = x.mean(dim=1) 
        return self.classifier(pooled)

class _Identification_Service:
    """
    This class wrap all functionalitites of the identification system
    """

    # Service instance and models
    _instance = None
    _embedding_model = None
    _language_model = None
    _language_processor = None
    _gender_model = None
    _age_model = None
    _age_bins = ["0-19", "20-39", "40-59", "60+"]
    _gender_processor = None
    _emotion_model = None     
    _emotion_processor = None 
    _device = None

    # ChromaDB client and collection
    _chroma_client = None
    _collection = None

    def clear_collection(self) -> bool:
        """Deletes the current collection and recreates an empty one."""
        try:
            collection_name = self._collection.name
            # Delete the existing collection to wipe all embeddings
            self._chroma_client.delete_collection(collection_name)
            # Recreate an empty collection with the same parameters
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection '{collection_name}' successfully cleared.")
            return True
        except Exception as e:
            print(f"Failed to clear collection: {e}")
            return False

    def compute_embeddings(self, wavlist):
        """
        This function only compute the Speaker embedding using Titanet on the input audio files corresponding to a certain User.
        
        Parameters:
        ----------
        wavlist (list): list of audio paths for which the function computes embeddings
        """
        for i,wav in enumerate(wavlist):
            x, sr =librosa.load(wav,sr=16000,mono=True)
            sig = torch.from_numpy(x).unsqueeze(0).to(self._device) 
            audio_signal_len = torch.tensor([sig.shape[1]]).to(self._device) 
            with torch.no_grad():
                _ , embedding = self._embedding_model.forward(input_signal=sig, input_signal_length=audio_signal_len)  
            embeddings = embedding if i==0 else torch.cat([embeddings, embedding])    
        return embeddings
    
    def extract_audio_attributes(self, audio_path: str) -> Dict[str, Any]:
        """
        Extracts language and transcription from an audio file using Whisper.
        Age extimation with a custom trained classifier and Gender classification with a Wav2vec finetuned model (from HF)
        """
        try:
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Failed to load audio file {audio_path}: {e}")
            return {
                "gender": "error", "gender_confidence": 0.0,
                "language": "error", "language_confidence": 0.0,
                "age": "error", "age_confidence": 0.0,
                "transcription": "" # NEW
            }

        # --- 1. Whisper-based Features (Age, Language, Transcription) ---
        transcription = ""
        try:
            inputs = self._language_processor(waveform, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self._device)

            # --- 1a. Age Estimation  ---
            if self._age_model is not None:
                try:
                    with torch.no_grad():
                        encoder_outputs = self._language_model.get_encoder()(input_features)
                        hidden_states = encoder_outputs.last_hidden_state
                        age_logits = self._age_model(hidden_states)
                        age_probs = torch.nn.functional.softmax(age_logits, dim=-1)
                        age_idx = torch.argmax(age_probs, dim=-1).item()
                        age_label = self._age_bins[age_idx]
                        age_confidence = round(age_probs[0, age_idx].item(), 3)
                except Exception as age_e:
                    print(f"Age estimation inference failed for {audio_path}: {age_e}")
            else:
                print(f"Age model is not initialized. Skipping age estimation for {audio_path}.")

            # --- 1b. Language Identification ---
            try:
                with torch.no_grad():
                    decoder_input_ids = torch.tensor(
                        [[self._language_model.config.decoder_start_token_id]]
                    ).to(self._device)

                    outputs = self._language_model.generate(
                        input_features,
                        decoder_input_ids=decoder_input_ids,
                        max_new_tokens=1,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                    language_token_id = outputs.sequences[0][-1].item() 
                    language = self._language_processor.decode(language_token_id).strip('<|>')
                    probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
                    language_confidence = round(probs[0, language_token_id].item(), 3)
            except Exception as lang_e:
                print(f"Language generation failed for {audio_path}: {lang_e}")
                
            # --- 1c. Full Transcription  ---
            try:
                with torch.no_grad():
                    trans_ids = self._language_model.generate(input_features, max_new_tokens=256)
                    transcription = self._language_processor.batch_decode(trans_ids, skip_special_tokens=True)[0].strip()
            except Exception as trans_e:
                print(f"Transcription failed for {audio_path}: {trans_e}")

        except Exception as whisper_e:
            print(f"Whisper processing block failed for {audio_path}: {whisper_e}")

         # --- 2. Gender Classification with Wav2Vec2 ---
        try:
            gender_inputs = self._gender_processor(
                waveform, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = gender_inputs.input_values.to(self._device)

            with torch.no_grad():
                gender_outputs = self._gender_model(input_values)
                gender_probs = torch.nn.functional.softmax(gender_outputs.logits, dim=1).squeeze().tolist()
            
            if gender_probs[0] > gender_probs[1]:
                gender = "female"
                gender_confidence = round(gender_probs[0], 3)
            else:
                gender = "male"
                gender_confidence = round(gender_probs[1], 3)

        except Exception as e:
            print(f"Gender classification failed for {audio_path}: {e}")
            gender = "error"
            gender_confidence = 0.0

        # --- 3. Emotion Classification ---
        try:
            if self._emotion_model is not None and self._emotion_processor is not None:
                emo_inputs = self._emotion_processor(
                    waveform, sampling_rate=16000, return_tensors="pt", padding=True
                )
                emo_input_values = emo_inputs.input_values.to(self._device)
                
                with torch.no_grad():
                    emo_outputs = self._emotion_model(emo_input_values)
                    emo_logits = emo_outputs.logits
                    emo_idx = torch.argmax(emo_logits, dim=-1).item()
                    emotion = self._emotion_model.config.id2label[emo_idx]
            else:
                emotion = "Unknown"
        except Exception as e:
            print(f"Emotion classification failed for {audio_path}: {e}")
            emotion = "error"


        return {
            "gender": gender,      
            "gender_confidence": gender_confidence,     
            "language": language,
            "language_confidence": language_confidence,
            "age": age_label,        
            "age_confidence": age_confidence,
            "transcription": transcription,
            "emotion": emotion
        }


    def extract_attributes(self, user_audios_map: Dict[str, List[str]], data_type: str, session_id: str, case_id: str = None) -> Dict:
        """
        Computes embeddings and saves them to ChromaDB with specific metadata tags.
        Returns the attributes AND the generated ChromaDB ID for each file.
        
        Parameters:
        - data_type: "enroll" or "probe"
        - session_id: The session context (crucial for probes)
        """
        all_extracted_attributes = {}
        #iterate on each user
        for user_id, wavlist in user_audios_map.items():
            if not wavlist: continue

            try:
                # 1. Compute Embeddings
                embeddings = self.compute_embeddings(wavlist)
                if len(wavlist) == 1 and len(embeddings.shape) == 2: pass
                embedding_vectors = [emb.squeeze().cpu().tolist() for emb in embeddings]

                # 2. Generate IDs (Id identifying each embedding in the Chroma collection)
                generated_ids = [str(uuid.uuid4()) for _ in wavlist]

                documents = wavlist
                metadatas = []
                user_attributes_list = []

                for i, wav_path in enumerate(wavlist):
                    attributes = self.extract_audio_attributes(wav_path)
                    current_id = generated_ids[i]
                    
                    metadata = {
                        "user_id": user_id,       # a unique UUID (The "Speaker Label")
                        "audio_file": wav_path, 
                        "data_type": data_type,
                        "session_id": session_id,
                        "case_id": case_id,       
                        **attributes
                    }

                    #Filter out any None values so ChromaDB doesn't crash
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    metadatas.append(metadata)
                    
                    user_attributes_list.append({
                        "id": current_id,    
                        "filepath": wav_path, 
                        "attributes": attributes
                    })
                
                # 3. Add to ChromaDB
                self._collection.add(
                    documents=documents, 
                    metadatas=metadatas, 
                    ids=generated_ids, 
                    embeddings=embedding_vectors
                )
                all_extracted_attributes[user_id] = user_attributes_list
            except Exception as e:
                print(f"Failed to process user {user_id}: {e}")

        return all_extracted_attributes

    def identify_by_id(self, query_ids: List[str], top_k: int = 5, session_id: str = None, probe_only: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieves embeddings for query_ids and finds closest matches using filtered search.
        
        Parameters:
        - probe_only (bool): If True, restricts search strictly to 'probe' data within the current session_id. (for resolution)
                             If False, searches 'enroll' data AND 'probe' data within the current session_id. (for identification)
        """
        if not query_ids:
            return []
            
        try:
            # 1. Retrieve the embedding for the probe ID
            record = self._collection.get(ids=query_ids, include=["embeddings"])
            embs_found = record.get('embeddings')
            
            # Safe check for None or Empty 
            if embs_found is None or len(embs_found) == 0:
                print(f"No embeddings found for ids: {query_ids}")
                return []

            # 2. Construct the ChromaDB Filter
            if probe_only:
                # STRICT RESOLUTION: Probe data in THIS session only
                if not session_id:
                    print("Error: session_id is required for probe-only resolution.")
                    return []
                
                where_filter = {
                    "$and": [
                        {"data_type": "probe"},
                        {"session_id": session_id}
                    ]
                }
            else:
                # HYBRID IDENTIFICATION: Enroll data OR (Probe data in THIS session)
                where_filter = {
                    "$or": [
                        {"data_type": "enroll"},
                        {
                            "$and": [
                                {"data_type": "probe"},
                                {"session_id": session_id if session_id else "none"}
                            ]
                        }
                    ]
                }

            # 3. Query with Filter
            query_results = self._collection.query(
                query_embeddings=embs_found,
                n_results=top_k + 1,
                include=["metadatas", "distances"],
                where=where_filter
            )
            
            mapped_results = []
            fetched_ids = record["ids"] 
            
            for i, query_id in enumerate(fetched_ids):
                matches_for_this_id = []
                
                # Check if we have results for this specific query
                if (query_results.get("ids") and len(query_results["ids"]) > i):
                    found_ids = query_results["ids"][i]
                    found_metas = query_results["metadatas"][i]
                    found_dists = query_results["distances"][i]
                    
                    filtered_matches = []
                    
                    for res_id, res_meta, res_dist in zip(found_ids, found_metas, found_dists):
                        # Filter out self-match
                        if res_id != query_id:
                            # Convert Distance to Similarity
                            score = 1.0 - res_dist
                            score = max(0.0, score)
                            
                            filtered_matches.append({
                                "chroma_id": res_id,
                                "score": score,
                                "metadata": res_meta
                            })
                    
                    matches_for_this_id = filtered_matches[:top_k]

                mapped_results.append({
                    "query_id": query_id,
                    "matches": matches_for_this_id
                })

            return mapped_results

        except Exception as e:
            # Print error to server logs so we can debug
            print(f"Identification failed: {e}")
            return []

    def get_all_enrolled_users(self) -> Dict[str, Any]:
        """Fetches all enrolled users and their extracted attributes from ChromaDB."""
        try:
            results = self._collection.get(
                where={"data_type": "enroll"},
                include=["metadatas"]
            )
            users = {}
            if results and results.get("metadatas"):
                for meta in results["metadatas"]:
                    uid = meta.get("user_id", "Unknown")
                    if uid not in users:
                        users[uid] = {
                            "age": meta.get("age", "unknown"),
                            "gender": meta.get("gender", "unknown"),
                            "language": meta.get("language", "unknown"),
                            "audio_files": []
                        }
                    # Keep track of the files associated with this user
                    filename = meta.get("audio_file", "").split("/")[-1]
                    if filename and filename not in users[uid]["audio_files"]:
                        users[uid]["audio_files"].append(filename)
            return users
        except Exception as e:
            print(f"Failed to fetch enrolled users: {e}")
            return {}

    def delete_session(self, session_id: str):
        """Deletes all temporary probe embeddings for a given session to prevent DB pollution."""
        try:
            self._collection.delete(where={"session_id": session_id})
            print(f"Cleaned up temporary session: {session_id}")
        except Exception as e:
            print(f"Failed to clean up session {session_id}: {e}")



def Identification_Service(embedding_model, language_processor, language_model, gender_processor, gender_model, age_model, emotion_processor, emotion_model, device, chroma_client, collection_name):
    """
    Factory method for the singleton service.
    Accepts pre-loaded AI models and clients.
    """
    if _Identification_Service._instance is None:
        _Identification_Service._instance = _Identification_Service()
        _Identification_Service._device = device

        # 1. Assign pre-loaded models from endpoints.py
        _Identification_Service._embedding_model = embedding_model
        _Identification_Service._language_processor = language_processor
        _Identification_Service._language_model = language_model
        _Identification_Service._gender_processor = gender_processor
        _Identification_Service._age_model = age_model
        _Identification_Service._gender_model = gender_model
        _Identification_Service._emotion_processor = emotion_processor
        _Identification_Service._emotion_model = emotion_model
        
        # 2. Connect to ChromaDB
        _Identification_Service._chroma_client = PersistentClient(path=chroma_client)
        _Identification_Service._collection = _Identification_Service._chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
    return _Identification_Service._instance


if __name__ == "__main__":
    """
    Test script
    """

    import argparse

    parser = argparse.ArgumentParser(description='Script interface for testing the Diarization Service')
    parser.add_argument('--embedding_model', help='Nvidia Model used to compute Speaker Embedding', default='titanet_large')
    parser.add_argument('--device', help='Device used for processing: cuda or cpu', default="cpu")
    parser.add_argument('--chroma_collection', help='path to Chroma Collection data folder', default="fastapi/enrolled_data/chroma")
    
    args = parser.parse_args()
    embedding_model = args.embedding_model
    device = args.device
    chroma_collection = args.chroma_collection

    #load the titanet Embedding Model
    model = EncDecSpeakerLabelModel.from_pretrained(model_name=embedding_model)
    model.freeze()

    service = Identification_Service(model, device, chroma_collection)
    wavlist=['/home/lconcina/Music/test_audio.mp3']
    embeddings = service.compute_embeddings(wavlist)
    print(embeddings)