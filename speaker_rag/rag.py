import os
import uuid
from typing import List, Dict
from transformers import pipeline

from .model_loader import load_audio_models
from .identification_service import Identification_Service

class SpeakerRAG:
    def __init__(self, 
                 chroma_dir: str = "./enrolled_data/chroma", 
                 collection_name: str = "speaker_embeddings",
                 llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", 
                 device: str = "cuda"):
        """Initializes the Speaker RAG system."""
        print(f"Initializing SpeakerRAG on device: {device}...")
        
        audio_models = load_audio_models(device=device)
        self.audio_service = Identification_Service(
            **audio_models,
            device=device,
            chroma_client=chroma_dir,
            collection_name=collection_name
        )
        
        print(f"Loading LLM ({llm_model_id}) into memory...")
        self.llm = pipeline("text-generation", model=llm_model_id, device_map="auto")
        print("SpeakerRAG initialization complete. All models ready.")
    
    def clear_database(self) -> dict:
        """Clears all stored speakers from the vector database."""
        success = self.audio_service.clear_collection()
        if success:
            return {"status": "success", "message": "Vector database cleared successfully."}
        else:
            return {"status": "error", "message": "Failed to clear the database. Check server logs."}

    def index_audio(self, audio_paths: List[str], user_ids: List[str] = None, session_id: str = "rag_session") -> dict:
        """
        Enrolls audio files into the database.
        - If 1 user_id is provided, all files are grouped under it.
        - If N user_ids are provided, they are mapped 1-to-1 to the files.
        - If NO user_ids are provided, each file gets its own unique UUID.
        """
        user_audios_map = {}
        results = {"status": "success", "enrolled_users": {}}

        # --- Resolve the IDs ---
        if not user_ids:
            # User's Proposal: 1 unique UUID for every single audio file
            print("No user IDs provided. Assigning a unique UUID to each file.")
            assigned_ids = [str(uuid.uuid4()) for _ in audio_paths]
        elif len(user_ids) == 1:
            print(f"Assigning all files to user: {user_ids[0]}")
            assigned_ids = [user_ids[0]] * len(audio_paths)
        elif len(user_ids) == len(audio_paths):
            print("Mapping multiple user IDs 1-to-1 with audio files.")
            assigned_ids = user_ids
        else:
            raise ValueError("Length of user_ids must be 1, or exactly equal to the number of audio files.")
        
        # --- Build the Mapping ---
        for audio, uid in zip(audio_paths, assigned_ids):
            if uid not in user_audios_map:
                user_audios_map[uid] = []
            user_audios_map[uid].append(audio)

        # --- Batch Enroll ---
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=user_audios_map, data_type="enroll", session_id=session_id
        )
        
        for uid, res_list in extraction_results.items():
            results["enrolled_users"][uid] = [res["id"] for res in res_list]
            
        return results

    def _format_results_for_llm(self, raw_results: list) -> str:
        """Translates the backend JSON into a readable prompt context for multiple files."""
        if not raw_results:
            return "No matching speakers were found in the database."

        context_lines = []
        
        for result in raw_results:
            # Extract the actual filename from the ChromaDB metadata
            probe_meta = result.get("probe_metadata", {})
            audio_file = probe_meta.get("audio_file", "Unknown File")
            file_name = os.path.basename(audio_file)
            
            matches = result.get("matches", [])
            context_lines.append(f"--- Identification Results for: {file_name} ---")
            
            if not matches:
                context_lines.append("No matches found for this audio file.\n")
            else:
                for i, match in enumerate(matches):
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    
                    speaker_id = meta.get("user_id", "Unknown") 
                    age = meta.get("age", "unknown age")
                    gender = meta.get("gender", "unknown gender")
                    language = meta.get("language", "unknown language")
                    
                    context_lines.append(
                        f"  Match #{i+1}: Speaker ID '{speaker_id}' (Similarity: {score:.2f}). "
                        f"Attributes: {age} {gender} speaking {language}."
                    )
                context_lines.append("")
            
        return "\n".join(context_lines)

    def query(self, audio_paths: List[str], question: str, session_id: str = "rag_session") -> str:
        """Identifies multiple unknown audio files and asks the LLM about them."""
        # Use a temporary speaker ID to group the probe files
        temp_speaker_id = str(uuid.uuid4())
        audio_map = {temp_speaker_id: audio_paths}
        
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map,
            data_type="probe",
            session_id=session_id
        )
        
        chroma_ids = [res["id"] for res in extraction_results.get(temp_speaker_id, [])]

        raw_results = self.audio_service.identify_by_id(
            query_ids=chroma_ids,
            top_k=3,
            session_id=session_id,
            probe_only=False
        )

        context = self._format_results_for_llm(raw_results)
        
        prompt = f"""You are an intelligent audio-analysis assistant. 
                    Based on the following backend speaker recognition results, answer the user's question.

                    ### Backend Audio Analysis Results:
                    {context}

                    ### User Question:
                    {question}

                    ### Answer:"""

        outputs = self.llm(
            prompt, 
            max_new_tokens=200, 
            return_full_text=False,
            temperature=0.3
        )
        
        return outputs[0]["generated_text"].strip()