import os
import uuid
import torch
import librosa
from typing import List, Dict
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig

from .model_loader import load_audio_models
from .identification_service import Identification_Service
from .semantic_service import Semantic_Service # NEW

class SpeakerRAG:
    def __init__(self, 
                 chroma_dir: str = "./enrolled_data/chroma", 
                 collection_name: str = "speaker_embeddings",
                 semantic_collection_name: str = "semantic_embeddings",
                 llm_model_id: str = "Qwen/Qwen2-Audio-7B-Instruct", 
                 device: str = "cuda"):
        """Initializes the Dual-RAG system (Speaker ID + Semantic)."""
        print(f"Initializing SpeakerRAG on device: {device}...")
        
        audio_models = load_audio_models(device=device)
        semantic_model = audio_models.pop("semantic_model")
        
        self.audio_service = Identification_Service(
            **audio_models,
            device=device,
            chroma_client=chroma_dir,
            collection_name=collection_name
        )
        
        self.semantic_service = Semantic_Service(
            semantic_model=semantic_model,
            chroma_client=chroma_dir,
            collection_name=semantic_collection_name
        )
        
        print(f"Loading LLM ({llm_model_id}) into memory...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.processor = AutoProcessor.from_pretrained(llm_model_id)
        self.llm = Qwen2AudioForConditionalGeneration.from_pretrained(
            llm_model_id, 
            device_map="auto",
            quantization_config=bnb_config if device == "cuda" else None
        )
        self.device = device
        print("SpeakerRAG initialization complete. All models ready.")
    
    def clear_database(self) -> dict:
        """Clears all stored speakers and semantics from the vector databases."""
        succ_audio = self.audio_service.clear_collection()
        succ_sem = self.semantic_service.clear_collection()
        if succ_audio and succ_sem:
            return {"status": "success", "message": "Vector databases cleared successfully."}
        else:
            return {"status": "error", "message": "Failed to clear databases. Check server logs."}

    def index_audio(self, audio_paths: List[str], user_ids: List[str] = None, session_id: str = "rag_session") -> dict:
        """Enrolls audio files into the ID database and Semantic database."""
        user_audios_map = {}
        results = {"status": "success", "enrolled_users": {}}

        if not user_ids:
            assigned_ids = [str(uuid.uuid4()) for _ in audio_paths]
        elif len(user_ids) == 1:
            assigned_ids = [user_ids[0]] * len(audio_paths)
        elif len(user_ids) == len(audio_paths):
            assigned_ids = user_ids
        else:
            raise ValueError("Length of user_ids must be 1, or exactly equal to the number of audio files.")
        
        for audio, uid in zip(audio_paths, assigned_ids):
            if uid not in user_audios_map:
                user_audios_map[uid] = []
            user_audios_map[uid].append(audio)

        # 1. Extract IDs, Embeddings, Transcriptions
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=user_audios_map, data_type="enroll", session_id=session_id
        )
        
        # 2. Add to Semantic DB using the exact same Chroma IDs
        for uid, res_list in extraction_results.items():
            results["enrolled_users"][uid] = []
            
            sem_ids = []
            sem_transcripts = []
            sem_metas = []
            
            for res in res_list:
                chroma_id = res["id"]
                results["enrolled_users"][uid].append(chroma_id)
                
                transcription = res["attributes"].pop("transcription", "")
                if transcription:
                    sem_ids.append(chroma_id)
                    sem_transcripts.append(transcription)
                    meta = {"user_id": uid, "audio_file": res["filepath"], "data_type": "enroll", "session_id": session_id}
                    meta.update(res["attributes"])
                    sem_metas.append(meta)
            
            if sem_ids:
                self.semantic_service.index_transcriptions(sem_ids, sem_transcripts, sem_metas)
            
        return results

    def _prepare_qwen_audio_inputs(self, audio_paths: List[str], text_prompt: str):
        content = []
        raw_audios = []
        for i, path in enumerate(audio_paths):
            content.append({"type": "audio", "audio_url": f"audio_{i}"})
            sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
            audio_array, _ = librosa.load(path, sr=sr)
            raw_audios.append(audio_array)
            
        content.append({"type": "text", "text": text_prompt})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        inputs = self.processor(text=text, audio=raw_audios, return_tensors="pt", padding=True).to(self.llm.device)
        return inputs

    def direct_query(self, audio_paths: List[str], question: str) -> str:
        """Directly asks the Multimodal LLM about the audio (No RAG)."""
        prompt = f"Please listen to the provided audio files and answer the following question: {question}"
        
        inputs = self._prepare_qwen_audio_inputs(audio_paths, prompt)
        
        generated_ids = self.llm.generate(**inputs, max_new_tokens=256, temperature=0.3)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def query(self, audio_paths: List[str], question: str) -> str:
        """Dual Retrieval: Independent Resolution for multiple files with auto-cleanup."""
        session_id = f"session_{uuid.uuid4()}"
        
        audio_map = {}
        for path in audio_paths:
            audio_map[str(uuid.uuid4())] = [path]
            
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map, data_type="probe", session_id=session_id
        )
        
        chroma_ids = []
        id_to_path = {}
        for uid, results in extraction_results.items():
            for res in results:
                chroma_ids.append(res["id"])
                id_to_path[res["id"]] = res["filepath"]

        raw_id_results = self.audio_service.identify_by_id(
            query_ids=chroma_ids, top_k=5, session_id=session_id, probe_only=False
        )

        context_lines = ["=== FACT SHEET: AUDIO FILE RESOLUTION ==="]
        identified_users = set() 
        SIMILARITY_THRESHOLD = 0.65 
        
        for result in raw_id_results:
            query_id = result["query_id"]
            filepath = id_to_path.get(query_id, "")
            
            raw_file_name = os.path.basename(filepath) if filepath else "Unknown File"
            file_name = raw_file_name.split("_", 1)[-1] if "_" in raw_file_name else raw_file_name
            
            try:
                audio_index = audio_paths.index(filepath) + 1 
            except ValueError:
                audio_index = "?"

            context_lines.append(f"--- Audio Track {audio_index} (File: {file_name}) ---")

            matches = result.get("matches", [])
            
            # 1. Filter out exact self-matches
            valid_matches = [m for m in matches if m.get("score", 0) < 0.99]
            
            # 2. Apply Threshold
            candidates = [m for m in valid_matches if m.get("score", 0) >= SIMILARITY_THRESHOLD]
            
            best_match = None
            if candidates:
                # THE FIX: Separate enrolled users from anonymous probes
                enrolled_candidates = [m for m in candidates if m.get("metadata", {}).get("data_type") == "enroll"]
                probe_candidates = [m for m in candidates if m.get("metadata", {}).get("data_type") == "probe"]
                
                # Prioritize Enrolled users over Probes!
                if enrolled_candidates:
                    enrolled_candidates = sorted(enrolled_candidates, key=lambda x: x.get("score", 0), reverse=True)
                    best_match = enrolled_candidates[0]
                elif probe_candidates:
                    probe_candidates = sorted(probe_candidates, key=lambda x: x.get("score", 0), reverse=True)
                    best_match = probe_candidates[0]
            
            if best_match:
                meta = best_match.get("metadata", {})
                data_type = meta.get("data_type")
                match_uid = meta.get("user_id", "Unknown")
                
                if data_type == "enroll":
                    identity = match_uid
                else:
                    # Successfully grouped with an unknown probe (Diarization)
                    identity = f"Unknown_Speaker_{str(match_uid)[:8]}"
                    
                identified_users.add(match_uid)
                age = meta.get("age", "Unknown")
                gender = meta.get("gender", "Unknown")
                language = meta.get("language", "Unknown")
            else:
                identity = "UNIDENTIFIED SPEAKER"
                age = "Unknown"
                gender = "Unknown"
                language = "Unknown"
                
            context_lines.append(f"Identity: {identity}")
            context_lines.append(f"Age Bracket: {age}")
            context_lines.append(f"Gender: {gender}")
            context_lines.append(f"Language: {language}")
        
        context_lines.append("")

        context_lines.append("=== FACT SHEET: RELEVANT PAST STATEMENTS ===")
        
        semantic_matches = self.semantic_service.search_by_text(
            query_text=question, 
            top_k=3, 
            session_id=session_id,
            target_user_ids=list(identified_users) if identified_users else None 
        )
        
        if not semantic_matches:
            context_lines.append("No relevant past statements found for the identified speaker(s).")
        else:
            for match in semantic_matches:
                meta = match.get("metadata", {})
                context_lines.append(f"Author: {meta.get('user_id', 'Unknown')} | Statement: \"{match['text']}\"")
        context_lines.append("")

        final_context = "\n".join(context_lines)

        prompt = f"""You are a strict data-reporting AI performing Speaker Resolution. 
                You are receiving {len(audio_paths)} distinct audio tracks.
                You must answer the User Question using strictly the FACT SHEET data provided below. 

                {final_context}

                CRITICAL RULES:
                1. Map your audio inputs directly to the "Audio Track" numbers in the FACT SHEET.
                2. If the user asks who is speaking, list the Identity for EVERY Audio Track individually. 
                3. If multiple tracks share the same identity, explicitly state that those tracks contain the same person.

                ### User Question:
                {question}

                ### Answer:"""

        inputs = self._prepare_qwen_audio_inputs(audio_paths, prompt)
        
        generated_ids = self.llm.generate(**inputs, max_new_tokens=300, temperature=0.1)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        self.audio_service.delete_session(session_id)
        
        return response, final_context

    def get_enrolled_users(self) -> dict:
        """Retrieves all enrolled users from the database."""
        users = self.audio_service.get_all_enrolled_users()
        return {"status": "success", "enrolled_users": users}