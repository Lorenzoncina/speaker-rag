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
        """Dual Retrieval: Independent Resolution for multiple files with Graph-Based Clustering."""
        session_id = f"session_{uuid.uuid4()}"
        
        audio_map = {}
        for path in audio_paths:
            audio_map[str(uuid.uuid4())] = [path]
            
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map, data_type="probe", session_id=session_id
        )
        
        chroma_ids = []
        id_to_path = {}
        id_to_uid = {}  # Track the internal User IDs
        for uid, results in extraction_results.items():
            for res in results:
                chroma_ids.append(res["id"])
                id_to_path[res["id"]] = res["filepath"]
                id_to_uid[res["id"]] = uid

        raw_id_results = self.audio_service.identify_by_id(
            query_ids=chroma_ids, top_k=5, session_id=session_id, probe_only=False
        )

        context_lines = ["=== FACT SHEET: AUDIO FILE RESOLUTION ==="]
        identified_users = set() 
        SIMILARITY_THRESHOLD = 0.50

        # --- Graph-Based Speaker Clustering (Union-Find) ---
        parent = {uid: uid for uid in id_to_uid.values()}
        enroll_links = {} 
        
        def find(i):
            if parent.get(i, i) == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                # Merge clusters predictably
                if root_i < root_j:
                    parent[root_j] = root_i
                else:
                    parent[root_i] = root_j

        # Pass 1: Build the clusters
        for result in raw_id_results:
            query_id = result["query_id"]
            my_uid = id_to_uid.get(query_id)
            if not my_uid: continue
            
            matches = result.get("matches", [])
            valid_matches = [m for m in matches if m.get("score", 0) < 0.99]
            candidates = [m for m in valid_matches if m.get("score", 0) >= SIMILARITY_THRESHOLD]
            
            if candidates:
                enrolled_candidates = [m for m in candidates if m.get("metadata", {}).get("data_type") == "enroll"]
                probe_candidates = [m for m in candidates if m.get("metadata", {}).get("data_type") == "probe"]
                
                if enrolled_candidates:
                    # If any file matches an enrolled user, link the entire cluster to that user!
                    best = sorted(enrolled_candidates, key=lambda x: x.get("score", 0), reverse=True)[0]
                    enrolled_uid = best.get("metadata", {}).get("user_id")
                    root = find(my_uid)
                    enroll_links[root] = enrolled_uid
                elif probe_candidates:
                    # If files match each other, merge them into the same cluster
                    best = sorted(probe_candidates, key=lambda x: x.get("score", 0), reverse=True)[0]
                    match_uid = best.get("metadata", {}).get("user_id")
                    if match_uid:
                        union(my_uid, match_uid)

        # Pass 2: Output the unified Fact Sheet
        for result in raw_id_results:
            query_id = result["query_id"]
            filepath = id_to_path.get(query_id, "")
            my_uid = id_to_uid.get(query_id)
            
            raw_file_name = os.path.basename(filepath) if filepath else "Unknown File"
            file_name = raw_file_name.split("_", 1)[-1] if "_" in raw_file_name else raw_file_name
            
            try:
                audio_index = audio_paths.index(filepath) + 1 
            except ValueError:
                audio_index = "?"

            context_lines.append(f"--- Audio Track {audio_index} (File: {file_name}) ---")

            # Resolve the final Identity using the cluster root
            root = find(my_uid)
            if root in enroll_links:
                identity = enroll_links[root]
            else:
                identity = f"Unknown_Speaker_{str(root)[:8]}"
                
            identified_users.add(identity)

            # Pull Age/Gender natively from this specific audio file's extraction
            attributes = {}
            for res in extraction_results.get(my_uid, []):
                if res["id"] == query_id:
                    attributes = res.get("attributes", {})
                    break

            age = attributes.get("age", "Unknown")
            gender = attributes.get("gender", "Unknown")
            language = attributes.get("language", "Unknown")
            acoustic_emotion = attributes.get("acoustic_emotion", "Unknown")
            text_emotion = attributes.get("text_emotion", "Unknown")
            transcription = attributes.get("transcription", "No transcription available.")

            context_lines.append(f"Speaker Profile:")
            context_lines.append(f"Identity: {identity}")
            context_lines.append(f"Age Bracket: {age}")
            context_lines.append(f"Gender: {gender}")
            context_lines.append(f"Language: {language}")
            context_lines.append(f"Current Utterance Analysis:")
            context_lines.append(f"  - Vocal Tone: {acoustic_emotion}")
            context_lines.append(f"  - Semantic Tone: {text_emotion}")
            context_lines.append(f"  - Transcription: \"{transcription}\"")
        
        context_lines.append("")

        # 4. Semantic Search (Filtered by Identity)
        context_lines.append("=== FACT SHEET: RELEVANT PAST STATEMENTS ===")
        
        semantic_matches = self.semantic_service.search_by_text(
            query_text=question, 
            top_k=5, 
            session_id=session_id,
            target_user_ids=list(identified_users) if identified_users else None 
        )
        
        if not semantic_matches:
            context_lines.append("No relevant past statements found for the identified speaker(s).")
        else:
            # Keep track of dialogue IDs we've already printed so we don't duplicate 
            # if two semantic matches happen to come from the exact same conversation!
            printed_dialogues = set()

            for match in semantic_matches:
                meta = match.get("metadata", {})
                dialogue_id = meta.get("dialogue_id")
                
                # --- SCENARIO A: This match belongs to a MELD conversation ---
                if dialogue_id and dialogue_id not in printed_dialogues:
                    printed_dialogues.add(dialogue_id)
                    
                    try:
                        # 1. Fetch ALL utterances from this specific conversation
                        dialogue_records = self.semantic_service._collection.get(
                            where={"dialogue_id": dialogue_id},
                            include=["metadatas", "documents"]
                        )
                        
                        # 2. Extract and format them
                        turns = []
                        for doc, d_meta in zip(dialogue_records["documents"], dialogue_records["metadatas"]):
                            turns.append({
                                "u_id": int(d_meta.get("utterance_id", 0)),
                                "speaker": d_meta.get("user_id", "Unknown").capitalize(),
                                "text": doc,
                                "ac_emo": d_meta.get("acoustic_emotion", "Unknown"),
                                "txt_emo": d_meta.get("text_emotion", "Unknown")
                            })
                            
                        # Sort chronologically by the utterance turn
                        turns = sorted(turns, key=lambda x: x["u_id"])
                        
                        # 3. Print the reconstructed conversation block
                        context_lines.append(f"\n--- Reconstructed Conversation (Dialogue ID: {dialogue_id}) ---")
                        for turn in turns:
                            # Add an arrow (►) to highlight the exact quote that triggered the semantic match!
                            prefix = "►" if turn["text"] == match.get("text", "") else " "
                            context_lines.append(f"{prefix} {turn['speaker']} (Vocal: {turn['ac_emo']} | Semantic: {turn['txt_emo']}): \"{turn['text']}\"")
                            
                    except Exception as e:
                        print(f"Failed to reconstruct dialogue {dialogue_id}: {e}")
                        
                # --- SCENARIO B: Fallback for old/standalone enrollments (no dialogue_id) ---
                elif not dialogue_id:
                    ac_emo = meta.get("acoustic_emotion", "Unknown")
                    txt_emo = meta.get("text_emotion", "Unknown")
                    context_lines.append(f"Author: {meta.get('user_id', 'Unknown')} | Vocal Tone: {ac_emo} | Semantic Tone: {txt_emo} | Statement: \"{match.get('text', '')}\"")
                    
        context_lines.append("")

        final_context = "\n".join(context_lines)

        # 5. Ask LLM
        prompt = f"""You are a strict data-reporting AI performing Speaker Resolution and Sentiment Analysis. 
            You are receiving {len(audio_paths)} distinct audio tracks.
            You must answer the User Question using strictly the FACT SHEET data provided below. 

            {final_context}

            CRITICAL RULES:
            1. Map your audio inputs directly to the "Audio Track" numbers in the FACT SHEET.
            2. The "Identity" field is the ABSOLUTE GROUND TRUTH. If the "Identity" string is exactly the same across multiple tracks, it is GUARANTEED to be the exact same person. 
            3. You must ignore discrepancies in "Age Bracket" or "Gender" if the "Identity" field matches. 
            4. Use the "Emotion" tags attached to the Current Tracks and Past Statements to explain HOW the speaker was feeling or WHY they might have said something. 

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