import os
import uuid
from typing import List, Dict

from .model_loader import load_audio_models
from .identification_service import Identification_Service
from .semantic_service import Semantic_Service 

class SpeakerRAG:
    def __init__(self, 
                 chroma_dir: str = "./enrolled_data/chroma", 
                 collection_name: str = "speaker_embeddings",
                 semantic_collection_name: str = "semantic_embeddings",
                 device: str = "cuda"):
        """Initializes the Dual-RAG System (Speaker ID + Semantic) without a hardcoded LLM."""
        print(f"Initializing SpeakerRAG Engine on device: {device}...")
        
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
        
        self.device = device
        print("SpeakerRAG initialization complete. Ready to build context.")
    
    def clear_database(self) -> dict:
        """Clears all stored speakers and semantics from the vector databases."""
        succ_audio = self.audio_service.clear_collection()
        succ_sem = self.semantic_service.clear_collection()
        if succ_audio and succ_sem:
            return {"status": "success", "message": "Vector databases cleared successfully."}
        else:
            return {"status": "error", "message": "Failed to clear databases."}

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

        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=user_audios_map, data_type="enroll", session_id=session_id
        )
        
        for uid, res_list in extraction_results.items():
            results["enrolled_users"][uid] = []
            sem_ids, sem_transcripts, sem_metas = [], [], []
            
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

    def build_context(self, audio_paths: List[str], question: str) -> dict:
        """Processes audio, clusters identities, retrieves semantic history, and builds the Fact Sheet.

        Returns a dict with:
          - fact_sheet: structured context string for the LLM
          - recommended_prompt: ready-to-use prompt combining the fact sheet and the question
        """
        session_id = f"session_{uuid.uuid4()}"
        audio_map = {str(uuid.uuid4()): [path] for path in audio_paths}
            
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map, data_type="probe", session_id=session_id
        )
        
        chroma_ids, id_to_path, id_to_uid = [], {}, {}
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
            if parent.get(i, i) == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                if root_i < root_j: parent[root_j] = root_i
                else: parent[root_i] = root_j

        for result in raw_id_results:
            my_uid = id_to_uid.get(result["query_id"])
            if not my_uid: continue
            
            candidates = [m for m in result.get("matches", []) if 0.99 > m.get("score", 0) >= SIMILARITY_THRESHOLD]
            if candidates:
                enrolled_candidates = [m for m in candidates if m.get("metadata", {}).get("data_type") == "enroll"]
                probe_candidates = [m for m in candidates if m.get("metadata", {}).get("data_type") == "probe"]
                
                if enrolled_candidates:
                    best = sorted(enrolled_candidates, key=lambda x: x.get("score", 0), reverse=True)[0]
                    enroll_links[find(my_uid)] = best.get("metadata", {}).get("user_id")
                elif probe_candidates:
                    best = sorted(probe_candidates, key=lambda x: x.get("score", 0), reverse=True)[0]
                    if match_uid := best.get("metadata", {}).get("user_id"):
                        union(my_uid, match_uid)

        # --- Output the Unified Fact Sheet ---
        for result in raw_id_results:
            query_id = result["query_id"]
            filepath = id_to_path.get(query_id, "")
            my_uid = id_to_uid.get(query_id)
            
            raw_file_name = os.path.basename(filepath) if filepath else "Unknown File"
            file_name = raw_file_name.split("_", 1)[-1] if "_" in raw_file_name else raw_file_name
            audio_index = audio_paths.index(filepath) + 1 if filepath in audio_paths else "?"

            context_lines.append(f"--- Audio Track {audio_index} (File: {file_name}) ---")

            root = find(my_uid)
            identity = enroll_links[root] if root in enroll_links else f"Unknown_Speaker_{str(root)[:8]}"
            identified_users.add(identity)

            attributes = next((res.get("attributes", {}) for res in extraction_results.get(my_uid, []) if res["id"] == query_id), {})

            context_lines.append(f"Speaker Profile:")
            context_lines.append(f"Identity: {identity}")
            context_lines.append(f"Age Bracket: {attributes.get('age', 'Unknown')}")
            context_lines.append(f"Gender: {attributes.get('gender', 'Unknown')}")
            context_lines.append(f"Language: {attributes.get('language', 'Unknown')}")
            context_lines.append(f"Current Utterance Analysis:")
            context_lines.append(f"  - Vocal Tone: {attributes.get('acoustic_emotion', 'Unknown')}")
            context_lines.append(f"  - Semantic Tone: {attributes.get('text_emotion', 'Unknown')}")
            context_lines.append(f"  - Transcription: \"{attributes.get('transcription', 'No transcription.')}\"")
        
        context_lines.append("")

        # --- Semantic Search (Filtered by Identity) ---
        context_lines.append("=== FACT SHEET: RELEVANT PAST STATEMENTS ===")
        semantic_matches = self.semantic_service.search_by_text(
            query_text=question, top_k=5, session_id=session_id,
            target_user_ids=list(identified_users) if identified_users else None 
        )
        
        if not semantic_matches:
            context_lines.append("No relevant past statements found for the identified speaker(s).")
        else:
            printed_dialogues = set()
            for match in semantic_matches:
                meta = match.get("metadata", {})
                dialogue_id = meta.get("dialogue_id")
                
                if dialogue_id and dialogue_id not in printed_dialogues:
                    printed_dialogues.add(dialogue_id)
                    try:
                        dialogue_records = self.semantic_service._collection.get(
                            where={"dialogue_id": dialogue_id}, include=["metadatas", "documents"]
                        )
                        turns = sorted([
                            {"u_id": int(d_meta.get("utterance_id", 0)),
                             "speaker": d_meta.get("user_id", "Unknown").capitalize(),
                             "text": doc,
                             "ac_emo": d_meta.get("acoustic_emotion", "Unknown"),
                             "txt_emo": d_meta.get("text_emotion", "Unknown")}
                            for doc, d_meta in zip(dialogue_records["documents"], dialogue_records["metadatas"])
                        ], key=lambda x: x["u_id"])
                        
                        context_lines.append(f"\n--- Reconstructed Conversation (Dialogue ID: {dialogue_id}) ---")
                        for turn in turns:
                            prefix = "►" if turn["text"] == match.get("text", "") else " "
                            context_lines.append(f"{prefix} {turn['speaker']} (Vocal: {turn['ac_emo']} | Semantic: {turn['txt_emo']}): \"{turn['text']}\"")
                    except Exception as e:
                        print(f"Failed to reconstruct dialogue {dialogue_id}: {e}")
                        
                elif not dialogue_id:
                    context_lines.append(f"Author: {meta.get('user_id', 'Unknown')} | Vocal Tone: {meta.get('acoustic_emotion', 'Unknown')} | Statement: \"{match.get('text', '')}\"")
                    
        context_lines.append("")
        
        # Cleanup
        self.audio_service.delete_session(session_id)

        fact_sheet = "\n".join(context_lines)

        recommended_prompt = f"""You are a strict data-reporting AI performing Speaker Resolution.
                You must answer the User Question using strictly the FACT SHEET data provided below.

                {fact_sheet}

                CRITICAL RULES:
                1. The "Identity" field is the ABSOLUTE GROUND TRUTH.
                2. If the "Identity" strings are identical across tracks, it is GUARANTEED to be the same person. Disregard differing Age/Gender.
                3. If the "Identity" strings differ, they are DIFFERENT people.
                4. Use the "Emotion" tags to explain HOW the speaker was feeling.
                5. EMOTION EVALUATION: The "Current Utterance Analysis" emotion tags are generated by imperfect zero-shot AI. Do NOT blindly trust them.
                6. CONTEXTUAL REASONING: You must read the "Reconstructed Conversations" to understand the speaker's history. If the past dialogue shows the speaker was arguing, sarcastic, annoyed, etc., you must correct the imperfect zero-shot tags and state their TRUE emotion based on that history.

                MANDATORY REASONING FORMAT:
                If asked if speakers are the same, start by explicitly stating the Identity of each track.

                ### User Question:
                {question}

                ### Answer:"""

        return {"fact_sheet": fact_sheet, "recommended_prompt": recommended_prompt}

    def get_enrolled_users(self) -> dict:
        """Retrieves all enrolled users from the database."""
        users = self.audio_service.get_all_enrolled_users()
        return {"status": "success", "enrolled_users": users}