import os
import re
import uuid
from typing import List, Dict, Optional

from .model_loader import load_audio_models
from .identification_service import Identification_Service
from .semantic_service import Semantic_Service
from .diarization_service import Diarization_Service, parse_vtt

class SpeakerRAG:
    def __init__(self,
                 chroma_dir: str = "./enrolled_data/chroma",
                 collection_name: str = "speaker_embeddings",
                 semantic_collection_name: str = "semantic_embeddings",
                 device: str = "cuda",
                 hf_auth_token: str = None):
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

        self._diarization_embedding_model = audio_models["embedding_model"]
        self._hf_auth_token = hf_auth_token
        self._diarization_service = None  # Lazy-loaded on first use

        self.device = device
        print("SpeakerRAG initialization complete. Ready to build context.")

    @property
    def diarization_service(self):
        if self._diarization_service is None:
            self._diarization_service = Diarization_Service(
                embedding_model=self._diarization_embedding_model,
                device=self.device,
                hf_auth_token=self._hf_auth_token,
            )
        return self._diarization_service
    
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

    def index_conversation(
        self,
        audio_path: str,
        speaker_map: Optional[Dict[str, str]] = None,
        num_speakers: Optional[int] = None,
        vtt_path: Optional[str] = None,
    ) -> dict:
        """Enrolls a full multi-speaker conversation into the system.

        Pipeline:
          1. Diarize the audio (VAD + speaker turn segmentation), or use VTT turns if provided.
          2. Extract Titanet embeddings per segment.
          3. Cluster segments into speaker groups (or apply speaker_map / VTT names).
          4. Match cluster centroids against enrolled speakers for cross-call identity.
          5. Extract audio attributes (transcription, emotion, etc.) per segment.
             If VTT provides transcriptions, those are used instead of re-running Whisper.
          6. Enroll each segment into both the speaker and semantic collections.

        Parameters:
        - audio_path: path to the full conversation audio.
        - speaker_map: optional dict mapping pyannote labels to known names.
        - num_speakers: optional hint for the number of speakers.
        - vtt_path: optional path to a WebVTT (.vtt) file (e.g. from Teams/Meet).
                    When provided, pyannote is skipped; speaker names and timestamps
                    come from the VTT, and transcriptions are carried over.

        Returns a dict with status, dialogue_id, identity_map, and per-segment enrollment results.
        """
        vtt_turns = None
        if vtt_path:
            vtt_turns = parse_vtt(vtt_path)
            if not vtt_turns:
                print(f"[Conversation] VTT file was empty or unparseable: {vtt_path}")

        diar_result = self.diarization_service.process_conversation(
            audio_path=audio_path,
            speaker_map=speaker_map,
            num_speakers=num_speakers,
            identification_service=self.audio_service,
            vtt_turns=vtt_turns,
        )

        dialogue_id = diar_result["dialogue_id"]
        segments = diar_result["segments"]
        identity_map = diar_result["identity_map"]

        if not segments:
            return {
                "status": "success",
                "dialogue_id": dialogue_id,
                "message": "No speech segments found in the audio.",
                "identity_map": identity_map,
                "enrolled_segments": 0,
            }

        user_audios_map: Dict[str, List[str]] = {}
        seg_lookup: Dict[str, Dict[str, Any]] = {}
        for seg in segments:
            uid = seg["identity"]
            path = seg["segment_path"]
            user_audios_map.setdefault(uid, []).append(path)
            seg_lookup[path] = seg

        session_id = f"conv_{dialogue_id}"
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=user_audios_map,
            data_type="enroll",
            session_id=session_id,
        )

        enrolled_count = 0
        for uid, res_list in extraction_results.items():
            sem_ids, sem_transcripts, sem_metas = [], [], []

            for res in res_list:
                chroma_id = res["id"]
                seg_info = seg_lookup.get(res["filepath"], {})

                # Prefer VTT transcription over Whisper when available
                vtt_transcript = seg_info.get("vtt_transcription", "")
                whisper_transcript = res["attributes"].pop("transcription", "")
                transcription = vtt_transcript if vtt_transcript else whisper_transcript

                if transcription:
                    sem_ids.append(chroma_id)
                    sem_transcripts.append(transcription)
                    meta = {
                        "user_id": uid,
                        "audio_file": res["filepath"],
                        "data_type": "enroll",
                        "session_id": session_id,
                        "dialogue_id": dialogue_id,
                        "utterance_id": seg_info.get("utterance_id", 0),
                    }
                    meta.update(res["attributes"])
                    if vtt_transcript:
                        meta["transcription"] = vtt_transcript
                    sem_metas.append(meta)

                enrolled_count += 1

            if sem_ids:
                self.semantic_service.index_transcriptions(sem_ids, sem_transcripts, sem_metas)

        print(f"[Conversation] Enrolled {enrolled_count} segments from dialogue {dialogue_id}.")

        return {
            "status": "success",
            "dialogue_id": dialogue_id,
            "identity_map": identity_map,
            "enrolled_segments": enrolled_count,
        }

    def extract_speakers_from_question(self, question: str) -> List[str]:
        """Scans the question for mentions of enrolled user IDs (case-insensitive, whole-word match).

        Returns a list of enrolled user_ids detected in the question text. If none match, returns [].
        UUID-like identifiers are skipped to avoid spurious matches on random enrollment IDs.
        """
        if not question:
            return []

        try:
            enrolled = self.audio_service.get_all_enrolled_users()
        except Exception as e:
            print(f"Failed to fetch enrolled users for NER: {e}")
            return []

        if not enrolled:
            return []

        uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
        matched = []
        for uid in enrolled.keys():
            if not uid or uuid_pattern.match(str(uid)):
                continue
            pattern = r"\b" + re.escape(str(uid)) + r"\b"
            if re.search(pattern, question, flags=re.IGNORECASE):
                matched.append(uid)
        return matched

    def build_context(
        self,
        audio_paths: Optional[List[str]] = None,
        question: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        """Hybrid context builder supporting two modes:

        1. Audio provided -> full pipeline: identification + filtered semantic search.
        2. No audio -> extract speaker names from the question via NER;
           if any match, filter by them, else run an unfiltered semantic search.

        Parameters:
        - conversation_history: optional list of past {"question": str, "answer": str} pairs.
          When provided, it is rendered into the recommended prompt as prior-turn context.
          The pipeline never stores it — memory is the caller's responsibility.

        Returns a dict with:
          - fact_sheet: structured context string for the LLM
          - recommended_prompt: ready-to-use prompt combining the fact sheet, history, and question
        """
        if audio_paths:
            return self._build_context_from_audio(
                audio_paths=audio_paths,
                question=question,
                conversation_history=conversation_history,
            )
        return self._build_context_from_text(
            question=question,
            conversation_history=conversation_history,
        )

    def _build_context_from_text(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        """Text-only context builder. Extracts speaker names from the question via NER;
        falls back to an unfiltered semantic search if no names match enrolled users."""
        context_lines = ["=== FACT SHEET: AUDIO FILE RESOLUTION ==="]
        context_lines.append("No audio provided. This is a text-only query.")

        extracted = self.extract_speakers_from_question(question)
        if extracted:
            target_user_ids = extracted
            resolution_note = f"Speaker filter (extracted from question): {', '.join(target_user_ids)}"
        else:
            target_user_ids = []
            resolution_note = "Speaker filter: none (unfiltered semantic search across all enrolled speakers)."

        context_lines.append(resolution_note)
        context_lines.append("")

        fact_sheet = self._append_semantic_section(
            context_lines=context_lines,
            question=question,
            target_user_ids=target_user_ids if target_user_ids else None,
            session_id=None,
        )

        recommended_prompt = self._build_prompt(
            fact_sheet=fact_sheet,
            question=question,
            conversation_history=conversation_history,
        )
        return {"fact_sheet": fact_sheet, "recommended_prompt": recommended_prompt}

    def _build_context_from_audio(
        self,
        audio_paths: List[str],
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        """Processes audio, clusters identities, retrieves semantic history, and builds the Fact Sheet."""
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
        fact_sheet = self._append_semantic_section(
            context_lines=context_lines,
            question=question,
            target_user_ids=list(identified_users) if identified_users else None,
            session_id=session_id,
        )

        # Cleanup
        self.audio_service.delete_session(session_id)

        recommended_prompt = self._build_prompt(
            fact_sheet=fact_sheet,
            question=question,
            conversation_history=conversation_history,
        )
        return {"fact_sheet": fact_sheet, "recommended_prompt": recommended_prompt}

    def _append_semantic_section(
        self,
        context_lines: List[str],
        question: str,
        target_user_ids: Optional[List[str]],
        session_id: Optional[str],
    ) -> str:
        """Runs semantic search, appends the 'Relevant Past Statements' section to context_lines,
        and returns the final fact sheet string."""
        context_lines.append("=== FACT SHEET: RELEVANT PAST STATEMENTS ===")
        semantic_matches = self.semantic_service.search_by_text(
            query_text=question, top_k=5, session_id=session_id,
            target_user_ids=target_user_ids,
        )

        if not semantic_matches:
            if target_user_ids:
                context_lines.append("No relevant past statements found for the identified speaker(s).")
            else:
                context_lines.append("No relevant past statements found across the enrolled database.")
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
        return "\n".join(context_lines)

    def _build_prompt(
        self,
        fact_sheet: str,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Builds the recommended system prompt combining the fact sheet, optional
        prior conversation turns, and the current user question."""
        history_block = ""
        if conversation_history:
            rendered_turns = []
            for i, turn in enumerate(conversation_history, start=1):
                q = (turn.get("question") or "").strip()
                a = (turn.get("answer") or "").strip()
                if not q and not a:
                    continue
                rendered_turns.append(f"Turn {i}:\nUser: {q}\nAssistant: {a}")
            if rendered_turns:
                history_block = (
                    "\n\n### Conversation History (prior turns, for context only):\n"
                    + "\n\n".join(rendered_turns)
                )

        return f"""You are a strict data-reporting AI performing Speaker Resolution.
                You must answer the User Question using strictly the FACT SHEET data provided below.

                {fact_sheet}

                CRITICAL RULES:
                1. The "Identity" field is the ABSOLUTE GROUND TRUTH and the ONLY thing that determines whether two tracks are the same person.
                2. SAME-PERSON DECISION: Compare ONLY the "Identity" strings.
                   - If the "Identity" strings are identical across tracks, they ARE the same person. This is GUARANTEED and final.
                   - If the "Identity" strings differ, they are DIFFERENT people.
                3. Differences in Age, Gender, Language, Vocal Tone, Semantic Tone, or Emotion are NEVER evidence that the speakers are different. The same person naturally sounds different across utterances. You must NEVER conclude "different people" because emotions or tones differ. Identity is the sole criterion.
                4. Use the "Emotion" tags only to explain HOW the speaker was feeling, never to decide WHO the speaker is.
                5. EMOTION EVALUATION: The "Current Utterance Analysis" emotion tags are generated by imperfect zero-shot AI. Do NOT blindly trust them.
                6. CONTEXTUAL REASONING: You must read the "Reconstructed Conversations" to understand the speaker's history. If the past dialogue shows the speaker was arguing, sarcastic, annoyed, etc., you must correct the imperfect zero-shot tags and state their TRUE emotion based on that history.
                7. CONVERSATION HISTORY: If prior turns are provided below, use them to resolve follow-up references (e.g., "he", "that speaker", "the previous one"), but the FACT SHEET above remains the source of truth.
{history_block}

                MANDATORY REASONING FORMAT:
                If asked if speakers are the same, FIRST state the Identity of each track verbatim, THEN compare ONLY those Identity strings to reach your conclusion. Do not mention emotion or tone as part of the same/different decision.

                ### User Question:
                {question}

                ### Answer:"""

    def get_enrolled_users(self) -> dict:
        """Retrieves all enrolled users from the database."""
        users = self.audio_service.get_all_enrolled_users()
        return {"status": "success", "enrolled_users": users}