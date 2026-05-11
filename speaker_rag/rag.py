import os
import re
import uuid
from typing import Any, List, Dict, Optional, Tuple

from chromadb import PersistentClient

from .model_loader import load_audio_models
from .identification_service import Identification_Service
from .semantic_service import Semantic_Service
from .diarization_service import Diarization_Service, parse_vtt
from .graph_service import GraphService
from .llm_extractor import LLMExtractor
from .entity_resolver import EntityResolver

# ---------------------------------------------------------------------------
# Question classification — routes the FACT SHEET KG section to the right
# emit branch instead of always dumping the entity-centric default.
# ---------------------------------------------------------------------------
_EMOTION_VOCAB = ("joy", "anger", "sadness", "fear", "disgust", "surprise", "neutral")

_PROFILE_COUNT_PAT = re.compile(
    r"\b(how many (?:times|utterances)|how often)\b", re.IGNORECASE
)
_PROFILE_RANK_PAT = re.compile(
    r"\b(most often|more often|most frequently|"
    r"(?:who|which) .*?(?:shows?|expresses?|expressing|displays?) .*? most)\b",
    re.IGNORECASE,
)
_PARTICIPANT_PAT = re.compile(
    r"(?:"
    r"who else"
    r"|who (?:was|were) (?:with|present|in|there)"
    r"|other (?:people|participants?|speakers?|friends?)"
    r"|in (?:the|that) (?:room|conversation|scene)"
    r"|(?:part of|in) (?:that|the) conversation"
    r"|(?:of|which of) (?:his|her|their) friends?"
    r"|with (?:him|her|them)"
    r"|besides (?:the )?two"
    r"|friends? (?:were|was) with"
    r")",
    re.IGNORECASE,
)


def _detect_emotion(question: str) -> Optional[str]:
    """Return the first emotion keyword found in the question, lowercase, or None."""
    if not question:
        return None
    q = question.lower()
    for emo in _EMOTION_VOCAB:
        if re.search(rf"\b{emo}\b", q):
            return emo
    return None


def _classify_question(question: str) -> str:
    """Map a question to one of: profile_count, profile_rank, dialogue_participants, default.

    Profile branches require an emotion keyword to fire — otherwise a question like
    'how many utterances are stored?' would be misrouted.
    """
    if not question:
        return "default"
    has_emotion = _detect_emotion(question) is not None
    if has_emotion and _PROFILE_COUNT_PAT.search(question):
        return "profile_count"
    if has_emotion and _PROFILE_RANK_PAT.search(question):
        return "profile_rank"
    if _PARTICIPANT_PAT.search(question):
        return "dialogue_participants"
    return "default"

class SpeakerRAG:
    def __init__(self,
                 chroma_dir: str = "./enrolled_data/chroma",
                 collection_name: str = "speaker_embeddings",
                 semantic_collection_name: str = "semantic_embeddings",
                 entity_collection_name: str = "entity_embeddings",
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

        # Dedicated Chroma collection for entity-level cosine search (used by
        # EntityResolver to dedupe newly-extracted entities against existing ones).
        self._entity_chroma_client = PersistentClient(path=chroma_dir)
        self._entity_collection_name = entity_collection_name
        self._entity_collection = self._entity_chroma_client.get_or_create_collection(
            name=entity_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._diarization_embedding_model = audio_models["embedding_model"]
        self._hf_auth_token = hf_auth_token
        self._diarization_service = None  # Lazy-loaded on first use

        graph_path = os.path.join(
            os.path.dirname(os.path.normpath(chroma_dir)), "knowledge_graph.json"
        )
        # LLMExtractor self-disables if ANTHROPIC_API_KEY is missing or the
        # anthropic SDK isn't installed. GraphService will print which mode
        # it ended up in.
        llm_extractor = LLMExtractor()
        entity_resolver = EntityResolver(
            chroma_collection=self._entity_collection,
            embedding_model=semantic_model,
            llm_extractor=llm_extractor,
        )
        self.graph_service = GraphService(
            graph_path=graph_path,
            llm_extractor=llm_extractor,
            entity_resolver=entity_resolver,
        )

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
        """Clears all stored speakers, semantics, entities, and the knowledge graph."""
        succ_audio = self.audio_service.clear_collection()
        succ_sem = self.semantic_service.clear_collection()
        succ_ent = self._clear_entity_collection()
        self.graph_service.clear()
        if succ_audio and succ_sem and succ_ent:
            return {"status": "success", "message": "Vector databases and knowledge graph cleared successfully."}
        else:
            return {"status": "error", "message": "Failed to clear one or more databases."}

    def _clear_entity_collection(self) -> bool:
        try:
            self._entity_chroma_client.delete_collection(self._entity_collection_name)
            self._entity_collection = self._entity_chroma_client.get_or_create_collection(
                name=self._entity_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # Re-bind the resolver's collection reference so it sees the fresh handle.
            if self.graph_service and self.graph_service._entity_resolver:
                self.graph_service._entity_resolver._collection = self._entity_collection
            print(f"Collection '{self._entity_collection_name}' successfully cleared.")
            return True
        except Exception as e:
            print(f"Failed to clear entity collection: {e}")
            return False

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
                    self.graph_service.index_utterance(
                        speaker_id=uid, chroma_id=chroma_id, text=transcription, metadata=meta
                    )

            if sem_ids:
                self.semantic_service.index_transcriptions(sem_ids, sem_transcripts, sem_metas)

        self.graph_service.persist()
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
                    self.graph_service.index_utterance(
                        speaker_id=uid, chroma_id=chroma_id, text=transcription, metadata=meta
                    )

                enrolled_count += 1

            if sem_ids:
                self.semantic_service.index_transcriptions(sem_ids, sem_transcripts, sem_metas)

        self.graph_service.persist()
        print(f"[Conversation] Enrolled {enrolled_count} segments from dialogue {dialogue_id}.")

        return {
            "status": "success",
            "dialogue_id": dialogue_id,
            "identity_map": identity_map,
            "enrolled_segments": enrolled_count,
        }

    def index_chat_session(
        self,
        messages: List[Dict[str, str]],
        user_id: str = "user",
        assistant_id: str = "assistant",
        session_id: Optional[str] = None,
    ) -> dict:
        """Enroll a chat session (user/assistant exchanges) into the memory layer.

        Mem0/Zep-style: chat turns become first-class enrollments so future
        queries can retrieve from them just like from audio dialogues.

        Each message is turned into one utterance:
          - User messages   → speaker_id=user_id, KG entity/relation extraction enabled.
          - Assistant messages → speaker_id=assistant_id, semantic search only
            (KG extraction skipped: assistant responses paraphrase the FACT
            SHEET and would re-extract already-known entities).

        All messages share a single dialogue_id so the FACT SHEET's
        'Reconstructed Conversation' branch can replay the full exchange at
        query time.

        Parameters:
          messages: list of {"role": "user"|"assistant", "content": str,
                            optional "timestamp": ISO string}
          user_id, assistant_id: speaker IDs in the graph/semantic store.
          session_id: optional dialogue_id. When omitted, a fresh UUID is
                      assigned — each call creates a new dialogue.

        Returns: {status, dialogue_id, enrolled_count, user_turns, assistant_turns}
        """
        if not messages:
            return {
                "status": "success",
                "dialogue_id": None,
                "enrolled_count": 0,
                "user_turns": 0,
                "assistant_turns": 0,
                "message": "No messages provided.",
            }

        dialogue_id = session_id or f"chat_{uuid.uuid4().hex[:12]}"

        sem_ids: List[str] = []
        sem_docs: List[str] = []
        sem_metas: List[Dict[str, Any]] = []

        user_turns = 0
        assistant_turns = 0

        for idx, msg in enumerate(messages):
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue

            speaker_id = user_id if role == "user" else assistant_id
            chroma_id = str(uuid.uuid4())

            meta: Dict[str, Any] = {
                "user_id": speaker_id,
                "audio_file": "",
                "data_type": "enroll",
                "session_id": f"chat_{dialogue_id}",
                "dialogue_id": dialogue_id,
                "utterance_id": idx,
                "transcription": content,
                "text_emotion": "Unknown",
                "acoustic_emotion": "Unknown",
                "language": "en",
                "source": "chat",
                "role": role,
            }
            if "timestamp" in msg and msg["timestamp"]:
                meta["timestamp"] = str(msg["timestamp"])

            sem_ids.append(chroma_id)
            sem_docs.append(content)
            sem_metas.append(meta)

            # User turns get KG extraction; assistant turns do not.
            self.graph_service.index_utterance(
                speaker_id=speaker_id,
                chroma_id=chroma_id,
                text=content,
                metadata=meta,
                mark_pending_extraction=(role == "user"),
            )

            if role == "user":
                user_turns += 1
            else:
                assistant_turns += 1

        if sem_ids:
            self.semantic_service.index_transcriptions(sem_ids, sem_docs, sem_metas)

        self.graph_service.persist()

        print(
            f"[Chat] Enrolled {len(sem_ids)} chat turns "
            f"({user_turns} user, {assistant_turns} assistant) "
            f"as dialogue {dialogue_id}."
        )

        return {
            "status": "success",
            "dialogue_id": dialogue_id,
            "enrolled_count": len(sem_ids),
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
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

        # In text-only mode, names matched in the question are an unreliable speaker
        # signal — a name appearing inside a quoted span is the *object* of the quote,
        # not the speaker. We always run unfiltered semantic search here and let each
        # retrieved turn carry its own speaker label so the LLM can attribute correctly.
        # The extracted names are still useful as a hint to the graph branches
        # (profile_rank restriction, entity-centric traversal), so we keep them for
        # the KG section only.
        extracted = self.extract_speakers_from_question(question)
        if extracted:
            resolution_note = (
                f"Names referenced in question: {', '.join(extracted)} "
                f"(used as graph hint only — semantic search runs unfiltered)."
            )
        else:
            resolution_note = "Semantic search runs unfiltered across all enrolled speakers."

        context_lines.append(resolution_note)
        context_lines.append("")

        fact_sheet, retrieved_dialogue_ids = self._append_semantic_section(
            context_lines=context_lines,
            question=question,
            target_user_ids=None,
            session_id=None,
        )
        fact_sheet = self._append_graph_section(
            fact_sheet=fact_sheet,
            question=question,
            speaker_ids=extracted if extracted else None,
            retrieved_dialogue_ids=retrieved_dialogue_ids,
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
        SIMILARITY_THRESHOLD = 0.45

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
        fact_sheet, retrieved_dialogue_ids = self._append_semantic_section(
            context_lines=context_lines,
            question=question,
            target_user_ids=list(identified_users) if identified_users else None,
            session_id=session_id,
        )
        fact_sheet = self._append_graph_section(
            fact_sheet=fact_sheet,
            question=question,
            speaker_ids=list(identified_users) if identified_users else None,
            retrieved_dialogue_ids=retrieved_dialogue_ids,
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
    ) -> Tuple[str, List[str]]:
        """Runs semantic search, appends the 'Relevant Past Statements' section to context_lines,
        and returns (fact_sheet_string, retrieved_dialogue_ids).

        The returned dialogue_ids preserve the order they were first surfaced and are used
        downstream by the KG section to resolve participant-style questions.
        """
        context_lines.append("=== FACT SHEET: RELEVANT PAST STATEMENTS ===")
        semantic_matches = self.semantic_service.search_by_text(
            query_text=question, top_k=5, session_id=session_id,
            target_user_ids=target_user_ids,
        )

        retrieved_dialogue_ids: List[str] = []
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
                    retrieved_dialogue_ids.append(str(dialogue_id))
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
        return "\n".join(context_lines), retrieved_dialogue_ids

    def _append_graph_section(
        self,
        fact_sheet: str,
        question: str,
        speaker_ids: Optional[List[str]] = None,
        retrieved_dialogue_ids: Optional[List[str]] = None,
    ) -> str:
        """Queries the knowledge graph and appends a KG context section to the fact sheet.

        The output is shaped by the question type so the FACT SHEET stays focused:
          - profile_count          → exact count for (speaker, emotion)
          - profile_rank           → ranked list of speakers for an emotion
          - dialogue_participants  → participants of the dialogue(s) surfaced by semantic search
          - default                → trimmed entity-centric traversal (no global co-speaker dump)
        """
        lines = ["\n=== FACT SHEET: KNOWLEDGE GRAPH CONTEXT ==="]
        qtype = _classify_question(question)
        emotion = _detect_emotion(question)
        mentioned_speakers = self.extract_speakers_from_question(question)

        # ---- profile_count: "How many times does X express Y?" ----
        if qtype == "profile_count":
            targets = list(dict.fromkeys((speaker_ids or []) + mentioned_speakers))
            if not targets:
                lines.append(
                    f"Profile-count question detected (emotion='{emotion}') but no enrolled "
                    f"speaker was identified in the question."
                )
            else:
                for spk in targets:
                    n_text = self.graph_service.count_emotion(spk, emotion, channel="text")
                    n_acou = self.graph_service.count_emotion(spk, emotion, channel="acoustic")
                    lines.append(
                        f"Aggregate: {spk.capitalize()} expressed '{emotion}' "
                        f"{n_text} time(s) by text-emotion and {n_acou} time(s) by acoustic-emotion "
                        f"across the enrolled corpus."
                    )
            lines.append("")
            return fact_sheet + "\n".join(lines)

        # ---- profile_rank: "Who shows X most often?" / "Joey or Ross?" ----
        if qtype == "profile_rank":
            ranked = self.graph_service.rank_speakers_by_emotion(
                emotion,
                restrict_to=mentioned_speakers if mentioned_speakers else None,
                channel="text",
                top_k=10,
            )
            if not ranked:
                lines.append(f"No speakers with text-emotion '{emotion}' found in the corpus.")
            else:
                scope = (", ".join(s.capitalize() for s in mentioned_speakers)
                         if mentioned_speakers else "all enrolled speakers")
                lines.append(
                    f"Ranking by '{emotion}' frequency (text-emotion channel) across {scope}:"
                )
                for s, n in ranked:
                    lines.append(f"  - {s.capitalize()}: {n} utterance(s)")
            lines.append("")
            return fact_sheet + "\n".join(lines)

        # ---- dialogue_participants: "who else was in the room / conversation?" ----
        if qtype == "dialogue_participants":
            if not retrieved_dialogue_ids:
                lines.append(
                    "Participant question detected but no dialogue could be resolved "
                    "from semantic search."
                )
            else:
                for dlg_id in retrieved_dialogue_ids:
                    participants = self.graph_service.dialogue_participants(dlg_id)
                    if participants:
                        names = ", ".join(s.capitalize() for s in sorted(participants))
                        lines.append(f"Dialogue {dlg_id} participants: {names}")
                    else:
                        lines.append(f"Dialogue {dlg_id}: no participants recorded in the graph.")
            lines.append("")
            return fact_sheet + "\n".join(lines)

        # ---- default: trimmed entity-centric traversal ----
        kg = self.graph_service.query(question, speaker_ids=speaker_ids)

        if not kg["matched_entities"]:
            lines.append("No named entities detected in the question for graph traversal.")
            lines.append("")
            return fact_sheet + "\n".join(lines)

        ent_str = ", ".join(f"{e['text']} ({e['label']})" for e in kg["matched_entities"][:10])
        if kg.get("fallback_mode"):
            lines.append(f"Entities mentioned by speaker(s): {ent_str}")
        else:
            lines.append(f"Entities in question: {ent_str}")

        if kg["related_speakers"]:
            lines.append(
                "Related speakers via graph: "
                + ", ".join(s.capitalize() for s in kg["related_speakers"])
            )

        # Typed-fact edges (Phase 4) — surface the relations that radiate from
        # the matched entities/speakers so questions like "where does Jill
        # Goodacre work?" can be answered from the graph rather than only the
        # utterances that mention her.
        typed_facts = kg.get("typed_facts") or []
        if typed_facts:
            lines.append("\nKnown typed facts about matched entities:")
            for f in typed_facts:
                lines.append(
                    f"  - {f['subject']} {f['predicate']} {f['object']}"
                )
                if f.get("fact"):
                    lines.append(f'      "{f["fact"]}"')

        # Skip the global co_speakers dump in default mode — it was the main FACT SHEET
        # noise contributor. Participant questions get routed to the dialogue branch above.

        if kg["utterances"]:
            lines.append("\nGraph-retrieved utterances:")
            for utt in kg["utterances"][:5]:
                lines.append(
                    f"  {utt['speaker'].capitalize()} "
                    f"(Vocal: {utt['acoustic_emotion']} | Semantic: {utt['text_emotion']}): "
                    f"\"{utt['text']}\""
                )

        lines.append("")
        return fact_sheet + "\n".join(lines)

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
                1. The "Identity" field is the ABSOLUTE GROUND TRUTH.
                2. If the "Identity" strings are identical across tracks, it is GUARANTEED to be the same person. Disregard differing Age/Gender.
                3. If the "Identity" strings differ, they are DIFFERENT people.
                4. Use the "Emotion" tags to explain HOW the speaker was feeling.
                5. EMOTION EVALUATION: The "Current Utterance Analysis" emotion tags are generated by imperfect zero-shot AI. Do NOT blindly trust them.
                6. CONTEXTUAL REASONING: You must read the "Reconstructed Conversations" to understand the speaker's history. If the past dialogue shows the speaker was arguing, sarcastic, annoyed, etc., you must correct the imperfect zero-shot tags and state their TRUE emotion based on that history.
                7. CONVERSATION HISTORY: If prior turns are provided below, use them to resolve follow-up references (e.g., "he", "that speaker", "the previous one"), but the FACT SHEET above remains the source of truth.
{history_block}

                MANDATORY REASONING FORMAT:
                If asked if speakers are the same, start by explicitly stating the Identity of each track.

                ### User Question:
                {question}

                ### Answer:"""

    def get_enrolled_users(self) -> dict:
        """Retrieves all enrolled users from the database."""
        users = self.audio_service.get_all_enrolled_users()
        return {"status": "success", "enrolled_users": users}