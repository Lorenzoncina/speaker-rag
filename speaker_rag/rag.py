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
        
        # NEW: Initialize Semantic Service
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

    def query(self, audio_paths: List[str], question: str, session_id: str = "rag_session") -> str:
        """Dual Retrieval: Uses Audio Probes to identify voices AND Question to find text."""
        # -----------------------------
        # 1. Audio Probe Retrieval (Voice Similarity)
        # -----------------------------
        temp_speaker_id = str(uuid.uuid4())
        audio_map = {temp_speaker_id: audio_paths}
        
        # Extract attributes and embeddings for the probe audio
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map, data_type="probe", session_id=session_id
        )
        chroma_ids = [res["id"] for res in extraction_results.get(temp_speaker_id, [])]

        # Search the database for voice matches
        raw_id_results = self.audio_service.identify_by_id(
            query_ids=chroma_ids, top_k=3, session_id=session_id, probe_only=False
        )

        context_lines = ["### [1] SPEAKER IDENTIFICATION MATCHES (Based on Voice Similiarity)"]
        for result in raw_id_results:
            file_name = os.path.basename(result.get("probe_metadata", {}).get("audio_file", "Unknown File"))
            matches = result.get("matches", [])
            
            if not matches:
                context_lines.append(f"Probe: {file_name} -> No voice matches found.")
            else:
                matched_ids = [m["chroma_id"] for m in matches]
                # Get the transcriptions of the matched identities from the Semantic DB
                transcripts_dict = self.semantic_service.get_texts_by_ids(matched_ids)
                
                context_lines.append(f"--- Probe File: {file_name} ---")
                for i, match in enumerate(matches):
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    
                    # Extract Attributes carefully
                    speaker_id = meta.get("user_id", "Unknown") 
                    age = meta.get("age", "unknown age")
                    gender = meta.get("gender", "unknown gender")
                    language = meta.get("language", "unknown language")
                    transcription = transcripts_dict.get(match["chroma_id"], "[No transcription available]")
                    
                    context_lines.append(
                        f"  Match #{i+1}: Speaker '{speaker_id}' (Voice Similarity: {score:.2f}). "
                        f"Attributes: {age} {gender} speaking {language}. "
                        f"What they previously said: \"{transcription}\""
                    )
                context_lines.append("")

        # -----------------------------
        # 2. Semantic Question Retrieval (Text Similarity)
        # -----------------------------
        context_lines.append("### [2] SEMANTIC SEARCH MATCHES (Based on the text of the question)")
        
        semantic_matches = self.semantic_service.search_by_text(query_text=question, top_k=3, session_id=session_id)
        
        if not semantic_matches:
            context_lines.append("No semantic matches found for the question.\n")
        else:
            for i, match in enumerate(semantic_matches):
                meta = match.get("metadata", {})
                
                # Extract Attributes for Semantic Matches
                speaker_id = meta.get("user_id", "Unknown") 
                age = meta.get("age", "unknown age")
                gender = meta.get("gender", "unknown gender")
                language = meta.get("language", "unknown language")
                
                context_lines.append(
                    f"  Match #{i+1}: Speaker '{speaker_id}' (Semantic Sim: {match['score']:.2f}). "
                    f"Attributes: {age} {gender} speaking {language}. "
                    f"They said: \"{match['text']}\""
                )
            context_lines.append("")

        final_context = "\n".join(context_lines)

        # -----------------------------
        # 3. Ask Multimodal LLM
        # -----------------------------
        prompt = f"""You are an intelligent audio-analysis assistant. 
                    Listen to the provided audio recordings and analyze the dual-backend retrieval results below.
                    The retrieval includes both Voice Similarity (identifying the speakers in the audio) and Semantic Similarity (identifying previous discussions matching your question).

                    {final_context}

                    ### User Question:
                    {question}

                    ### Answer:"""

        inputs = self._prepare_qwen_audio_inputs(audio_paths, prompt)
        
        generated_ids = self.llm.generate(**inputs, max_new_tokens=300, temperature=0.3)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response