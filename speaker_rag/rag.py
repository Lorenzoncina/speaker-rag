import uuid
from transformers import pipeline

# Internal imports hidden from the end-user
from .model_loader import load_audio_models
from .identification_service import Identification_Service

class SpeakerRAG:
    def __init__(self, 
                 chroma_dir: str = "./speaker_chroma_db", 
                 collection_name: str = "speaker_embeddings",
                 llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", 
                 device: str = "cuda"):
        """
        Initializes the Speaker RAG system. 
        Automatically loads all required audio models, connects to ChromaDB, and loads the LLM.
        """
        print(f"Initializing SpeakerRAG on device: {device}...")
        
        # 1. Load Audio Models internally
        audio_models = load_audio_models(device=device)
        
        # 2. Instantiate the private Service internally
        self.audio_service = Identification_Service(
            **audio_models,
            device=device,
            chroma_client=chroma_dir,
            collection_name=collection_name
        )
        
        # 3. Load the LLM internally
        print(f"Loading LLM ({llm_model_id}) into memory...")
        self.llm = pipeline(
            "text-generation", 
            model=llm_model_id, 
            device_map="auto"
        )
        print("SpeakerRAG initialization complete. All models ready.")

    def index_audio(self, audio_path: str, user_id: str = None, session_id: str = "rag_session") -> dict:
        """Enrolls a known speaker into the vector database."""
        if user_id is None:
            user_id = str(uuid.uuid4())
            
        audio_map = {user_id: [audio_path]}
        
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map,
            data_type="enroll",
            session_id=session_id
        )
        
        return {
            "status": "success", 
            "user_id": user_id, 
            "chroma_id": extraction_results[user_id][0]["id"]
        }

    def _format_results_for_llm(self, raw_results: list) -> str:
        """Translates the backend JSON into a readable prompt context."""
        if not raw_results or not raw_results[0].get("matches"):
            return "No matching speakers were found in the database."

        matches = raw_results[0]["matches"]
        context_lines = []
        
        for i, match in enumerate(matches):
            score = match.get("score", 0.0)
            meta = match.get("metadata", {})
            
            speaker_id = meta.get("user_id", "Unknown") 
            age = meta.get("age", "unknown age")
            gender = meta.get("gender", "unknown gender")
            language = meta.get("language", "unknown language")
            
            context_lines.append(
                f"Match #{i+1}: Speaker ID '{speaker_id}' (Similarity: {score:.2f}). "
                f"Attributes: {age} {gender} speaking {language}."
            )
            
        return "\n".join(context_lines)

    def query(self, audio_path: str, question: str, session_id: str = "rag_session") -> str:
        """Identifies an unknown audio file and asks the LLM about it."""
        temp_speaker_id = str(uuid.uuid4())
        audio_map = {temp_speaker_id: [audio_path]}
        
        extraction_results = self.audio_service.extract_attributes(
            user_audios_map=audio_map,
            data_type="probe",
            session_id=session_id
        )
        chroma_id = extraction_results[temp_speaker_id][0]["id"]

        raw_results = self.audio_service.identify_by_id(
            query_ids=[chroma_id],
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
            max_new_tokens=150, 
            return_full_text=False,
            temperature=0.3
        )
        
        return outputs[0]["generated_text"].strip()