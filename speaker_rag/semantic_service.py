"""
Script Name: semantic_service.py

Description:
Handles the core logic for Semantic indexing and retrieval. Uses a SentenceTransformer 
model to compute embeddings for the text transcribed from the audio files. Stores these 
embeddings in a parallel ChromaDB collection using the exact same IDs as the Speaker 
collection, allowing 1-to-1 cross-referencing between identity and semantic content.
"""
from typing import List, Dict, Any
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class _Semantic_Service:
    _instance = None
    _model = None
    _chroma_client = None
    _collection = None

    def clear_collection(self) -> bool:
        """Deletes the current collection and recreates an empty one."""
        try:
            collection_name = self._collection.name
            self._chroma_client.delete_collection(collection_name)
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection '{collection_name}' successfully cleared.")
            return True
        except Exception as e:
            print(f"Failed to clear semantic collection: {e}")
            return False

    def index_transcriptions(self, ids: List[str], transcriptions: List[str], metadatas: List[Dict[str, Any]]):
        """Computes semantic embeddings for the texts and stores them using the provided IDs."""
        if not ids or not transcriptions:
            return
        
        try:
            # Encode texts to dense vectors
            embeddings = self._model.encode(transcriptions, show_progress_bar=False).tolist()
            
            # Add to ChromaDB
            self._collection.add(
                documents=transcriptions,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"Failed to index transcriptions into Semantic DB: {e}")

    def search_by_text(self, query_text: str, top_k: int = 3, session_id: str = None) -> List[Dict[str, Any]]:
        """Searches the semantic database for the closest matching transcriptions to the query text."""
        try:
            query_embedding = self._model.encode([query_text]).tolist()
            
            # We want to search enrolled data + probe data from this session
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

            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where=where_filter
            )

            matches = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for res_id, doc, meta, dist in zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]):
                    score = max(0.0, 1.0 - dist)
                    matches.append({
                        "chroma_id": res_id,
                        "score": score,
                        "text": doc,
                        "metadata": meta
                    })
            return matches

        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
            
    def get_texts_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Fetches the actual transcriptions for specific Chroma IDs."""
        if not ids:
            return {}
        try:
            results = self._collection.get(ids=ids, include=["documents"])
            return {doc_id: doc for doc_id, doc in zip(results["ids"], results["documents"])}
        except Exception as e:
            print(f"Failed to fetch texts by ID: {e}")
            return {}

def Semantic_Service(semantic_model, chroma_client: str, collection_name: str):
    """Factory method for the singleton service."""
    if _Semantic_Service._instance is None:
        _Semantic_Service._instance = _Semantic_Service()
        _Semantic_Service._model = semantic_model
        _Semantic_Service._chroma_client = PersistentClient(path=chroma_client)
        _Semantic_Service._collection = _Semantic_Service._chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
    return _Semantic_Service._instance