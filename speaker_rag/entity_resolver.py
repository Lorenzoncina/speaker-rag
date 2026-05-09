"""
Entity resolution for SpeakerRAG knowledge graph.

Given a newly extracted entity (name, type, description), decide whether it
refers to the same real-world entity as one already in the graph. Pipeline:

  1. Embed `name (type): description` via SentenceTransformer (reused from
     semantic_service to avoid loading a second model).
  2. Cosine-search top-K existing entities of the SAME type in a dedicated
     Chroma collection (`entity_embeddings`).
  3. Auto-merge when top similarity ≥ HIGH_THRESHOLD (very confident).
  4. Ask Sonnet to judge when similarity falls in the AMBIGUOUS band.
  5. Otherwise treat as new.

The resolver does not mutate the NetworkX graph — it returns a Decision the
GraphService applies. This keeps Chroma and graph state changes co-located in
GraphService and makes the resolver easy to unit-test in isolation.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResolutionDecision:
    is_duplicate: bool
    target_node_id: Optional[str] = None
    canonical_name: Optional[str] = None
    canonical_description: Optional[str] = None
    top_score: float = 0.0
    method: str = "none"  # 'cosine_high' | 'llm_judge' | 'none'


class EntityResolver:
    # Tuned 2026-05-09 after observing a wrong silent merge ('GR-6' → 'KL-5')
    # at score 0.942 and many same-name candidates falling through below 0.78
    # because their per-dialogue descriptions diverged.
    HIGH_THRESHOLD = 0.96      # auto-merge above this (very near-identical)
    AMBIGUOUS_LOWER = 0.60     # ask LLM judge in [0.60, 0.96)
    TOP_K = 5                  # number of candidates to fetch
    JUDGE_TOP_K = 3            # number of candidates passed to the LLM judge

    def __init__(
        self,
        chroma_collection,
        embedding_model,
        llm_extractor=None,
    ):
        self._collection = chroma_collection
        self._model = embedding_model
        self._llm = llm_extractor

    @property
    def available(self) -> bool:
        return self._collection is not None and self._model is not None

    def _encode(self, name: str, etype: str, description: str) -> List[float]:
        text = f"{name} ({etype}): {description or 'no description'}"
        emb = self._model.encode(text)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def add(
        self,
        node_id: str,
        name: str,
        etype: str,
        description: str,
    ) -> None:
        """Insert (or upsert) an entity into the Chroma collection.

        Uses upsert so re-syncing an existing entity (e.g. during a backfill
        pass on a graph that predates the resolver) is a no-op rather than an
        error.
        """
        if not self.available:
            return
        emb = self._encode(name, etype, description)
        try:
            self._collection.upsert(
                ids=[node_id],
                embeddings=[emb],
                metadatas=[{
                    "name": name,
                    "type": etype,
                    "description": description or "",
                }],
                documents=[f"{name} ({etype}): {description}"],
            )
        except Exception as e:
            print(f"[EntityResolver] add failed for {node_id}: {e}")

    def update(
        self,
        node_id: str,
        name: str,
        etype: str,
        description: str,
    ) -> None:
        """Update an existing entity's embedding/metadata after a merge."""
        if not self.available:
            return
        emb = self._encode(name, etype, description)
        try:
            self._collection.update(
                ids=[node_id],
                embeddings=[emb],
                metadatas=[{
                    "name": name,
                    "type": etype,
                    "description": description or "",
                }],
                documents=[f"{name} ({etype}): {description}"],
            )
        except Exception as e:
            print(f"[EntityResolver] update failed for {node_id}: {e}")

    def delete(self, node_id: str) -> None:
        if not self.available:
            return
        try:
            self._collection.delete(ids=[node_id])
        except Exception as e:
            print(f"[EntityResolver] delete failed for {node_id}: {e}")

    def resolve(
        self,
        name: str,
        etype: str,
        description: str,
        dialogue_context: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> ResolutionDecision:
        """Decide whether this candidate is a duplicate of an existing entity.

        exclude_ids: optional list of entity ids to exclude from candidates
        (useful during a retrospective pass to avoid matching the entity to
        itself).
        """
        if not self.available:
            return ResolutionDecision(is_duplicate=False)

        emb = self._encode(name, etype, description)
        try:
            results = self._collection.query(
                query_embeddings=[emb],
                n_results=self.TOP_K,
                where={"type": etype},
                include=["metadatas", "distances", "documents"],
            )
        except Exception as e:
            print(f"[EntityResolver] query failed: {e}")
            return ResolutionDecision(is_duplicate=False)

        ids = (results.get("ids") or [[]])[0]
        if not ids:
            return ResolutionDecision(is_duplicate=False)

        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        scores = [1.0 - d for d in dists]

        # Filter excluded ids
        if exclude_ids:
            keep = [(i, m, s) for i, m, s in zip(ids, metas, scores) if i not in exclude_ids]
            if not keep:
                return ResolutionDecision(is_duplicate=False)
            ids, metas, scores = zip(*keep)
            ids, metas, scores = list(ids), list(metas), list(scores)

        top_id, top_meta, top_score = ids[0], metas[0], scores[0]

        if top_score >= self.HIGH_THRESHOLD:
            return ResolutionDecision(
                is_duplicate=True,
                target_node_id=top_id,
                canonical_name=self._pick_canonical_name(name, top_meta.get("name", name)),
                canonical_description=self._merge_descriptions(description, top_meta.get("description", "")),
                top_score=top_score,
                method="cosine_high",
            )

        if top_score < self.AMBIGUOUS_LOWER:
            return ResolutionDecision(is_duplicate=False, top_score=top_score)

        if not (self._llm and self._llm.available):
            return ResolutionDecision(is_duplicate=False, top_score=top_score)

        candidates: List[Dict[str, Any]] = []
        for i in range(min(self.JUDGE_TOP_K, len(ids))):
            candidates.append({
                "id": ids[i],
                "name": metas[i].get("name", ""),
                "type": metas[i].get("type", ""),
                "description": metas[i].get("description", ""),
                "score": round(scores[i], 3),
            })

        try:
            judge = self._llm.judge_entity_duplicate(
                candidate={"name": name, "type": etype, "description": description},
                existing=candidates,
                dialogue_context=dialogue_context or "",
            )
        except Exception as e:
            print(f"[EntityResolver] LLM judge failed: {e}")
            return ResolutionDecision(is_duplicate=False, top_score=top_score)

        if not judge or not judge.get("is_duplicate"):
            return ResolutionDecision(is_duplicate=False, top_score=top_score, method="llm_judge")

        target_id = judge.get("target_id")
        if not target_id or target_id not in ids:
            return ResolutionDecision(is_duplicate=False, top_score=top_score, method="llm_judge")

        return ResolutionDecision(
            is_duplicate=True,
            target_node_id=target_id,
            canonical_name=judge.get("canonical_name") or name,
            canonical_description=judge.get("canonical_description") or description,
            top_score=top_score,
            method="llm_judge",
        )

    @staticmethod
    def _pick_canonical_name(name_a: str, name_b: str) -> str:
        a = (name_a or "").strip()
        b = (name_b or "").strip()
        if not a:
            return b
        if not b:
            return a
        # Prefer the longer/more-complete name. Tie-break: keep the existing one (b).
        return a if len(a) > len(b) else b

    @staticmethod
    def _merge_descriptions(desc_a: str, desc_b: str) -> str:
        a = (desc_a or "").strip()
        b = (desc_b or "").strip()
        if not a:
            return b
        if not b:
            return a
        if a in b:
            return b
        if b in a:
            return a
        return f"{b} | {a}"
