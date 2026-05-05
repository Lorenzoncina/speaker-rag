"""
Knowledge graph layer for SpeakerRAG.

Node types:
  speaker   — an enrolled or diarized speaker identity
  utterance — a single transcribed audio segment (keyed by its ChromaDB ID)
  entity    — a named entity extracted from transcription text

Edge relations:
  Speaker  --SAID-->     Utterance   {dialogue_id}
  Utterance --MENTIONS--> Entity     {label: ORG/PERSON/GPE/…}
  Speaker  --KNOWS-->    Speaker     {n_dialogues: int}  (bidirectional)

Persistence: JSON file via networkx node_link serialization.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Optional spaCy NER — falls back to regex when not installed
# ---------------------------------------------------------------------------
try:
    import spacy as _spacy_module
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

_SPACY_NLP = None  # lazy-loaded on first call

# Regex patterns for the fallback extractor
_TITLE_RE = re.compile(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]+)*)\b")
_CAPS_RE = re.compile(r"\b([A-Z]{2,})\b")
_STOP = {
    "The", "A", "An", "I", "We", "He", "She", "They", "It", "You",
    "This", "That", "These", "Those", "Is", "Are", "Was", "Were",
    "Has", "Have", "Had", "Do", "Does", "Did", "Will", "Would",
    "Can", "Could", "Should", "May", "Might", "Must", "Shall",
    "And", "But", "Or", "So", "For", "Yet", "Nor",
    "In", "On", "At", "By", "To", "Of", "Up", "As", "If",
}


def _regex_entities(text: str) -> List[Tuple[str, str]]:
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for m in _TITLE_RE.finditer(text):
        w = m.group(1)
        if w not in _STOP and w not in seen:
            seen.add(w)
            out.append((w, "UNKNOWN"))
    for m in _CAPS_RE.finditer(text):
        w = m.group(1)
        if w not in seen:
            seen.add(w)
            out.append((w, "ORG"))
    return out


def _spacy_entities(text: str) -> List[Tuple[str, str]]:
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = _spacy_module.load("en_core_web_sm")
        except Exception:
            return _regex_entities(text)
    keep = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC"}
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for ent in _SPACY_NLP(text).ents:
        if ent.label_ in keep:
            k = ent.text.strip()
            if k and k not in seen:
                seen.add(k)
                out.append((k, ent.label_))
    return out


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Return (surface_text, label) pairs for named entities found in text."""
    return _spacy_entities(text) if _SPACY_AVAILABLE else _regex_entities(text)


# ---------------------------------------------------------------------------
# GraphService
# ---------------------------------------------------------------------------

class GraphService:
    """Lightweight in-process knowledge graph for speaker-utterance-entity relations."""

    def __init__(self, graph_path: Optional[str] = None):
        self._graph_path = Path(graph_path) if graph_path else None
        self._g: nx.DiGraph = nx.DiGraph()
        if self._graph_path and self._graph_path.exists():
            self._load()
        n, e = self._g.number_of_nodes(), self._g.number_of_edges()
        if n:
            print(f"[GraphService] Loaded graph: {n} nodes, {e} edges.")
        else:
            print("[GraphService] Initialized empty knowledge graph.")

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def index_utterance(
        self,
        speaker_id: str,
        chroma_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a speaker, utterance, and its extracted entities to the graph.

        Does NOT auto-persist — call persist() explicitly after a batch.
        """
        if not text or not text.strip():
            return
        meta = metadata or {}
        dialogue_id = str(meta.get("dialogue_id", ""))

        # Speaker node
        if not self._g.has_node(speaker_id):
            self._g.add_node(speaker_id, type="speaker")

        # Utterance node (cap stored text to keep graph file manageable)
        self._g.add_node(
            chroma_id,
            type="utterance",
            text=text[:500],
            dialogue_id=dialogue_id,
            utterance_id=int(meta.get("utterance_id", 0)),
            acoustic_emotion=str(meta.get("acoustic_emotion", "")),
            text_emotion=str(meta.get("text_emotion", "")),
        )

        # Speaker --SAID--> Utterance
        self._g.add_edge(
            speaker_id,
            chroma_id,
            relation="SAID",
            dialogue_id=dialogue_id,
        )

        # Entity nodes + MENTIONS edges
        for ent_text, ent_label in extract_entities(text):
            ent_id = f"ent::{ent_label}::{ent_text.lower()}"
            if not self._g.has_node(ent_id):
                self._g.add_node(
                    ent_id,
                    type="entity",
                    label=ent_label,
                    surface=ent_text,
                )
            if not self._g.has_edge(chroma_id, ent_id):
                self._g.add_edge(chroma_id, ent_id, relation="MENTIONS", label=ent_label)

        # KNOWS edges: link co-speakers who share this dialogue
        if dialogue_id:
            for node, attrs in self._g.nodes(data=True):
                if (
                    attrs.get("type") == "utterance"
                    and attrs.get("dialogue_id") == dialogue_id
                    and node != chroma_id
                ):
                    for pred in list(self._g.predecessors(node)):
                        if (
                            self._g.nodes.get(pred, {}).get("type") == "speaker"
                            and pred != speaker_id
                        ):
                            self._add_knows(speaker_id, pred)

    def _add_knows(self, a: str, b: str) -> None:
        """Add or increment a bidirectional KNOWS edge between two speakers."""
        for src, dst in ((a, b), (b, a)):
            if self._g.has_edge(src, dst) and self._g[src][dst].get("relation") == "KNOWS":
                self._g[src][dst]["n_dialogues"] += 1
            else:
                self._g.add_edge(src, dst, relation="KNOWS", n_dialogues=1)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        speaker_ids: Optional[List[str]] = None,
        max_utterances: int = 5,
    ) -> Dict[str, Any]:
        """Walk the graph from entities found in the question.

        Returns:
          matched_entities  — entities found in both question and graph
          related_speakers  — speakers connected to those entities
          utterances        — top matching utterances (by entity overlap count)
          co_speakers       — KNOWS pairs among related speakers
        """
        out: Dict[str, Any] = {
            "matched_entities": [],
            "related_speakers": [],
            "utterances": [],
            "co_speakers": [],
        }

        if not question.strip() or self._g.number_of_nodes() == 0:
            return out

        q_entities = extract_entities(question)
        if not q_entities:
            return out

        # Match entity nodes in graph (exact key first, then partial surface match)
        matched_ent_ids: List[str] = []
        for ent_text, ent_label in q_entities:
            exact = f"ent::{ent_label}::{ent_text.lower()}"
            if self._g.has_node(exact):
                matched_ent_ids.append(exact)
                out["matched_entities"].append({"text": ent_text, "label": ent_label})
            else:
                # partial match against stored surface text (label-agnostic)
                q_lower = ent_text.lower()
                for nid, ndata in self._g.nodes(data=True):
                    if ndata.get("type") == "entity":
                        surface = ndata.get("surface", "").lower()
                        if q_lower in surface or surface in q_lower:
                            matched_ent_ids.append(nid)
                            out["matched_entities"].append(
                                {"text": ndata["surface"], "label": ndata.get("label", "")}
                            )
                            break  # one match per query entity is enough

        if not matched_ent_ids:
            return out

        # Entity → predecessor utterances, scored by number of matched entities
        utt_scores: Dict[str, int] = {}
        for eid in set(matched_ent_ids):
            for uid in self._g.predecessors(eid):
                if self._g.nodes.get(uid, {}).get("type") == "utterance":
                    utt_scores[uid] = utt_scores.get(uid, 0) + 1

        # Optionally restrict to utterances from specific speakers
        if speaker_ids:
            spk_set = set(speaker_ids)
            utt_scores = {
                uid: score
                for uid, score in utt_scores.items()
                if any(
                    self._g.nodes.get(p, {}).get("type") == "speaker" and p in spk_set
                    for p in self._g.predecessors(uid)
                )
            }

        top = sorted(utt_scores, key=lambda x: utt_scores[x], reverse=True)[:max_utterances]

        seen_speakers: set = set()
        for uid in top:
            ua = self._g.nodes[uid]
            spk = next(
                (
                    p
                    for p in self._g.predecessors(uid)
                    if self._g.nodes.get(p, {}).get("type") == "speaker"
                ),
                "Unknown",
            )
            seen_speakers.add(spk)
            out["utterances"].append(
                {
                    "chroma_id": uid,
                    "text": ua.get("text", ""),
                    "speaker": spk,
                    "acoustic_emotion": ua.get("acoustic_emotion", ""),
                    "text_emotion": ua.get("text_emotion", ""),
                }
            )

        out["related_speakers"] = list(seen_speakers)

        # KNOWS edges among the speakers we just found
        for spk in seen_speakers:
            for nbr in self._g.successors(spk):
                edge = self._g[spk].get(nbr, {})
                if (
                    self._g.nodes.get(nbr, {}).get("type") == "speaker"
                    and edge.get("relation") == "KNOWS"
                ):
                    out["co_speakers"].append(
                        {
                            "speaker_a": spk,
                            "speaker_b": nbr,
                            "n_dialogues": edge.get("n_dialogues", 0),
                        }
                    )

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Serialize the graph to JSON. No-op if no graph_path was given."""
        if not self._graph_path:
            return
        try:
            self._graph_path.parent.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self._g, edges="links")
            self._graph_path.write_text(json.dumps(data))
        except Exception as e:
            print(f"[GraphService] Persist failed: {e}")

    def _load(self) -> None:
        try:
            data = json.loads(self._graph_path.read_text())
            self._g = nx.node_link_graph(data, directed=True, edges="links")
        except Exception as e:
            print(f"[GraphService] Load failed ({e}), starting empty.")
            self._g = nx.DiGraph()

    def clear(self) -> None:
        """Wipe all nodes/edges and delete the persisted file."""
        self._g = nx.DiGraph()
        if self._graph_path and self._graph_path.exists():
            self._graph_path.unlink()
        print("[GraphService] Knowledge graph cleared.")

    def get_viz_data(self, max_entities: int = 40) -> Dict[str, Any]:
        """Return a simplified speaker-entity bipartite graph for visualization.

        Utterance nodes are not returned (too many to render usefully).
        Entity nodes are limited to the top-N by total mention count.
        Edges returned:
          - KNOWS   between speakers (with n_dialogues weight)
          - MENTIONS from each speaker to its top entities (aggregated through utterances)
        """
        g = self._g

        speakers = [n for n, d in g.nodes(data=True) if d.get("type") == "speaker"]

        # For each speaker accumulate entity mention counts through utterance nodes
        speaker_entity_counts: Dict[str, Dict[str, int]] = {spk: {} for spk in speakers}
        for spk in speakers:
            for utt in g.successors(spk):
                if g.nodes.get(utt, {}).get("type") != "utterance":
                    continue
                for ent in g.successors(utt):
                    if g.nodes.get(ent, {}).get("type") == "entity":
                        speaker_entity_counts[spk][ent] = (
                            speaker_entity_counts[spk].get(ent, 0) + 1
                        )

        # Pick top entities by global mention count
        entity_total: Dict[str, int] = {}
        for spk_ents in speaker_entity_counts.values():
            for ent, cnt in spk_ents.items():
                entity_total[ent] = entity_total.get(ent, 0) + cnt
        top_entities = sorted(entity_total, key=lambda x: entity_total[x], reverse=True)[
            :max_entities
        ]
        top_entity_set = set(top_entities)

        # Build node list
        nodes: List[Dict[str, Any]] = []
        for spk in speakers:
            utt_count = sum(
                1 for n in g.successors(spk)
                if g.nodes.get(n, {}).get("type") == "utterance"
            )
            nodes.append({"id": spk, "type": "speaker", "utterance_count": utt_count})
        for ent in top_entities:
            nd = g.nodes[ent]
            nodes.append(
                {
                    "id": ent,
                    "type": "entity",
                    "label": nd.get("label", "UNKNOWN"),
                    "surface": nd.get("surface", ent),
                    "total_mentions": entity_total[ent],
                }
            )

        # Build edge list
        edges: List[Dict[str, Any]] = []
        seen_knows: set = set()
        for spk in speakers:
            # KNOWS edges (deduplicated — stored bidirectionally)
            for nbr in g.successors(spk):
                if g.nodes.get(nbr, {}).get("type") == "speaker":
                    ed = g[spk][nbr]
                    if ed.get("relation") == "KNOWS":
                        pair = tuple(sorted([spk, nbr]))
                        if pair not in seen_knows:
                            seen_knows.add(pair)
                            edges.append(
                                {
                                    "source": spk,
                                    "target": nbr,
                                    "relation": "KNOWS",
                                    "n_dialogues": ed.get("n_dialogues", 0),
                                }
                            )
            # MENTIONS edges (aggregated through utterances)
            for ent, cnt in speaker_entity_counts[spk].items():
                if ent in top_entity_set:
                    edges.append(
                        {"source": spk, "target": ent, "relation": "MENTIONS", "count": cnt}
                    )

        return {"stats": self.stats, "nodes": nodes, "edges": edges}

    @property
    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, attrs in self._g.nodes(data=True):
            t = attrs.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            **counts,
        }
