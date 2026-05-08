"""
Knowledge graph layer for SpeakerRAG.

Node types:
  speaker   — an enrolled or diarized speaker identity
  utterance — a single transcribed audio segment (keyed by its ChromaDB ID)
  entity    — a named entity extracted from transcription text
  dialogue  — a single multi-speaker conversation (keyed by dialogue_id)

Edge relations:
  Speaker  --SAID-->            Utterance   {dialogue_id}
  Utterance --MENTIONS-->       Entity      {label: ORG/PERSON/GPE/…}
  Speaker  --KNOWS-->           Speaker     {n_dialogues: int}  (bidirectional)
  Dialogue --CONTAINS-->        Utterance
  Dialogue --HAS_PARTICIPANT--> Speaker

Persistence: JSON file via networkx node_link serialization.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .llm_extractor import LLMExtractor

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

    def __init__(
        self,
        graph_path: Optional[str] = None,
        llm_extractor: Optional[LLMExtractor] = None,
    ):
        self._graph_path = Path(graph_path) if graph_path else None
        self._g: nx.DiGraph = nx.DiGraph()
        self._llm_extractor = llm_extractor
        if self._graph_path and self._graph_path.exists():
            self._load()
            self._backfill_dialogue_nodes()
        n, e = self._g.number_of_nodes(), self._g.number_of_edges()
        if n:
            print(f"[GraphService] Loaded graph: {n} nodes, {e} edges.")
        else:
            print("[GraphService] Initialized empty knowledge graph.")
        if self._llm_extractor and self._llm_extractor.available:
            print(f"[GraphService] LLM entity extraction enabled (model={self._llm_extractor.model}).")
        else:
            reason = (
                self._llm_extractor.unavailable_reason
                if self._llm_extractor else "no extractor passed"
            )
            print(f"[GraphService] LLM extraction disabled — falling back to regex/spaCy ({reason}).")

    def _backfill_dialogue_nodes(self) -> None:
        """One-time migration: older graphs stored dialogue_id only as an utterance
        attribute. This adds Dialogue nodes + CONTAINS/HAS_PARTICIPANT edges so
        per-dialogue traversal is O(1)."""
        has_dialogue_node = any(
            d.get("type") == "dialogue" for _, d in self._g.nodes(data=True)
        )
        if has_dialogue_node:
            return

        added = 0
        for uid, ud in list(self._g.nodes(data=True)):
            if ud.get("type") != "utterance":
                continue
            dlg_raw = ud.get("dialogue_id", "")
            if not dlg_raw:
                continue
            dlg_node = f"dlg::{dlg_raw}"
            if not self._g.has_node(dlg_node):
                self._g.add_node(dlg_node, type="dialogue", dialogue_id=str(dlg_raw))
                added += 1
            self._g.add_edge(dlg_node, uid, relation="CONTAINS")
            for pred in self._g.predecessors(uid):
                if (self._g.nodes.get(pred, {}).get("type") == "speaker"
                        and not self._g.has_edge(dlg_node, pred)):
                    self._g.add_edge(dlg_node, pred, relation="HAS_PARTICIPANT")

        if added:
            print(f"[GraphService] Backfilled {added} dialogue nodes from existing utterances.")
            self.persist()

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

        # Entity nodes + MENTIONS edges.
        # When an LLM extractor is wired up, defer extraction — we'll batch
        # per dialogue at persist time so the model sees full context.
        # Otherwise fall back to the inline regex/spaCy path.
        if self._llm_extractor and self._llm_extractor.available:
            self._g.nodes[chroma_id]["entities_pending"] = True
        else:
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

        # Dialogue node + CONTAINS/HAS_PARTICIPANT edges, plus KNOWS via existing participants.
        # Using the dialogue's participant set keeps this O(participants) instead of O(all utterances).
        if dialogue_id:
            dlg_node = f"dlg::{dialogue_id}"
            if not self._g.has_node(dlg_node):
                self._g.add_node(dlg_node, type="dialogue", dialogue_id=dialogue_id)

            existing_participants = [
                p for p in self._g.successors(dlg_node)
                if self._g.nodes.get(p, {}).get("type") == "speaker"
            ]

            self._g.add_edge(dlg_node, chroma_id, relation="CONTAINS")
            if not self._g.has_edge(dlg_node, speaker_id):
                self._g.add_edge(dlg_node, speaker_id, relation="HAS_PARTICIPANT")

            for other in existing_participants:
                if other != speaker_id:
                    self._add_knows(speaker_id, other)

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

        Two traversal modes:
          1. Entity-centric (default): extract entities from the question, match them
             in the graph, walk back to utterances and speakers.
          2. Speaker-centric fallback: when speaker_ids are known but the question
             contains no entity names (e.g. "Does Jan mention any city?"), traverse
             directly from the speaker node to all entities it mentions.

        Returns:
          matched_entities  — entities found in the graph (from question NER or speaker traversal)
          related_speakers  — speakers connected to those entities
          utterances        — top matching utterances
          co_speakers       — KNOWS pairs among related speakers
          fallback_mode     — True when speaker-centric fallback was used
        """
        out: Dict[str, Any] = {
            "matched_entities": [],
            "related_speakers": [],
            "utterances": [],
            "co_speakers": [],
            "fallback_mode": False,
        }

        if self._g.number_of_nodes() == 0:
            return out

        # --- 1. Entity-centric traversal ---
        q_entities = extract_entities(question) if question.strip() else []

        matched_ent_ids: List[str] = []
        for ent_text, ent_label in q_entities:
            exact = f"ent::{ent_label}::{ent_text.lower()}"
            if self._g.has_node(exact):
                matched_ent_ids.append(exact)
                out["matched_entities"].append({"text": ent_text, "label": ent_label})
            else:
                q_lower = ent_text.lower()
                for nid, ndata in self._g.nodes(data=True):
                    if ndata.get("type") == "entity":
                        surface = ndata.get("surface", "").lower()
                        if q_lower in surface or surface in q_lower:
                            matched_ent_ids.append(nid)
                            out["matched_entities"].append(
                                {"text": ndata["surface"], "label": ndata.get("label", "")}
                            )
                            break

        if matched_ent_ids:
            # Entity → utterances, scored by number of matched entities
            utt_scores: Dict[str, int] = {}
            for eid in set(matched_ent_ids):
                for uid in self._g.predecessors(eid):
                    if self._g.nodes.get(uid, {}).get("type") == "utterance":
                        utt_scores[uid] = utt_scores.get(uid, 0) + 1

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
                    (p for p in self._g.predecessors(uid)
                     if self._g.nodes.get(p, {}).get("type") == "speaker"),
                    "Unknown",
                )
                seen_speakers.add(spk)
                out["utterances"].append({
                    "chroma_id": uid,
                    "text": ua.get("text", ""),
                    "speaker": spk,
                    "acoustic_emotion": ua.get("acoustic_emotion", ""),
                    "text_emotion": ua.get("text_emotion", ""),
                })

            out["related_speakers"] = list(seen_speakers)
            self._append_knows(out, seen_speakers)
            return out

        # --- 2. Speaker-centric fallback ---
        # No entity found in question, but we know which speakers to look at.
        # Traverse: speaker → utterances → entities.
        if not speaker_ids:
            return out

        out["fallback_mode"] = True
        spk_set = set(speaker_ids)
        seen_ents: set = set()
        seen_speakers_fb: set = set()

        for spk in spk_set:
            if not self._g.has_node(spk) or self._g.nodes[spk].get("type") != "speaker":
                continue
            for uid in self._g.successors(spk):
                if self._g.nodes.get(uid, {}).get("type") != "utterance":
                    continue
                ua = self._g.nodes[uid]
                seen_speakers_fb.add(spk)
                if len(out["utterances"]) < max_utterances:
                    out["utterances"].append({
                        "chroma_id": uid,
                        "text": ua.get("text", ""),
                        "speaker": spk,
                        "acoustic_emotion": ua.get("acoustic_emotion", ""),
                        "text_emotion": ua.get("text_emotion", ""),
                    })
                for ent_id in self._g.successors(uid):
                    if (self._g.nodes.get(ent_id, {}).get("type") == "entity"
                            and ent_id not in seen_ents):
                        seen_ents.add(ent_id)
                        nd = self._g.nodes[ent_id]
                        out["matched_entities"].append({
                            "text": nd.get("surface", ""),
                            "label": nd.get("label", ""),
                        })

        out["related_speakers"] = list(seen_speakers_fb)
        self._append_knows(out, seen_speakers_fb)
        return out

    def _append_knows(self, out: Dict[str, Any], seen_speakers: set) -> None:
        """Populate co_speakers from KNOWS edges among a set of speaker nodes."""
        for spk in seen_speakers:
            for nbr in self._g.successors(spk):
                edge = self._g[spk].get(nbr, {})
                if (self._g.nodes.get(nbr, {}).get("type") == "speaker"
                        and edge.get("relation") == "KNOWS"):
                    out["co_speakers"].append({
                        "speaker_a": spk,
                        "speaker_b": nbr,
                        "n_dialogues": edge.get("n_dialogues", 0),
                    })

    # ------------------------------------------------------------------
    # Dialogue and emotion-aggregate queries
    # ------------------------------------------------------------------

    def dialogue_participants(self, dialogue_id: Any) -> List[str]:
        """Return all speakers who participated in a given dialogue. O(1) hop."""
        if dialogue_id in (None, ""):
            return []
        key = str(dialogue_id)
        dlg_node = key if key.startswith("dlg::") else f"dlg::{key}"
        if not self._g.has_node(dlg_node):
            return []
        return [
            n for n in self._g.successors(dlg_node)
            if self._g.nodes.get(n, {}).get("type") == "speaker"
        ]

    def count_emotion(self, speaker: str, emotion: str, channel: str = "text") -> int:
        """Count utterances by `speaker` tagged with `emotion`.

        channel: 'text'   — uses the text_emotion attribute (default)
                 'acoustic' — uses acoustic_emotion
                 'either' — counts an utterance if either channel matches
        """
        if not speaker or not emotion:
            return 0
        if not self._g.has_node(speaker) or self._g.nodes[speaker].get("type") != "speaker":
            return 0
        emo = emotion.strip().lower()
        n = 0
        for uid in self._g.successors(speaker):
            ud = self._g.nodes.get(uid, {})
            if ud.get("type") != "utterance":
                continue
            te = str(ud.get("text_emotion", "")).lower()
            ae = str(ud.get("acoustic_emotion", "")).lower()
            if channel == "text" and te == emo:
                n += 1
            elif channel == "acoustic" and ae == emo:
                n += 1
            elif channel == "either" and (te == emo or ae == emo):
                n += 1
        return n

    def speaker_emotion_breakdown(self, speaker: str, channel: str = "text") -> Dict[str, int]:
        """Return {emotion: count} across all utterances of `speaker`."""
        out: Dict[str, int] = {}
        if not self._g.has_node(speaker) or self._g.nodes[speaker].get("type") != "speaker":
            return out
        attr = "text_emotion" if channel == "text" else "acoustic_emotion"
        for uid in self._g.successors(speaker):
            ud = self._g.nodes.get(uid, {})
            if ud.get("type") != "utterance":
                continue
            tag = str(ud.get(attr, "")).lower()
            if tag:
                out[tag] = out.get(tag, 0) + 1
        return out

    def rank_speakers_by_emotion(
        self,
        emotion: str,
        restrict_to: Optional[List[str]] = None,
        channel: str = "text",
        top_k: int = 10,
    ) -> List[Tuple[str, int]]:
        """Rank speakers by how often they expressed `emotion`.

        restrict_to: optional list of speaker ids to limit the ranking to.
                     When omitted, ranks across all enrolled speakers.
        Returns a list of (speaker, count) tuples sorted descending. Speakers
        with zero matches are filtered out.
        """
        if restrict_to:
            speakers = [s for s in restrict_to if self._g.has_node(s)]
        else:
            speakers = [
                n for n, d in self._g.nodes(data=True) if d.get("type") == "speaker"
            ]
        scored = [(s, self.count_emotion(s, emotion, channel)) for s in speakers]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if top_k else scored

    # ------------------------------------------------------------------
    # Deferred LLM-backed entity extraction
    # ------------------------------------------------------------------

    def set_llm_extractor(self, llm_extractor: Optional[LLMExtractor]) -> Dict[str, Any]:
        """Swap the LLM extractor at runtime (e.g. when the user provides a key
        via the UI). Returns a small status dict for the caller to surface."""
        self._llm_extractor = llm_extractor
        if llm_extractor and llm_extractor.available:
            print(f"[GraphService] LLM extractor set (model={llm_extractor.model}).")
            return {"enabled": True, "model": llm_extractor.model, "reason": None}
        reason = (
            llm_extractor.unavailable_reason if llm_extractor
            else "extractor cleared"
        )
        print(f"[GraphService] LLM extractor disabled ({reason}).")
        return {"enabled": False, "model": None, "reason": reason}

    def llm_status(self) -> Dict[str, Any]:
        """Report whether LLM extraction is currently active."""
        if self._llm_extractor and self._llm_extractor.available:
            return {"enabled": True, "model": self._llm_extractor.model, "reason": None}
        reason = (
            self._llm_extractor.unavailable_reason if self._llm_extractor
            else "no extractor configured"
        )
        return {"enabled": False, "model": None, "reason": reason}

    def extract_pending_entities(self) -> Dict[str, int]:
        """Run LLM entity extraction for any utterances marked entities_pending.

        Utterances are grouped by dialogue_id (one LLM call per dialogue) so the
        model sees full conversational context. Free-floating utterances with no
        dialogue_id are batched together as a single pseudo-dialogue.

        Returns a small stats dict: {dialogues_processed, utterances_extracted,
        entities_added, errors}.
        """
        stats = {
            "dialogues_processed": 0,
            "utterances_extracted": 0,
            "entities_added": 0,
            "errors": 0,
        }
        if not (self._llm_extractor and self._llm_extractor.available):
            return stats

        # Bucket pending utterances by dialogue_id
        buckets: Dict[str, List[str]] = {}
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("type") == "utterance" and attrs.get("entities_pending"):
                dlg = attrs.get("dialogue_id") or "_no_dialogue"
                buckets.setdefault(dlg, []).append(nid)

        if not buckets:
            return stats

        for dlg_id, utterance_ids in buckets.items():
            # Sort by utterance_id when present so the LLM sees turns in order
            utterance_ids.sort(
                key=lambda x: self._g.nodes[x].get("utterance_id", 0)
            )
            payload = []
            for uid in utterance_ids:
                ud = self._g.nodes[uid]
                speaker = next(
                    (p for p in self._g.predecessors(uid)
                     if self._g.nodes.get(p, {}).get("type") == "speaker"),
                    "Unknown",
                )
                payload.append({
                    "speaker": speaker.capitalize() if speaker else "Unknown",
                    "text": ud.get("text", ""),
                })

            try:
                results = self._llm_extractor.extract_dialogue_entities(payload)
            except Exception as e:
                print(f"[GraphService] extraction failed for dialogue {dlg_id}: {e}")
                stats["errors"] += 1
                continue

            for uid, entities in zip(utterance_ids, results):
                for ent_text, ent_label in entities:
                    ent_id = f"ent::{ent_label}::{ent_text.lower()}"
                    if not self._g.has_node(ent_id):
                        self._g.add_node(
                            ent_id,
                            type="entity",
                            label=ent_label,
                            surface=ent_text,
                        )
                        stats["entities_added"] += 1
                    if not self._g.has_edge(uid, ent_id):
                        self._g.add_edge(
                            uid, ent_id, relation="MENTIONS", label=ent_label
                        )
                # Clear the pending flag whether or not entities were found —
                # an empty result is a valid answer ("no entities here").
                self._g.nodes[uid].pop("entities_pending", None)
                stats["utterances_extracted"] += 1

            stats["dialogues_processed"] += 1

        print(
            f"[GraphService] LLM extraction: {stats['dialogues_processed']} dialogues, "
            f"{stats['utterances_extracted']} utterances, "
            f"{stats['entities_added']} new entity nodes, "
            f"{stats['errors']} errors."
        )
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Serialize the graph to JSON. No-op if no graph_path was given.

        Before writing, runs deferred LLM entity extraction so persisted graphs
        always have their MENTIONS edges populated.
        """
        if self._llm_extractor and self._llm_extractor.available:
            self.extract_pending_entities()

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
