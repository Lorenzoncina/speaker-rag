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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .llm_extractor import LLMExtractor
from .entity_resolver import EntityResolver, ResolutionDecision


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

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
        entity_resolver: Optional[EntityResolver] = None,
    ):
        self._graph_path = Path(graph_path) if graph_path else None
        self._g: nx.DiGraph = nx.DiGraph()
        self._llm_extractor = llm_extractor
        self._entity_resolver = entity_resolver
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
        if self._entity_resolver and self._entity_resolver.available:
            print("[GraphService] Entity resolver enabled (cosine + LLM judge).")
        else:
            print("[GraphService] Entity resolver disabled.")

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
        mark_pending_extraction: bool = True,
    ) -> None:
        """Add a speaker, utterance, and its extracted entities to the graph.

        Does NOT auto-persist — call persist() explicitly after a batch.

        mark_pending_extraction: when False, the utterance is added with all
        structural edges (SAID/CONTAINS/HAS_PARTICIPANT/KNOWS) but is NOT
        queued for LLM entity/relation extraction. Used for utterances we
        want stored for retrieval but not parsed into KG facts — e.g.,
        assistant messages in a chat enrollment, where the content is
        usually a paraphrase of the FACT SHEET and would just re-extract
        already-known entities.
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
        # When mark_pending_extraction=False we skip extraction entirely.
        if not mark_pending_extraction:
            pass  # caller opted out of KG extraction for this utterance
        elif self._llm_extractor and self._llm_extractor.available:
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

    def _display_name(self, node_id: str) -> str:
        """Human-readable label for a node id (speaker → capitalized id, entity → surface)."""
        nd = self._g.nodes.get(node_id, {})
        ntype = nd.get("type", "")
        if ntype == "speaker":
            return str(node_id).capitalize()
        if ntype == "entity":
            return nd.get("surface", node_id)
        return str(node_id)

    def _collect_typed_facts(
        self,
        node_ids: List[str],
        max_facts: int = 15,
        include_invalidated: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect typed-fact edges incident to any node in `node_ids`.

        Walks both directions (the node may be subject OR object of a fact).
        Filters out invalidated facts (valid_to set) unless include_invalidated.
        """
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for nid in node_ids:
            if not self._g.has_node(nid):
                continue
            # Outgoing — node is subject
            for _, tgt, attrs in self._g.out_edges(nid, data=True):
                for f in attrs.get("facts", []) or []:
                    if not include_invalidated and f.get("valid_to"):
                        continue
                    key = (nid, tgt, f.get("type", ""), f.get("fact", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "subject": self._display_name(nid),
                        "predicate": f.get("type", ""),
                        "object": self._display_name(tgt),
                        "fact": f.get("fact", ""),
                        "source_utterances": f.get("source_utterances", []),
                        "valid_from": f.get("valid_from"),
                        "valid_to": f.get("valid_to"),
                    })
                    if len(out) >= max_facts:
                        return out
            # Incoming — node is object
            for src, _, attrs in self._g.in_edges(nid, data=True):
                for f in attrs.get("facts", []) or []:
                    if not include_invalidated and f.get("valid_to"):
                        continue
                    key = (src, nid, f.get("type", ""), f.get("fact", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "subject": self._display_name(src),
                        "predicate": f.get("type", ""),
                        "object": self._display_name(nid),
                        "fact": f.get("fact", ""),
                        "source_utterances": f.get("source_utterances", []),
                        "valid_from": f.get("valid_from"),
                        "valid_to": f.get("valid_to"),
                    })
                    if len(out) >= max_facts:
                        return out
        return out

    def query(
        self,
        question: str,
        speaker_ids: Optional[List[str]] = None,
        max_utterances: int = 5,
        max_facts: int = 10,
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
          typed_facts       — typed-relation edges incident to matched entities/speakers
          fallback_mode     — True when speaker-centric fallback was used
        """
        out: Dict[str, Any] = {
            "matched_entities": [],
            "related_speakers": [],
            "utterances": [],
            "co_speakers": [],
            "typed_facts": [],
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

            # Surface typed-fact edges incident to the matched entities AND the
            # speakers we found. This is the Phase 4 hook: WORKS_AT, HAS_AGENT,
            # IS_FRIENDS_WITH, etc. become first-class context for the LLM.
            fact_seed = list(set(matched_ent_ids) | seen_speakers)
            out["typed_facts"] = self._collect_typed_facts(fact_seed, max_facts=max_facts)
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

        # Typed facts on the speakers themselves (e.g. Joey HAS_AGENT Estelle
        # — useful even when the question has no entity names).
        out["typed_facts"] = self._collect_typed_facts(
            list(seen_speakers_fb), max_facts=max_facts
        )
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

    def _upsert_entity(
        self,
        name: str,
        etype: str,
        description: str,
        stats: Dict[str, int],
        dialogue_context: str = "",
    ) -> str:
        """Resolve a candidate entity, then create-or-merge into the graph.

        Returns the canonical node id (either newly created or the target of a
        merge decision). When the resolver judges the candidate to be a duplicate
        of an existing entity, that entity's description, alias list, and Chroma
        embedding are updated; no new node is created.
        """
        # 1. Try resolution (cosine + optional LLM judge).
        decision = ResolutionDecision(is_duplicate=False)
        if self._entity_resolver and self._entity_resolver.available:
            decision = self._entity_resolver.resolve(
                name=name,
                etype=etype,
                description=description,
                dialogue_context=dialogue_context,
            )

        # 2. Duplicate path: merge candidate metadata into the existing target.
        if (
            decision.is_duplicate
            and decision.target_node_id
            and self._g.has_node(decision.target_node_id)
        ):
            target_id = decision.target_node_id
            nd = self._g.nodes[target_id]
            new_surface = decision.canonical_name or nd.get("surface", name)
            new_desc = decision.canonical_description or nd.get("description", description)
            old_surface = nd.get("surface", "")
            nd["surface"] = new_surface
            nd["description"] = new_desc
            nd["last_seen"] = _now_iso()
            nd["n_mentions"] = nd.get("n_mentions", 0) + 1
            aliases = set(nd.get("aliases", []))
            for alt in (name, old_surface):
                if alt and alt != new_surface:
                    aliases.add(alt)
            nd["aliases"] = sorted(aliases)
            if self._entity_resolver:
                self._entity_resolver.update(
                    target_id, new_surface, nd.get("label", etype), new_desc
                )
            stats["entities_merged"] = stats.get("entities_merged", 0) + 1
            method_key = f"merges_{decision.method}" if decision.method else "merges_other"
            stats[method_key] = stats.get(method_key, 0) + 1
            print(
                f"[Resolver] merge ({decision.method}, score={decision.top_score:.3f}): "
                f"'{name}' → '{new_surface}' [{etype}]"
            )
            return target_id

        # 3. New entity path. Use a stable conventional id; if a node already
        #    exists at that id (very near-duplicate the resolver missed) we
        #    fall back to a description-merge — same shape as before.
        ent_id = f"ent::{etype}::{name.lower()}"
        if not self._g.has_node(ent_id):
            self._g.add_node(
                ent_id,
                type="entity",
                label=etype,
                surface=name,
                description=description or "",
                n_mentions=1,
                first_seen=_now_iso(),
                last_seen=_now_iso(),
                aliases=[],
            )
            stats["entities_added"] = stats.get("entities_added", 0) + 1
            if self._entity_resolver:
                self._entity_resolver.add(ent_id, name, etype, description)
            return ent_id

        # Same conventional id exists. Cosine missed the match — likely because
        # descriptions diverged across dialogues. Escalate to the LLM judge before
        # silently merging or splitting.
        existing_node = self._g.nodes[ent_id]

        judge_result = None
        if self._llm_extractor and self._llm_extractor.available:
            judge_candidates = [{
                "id": ent_id,
                "name": existing_node.get("surface", ""),
                "type": existing_node.get("label", etype),
                "description": existing_node.get("description", ""),
                "score": 1.0,  # synthetic — name/type match exactly
            }]
            try:
                judge_result = self._llm_extractor.judge_entity_duplicate(
                    candidate={"name": name, "type": etype, "description": description},
                    existing=judge_candidates,
                    dialogue_context=dialogue_context,
                )
            except Exception as e:
                print(f"[GraphService] id-collision judge failed: {e}")

        if judge_result and judge_result.get("is_duplicate"):
            # Judge confirmed: merge into existing node.
            new_surface = judge_result.get("canonical_name") or existing_node.get("surface", name)
            new_desc = judge_result.get("canonical_description") or existing_node.get("description", description)
            old_surface = existing_node.get("surface", "")
            existing_node["surface"] = new_surface
            existing_node["description"] = new_desc
            existing_node["last_seen"] = _now_iso()
            existing_node["n_mentions"] = existing_node.get("n_mentions", 0) + 1
            aliases = set(existing_node.get("aliases", []))
            for alt in (name, old_surface):
                if alt and alt != new_surface:
                    aliases.add(alt)
            existing_node["aliases"] = sorted(aliases)
            if self._entity_resolver:
                self._entity_resolver.update(
                    ent_id, new_surface, existing_node.get("label", etype), new_desc
                )
            stats["entities_merged"] = stats.get("entities_merged", 0) + 1
            stats["merges_id_collision_judge"] = stats.get("merges_id_collision_judge", 0) + 1
            print(
                f"[Resolver] id-collision merge (judge): "
                f"'{name}' → '{new_surface}' [{etype}]"
            )
            return ent_id

        if judge_result is not None:
            # Judge said different — disambiguate with a numeric suffix so both
            # entities live as separate nodes (e.g. two unrelated 'Janice's).
            suffix = 2
            unique_id = f"{ent_id}#{suffix}"
            while self._g.has_node(unique_id):
                suffix += 1
                unique_id = f"{ent_id}#{suffix}"
            self._g.add_node(
                unique_id,
                type="entity",
                label=etype,
                surface=name,
                description=description or "",
                n_mentions=1,
                first_seen=_now_iso(),
                last_seen=_now_iso(),
                aliases=[],
            )
            stats["entities_added"] = stats.get("entities_added", 0) + 1
            stats["disambiguated_id_collisions"] = stats.get("disambiguated_id_collisions", 0) + 1
            if self._entity_resolver:
                self._entity_resolver.add(unique_id, name, etype, description)
            print(
                f"[Resolver] id-collision split (judge said different): "
                f"'{name}' → new node '{unique_id}' [{etype}]"
            )
            return unique_id

        # Judge unavailable — fall back to the legacy description-merge so we
        # don't lose data, but log so the situation is visible.
        nd = self._g.nodes[ent_id]
        nd["last_seen"] = _now_iso()
        nd["n_mentions"] = nd.get("n_mentions", 0) + 1
        existing = (nd.get("description") or "").strip()
        new_desc = (description or "").strip()
        if new_desc and new_desc not in existing:
            nd["description"] = (
                f"{existing} | {new_desc}" if existing else new_desc
            )
            stats["entities_updated"] = stats.get("entities_updated", 0) + 1
            if self._entity_resolver:
                self._entity_resolver.update(
                    ent_id, nd.get("surface", name), nd.get("label", etype), nd["description"]
                )
        return ent_id

    @staticmethod
    def _merge_edge_attrs(target_attrs: Dict[str, Any], source_attrs: Dict[str, Any]) -> None:
        """In-place merge of edge attribute dicts when rerouting edges.

        - `facts` lists are concatenated (preserves all typed-fact entries).
        - `n_dialogues` (KNOWS) and `count` (MENTIONS) are summed.
        - Other attrs are kept from target if present, else copied from source.
        """
        if "facts" in source_attrs and source_attrs["facts"]:
            target_attrs.setdefault("facts", []).extend(source_attrs["facts"])
        if "n_dialogues" in source_attrs:
            target_attrs["n_dialogues"] = (
                target_attrs.get("n_dialogues", 0) + source_attrs["n_dialogues"]
            )
        if "count" in source_attrs:
            target_attrs["count"] = (
                target_attrs.get("count", 0) + source_attrs["count"]
            )
        for k, v in source_attrs.items():
            if k in ("facts", "n_dialogues", "count"):
                continue
            target_attrs.setdefault(k, v)

    def _merge_node(
        self,
        duplicate: str,
        target: str,
        canonical_name: str,
        canonical_description: str,
        stats: Dict[str, int],
    ) -> None:
        """Reroute every edge from `duplicate` onto `target`, then drop `duplicate`.

        Used by the retrospective dedup pass — the eager path applies merges
        directly in `_upsert_entity` without ever creating the duplicate node.
        """
        if duplicate == target or not self._g.has_node(duplicate):
            return

        dup_node = self._g.nodes[duplicate]
        dup_surface = dup_node.get("surface", "")
        dup_n_mentions = dup_node.get("n_mentions", 0)
        dup_aliases = list(dup_node.get("aliases", []))

        # Reroute in-edges
        for u, _, attrs in list(self._g.in_edges(duplicate, data=True)):
            if u == target:
                continue
            if self._g.has_edge(u, target):
                self._merge_edge_attrs(self._g[u][target], attrs)
            else:
                self._g.add_edge(u, target, **attrs)

        # Reroute out-edges
        for _, v, attrs in list(self._g.out_edges(duplicate, data=True)):
            if v == target:
                continue
            if self._g.has_edge(target, v):
                self._merge_edge_attrs(self._g[target][v], attrs)
            else:
                self._g.add_edge(target, v, **attrs)

        # Update target metadata
        nd = self._g.nodes[target]
        new_surface = canonical_name or nd.get("surface", "")
        nd["surface"] = new_surface
        nd["description"] = canonical_description or nd.get("description", "")
        nd["last_seen"] = _now_iso()
        nd["n_mentions"] = nd.get("n_mentions", 0) + dup_n_mentions
        aliases = set(nd.get("aliases", []))
        if dup_surface and dup_surface != new_surface:
            aliases.add(dup_surface)
        for a in dup_aliases:
            if a and a != new_surface:
                aliases.add(a)
        nd["aliases"] = sorted(aliases)

        # Drop the duplicate from the graph and Chroma
        self._g.remove_node(duplicate)
        if self._entity_resolver:
            self._entity_resolver.delete(duplicate)
            self._entity_resolver.update(
                target, new_surface, nd.get("label", "UNKNOWN"), nd["description"]
            )

        stats["entities_merged"] = stats.get("entities_merged", 0) + 1

    def dedupe_all_entities(self) -> Dict[str, int]:
        """Retrospectively dedupe entity nodes already in the graph.

        Walks entities from least- to most-mentioned and asks the resolver
        whether each could merge into another existing entity. When the
        resolver says yes, the lower-mention entity is merged into the higher.
        This is a one-shot maintenance pass — typical use is after a bulk
        ingest pre-dating the resolver, or after schema changes.
        """
        stats: Dict[str, int] = {
            "candidates_examined": 0,
            "entities_merged": 0,
            "merges_cosine_high": 0,
            "merges_llm_judge": 0,
        }
        if not (self._entity_resolver and self._entity_resolver.available):
            print("[GraphService] dedupe_all_entities: resolver unavailable, no-op.")
            return stats

        entities = [
            (nid, attrs)
            for nid, attrs in self._g.nodes(data=True)
            if attrs.get("type") == "entity"
        ]
        # Backfill Chroma from the graph in case entities predate the resolver
        # (e.g. were created by an earlier ingest before this collection existed).
        # `add` is idempotent (upsert), so this is safe even when the collection
        # is already up to date.
        for nid, attrs in entities:
            self._entity_resolver.add(
                nid,
                attrs.get("surface", nid),
                attrs.get("label", "UNKNOWN"),
                attrs.get("description", ""),
            )

        # Process least-mentioned first so they get merged INTO higher-mention nodes.
        entities.sort(key=lambda x: x[1].get("n_mentions", 0))

        for nid, attrs in entities:
            if not self._g.has_node(nid):
                # already merged away in an earlier iteration
                continue
            stats["candidates_examined"] += 1
            decision = self._entity_resolver.resolve(
                name=attrs.get("surface", ""),
                etype=attrs.get("label", "UNKNOWN"),
                description=attrs.get("description", ""),
                exclude_ids=[nid],
            )
            if not (
                decision.is_duplicate
                and decision.target_node_id
                and self._g.has_node(decision.target_node_id)
            ):
                continue
            target_surface = self._g.nodes[decision.target_node_id].get("surface", "")
            self._merge_node(
                duplicate=nid,
                target=decision.target_node_id,
                canonical_name=decision.canonical_name or attrs.get("surface", ""),
                canonical_description=decision.canonical_description or attrs.get("description", ""),
                stats=stats,
            )
            method_key = f"merges_{decision.method}" if decision.method else "merges_other"
            stats[method_key] = stats.get(method_key, 0) + 1
            print(
                f"[Resolver:retro] merge ({decision.method}, score={decision.top_score:.3f}): "
                f"'{attrs.get('surface', nid)}' → '{target_surface}' "
                f"[{attrs.get('label', 'UNKNOWN')}]"
            )

        print(
            f"[GraphService] dedupe_all_entities: examined {stats['candidates_examined']}, "
            f"merged {stats['entities_merged']}."
        )
        return stats

    def _resolve_to_node(
        self,
        name: str,
        speaker_ids: set,
    ) -> Optional[str]:
        """Map an LLM-returned entity name to either a speaker node id or an
        entity node id already present in the graph. Returns None if nothing
        matches — caller should ignore that relation."""
        if not name:
            return None
        lc = name.lower().strip()
        for spk in speaker_ids:
            if spk.lower() == lc:
                return spk
        # Otherwise look for an existing entity node by surface match
        for nid, nd in self._g.nodes(data=True):
            if nd.get("type") != "entity":
                continue
            if (nd.get("surface") or "").lower() == lc:
                return nid
        return None

    def _add_fact(
        self,
        src: str,
        tgt: str,
        fact: Dict[str, Any],
        dialogue_context: str = "",
        stats: Optional[Dict[str, int]] = None,
    ) -> None:
        """Append a typed-fact dict to the edge between src and tgt.

        Creates the edge with relation='FACT' if it doesn't exist. When the
        edge already exists (e.g. a Speaker--KNOWS--Speaker structural edge),
        the existing relation field is preserved and the fact is appended to
        the edge's `facts` list.

        Phase 3 — bi-temporal invalidation: before adding, scan all OUT-edges
        from `src` for live (non-invalidated) facts with the same predicate
        type. Ask the LLM judge whether the new fact contradicts each one;
        when it does, set the old fact's `valid_to` to the new fact's
        `extracted_at`. The old fact is preserved as history, not deleted.
        """
        if src == tgt:
            return

        # Invalidation pass — only when the LLM judge is wired up.
        if self._llm_extractor and self._llm_extractor.available:
            new_type = (fact.get("type") or "").upper()
            new_fact_text = (fact.get("fact") or "").strip()
            for _, other_tgt, attrs in list(self._g.out_edges(src, data=True)):
                for old in attrs.get("facts", []) or []:
                    if old.get("valid_to"):
                        continue  # already invalidated
                    if (old.get("type") or "").upper() != new_type:
                        continue
                    # Identical fact text — treat as same fact, no contradiction.
                    if (old.get("fact") or "").strip() == new_fact_text and other_tgt == tgt:
                        continue
                    try:
                        verdict = self._llm_extractor.judge_fact_contradiction(
                            old_fact=old,
                            new_fact=fact,
                            dialogue_context=dialogue_context,
                        )
                    except Exception as e:
                        print(f"[GraphService] contradiction judge failed: {e}")
                        verdict = None
                    if verdict and verdict.get("invalidates"):
                        old["valid_to"] = fact.get("extracted_at") or _now_iso()
                        old["invalidated_by"] = new_fact_text
                        old["invalidation_reason"] = verdict.get("reason", "")
                        if stats is not None:
                            stats["facts_invalidated"] = stats.get("facts_invalidated", 0) + 1
                        print(
                            f"[Resolver] invalidated fact: '{self._display_name(src)} "
                            f"{old.get('type', '')} {self._display_name(other_tgt)}' "
                            f"superseded by '{new_fact_text}' "
                            f"(reason: {verdict.get('reason', '')})"
                        )

        if self._g.has_edge(src, tgt):
            ed = self._g[src][tgt]
            ed.setdefault("facts", []).append(fact)
        else:
            self._g.add_edge(src, tgt, relation="FACT", facts=[fact])

    def _extract_entities_window(
        self,
        utterance_ids: List[str],
        dlg_id: str,
        all_speakers: set,
        stats: Dict[str, int],
    ) -> None:
        """Run the two-pass extraction (entities then relations) on a single
        window of utterance ids belonging to one dialogue.

        Mutates the graph in place and updates `stats` counters. Clears the
        `entities_pending` flag on every utterance in the window once Pass 1
        has run (or failed) so the same window is never reprocessed.
        """
        # Build window payload, recording 1-indexed -> chroma_id
        payload: List[Dict[str, Any]] = []
        idx_to_chroma: Dict[int, str] = {}
        speaker_set: set = set()
        for i, uid in enumerate(utterance_ids, start=1):
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
            idx_to_chroma[i] = uid
            if speaker and speaker != "Unknown":
                speaker_set.add(speaker)

        # --- Pass 1: entities ---
        try:
            entities = self._llm_extractor.extract_dialogue_entities(payload)
        except Exception as e:
            print(f"[GraphService] entity extraction failed for dialogue {dlg_id}: {e}")
            stats["errors"] += 1
            # Even on failure, clear the pending flag so we don't loop forever
            for uid in utterance_ids:
                self._g.nodes[uid].pop("entities_pending", None)
                stats["utterances_extracted"] += 1
            return

        # name -> node_id mapping for the relation pass.
        # Speakers take priority over entity nodes when names collide
        # (e.g. extracted entity "Joey" should map to speaker node `joey`).
        name_to_node: Dict[str, str] = {}
        for spk in all_speakers:
            name_to_node[spk.lower()] = spk

        # Dialogue-context string for the LLM judge (so it can disambiguate
        # ambiguous cosine matches with grounded text). Cap at ~1500 chars
        # to keep judge input small.
        dialogue_context = "\n".join(
            f"[{i}] {u.get('speaker', '')}: {u.get('text', '')}"
            for i, u in enumerate(payload, start=1)
        )[:1500]

        for ent in entities:
            ent_name = ent["name"]
            ent_type = ent["type"]
            # If this entity name matches a known speaker (any dialogue),
            # route mentions to the speaker node rather than creating a
            # duplicate ent::PERSON::<name> entity.
            speaker_match = next(
                (s for s in all_speakers if s.lower() == ent_name.lower()),
                None,
            )
            if speaker_match and ent_type == "PERSON":
                target_node = speaker_match
            else:
                target_node = self._upsert_entity(
                    name=ent_name,
                    etype=ent_type,
                    description=ent["description"],
                    stats=stats,
                    dialogue_context=dialogue_context,
                )
            # MENTIONS edges
            for utt_idx in ent["mentioned_in"]:
                chroma_id = idx_to_chroma.get(utt_idx)
                if chroma_id and not self._g.has_edge(chroma_id, target_node):
                    self._g.add_edge(
                        chroma_id, target_node, relation="MENTIONS", label=ent_type
                    )
            name_to_node[ent_name.lower()] = target_node

        # Clear entities_pending on every utterance now that pass 1 ran
        for uid in utterance_ids:
            self._g.nodes[uid].pop("entities_pending", None)
            stats["utterances_extracted"] += 1

        # --- Pass 2: relations (Sonnet) ---
        relation_entity_list: List[Dict[str, Any]] = []
        seen_lc: set = set()
        for ent in entities:
            key = ent["name"].lower()
            if key in seen_lc:
                continue
            seen_lc.add(key)
            relation_entity_list.append({
                "name": ent["name"],
                "type": ent["type"],
                "description": ent["description"],
            })
        for spk in speaker_set:
            if spk.lower() in seen_lc:
                continue
            seen_lc.add(spk.lower())
            relation_entity_list.append({
                "name": spk.capitalize(),
                "type": "PERSON",
                "description": "Speaker in this dialogue.",
            })

        if len(relation_entity_list) >= 2:
            try:
                relations = self._llm_extractor.extract_relations(
                    payload, relation_entity_list
                )
            except Exception as e:
                print(f"[GraphService] relation extraction failed for dialogue {dlg_id}: {e}")
                stats["errors"] += 1
                relations = []

            for rel in relations:
                src_node = name_to_node.get(rel["source"].lower())
                tgt_node = name_to_node.get(rel["target"].lower())
                if not src_node or not tgt_node or src_node == tgt_node:
                    continue
                src_chromas = [
                    idx_to_chroma[i] for i in rel["source_utterances"]
                    if i in idx_to_chroma
                ]
                extracted_at = _now_iso()
                fact = {
                    "type": rel["type"],
                    "fact": rel["fact"],
                    "source_utterances": src_chromas,
                    "dialogue_id": str(dlg_id) if dlg_id != "_no_dialogue" else "",
                    "extracted_at": extracted_at,
                    "valid_from": extracted_at,
                    "valid_to": None,
                }
                self._add_fact(
                    src_node,
                    tgt_node,
                    fact,
                    dialogue_context=dialogue_context,
                    stats=stats,
                )
                stats["relations_added"] += 1

    def extract_pending_entities(self) -> Dict[str, int]:
        """Two-pass LLM extraction over utterances marked entities_pending.

        Pass 1 (Haiku, per dialogue): canonical entities with descriptions and
        which utterances they appear in. Adds Entity nodes and Utterance--MENTIONS-->Entity edges.
        Pass 2 (Sonnet, per dialogue, only when ≥2 candidate entities including
        speakers): typed entity-entity relations. Adds typed-fact edges between
        the two entity/speaker nodes.

        Returns a stats dict: {dialogues_processed, utterances_extracted,
        entities_added, entities_updated, relations_added, errors}.
        """
        stats = {
            "dialogues_processed": 0,
            "utterances_extracted": 0,
            "entities_added": 0,
            "entities_updated": 0,
            "entities_merged": 0,
            "merges_cosine_high": 0,
            "merges_llm_judge": 0,
            "merges_id_collision_judge": 0,
            "disambiguated_id_collisions": 0,
            "relations_added": 0,
            "facts_invalidated": 0,
            "errors": 0,
        }
        if not (self._llm_extractor and self._llm_extractor.available):
            return stats

        # All speakers in the graph — used to route LLM-extracted PERSON entities
        # to the existing speaker node rather than creating a duplicate
        # ent::PERSON::<name> node. Computed once per pass so cross-dialogue
        # speaker references resolve correctly.
        all_speakers: set = {
            nid for nid, attrs in self._g.nodes(data=True)
            if attrs.get("type") == "speaker"
        }

        # Bucket pending utterances by dialogue_id
        buckets: Dict[str, List[str]] = {}
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("type") == "utterance" and attrs.get("entities_pending"):
                dlg = attrs.get("dialogue_id") or "_no_dialogue"
                buckets.setdefault(dlg, []).append(nid)

        if not buckets:
            return stats

        # Chunking guard for very long dialogues. A meeting-length dialogue
        # (e.g. 224 utterances under one dialogue_id) overflows the local
        # model's 32K context when sent as one prompt, producing truncated
        # JSON and zero entities. When a dialogue exceeds CHUNK_SIZE we run
        # the two-pass extraction over sliding windows and let the
        # EntityResolver merge duplicates across windows. Shorter dialogues
        # (MELD-style) are sent as a single call exactly as before.
        # The dialogue_id is preserved across all windows — chunking is a
        # backend batching detail, not a re-definition of "conversation."
        CHUNK_SIZE = 40

        for dlg_id, utterance_ids in buckets.items():
            # Sort by utterance_id when present so the LLM sees turns in order
            utterance_ids.sort(
                key=lambda x: self._g.nodes[x].get("utterance_id", 0)
            )

            if len(utterance_ids) <= CHUNK_SIZE:
                windows = [utterance_ids]
            else:
                windows = [
                    utterance_ids[i : i + CHUNK_SIZE]
                    for i in range(0, len(utterance_ids), CHUNK_SIZE)
                ]
                print(
                    f"[GraphService] Dialogue {dlg_id}: {len(utterance_ids)} "
                    f"utterances → splitting into {len(windows)} windows of "
                    f"≤{CHUNK_SIZE} for LLM extraction."
                )

            for window in windows:
                self._extract_entities_window(
                    utterance_ids=window,
                    dlg_id=dlg_id,
                    all_speakers=all_speakers,
                    stats=stats,
                )

            stats["dialogues_processed"] += 1

        print(
            f"[GraphService] LLM extraction: {stats['dialogues_processed']} dialogues, "
            f"{stats['utterances_extracted']} utterances, "
            f"{stats['entities_added']} new entities, "
            f"{stats['entities_merged']} merged "
            f"(cosine={stats.get('merges_cosine_high', 0)}, "
            f"llm={stats.get('merges_llm_judge', 0)}, "
            f"id_collision={stats.get('merges_id_collision_judge', 0)}), "
            f"{stats.get('disambiguated_id_collisions', 0)} id-collisions split, "
            f"{stats['entities_updated']} desc-only updates, "
            f"{stats['relations_added']} typed relations, "
            f"{stats.get('facts_invalidated', 0)} facts invalidated, "
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
                    "description": nd.get("description", ""),
                    "total_mentions": entity_total[ent],
                }
            )

        displayed_nodes: set = set(speakers) | top_entity_set

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

        # Typed FACT edges — one viz-edge per fact so distinct relation types
        # between the same pair render separately. Only include edges whose
        # endpoints are already in the displayed node set.
        for src, tgt, ed in g.edges(data=True):
            if src not in displayed_nodes or tgt not in displayed_nodes:
                continue
            for fact in ed.get("facts", []) or []:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relation": fact.get("type", "FACT"),
                    "fact": fact.get("fact", ""),
                    "source_utterances": fact.get("source_utterances", []),
                    "is_typed_fact": True,
                })

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
