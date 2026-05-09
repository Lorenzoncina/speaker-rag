"""
LLM-backed entity and relation extraction for SpeakerRAG.

Two-pass extraction per dialogue:
  1. Entity pass (Haiku): extracts named entities with type and a 1-2 sentence
     description grounded in the dialogue. Returns per-dialogue canonical list.
  2. Relation pass (Sonnet): typed relations between extracted entities.
     Skipped when fewer than two entities are present.

Output shapes are stable so GraphService can index without re-parsing.
Note: Anthropic prompt caching is silently disabled on Haiku below ~2048 tokens
and isn't worth the prompt bulk for these extraction prompts; we omit
cache_control entirely.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional


_ENTITY_SYSTEM_PROMPT = """You extract named entities from a conversation transcript.

Input format:
A dialogue is given as a numbered list of utterances. Each line shows the speaker and the utterance text.

Output format:
Return ONLY a JSON object: {"entities": [...]}
Each entity is: {"name": str, "type": str, "description": str, "mentioned_in": [int, ...]}

Field rules:
- name: canonical/most complete form (prefer "Joey Tribbiani" over "Joey" if both appear; "Phoebe" over "Pheebs").
- type: one of PERSON, LOCATION, ORGANIZATION, EVENT, OBJECT, CONCEPT, WORK.
- description: 1-2 sentence summary of what THIS DIALOGUE reveals about the entity. Do not use general/world knowledge — only what the dialogue says or strongly implies. If the dialogue reveals nothing about the entity beyond its name, write a minimal description like "Mentioned by <Speaker>." or "<Type> referenced in dialogue."
- mentioned_in: list of utterance numbers (the [N] indices) where the entity is named. Pronouns do not count.

Extraction rules:
- Extract proper nouns and meaningful named entities only (people, places, brands, events, named objects, titles, named concepts).
- The speaker of an utterance is NOT an entity OF THEIR OWN UTTERANCE. But when one speaker addresses or refers to another speaker BY NAME, that other speaker IS an entity (type PERSON) for the utterance where they were mentioned.
- Do NOT include pronouns, common nouns, generic descriptors, or filler.
- Do NOT include temporal expressions (dates, days, times, "yesterday", "Friday").
- Merge alternate forms of the same name into one entity (use canonical name, list all utterances where any form appears).
- If no entities, return {"entities": []}.
- Output strictly valid JSON. No prose, no markdown fences."""


_CONTRADICTION_SYSTEM_PROMPT = """You decide whether a NEW fact INVALIDATES an OLDER fact in a knowledge graph.

Two facts conflict (the new one invalidates the old one) when:
- They share the same SUBJECT and the same PREDICATE TYPE.
- The new fact's object/value makes the old fact's claim no longer true.
- Examples: "Phoebe DATES David" then later "Phoebe DATES Mike" (assuming exclusive dating).
  "Joey LIVES_IN New York" then "Joey LIVES_IN Los Angeles".
  "Rachel WORKS_AT Central Perk" then "Rachel WORKS_AT Bloomingdales".

Two facts do NOT conflict when:
- They could both be simultaneously true. Examples: "Joey IS_FRIENDS_WITH Chandler" and "Joey IS_FRIENDS_WITH Ross".
  "Phoebe HAS_BROTHER Frank" and "Phoebe HAS_BROTHER Frank Jr" (different people).
  "Rachel KNOWS New York" and "Rachel KNOWS Los Angeles".
- The relation type is naturally one-to-many (HAS_FRIEND, OWNS, MENTIONS, KNOWS, IS_RELATED_TO, etc.).
- The new fact is just a re-statement of the old one with different wording.

Output format:
Return ONLY a JSON object:
{
  "invalidates": true | false,
  "reason": "<one short sentence>"
}

When in doubt, default to false. False positives are cheap (we keep both facts), false negatives silently delete history.
Output strictly valid JSON. No prose, no markdown fences."""


_JUDGE_SYSTEM_PROMPT = """You decide whether a NEW entity is a duplicate of any EXISTING entity already in a knowledge graph.

Input format:
1. NEW entity: {name, type, description}.
2. EXISTING entities: top similarity candidates already in the graph, each with {id, name, type, description, score}. Score is cosine similarity (0–1) of the candidate to the new entity.
3. Optional DIALOGUE CONTEXT: utterances where the new entity was extracted, for disambiguation.

Output format:
Return ONLY a JSON object:
{
  "is_duplicate": true | false,
  "target_id": "<existing entity id>"           // only when is_duplicate is true
  "canonical_name": "<best/most complete name>" // only when is_duplicate is true
  "canonical_description": "<merged 1–3 sentence description>"  // only when is_duplicate is true
}

Decision rules:
- Two entities are duplicates ONLY when they refer to the SAME real-world entity.
- A shared name does NOT imply duplicate. Different people, places, or things can share a name.
- Use descriptions and the dialogue context to disambiguate.
- A more complete name (e.g. "Joey Tribbiani") may still be a duplicate of a shorter form ("Joey") IF the descriptions are mutually consistent and there is no contradicting evidence.
- When in doubt, default to is_duplicate=false. Wrong merges are very hard to undo.
- canonical_name: prefer the most complete form; otherwise pick the best canonical surface.
- canonical_description: combine unique factual content from both descriptions in 1–3 sentences. Do NOT introduce general/world knowledge that wasn't already in either description.
- Output strictly valid JSON. No prose, no markdown fences."""


_RELATION_SYSTEM_PROMPT = """You extract typed relationships between entities mentioned in a conversation.

Input format:
1. An ENTITY LIST: canonical entities already identified in the dialogue, with their types and descriptions. The list also includes the dialogue's speakers (type PERSON).
2. The DIALOGUE: the same numbered list of utterances those entities came from.

Output format:
Return ONLY a JSON object: {"relations": [...]}
Each relation: {"source": str, "target": str, "type": str, "fact": str, "source_utterances": [int, ...]}

Field rules:
- source, target: must be EXACT name strings copied from the input ENTITY LIST. Do not invent new entities.
- type: UPPER_SNAKE_CASE concise predicate (AGENT_OF, IS_FRIENDS_WITH, WORKS_AT, HAS_INJURY, DATES, MARRIED_TO, OWNS, LIVES_WITH, HAS_CHARACTER, MENTIONS_AS_FAVORITE, etc.). Reuse common predicates rather than inventing one-off phrasings.
- fact: ONE full natural-language sentence describing the relationship.
- source_utterances: list of utterance numbers (the [N] indices) that support this fact.

Extraction rules:
- Each relation must hold between two DISTINCT entities from the ENTITY LIST.
- Only extract facts that are stated or strongly implied IN THE DIALOGUE. Do NOT use general/world knowledge.
- Skip pure temporal annotations (when X happened) — those are handled separately.
- Skip relations where the only support is "X said Y" — that is a SAID edge, not a semantic fact.
- A given entity pair may have multiple distinct relations; emit one entry per distinct fact.
- If no relations can be supported by the dialogue, return {"relations": []}.
- Output strictly valid JSON. No prose, no markdown fences."""


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _JSON_FENCE_RE.sub("", raw.strip()).strip()


class LLMExtractor:
    """Two-pass entity + relation extractor using Anthropic's Messages API.

    The constructor reads ANTHROPIC_API_KEY from the env (or accepts one
    directly). If the key is missing or the SDK isn't installed, `available`
    returns False and callers should fall back to regex/spaCy.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        relation_model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.relation_model = relation_model
        self.max_tokens = max_tokens
        self._client = None
        self._unavailable_reason: Optional[str] = None

        key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        if not key:
            self._unavailable_reason = "ANTHROPIC_API_KEY not set"
            return
        try:
            from anthropic import Anthropic
        except ImportError:
            self._unavailable_reason = "anthropic SDK not installed (pip install anthropic)"
            return
        try:
            self._client = Anthropic(api_key=key)
        except Exception as e:
            self._unavailable_reason = f"failed to init Anthropic client: {e}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    # ------------------------------------------------------------------
    # Pass 1 — Entities
    # ------------------------------------------------------------------

    def extract_dialogue_entities(
        self,
        utterances: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract canonical entities for one dialogue.

        utterances: ordered list of dicts with keys `speaker` and `text`.
        Returns a list of entity dicts: {name, type, description, mentioned_in}.
        On any failure, returns [].
        """
        if not self.available or not utterances:
            return []

        user_prompt = self._format_dialogue(utterances)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_ENTITY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"[LLMExtractor] entity API call failed: {e}")
            return []

        self._log_usage(response, label="entity")

        try:
            raw = response.content[0].text
        except (AttributeError, IndexError):
            return []

        return self._parse_entities(raw, n_utterances=len(utterances))

    # ------------------------------------------------------------------
    # Pass 2 — Relations
    # ------------------------------------------------------------------

    def extract_relations(
        self,
        utterances: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract typed entity-entity relations for one dialogue (Sonnet).

        utterances: same shape as in entity extraction.
        entities:   the canonical entity list to constrain extraction. Should
                    already include speakers (caller's responsibility).

        Returns a list of relation dicts:
        {source, target, type, fact, source_utterances}.
        Returns [] when fewer than two entities are present, when the model
        is unavailable, or on any parsing failure.
        """
        if not self.available or len(entities) < 2 or not utterances:
            return []

        user_prompt = self._format_relation_prompt(utterances, entities)
        try:
            response = self._client.messages.create(
                model=self.relation_model,
                max_tokens=self.max_tokens,
                system=_RELATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"[LLMExtractor] relation API call failed: {e}")
            return []

        self._log_usage(response, label="relation")

        try:
            raw = response.content[0].text
        except (AttributeError, IndexError):
            return []

        return self._parse_relations(raw, valid_names={e["name"] for e in entities})

    # ------------------------------------------------------------------
    # Pass 3 — Entity duplicate judge
    # ------------------------------------------------------------------

    def judge_entity_duplicate(
        self,
        candidate: Dict[str, str],
        existing: List[Dict[str, Any]],
        dialogue_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Ask Sonnet whether `candidate` duplicates any of `existing`.

        Returns dict {is_duplicate, target_id?, canonical_name?, canonical_description?}
        or None on parse/transport failure.
        """
        if not self.available or not existing:
            return None

        user_prompt = self._format_judge_prompt(candidate, existing, dialogue_context)
        try:
            response = self._client.messages.create(
                model=self.relation_model,
                max_tokens=1024,
                system=_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"[LLMExtractor] judge API call failed: {e}")
            return None

        self._log_usage(response, label="judge")

        try:
            raw = response.content[0].text
        except (AttributeError, IndexError):
            return None

        return self._parse_judge_response(raw)

    # ------------------------------------------------------------------
    # Pass 4 — Fact contradiction judge (Phase 3 invalidation)
    # ------------------------------------------------------------------

    def judge_fact_contradiction(
        self,
        old_fact: Dict[str, Any],
        new_fact: Dict[str, Any],
        dialogue_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Ask Sonnet whether `new_fact` invalidates `old_fact`.

        Both facts should already share the same subject + predicate type at
        the call site (caller filters candidates first). Returns
        {invalidates: bool, reason: str} or None on failure.
        """
        if not self.available:
            return None

        user_prompt = (
            "OLD FACT:\n"
            f'  predicate: {old_fact.get("type", "")}\n'
            f'  fact: {old_fact.get("fact", "")}\n'
            f'  extracted_at: {old_fact.get("extracted_at", "")}\n'
            "\nNEW FACT:\n"
            f'  predicate: {new_fact.get("type", "")}\n'
            f'  fact: {new_fact.get("fact", "")}\n'
            f'  extracted_at: {new_fact.get("extracted_at", "")}'
        )
        if dialogue_context:
            user_prompt += f"\n\nDIALOGUE CONTEXT (where the new fact was extracted):\n{dialogue_context}"
        user_prompt += "\n\nReturn the JSON object now."

        try:
            response = self._client.messages.create(
                model=self.relation_model,
                max_tokens=512,
                system=_CONTRADICTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"[LLMExtractor] contradiction-judge API call failed: {e}")
            return None

        self._log_usage(response, label="contradiction")

        try:
            raw = response.content[0].text
        except (AttributeError, IndexError):
            return None

        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[LLMExtractor] contradiction JSON parse failed: {cleaned[:200]!r}")
            return None
        if not isinstance(data, dict):
            return None
        return {
            "invalidates": bool(data.get("invalidates", False)),
            "reason": str(data.get("reason", "")).strip(),
        }

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_dialogue(self, utterances: List[Dict[str, Any]]) -> str:
        lines = ["Dialogue:"]
        for i, u in enumerate(utterances, start=1):
            speaker = (u.get("speaker") or "Unknown").strip()
            text = (u.get("text") or "").strip().replace("\n", " ")
            lines.append(f"[{i}] {speaker}: {text}")
        lines.append("\nReturn the JSON object now.")
        return "\n".join(lines)

    def _format_relation_prompt(
        self,
        utterances: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> str:
        ent_lines = ["ENTITY LIST:"]
        for ent in entities:
            name = (ent.get("name") or "").strip()
            etype = (ent.get("type") or "").strip()
            desc = (ent.get("description") or "").strip()
            ent_lines.append(f"- {name} ({etype}): {desc}")
        dlg_lines = [self._format_dialogue(utterances)]
        return "\n".join(ent_lines) + "\n\n" + "\n".join(dlg_lines)

    def _format_judge_prompt(
        self,
        candidate: Dict[str, str],
        existing: List[Dict[str, Any]],
        dialogue_context: str,
    ) -> str:
        cand_block = (
            "NEW ENTITY:\n"
            f'- name: {candidate.get("name", "")}\n'
            f'- type: {candidate.get("type", "")}\n'
            f'- description: {candidate.get("description", "")}'
        )
        ex_lines = ["EXISTING ENTITIES (top candidates):"]
        for ex in existing:
            ex_lines.append(
                f'- id: {ex.get("id", "")}\n'
                f'  name: {ex.get("name", "")}\n'
                f'  type: {ex.get("type", "")}\n'
                f'  description: {ex.get("description", "")}\n'
                f'  cosine_score: {ex.get("score", 0.0)}'
            )
        ex_block = "\n".join(ex_lines)
        ctx_block = ""
        if dialogue_context:
            ctx_block = f"\nDIALOGUE CONTEXT:\n{dialogue_context}"
        return f"{cand_block}\n\n{ex_block}{ctx_block}\n\nReturn the JSON object now."

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_entities(self, raw: str, n_utterances: int) -> List[Dict[str, Any]]:
        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[LLMExtractor] entity JSON parse failed: {cleaned[:200]!r}")
            return []
        if not isinstance(data, dict):
            return []
        items = data.get("entities", [])
        if not isinstance(items, list):
            return []

        out: List[Dict[str, Any]] = []
        seen: set = set()
        for ent in items:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            etype = str(ent.get("type", "")).strip().upper() or "CONCEPT"
            desc = str(ent.get("description", "")).strip()
            mentioned_raw = ent.get("mentioned_in", [])
            mentioned: List[int] = []
            if isinstance(mentioned_raw, list):
                for v in mentioned_raw:
                    try:
                        idx = int(v)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= idx <= n_utterances and idx not in mentioned:
                        mentioned.append(idx)
            if not name:
                continue
            key = (name.lower(), etype)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "name": name,
                "type": etype,
                "description": desc,
                "mentioned_in": mentioned,
            })
        return out

    def _parse_relations(
        self, raw: str, valid_names: set
    ) -> List[Dict[str, Any]]:
        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[LLMExtractor] relation JSON parse failed: {cleaned[:200]!r}")
            return []
        if not isinstance(data, dict):
            return []
        items = data.get("relations", [])
        if not isinstance(items, list):
            return []

        # Case-insensitive lookup back to canonical name.
        lookup = {n.lower(): n for n in valid_names}
        out: List[Dict[str, Any]] = []
        for rel in items:
            if not isinstance(rel, dict):
                continue
            src_raw = str(rel.get("source", "")).strip()
            tgt_raw = str(rel.get("target", "")).strip()
            src = lookup.get(src_raw.lower())
            tgt = lookup.get(tgt_raw.lower())
            if not src or not tgt or src == tgt:
                continue
            rtype = str(rel.get("type", "")).strip().upper().replace(" ", "_") or "RELATED_TO"
            fact = str(rel.get("fact", "")).strip()
            su_raw = rel.get("source_utterances", [])
            su: List[int] = []
            if isinstance(su_raw, list):
                for v in su_raw:
                    try:
                        idx = int(v)
                    except (TypeError, ValueError):
                        continue
                    if idx not in su:
                        su.append(idx)
            out.append({
                "source": src,
                "target": tgt,
                "type": rtype,
                "fact": fact,
                "source_utterances": su,
            })
        return out

    def _parse_judge_response(self, raw: str) -> Optional[Dict[str, Any]]:
        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[LLMExtractor] judge JSON parse failed: {cleaned[:200]!r}")
            return None
        if not isinstance(data, dict):
            return None
        is_dup = bool(data.get("is_duplicate", False))
        if not is_dup:
            return {"is_duplicate": False}
        target_id = str(data.get("target_id", "")).strip()
        if not target_id:
            return {"is_duplicate": False}
        return {
            "is_duplicate": True,
            "target_id": target_id,
            "canonical_name": str(data.get("canonical_name", "")).strip(),
            "canonical_description": str(data.get("canonical_description", "")).strip(),
        }

    # ------------------------------------------------------------------
    # Internal: usage logging
    # ------------------------------------------------------------------

    def _log_usage(self, response: Any, label: str) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        print(
            f"[LLMExtractor:{label}] usage: input={getattr(u, 'input_tokens', 0)} "
            f"output={getattr(u, 'output_tokens', 0)}"
        )
