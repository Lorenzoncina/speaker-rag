"""
Local LLM-backed entity and relation extraction for SpeakerRAG.

Drop-in replacement for `LLMExtractor` (Anthropic) using a quantized
Qwen/Qwen3-14B running in-process. Same four public methods so
`GraphService.set_llm_extractor()` can swap one for the other without any
graph-service changes:

  - extract_dialogue_entities(utterances)        — Pass 1 entity extraction
  - extract_relations(utterances, entities)      — Pass 2 typed relations
  - judge_entity_duplicate(candidate, existing)  — entity dedup arbiter
  - judge_fact_contradiction(old, new)           — bi-temporal invalidation arbiter

Plus a public `generate(prompt, ...)` helper used by the API's `/llm/generate`
endpoint so the UI can route fact-sheet answering through the same loaded
model when the user is in Local mode.

Ontology design (matches the user's preference: closed entity types, open
predicates):
  - Entities: PERSON / LOCATION / ORGANIZATION / EVENT / OBJECT / CONCEPT / WORK
    Anything else returned by the model is downcast to CONCEPT (rather than
    dropped) to avoid losing salient extractions on a smaller model.
  - Relation predicates: free UPPER_SNAKE_CASE — same shape as the Sonnet path.

Inference backend: vLLM with prefix caching + continuous batching. Since our
system prompts are byte-identical across every call, prefix caching means
vLLM skips re-prefilling them after the first request — the bulk of input
tokens cost only one prefill per process lifetime.

Quantization: 4-bit AWQ by default (Qwen3-14B-AWQ ≈ 8 GB VRAM, dedicated
fast CUDA kernels). Fits alongside the existing audio stack on a 46 GB GPU
with plenty of headroom for the KV cache. Pass `quantization=None` to load
the unquantized model, or `"gptq"` to use a GPTQ build.
"""
from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional


ALLOWED_ENTITY_TYPES = {
    "PERSON", "LOCATION", "ORGANIZATION",
    "EVENT", "OBJECT", "CONCEPT", "WORK",
}


# Same shape as the Anthropic prompts, with a few small-model adjustments:
#   - Explicit "do not write anything before/after the JSON" reminders.
#   - The allowed-entity-type list is restated mid-prompt where Qwen3 is most
#     likely to attend to it.
_ENTITY_SYSTEM_PROMPT = """You extract named entities from a conversation transcript.

Input format:
A dialogue is given as a numbered list of utterances. Each line shows the speaker and the utterance text.

Output format:
Return ONLY a JSON object: {"entities": [...]}
Each entity is: {"name": str, "type": str, "description": str, "mentioned_in": [int, ...]}

Allowed entity types (use EXACTLY one of these, uppercase):
PERSON, LOCATION, ORGANIZATION, EVENT, OBJECT, CONCEPT, WORK

Field rules:
- name: canonical/most complete form (prefer "Joey Tribbiani" over "Joey" if both appear; "Phoebe" over "Pheebs").
- type: must be one of the allowed types above.
- description: 1-2 sentence summary of what THIS DIALOGUE reveals about the entity. Do not use general/world knowledge — only what the dialogue says or strongly implies. If the dialogue reveals nothing about the entity beyond its name, write a minimal description like "Mentioned by <Speaker>." or "<Type> referenced in dialogue."
- mentioned_in: list of utterance numbers (the [N] indices) where the entity is named. Pronouns do not count.

Extraction rules:
- Extract proper nouns and meaningful named entities only (people, places, brands, events, named objects, titles, named concepts).
- The speaker of an utterance is NOT an entity OF THEIR OWN UTTERANCE. But when one speaker addresses or refers to another speaker BY NAME, that other speaker IS an entity (type PERSON) for the utterance where they were mentioned.
- Do NOT include pronouns, common nouns, generic descriptors, or filler.
- Do NOT include temporal expressions (dates, days, times, "yesterday", "Friday").
- Merge alternate forms of the same name into one entity (use canonical name, list all utterances where any form appears).
- If no entities, return {"entities": []}.
- Output strictly valid JSON. No prose, no markdown fences, no explanation before or after."""


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
- Output strictly valid JSON. No prose, no markdown fences, no explanation before or after."""


_JUDGE_SYSTEM_PROMPT = """You decide whether a NEW entity is a duplicate of any EXISTING entity already in a knowledge graph.

Input format:
1. NEW entity: {name, type, description}.
2. EXISTING entities: top similarity candidates already in the graph, each with {id, name, type, description, score}. Score is cosine similarity (0–1) of the candidate to the new entity.
3. Optional DIALOGUE CONTEXT: utterances where the new entity was extracted, for disambiguation.

Output format:
Return ONLY a JSON object:
{
  "is_duplicate": true | false,
  "target_id": "<existing entity id>",
  "canonical_name": "<best/most complete name>",
  "canonical_description": "<merged 1-3 sentence description>"
}
target_id, canonical_name, canonical_description are only required when is_duplicate is true.

Decision rules:
- Two entities are duplicates ONLY when they refer to the SAME real-world entity.
- A shared name does NOT imply duplicate. Different people, places, or things can share a name.
- Use descriptions and the dialogue context to disambiguate.
- A more complete name (e.g. "Joey Tribbiani") may still be a duplicate of a shorter form ("Joey") IF the descriptions are mutually consistent and there is no contradicting evidence.
- When in doubt, default to is_duplicate=false. Wrong merges are very hard to undo.
- canonical_name: prefer the most complete form; otherwise pick the best canonical surface.
- canonical_description: combine unique factual content from both descriptions in 1-3 sentences. Do NOT introduce general/world knowledge that wasn't already in either description.
- Output strictly valid JSON. No prose, no markdown fences, no explanation before or after."""


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
{"invalidates": true | false, "reason": "<one short sentence>"}

When in doubt, default to false. False positives are cheap (we keep both facts), false negatives silently delete history.
Output strictly valid JSON. No prose, no markdown fences, no explanation before or after."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.MULTILINE)
# Greedy first-{...}-block fallback for when the model wraps JSON in prose.
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _strip_fences(raw: str) -> str:
    return _JSON_FENCE_RE.sub("", raw.strip()).strip()


def _extract_json(raw: str) -> Optional[Any]:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


class LocalLLMExtractor:
    """Qwen3-14B-backed extractor with the same surface as `LLMExtractor`.

    The model is loaded eagerly in the constructor so callers can see init
    errors (OOM, missing weights, quantization mismatch) up front. Use
    `unload()` to free GPU memory when switching back to the Anthropic mode.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-14B-AWQ"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        device: str = "cuda",
        quantization: Optional[str] = "awq_marlin",
        gpu_memory_utilization: float = 0.55,
        max_model_len: int = 32768,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._device = device
        self._quantization = quantization
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()
        self._unavailable_reason: Optional[str] = None
        # Match `LLMExtractor.model` / `relation_model` attribute names so the
        # GraphService log line and status endpoints render correctly.
        self.model = model_name
        self.relation_model = model_name

        try:
            self._load()
        except Exception as e:
            self._unavailable_reason = f"failed to load {model_name}: {e}"
            print(f"[LocalLLMExtractor] {self._unavailable_reason}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        print(
            f"[LocalLLMExtractor] Loading {self.model_name} via vLLM "
            f"(quantization={self._quantization})..."
        )

        # Keep tokenizer separate so we can apply Qwen3's chat template with
        # `enable_thinking=False` (vLLM's chat helper exposes this through
        # `chat_template_kwargs`, but applying the template explicitly keeps
        # the call site identical to the previous transformers path).
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        llm_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "dtype": "float16",
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "max_model_len": self._max_model_len,
            # Prefix caching is the big win — our four system prompts are
            # byte-identical across every call, so vLLM skips re-prefilling
            # them after the first request.
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        }
        if self._quantization:
            llm_kwargs["quantization"] = self._quantization

        self._model = LLM(**llm_kwargs)
        print(f"[LocalLLMExtractor] {self.model_name} loaded via vLLM.")

    def unload(self) -> None:
        """Free GPU memory. Used when switching modes."""
        with self._lock:
            if self._model is not None:
                # vLLM holds onto the engine via several internal references;
                # explicit shutdown release the executor + KV cache.
                try:
                    shutdown = getattr(self._model.llm_engine, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                except Exception:
                    pass
                del self._model
                self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import contextlib
            from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
            with contextlib.suppress(Exception):
                destroy_model_parallel()
            with contextlib.suppress(Exception):
                destroy_distributed_environment()
        except Exception:
            pass
        print(f"[LocalLLMExtractor] {self.model_name} unloaded.")

    @property
    def available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    # ------------------------------------------------------------------
    # Core generation helper
    # ------------------------------------------------------------------

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        from vllm import SamplingParams

        if not self.available:
            return ""
        max_new_tokens = max_new_tokens or self.max_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Qwen3 exposes an `enable_thinking` flag on its chat template that
        # toggles the <think>...</think> CoT block. We want short
        # deterministic JSON output, so thinking is off.
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Stop as soon as Qwen finishes the JSON object — guards against
        # over-generation prose after the closing brace. The triple-newline
        # stop catches the common "explanation paragraph" tail.
        stop_seqs = ["\n\n\n", "```"]
        if stop:
            stop_seqs.extend(stop)

        sampling_params = SamplingParams(
            temperature=0.0,  # greedy — matches the prior do_sample=False intent
            max_tokens=max_new_tokens,
            stop=stop_seqs,
        )

        t0 = time.perf_counter()
        with self._lock:
            outputs = self._model.generate(
                [prompt_text],
                sampling_params,
                use_tqdm=False,
            )
        if not outputs or not outputs[0].outputs:
            return ""
        gen = outputs[0].outputs[0]
        elapsed = time.perf_counter() - t0
        # Single concise line per call so we can spot which path dominates.
        n_prompt = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0
        n_out = len(gen.token_ids) if gen.token_ids else 0
        tps = n_out / elapsed if elapsed > 0 else 0.0
        print(
            f"[LocalLLMExtractor] gen: {elapsed:.2f}s  "
            f"prompt_tok={n_prompt}  out_tok={n_out}  tok/s={tps:.1f}  "
            f"finish={gen.finish_reason}"
        )
        return gen.text

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Public generation helper for the FastAPI `/llm/generate` route.

        Uses a generic analytical-assistant system prompt so the route can be
        called for fact-sheet answering when the UI is in Local mode.
        """
        if not self.available:
            return ""
        return self._generate(
            system_prompt="You are a helpful and precise analytical assistant.",
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Pass 1 — Entities
    # ------------------------------------------------------------------

    def extract_dialogue_entities(
        self,
        utterances: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.available or not utterances:
            return []
        user_prompt = self._format_dialogue(utterances)
        try:
            # Entity JSON output is rarely >600 tokens in practice; cap at 768
            # to bound the worst case without leaving headroom for prose tails.
            raw = self._generate(_ENTITY_SYSTEM_PROMPT, user_prompt, max_new_tokens=768)
        except Exception as e:
            print(f"[LocalLLMExtractor] entity generate failed: {e}")
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
        if not self.available or len(entities) < 2 or not utterances:
            return []
        user_prompt = self._format_relation_prompt(utterances, entities)
        try:
            # Relations tend to be denser than entities; 1024 covers the
            # observed p99 output length on MELD-style transcripts.
            raw = self._generate(_RELATION_SYSTEM_PROMPT, user_prompt, max_new_tokens=1024)
        except Exception as e:
            print(f"[LocalLLMExtractor] relation generate failed: {e}")
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
        if not self.available or not existing:
            return None
        user_prompt = self._format_judge_prompt(candidate, existing, dialogue_context)
        try:
            raw = self._generate(_JUDGE_SYSTEM_PROMPT, user_prompt, max_new_tokens=1024)
        except Exception as e:
            print(f"[LocalLLMExtractor] judge generate failed: {e}")
            return None
        return self._parse_judge_response(raw)

    # ------------------------------------------------------------------
    # Pass 4 — Fact contradiction judge
    # ------------------------------------------------------------------

    def judge_fact_contradiction(
        self,
        old_fact: Dict[str, Any],
        new_fact: Dict[str, Any],
        dialogue_context: str = "",
    ) -> Optional[Dict[str, Any]]:
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
            raw = self._generate(_CONTRADICTION_SYSTEM_PROMPT, user_prompt, max_new_tokens=512)
        except Exception as e:
            print(f"[LocalLLMExtractor] contradiction generate failed: {e}")
            return None

        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        return {
            "invalidates": bool(data.get("invalidates", False)),
            "reason": str(data.get("reason", "")).strip(),
        }

    # ------------------------------------------------------------------
    # Prompt formatting (identical shape to LLMExtractor)
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
        data = _extract_json(raw)
        if not isinstance(data, dict):
            print(f"[LocalLLMExtractor] entity JSON parse failed: {raw[:200]!r}")
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
            etype_raw = str(ent.get("type", "")).strip().upper()
            # Closed-vocab enforcement: anything not in ALLOWED_ENTITY_TYPES
            # collapses to CONCEPT so we don't silently drop a salient name.
            etype = etype_raw if etype_raw in ALLOWED_ENTITY_TYPES else "CONCEPT"
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

    def _parse_relations(self, raw: str, valid_names: set) -> List[Dict[str, Any]]:
        data = _extract_json(raw)
        if not isinstance(data, dict):
            print(f"[LocalLLMExtractor] relation JSON parse failed: {raw[:200]!r}")
            return []
        items = data.get("relations", [])
        if not isinstance(items, list):
            return []

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
        data = _extract_json(raw)
        if not isinstance(data, dict):
            print(f"[LocalLLMExtractor] judge JSON parse failed: {raw[:200]!r}")
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
