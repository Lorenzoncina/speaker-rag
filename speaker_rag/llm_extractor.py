"""
LLM-backed entity extraction for SpeakerRAG.

Uses Anthropic's Haiku via the Messages API with prompt caching on the system
prompt. Designed to be called per-dialogue (batched) rather than per-utterance,
which keeps cost low and lets the model see surrounding context.

Output is the same shape as the regex/spaCy extractor — a list of (name, type)
tuples per utterance — so the rest of the graph code is unchanged.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# System prompt is cacheable — keep it stable across calls so the cache hits.
_SYSTEM_PROMPT = """You extract named entities from conversation transcripts.

Input format:
A dialogue is given as a numbered list of utterances. Each line shows the speaker and the utterance text.

Output format:
Return ONLY a JSON array. Each element corresponds to one input utterance, in input order. Each element is a list of {"name": str, "type": str} objects for entities mentioned in that utterance.

Allowed entity types: PERSON, LOCATION, ORGANIZATION, EVENT, OBJECT, CONCEPT, WORK.

Extraction rules:
- Extract proper nouns and meaningful named entities only (people, places, brands, events, named objects, titles).
- Do NOT include the speaker of an utterance as an entity of their own utterance — but DO include other people when mentioned by name.
- Do NOT include pronouns, common nouns, generic descriptors, or filler ("y'know", "the guy", "stuff").
- Use the canonical/most-complete form of a name visible in the dialogue (e.g., prefer "Joey" over "Joe" if both appear and refer to the same person).
- If an utterance has no entities, return [].
- Output strictly valid JSON. No prose, no markdown fences, no explanations."""


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMExtractor:
    """Wrapper around Anthropic's Messages API for batched entity extraction.

    Reads ANTHROPIC_API_KEY from the environment. If the key is missing or the
    SDK isn't installed, `available` is False and callers should fall back to
    the regex/spaCy extractor.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 2048,
        cache_ttl: str = "1h",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self._client = None
        self._unavailable_reason: Optional[str] = None

        # Explicit api_key arg wins; otherwise fall back to env var.
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

    def extract_dialogue_entities(
        self,
        utterances: List[Dict[str, Any]],
    ) -> List[List[Tuple[str, str]]]:
        """Extract entities for every utterance in a single dialogue, in one LLM call.

        utterances: ordered list of dicts with keys `speaker` and `text`.
        Returns a list of `(name, type)` tuples per utterance, in the same order.
        On any failure, returns empty lists so callers can degrade gracefully.
        """
        if not self.available or not utterances:
            return [[] for _ in utterances]

        user_prompt = self._format_dialogue(utterances)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
                    },
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            print(f"[LLMExtractor] API call failed: {e}")
            return [[] for _ in utterances]

        u = getattr(response, "usage", None)
        if u is not None:
            print(
                f"[LLMExtractor] usage: input={getattr(u, 'input_tokens', 0)} "
                f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} "
                f"cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
                f"output={getattr(u, 'output_tokens', 0)}"
            )

        try:
            raw = response.content[0].text
        except (AttributeError, IndexError):
            return [[] for _ in utterances]

        return self._parse_response(raw, expected_len=len(utterances))

    def _format_dialogue(self, utterances: List[Dict[str, Any]]) -> str:
        lines = ["Dialogue:"]
        for i, u in enumerate(utterances, start=1):
            speaker = (u.get("speaker") or "Unknown").strip()
            text = (u.get("text") or "").strip().replace("\n", " ")
            lines.append(f"[{i}] {speaker}: {text}")
        lines.append("\nReturn the JSON array now.")
        return "\n".join(lines)

    def _parse_response(
        self, raw: str, expected_len: int
    ) -> List[List[Tuple[str, str]]]:
        cleaned = _JSON_FENCE_RE.sub("", raw.strip()).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[LLMExtractor] failed to parse JSON: {cleaned[:200]!r}")
            return [[] for _ in range(expected_len)]

        if not isinstance(data, list):
            return [[] for _ in range(expected_len)]

        out: List[List[Tuple[str, str]]] = []
        for item in data:
            if not isinstance(item, list):
                out.append([])
                continue
            entities: List[Tuple[str, str]] = []
            seen: set = set()
            for ent in item:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("name", "")).strip()
                etype = str(ent.get("type", "")).strip().upper() or "CONCEPT"
                if not name or name.lower() in seen:
                    continue
                seen.add(name.lower())
                entities.append((name, etype))
            out.append(entities)

        # Pad or truncate to match expected length so callers can zip safely.
        if len(out) < expected_len:
            out.extend([[] for _ in range(expected_len - len(out))])
        elif len(out) > expected_len:
            out = out[:expected_len]
        return out
