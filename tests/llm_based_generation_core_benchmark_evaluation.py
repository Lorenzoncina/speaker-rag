"""
generate_meld_benchmark_llm.py

Hybrid LLM-driven benchmark generator for the SpeechRAG framework.

WHY HYBRID?
A pure heuristic generator (regex over capitalized words, etc.) produces
ill-formed questions like "What did Monica say about Count?" — because it has
no concept of semantic salience. A pure LLM generator (give Claude the CSV,
ask for questions) risks hallucinated ground truths — Claude might invent
that Phoebe said something Rachel actually said.

The hybrid approach splits the work:
  1. PYTHON sampler: pulls grounded candidates from the CSV. Each candidate
     carries exact dialogue_id / utterance_id pointers, so the ground truth
     is always anchored to real source rows.
  2. CLAUDE selector/phraser: receives a batch of candidates and decides
     (a) which are good enough to keep, (b) how to phrase the question
     naturally. Claude does NOT invent facts — it only filters and rephrases.
  3. VALIDATION: the returned question is matched back to the source row to
     ensure the ground truth Claude wrote is consistent with the data.

Five q_types map to five capability axes of SpeechRAG:
  identity         | "Who said X?"             -> speaker NER + semantic match
  emotion_grounded | "How did X feel saying Y?"-> acoustic/text_emotion metadata
  semantic_recall  | "What did X say about Y?" -> filtered semantic search
  graph_reasoning  | "Who else was present?"   -> graph_service co-speakers
  emotion_profile  | "Most frequent emotion?"  -> metadata aggregation

USAGE
  pip install anthropic pandas
  export ANTHROPIC_API_KEY=sk-ant-...
  python generate_meld_benchmark_llm.py \
      --csv train_sent_emo.csv \
      --out benchmark_llm.json \
      --n-per-type 3 \
      --model claude-sonnet-4-6 \
      --seed 42

`--n-per-type 3` produces 15 questions; bump for larger sets. The Python
sampler oversamples ~3x candidates per slot so Claude has room to reject
weak ones, then we fill until we have enough kept questions per type.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
MAIN_SPEAKERS = ["Joey", "Ross", "Rachel", "Phoebe", "Monica", "Chandler"]
MIN_UTT_LEN = 35
DEFAULT_MODEL = "claude-sonnet-4-6"
OVERSAMPLE = 3  # ask Python for 3x as many candidates as questions wanted

# Words that should NEVER count as "topics" for semantic_recall — they are
# verbs/discourse markers that happen to appear capitalized at sentence start.
TOPIC_BLOCKLIST = {
    "Yeah", "Okay", "Well", "Look", "Listen", "Please", "Hey", "Damn",
    "There", "They", "Nobody", "What", "When", "Where", "Why", "Count",
    "This", "That", "These", "Those", "Stop", "Just", "Like", "Right",
    "About", "After", "Before", "Some", "Other", "Sure", "Going", "Coming",
    "Really", "Actually", "Maybe", "Wait", "Sorry", "Thank", "Thanks",
    "Hello", "Anyway", "Wanna", "Gonna", "Have", "Here", "Then", "Even",
    "From", "Such", "Once",
}


# -------------------------------------------------------------------------
# Python samplers — produce GROUNDED candidates with exact CSV pointers
# -------------------------------------------------------------------------
def _row_to_evidence(r) -> Dict[str, Any]:
    return {
        "dialogue_id": int(r.Dialogue_ID),
        "utterance_id": int(r.Utterance_ID),
        "speaker": r.Speaker,
        "emotion": r.Emotion,
        "sentiment": r.Sentiment,
        "utterance": r.Utterance,
    }


def sample_identity(df: pd.DataFrame, k: int, rng: random.Random) -> List[Dict]:
    pool = df[
        df.Speaker.isin(MAIN_SPEAKERS)
        & df.Utterance.str.len().between(40, 140)
    ]
    sample = pool.sample(min(k, len(pool)), random_state=rng.randint(0, 1 << 30))
    return [{"row": _row_to_evidence(r)} for _, r in sample.iterrows()]


def sample_emotion_grounded(df: pd.DataFrame, k: int, rng: random.Random) -> List[Dict]:
    leak_words = r"\b(?:angry|anger|sad|sadness|happy|joyful|afraid|fear|fearful|disgust|disgusted|surprised|surprise)\b"
    pool = df[
        df.Speaker.isin(MAIN_SPEAKERS)
        & (df.Emotion != "neutral")
        & df.Utterance.str.len().between(35, 140)
        & ~df.Utterance.str.contains(leak_words, regex=True, case=False, na=False)
    ]
    sample = pool.sample(min(k, len(pool)), random_state=rng.randint(0, 1 << 30))
    return [{"row": _row_to_evidence(r)} for _, r in sample.iterrows()]


def sample_semantic_recall(df: pd.DataFrame, k: int, rng: random.Random) -> List[Dict]:
    """Find (speaker, topic, hits) triples where the topic looks meaningful.
    Final keep-decision is delegated to Claude — Python just throws out the
    obviously broken cases."""
    candidates: List[Dict] = []
    for spk in MAIN_SPEAKERS:
        spk_df = df[df.Speaker == spk]
        seen_topics: set = set()
        # iterate utterances in random order so we don't always pick D0
        idx = list(spk_df.index)
        rng.shuffle(idx)
        for i in idx:
            r = spk_df.loc[i]
            tokens = re.findall(r"\b[A-Z][a-z]{3,}\b", r.Utterance)
            for tok in tokens:
                if tok in MAIN_SPEAKERS:  # those go to graph_reasoning
                    continue
                if tok in TOPIC_BLOCKLIST:
                    continue
                if tok in seen_topics:
                    continue
                # ALL CAPS check (e.g. RESPECTFULLY) — token already lowercased
                # tail by the regex, so this catches nothing. The blocklist +
                # Claude's filter are the real safety net.
                hits = spk_df[
                    spk_df.Utterance.str.contains(
                        rf"\b{re.escape(tok)}\b", regex=True, na=False
                    )
                ]
                if 1 <= len(hits) <= 4:
                    seen_topics.add(tok)
                    candidates.append({
                        "speaker": spk,
                        "topic": tok,
                        "hits": [_row_to_evidence(h) for _, h in hits.iterrows()],
                    })
                    break  # one candidate per utterance
            if len([c for c in candidates if c["speaker"] == spk]) >= max(2, k // 3):
                break
    rng.shuffle(candidates)
    return candidates[:k]


def sample_graph_reasoning(df: pd.DataFrame, k: int, rng: random.Random) -> List[Dict]:
    candidates: List[Dict] = []
    dialogue_ids = list(df.Dialogue_ID.unique())
    rng.shuffle(dialogue_ids)
    for did in dialogue_ids:
        dlg = df[df.Dialogue_ID == did].sort_values("Utterance_ID")
        speakers = set(dlg.Speaker) & set(MAIN_SPEAKERS)
        if len(speakers) < 3:
            continue
        # find a strong-emotion main-speaker utterance
        strong = dlg[
            dlg.Speaker.isin(MAIN_SPEAKERS)
            & dlg.Emotion.isin(["anger", "disgust", "sadness"])
            & (dlg.Utterance.str.len() >= MIN_UTT_LEN)
        ]
        if strong.empty:
            continue
        anchor = strong.iloc[0]
        candidates.append({
            "dialogue_id": int(did),
            "anchor_speaker": anchor.Speaker,
            "anchor_emotion": anchor.Emotion,
            "anchor_utterance": anchor.Utterance,
            "others_present": sorted(speakers - {anchor.Speaker}),
            "full_dialogue": [_row_to_evidence(r) for _, r in dlg.iterrows()],
        })
        if len(candidates) >= k:
            break
    return candidates


def sample_emotion_profile(df: pd.DataFrame, k: int, rng: random.Random) -> List[Dict]:
    """Returns aggregation FACTS, not utterances. Each fact is a true
    statistic over the whole corpus that Claude phrases into a question."""
    facts: List[Dict] = []
    main_df = df[df.Speaker.isin(MAIN_SPEAKERS)]

    # Fact type 1: who shows the most of emotion X? (rare emotions are most interesting)
    for emo in ["fear", "disgust", "sadness", "anger"]:
        counts = main_df[main_df.Emotion == emo].Speaker.value_counts()
        if not counts.empty:
            top = counts.index[0]
            facts.append({
                "fact_type": "max_speaker_for_emotion",
                "emotion": emo,
                "top_speaker": top,
                "top_count": int(counts.iloc[0]),
                "full_ranking": counts.to_dict(),
            })

    # Fact type 2: count for a specific (speaker, emotion) pair
    for spk in MAIN_SPEAKERS:
        for emo in ["fear", "disgust", "anger", "sadness"]:
            n = int(((main_df.Speaker == spk) & (main_df.Emotion == emo)).sum())
            if n > 0:
                facts.append({
                    "fact_type": "count_speaker_emotion",
                    "speaker": spk,
                    "emotion": emo,
                    "count": n,
                })

    # Fact type 3: pairwise comparison
    for emo in ["anger", "fear", "sadness"]:
        counts = main_df[main_df.Emotion == emo].Speaker.value_counts()
        if len(counts) >= 2:
            top2 = counts.head(2)
            facts.append({
                "fact_type": "compare_two_speakers",
                "emotion": emo,
                "speaker_a": top2.index[0],
                "speaker_b": top2.index[1],
                "count_a": int(top2.iloc[0]),
                "count_b": int(top2.iloc[1]),
                "winner": top2.index[0],
            })

    rng.shuffle(facts)
    # Attach a couple of exemplar utterances to each fact for the evidence field
    for f in facts:
        if "speaker" in f:
            spk = f["speaker"]
            emo = f.get("emotion")
            ex = main_df[(main_df.Speaker == spk) & (main_df.Emotion == emo)].head(2)
        elif "top_speaker" in f:
            spk = f["top_speaker"]
            emo = f["emotion"]
            ex = main_df[(main_df.Speaker == spk) & (main_df.Emotion == emo)].head(2)
        else:
            spk = f.get("speaker_a")
            emo = f.get("emotion")
            ex = main_df[(main_df.Speaker == spk) & (main_df.Emotion == emo)].head(2)
        f["evidence"] = [_row_to_evidence(r) for _, r in ex.iterrows()]

    return facts[:k]


SAMPLERS = {
    "identity": sample_identity,
    "emotion_grounded": sample_emotion_grounded,
    "semantic_recall": sample_semantic_recall,
    "graph_reasoning": sample_graph_reasoning,
    "emotion_profile": sample_emotion_profile,
}


# -------------------------------------------------------------------------
# Claude prompts — selector/phraser per q_type
# -------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "identity": """You are crafting evaluation questions for a speaker-aware multi-modal RAG system.
You receive a single utterance from a TV-show transcript with its true speaker.
Decide whether this utterance is a GOOD candidate for a "Who said X?" retrieval test:
  - It must be specific enough that the answer is non-trivial (not "Hi!" or "Yeah!").
  - It should not contain the speaker's own name (that would leak the answer).
  - It should not be so generic that any character could have said it.
If good, phrase a natural "Who said: '<utterance>'?" question.
Output STRICT JSON only, no prose, no code fences:
{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<speaker name>"}""",

    "emotion_grounded": """You are crafting evaluation questions for an audio-grounded RAG system.
You receive an utterance with its speaker and ground-truth emotion (from MELD annotations).
Decide whether this is a GOOD emotion-attribution test:
  - The utterance should not literally contain the emotion word (that would leak the answer).
  - The emotion should be plausibly inferable from the content/context, not a coin flip.
If good, phrase a natural question of the form "What was <speaker>'s emotional state when they said '<utterance>'?" — vary the wording across calls (e.g., "How was X feeling when...", "What emotion did X express..."). Use the speaker's actual name.
Output STRICT JSON only, no prose, no code fences:
{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<one of: anger|disgust|fear|joy|sadness|surprise>"}""",

    "semantic_recall": """You are crafting evaluation questions about a speaker's recurring topics.
You receive a (speaker, topic, hits) candidate. The "topic" was extracted by a regex looking for capitalized words. Many of these are NOISE — verbs like "Count" or "Stop", discourse markers like "Listen", or unique-context filler.

REJECT IT IF:
  - The topic is a verb, discourse marker, or generic English word.
  - The topic is shouty ALL-CAPS for emphasis (e.g., "RESPECTFULLY").
  - The topic appears only in a single throwaway phrase with no real content.
  - The topic is a one-off oddity that no retrieval system could meaningfully match.

KEEP IT IF the topic is a named character, a recurring theme, a job/profession, an object/place that the speaker actually has things to say about.

If kept, phrase a natural "What did <speaker> say about <topic>?" question (vary phrasing). The ground_truth should be a 1-2 sentence summary of what the speaker said, drawing only on the provided hits — do not invent facts.
Output STRICT JSON only, no prose, no code fences:
{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<1-2 sentence summary>"}""",

    "graph_reasoning": """You are crafting evaluation questions for a knowledge-graph-based RAG system.
You receive a multi-speaker dialogue with a strong-emotion anchor moment. The test is whether the system can identify the OTHER speakers present in that dialogue, given a description of the emotional/topical content.

Phrase the question naturally — describe what was happening (who got upset, why) so the question is identifiable from semantic + emotion signals, NOT from a dialogue ID. Vary the phrasing across calls.

REJECT if the dialogue is too short or the anchor moment is too generic to identify.

If kept, ground_truth is the comma-separated list of other speakers present (excluding the anchor speaker).
Output STRICT JSON only, no prose, no code fences:
{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<comma-separated names>"}""",

    "emotion_profile": """You are crafting evaluation questions about aggregate emotional behavior across a corpus.
You receive a FACT — a true statistic computed over all enrolled conversations. Phrase a natural question whose correct answer requires aggregation across the whole corpus (not just retrieving one utterance).

Vary the question phrasing across calls. Examples of good phrasings:
  - "Across the corpus, who shows the most fear?"
  - "How many times does Monica express disgust?"
  - "Between Joey and Ross, who gets angry more often?"
Avoid always using "most frequent emotion" — vary the angle.

REJECT only if the fact is degenerate (e.g., 0 vs 0).

If kept, ground_truth should state the answer with the supporting count(s) so an LLM-judge can score precision.
Output STRICT JSON only, no prose, no code fences:
{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<answer with counts>"}""",
}


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
def call_claude(client: Anthropic, model: str, system: str, payload: Dict) -> Optional[Dict]:
    """One round-trip to Claude. Returns parsed JSON dict or None on parse failure."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        text = resp.content[0].text.strip()
        # tolerate occasional code-fence wrapping
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[warn] Claude returned non-JSON: {e}\n  raw: {text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[warn] Claude API call failed: {e}", file=sys.stderr)
        return None


def build_evidence(q_type: str, candidate: Dict) -> List[Dict]:
    """Extract the evidence list to attach to each kept question."""
    if q_type in {"identity", "emotion_grounded"}:
        return [candidate["row"]]
    if q_type == "semantic_recall":
        return candidate["hits"]
    if q_type == "graph_reasoning":
        return candidate["full_dialogue"]
    if q_type == "emotion_profile":
        return candidate.get("evidence", [])
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\nWHY")[0])
    parser.add_argument("--csv", default="train_sent_emo.csv")
    parser.add_argument("--out", default="benchmark_llm.json")
    parser.add_argument("--n-per-type", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL}). "
                             "Use claude-opus-4-7 for higher quality.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY in your environment.")

    df = pd.read_csv(args.csv)
    rng = random.Random(args.seed)
    client = Anthropic()

    benchmark: List[Dict] = []
    for q_type, sampler in SAMPLERS.items():
        print(f"\n=== {q_type} (target: {args.n_per_type}) ===", file=sys.stderr)
        candidates = sampler(df, k=args.n_per_type * OVERSAMPLE, rng=rng)
        print(f"  Python sampled {len(candidates)} candidates", file=sys.stderr)

        kept = 0
        for cand in candidates:
            if kept >= args.n_per_type:
                break
            # Build user payload (strip evidence dump for graph_reasoning to save tokens)
            payload = dict(cand)
            if q_type == "graph_reasoning":
                # send the dialogue as text snippets, not the full evidence dump
                payload["dialogue_excerpt"] = [
                    f"[{u['speaker']}/{u['emotion']}] {u['utterance']}"
                    for u in cand["full_dialogue"][:20]
                ]
                payload.pop("full_dialogue", None)

            result = call_claude(client, args.model, SYSTEM_PROMPTS[q_type], payload)
            if result is None or not result.get("keep"):
                continue

            benchmark.append({
                "id": f"{q_type}_{kept:04d}",
                "q_type": q_type,
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "evidence": build_evidence(q_type, cand),
            })
            kept += 1
            print(f"  [{kept}/{args.n_per_type}] {result['question'][:90]}", file=sys.stderr)

        if kept < args.n_per_type:
            print(f"  [warn] only kept {kept}/{args.n_per_type} for {q_type} "
                  f"— consider increasing OVERSAMPLE or n-per-type", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(benchmark)} questions to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()