"""
generate_meld_benchmark.py

Synthetic evaluation-set generator for the SpeechRAG framework, built from the
MELD train split (`train_sent_emo.csv`). Inspired by Microsoft's GraphRAG
"adaptive benchmarking" approach: rather than reusing a benchmark designed for
text-only chatbot memory, we generate questions that exercise the specific
retrieval capabilities of a speaker-aware, audio-grounded RAG system.

The benchmark targets five distinct capability axes, each mapped to a concrete
component of SpeechRAG:

  Q-type             | Tests                                  | System component
  -------------------|----------------------------------------|---------------------------
  identity           | "Who said X?"                          | Speaker NER + semantic
  emotion_grounded   | "How did X feel when saying Y?"        | acoustic/text_emotion metadata
  semantic_recall    | "What did X say about Y?"              | semantic_service (filtered)
  graph_reasoning    | "Who was X arguing with about Y?"      | graph_service (co-speakers)
  emotion_profile    | "Dominant emotion of X across all..."  | metadata aggregation

Each generated question carries:
  - `question`: the natural-language prompt to feed into SpeakerRAG.build_context
  - `ground_truth`: the answer derived directly from the CSV (the source of truth)
  - `evidence`: list of (Dialogue_ID, Utterance_ID, Speaker, Emotion, Utterance)
                tuples that support the ground truth, used by the LLM-judge to
                compute a grounded score.
  - `q_type`: one of the five categories above, useful for per-type metrics.

Usage:
    python generate_meld_benchmark.py \
        --csv train_sent_emo.csv \
        --out benchmark.json \
        --n_per_type 1 \
        --seed 42

Setting `--n_per_type 1` reproduces the seed set of 5 questions; bump it to
generate larger evaluation sets. The seed ensures reproducibility.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Speakers we restrict to ---------------------------------------------
# The six mains have ~1300+ utterances each; minor characters often appear in
# only a single dialogue and are bad fits for speaker-aware retrieval tests.
MAIN_SPEAKERS = ["Joey", "Ross", "Rachel", "Phoebe", "Monica", "Chandler"]

# Utterances shorter than this are typically backchannels ("Yeah!", "Ok.")
# and are ambiguous in retrieval — skip them when picking question evidence.
MIN_UTT_LEN = 35


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _content_words(text: str) -> List[str]:
    """Crude content-word extractor: capitalized non-stop tokens with len>=4.
    Used to seed `semantic_recall` and `graph_reasoning` questions with a
    concrete topic anchor (a name, place, or salient noun)."""
    import re
    stop = {"That", "This", "What", "When", "Where", "Why", "Who", "Yeah",
            "Well", "Okay", "Look", "Hey", "Damn", "There", "They", "Nobody",
            "Please", "Listen"}
    toks = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)
    return [t for t in toks if t not in stop]


def _format_evidence(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    """Standardised evidence record used by the judge."""
    return [
        {
            "dialogue_id": int(r.Dialogue_ID),
            "utterance_id": int(r.Utterance_ID),
            "speaker": r.Speaker,
            "emotion": r.Emotion,
            "sentiment": r.Sentiment,
            "utterance": r.Utterance,
        }
        for _, r in rows.iterrows()
    ]


# -------------------------------------------------------------------------
# Question generators (one per q_type)
# -------------------------------------------------------------------------

def gen_identity(df: pd.DataFrame, rng: random.Random) -> Optional[Dict[str, Any]]:
    """'Who said: \"<utterance>\"?' — exercises the speaker-attribution path
    (NER on the question + semantic match on the quoted text)."""
    pool = df[
        df.Speaker.isin(MAIN_SPEAKERS) &
        (df.Utterance.str.len() >= MIN_UTT_LEN) &
        (df.Utterance.str.len() <= 140)
    ]
    if pool.empty:
        return None
    row = pool.sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
    return {
        "q_type": "identity",
        "question": f'Who said: "{row.Utterance.strip()}"?',
        "ground_truth": row.Speaker,
        "evidence": _format_evidence(pd.DataFrame([row])),
    }


def gen_emotion_grounded(df: pd.DataFrame, rng: random.Random) -> Optional[Dict[str, Any]]:
    """'What was <speaker>'s emotional state when they said \"<utterance>\"?'
    — tests retrieval of the acoustic_emotion / text_emotion metadata
    attached to the matching segment. We bias towards non-neutral emotions
    so the answer is non-trivial."""
    pool = df[
        df.Speaker.isin(MAIN_SPEAKERS) &
        (df.Emotion != "neutral") &
        (df.Utterance.str.len() >= MIN_UTT_LEN)
    ]
    if pool.empty:
        return None
    row = pool.sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
    return {
        "q_type": "emotion_grounded",
        "question": (
            f'What was {row.Speaker}\'s emotional state when they said: '
            f'"{row.Utterance.strip()}"?'
        ),
        "ground_truth": row.Emotion,
        "evidence": _format_evidence(pd.DataFrame([row])),
    }


def gen_semantic_recall(df: pd.DataFrame, rng: random.Random) -> Optional[Dict[str, Any]]:
    """'What did <speaker> say about <topic>?' — exercises semantic search
    filtered by speaker. We pick a (speaker, topic) pair where the topic
    appears in at least one utterance by that speaker."""
    # Find (speaker, topic) pairs with at least one match
    candidates: List[Tuple[str, str, pd.DataFrame]] = []
    for spk in MAIN_SPEAKERS:
        spk_df = df[df.Speaker == spk]
        for _, r in spk_df.sample(min(200, len(spk_df)),
                                  random_state=rng.randint(0, 1 << 30)).iterrows():
            for tok in _content_words(r.Utterance):
                hits = spk_df[spk_df.Utterance.str.contains(
                    rf"\b{tok}\b", regex=True, case=False, na=False
                )]
                if len(hits) >= 1 and len(hits) <= 4:
                    # narrow enough to be a real recall test, not a generic word
                    candidates.append((spk, tok, hits))
                    break
            if candidates and candidates[-1][0] == spk:
                break
    if not candidates:
        return None
    spk, topic, hits = rng.choice(candidates)
    return {
        "q_type": "semantic_recall",
        "question": f"What did {spk} say about {topic}?",
        "ground_truth": (
            f"{spk} mentioned {topic} in: "
            + " | ".join(f'"{u.strip()}"' for u in hits.Utterance.tolist())
        ),
        "evidence": _format_evidence(hits),
    }


def gen_graph_reasoning(df: pd.DataFrame, rng: random.Random) -> Optional[Dict[str, Any]]:
    """'Who was <speaker> arguing with in the conversation about <topic>?' —
    exercises the knowledge graph: speaker--dialogue links and co-speaker
    edges, plus entity matching on the topic word."""
    # Find an angry/disgusted utterance from a main speaker that names another
    # main speaker by entity, in a dialogue with multiple participants.
    angry = df[
        df.Speaker.isin(MAIN_SPEAKERS) &
        df.Emotion.isin(["anger", "disgust"]) &
        (df.Utterance.str.len() >= MIN_UTT_LEN)
    ]
    angry = angry.sample(frac=1, random_state=rng.randint(0, 1 << 30))

    for _, r in angry.iterrows():
        # find named entities that ARE other main speakers
        mentioned = [w for w in _content_words(r.Utterance)
                     if w in MAIN_SPEAKERS and w != r.Speaker]
        if not mentioned:
            continue
        target = mentioned[0]
        dlg = df[df.Dialogue_ID == r.Dialogue_ID]
        co_speakers = sorted(set(dlg.Speaker.tolist()) - {r.Speaker})
        if not co_speakers:
            continue
        return {
            "q_type": "graph_reasoning",
            "question": (
                f"In the dialogue where {r.Speaker} got upset about {target}, "
                f"who else was present in the conversation?"
            ),
            "ground_truth": ", ".join(co_speakers),
            "evidence": _format_evidence(dlg),
        }
    return None


def gen_emotion_profile(df: pd.DataFrame, rng: random.Random) -> Optional[Dict[str, Any]]:
    """'What was <speaker>'s most frequent non-neutral emotion across all
    enrolled conversations?' — tests aggregation across the whole metadata
    store. This is the capability gap we want to highlight: the current
    semantic_service.search_by_text returns top-k matches, not aggregates."""
    spk = rng.choice(MAIN_SPEAKERS)
    spk_df = df[df.Speaker == spk]
    counts = Counter(
        e for e in spk_df.Emotion.tolist() if e != "neutral"
    )
    if not counts:
        return None
    top_emo, top_n = counts.most_common(1)[0]
    # Provide a few exemplar utterances as evidence
    exemplars = spk_df[spk_df.Emotion == top_emo].head(5)
    return {
        "q_type": "emotion_profile",
        "question": (
            f"Across all the conversations enrolled in the system, what is "
            f"{spk}'s most frequent non-neutral emotion?"
        ),
        "ground_truth": (
            f"{top_emo} (occurs {top_n} times across the enrolled corpus)"
        ),
        "evidence": _format_evidence(exemplars),
    }


GENERATORS = {
    "identity": gen_identity,
    "emotion_grounded": gen_emotion_grounded,
    "semantic_recall": gen_semantic_recall,
    "graph_reasoning": gen_graph_reasoning,
    "emotion_profile": gen_emotion_profile,
}


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def build_benchmark(df: pd.DataFrame, n_per_type: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    bench: List[Dict[str, Any]] = []
    for q_type, gen in GENERATORS.items():
        produced = 0
        attempts = 0
        while produced < n_per_type and attempts < n_per_type * 20:
            attempts += 1
            q = gen(df, rng)
            if q is None:
                continue
            q["id"] = f"{q_type}_{produced:04d}"
            bench.append(q)
            produced += 1
        if produced < n_per_type:
            print(f"[warn] only produced {produced}/{n_per_type} for {q_type}")
    return bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="train_sent_emo.csv",
                        help="Path to the MELD train_sent_emo.csv file.")
    parser.add_argument("--out", default="benchmark.json",
                        help="Output JSON path.")
    parser.add_argument("--n_per_type", type=int, default=1,
                        help="Number of questions per q_type. "
                             "Default 1 -> 5 questions total.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    bench = build_benchmark(df, n_per_type=args.n_per_type, seed=args.seed)

    Path(args.out).write_text(json.dumps(bench, indent=2, ensure_ascii=False))
    print(f"Wrote {len(bench)} questions to {args.out}")
    for q in bench:
        print(f"  [{q['q_type']}] {q['question'][:80]}")


if __name__ == "__main__":
    main()