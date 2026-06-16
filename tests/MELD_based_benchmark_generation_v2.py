"""
generate_meld_benchmark_v4.py

Hybrid LLM-driven benchmark generator for the SpeechRAG framework (GPT backend).

TAXONOMY: 6 q_types x 3 subtypes (audio_identity has 1) = 16 subtypes.
`--n-per-type` is the number of questions PER SUBTYPE.

  identity        : direct | pair_attribution | cross_call_linking
  emotion         : direct_attribution | sarcasm_detection | trajectory
  semantic_recall : open_summary | quote_completion | stance
  graph           : co_speakers | entity_cross_speaker | relationship_inference
  aggregate       : argmax_speaker | exact_count | pairwise_comparison
  audio_identity  : semantic            (cross-modal; uses a MELD dev clip)

DESIGN PRINCIPLE (why this is trustworthy):
  Python owns every DETERMINISTIC field — counts, speaker lists, the correct
  emotion label, the MCQ choices. GPT is only ever a phraser/summariser: it
  rewords the question naturally and, for open-ended subtypes, writes a summary
  or rubric FROM the evidence it is handed. GPT never decides a count, a winner,
  or who said what. So a ground truth can't be wrong because the model did bad
  arithmetic or hallucinated an attribution.

ANSWER FORMATS (so the eval harness can score each subtype correctly):
  mcq_single  -> ground_truth is one choice string; exact match.
  mcq_multi   -> ground_truth is a list (subset of choices); set match.
  numeric     -> ground_truth is an int; compare with `tolerance`.
  fuzzy_match -> ground_truth + accept_variants; normalized substring/fuzzy match.
  free_form   -> judge_rubric; score with an LLM judge.

CROSS-MODAL AXIS (audio_identity.semantic):
  Enrollment = MELD train. Probe = a dev clip (dev_splits_complete/dia{d}_utt{u}.*,
  extension-agnostic). The system must (1) identify the voice against enrolled
  speakers, then (2) answer what that speaker said about a topic/person mined
  from their ENROLLED (train) utterances. The question names the topic but never
  the speaker.

USAGE
  pip install openai pandas
  export OPENAI_API_KEY=sk-...
  python generate_meld_benchmark_v4.py \
      --meld-root ../../../datasets/MELD.Raw \
      --out ./new_bench_with_audio.json \
      --n-per-type 2 \
      --model gpt-4o \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai")


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
MAIN_SPEAKERS = ["Joey", "Ross", "Rachel", "Phoebe", "Monica", "Chandler"]
EMOTIONS_7 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NEG_EMOTIONS = {"anger", "disgust", "fear", "sadness"}
CROSSCALL_CHOICES = ["Yes, same speaker", "No, different speakers"]
DEFAULT_MODEL = "gpt-4o"
OVERSAMPLE = 4

# Rotated per question so wording is diverse within a subtype. These describe
# HOW to phrase the wrapper, never what to ask — the intent/constraints are
# fixed by each subtype's prompt; only tone/structure varies.
VARIATION_STYLES = [
    "plain and direct",
    "lead with the key detail, then ask",
    "slightly conversational and casual",
    "concise and clipped",
    "formal and precise",
    "framed as an instruction (e.g. 'Identify…', 'Name…', 'Tell me…')",
    "inverted word order (put the object first)",
    "set the scene in a short clause, then pose the question",
    "neutral, journalistic register",
    "curious / inquisitive tone",
    "second-person address ('Looking at the conversations, …')",
    "minimal — the shortest natural form",
]

POSITIVE_CUES = (
    r"\b(?:great|nice|wonderful|love|perfect|fantastic|amazing|awesome|"
    r"congratulations|thank|thanks|happy|glad|good|excited|fun|terrific|brilliant)\b"
)

TOPIC_BLOCKLIST = {
    "Yeah", "Okay", "Well", "Look", "Listen", "Please", "Hey", "Damn",
    "There", "They", "Nobody", "What", "When", "Where", "Why", "Count",
    "This", "That", "These", "Those", "Stop", "Just", "Like", "Right",
    "About", "After", "Before", "Some", "Other", "Sure", "Going", "Coming",
    "Really", "Actually", "Maybe", "Wait", "Sorry", "Thank", "Thanks",
    "Hello", "Anyway", "Wanna", "Gonna", "Have", "Here", "Then", "Even",
    "From", "Such", "Once", "Yes", "Now", "Oh", "And", "But", "God",
    "Okay", "Hmm", "Uh", "Ahh", "Ohh", "Guys", "Guy", "Man", "Mister",
}
# main-character surnames so topics about a main friend don't masquerade as
# generic entities (they belong in graph/identity, not semantic topics).
SURNAME_BLOCKLIST = {"Tribbiani", "Geller", "Green", "Bing", "Buffay"}


# -------------------------------------------------------------------------
# Evidence helpers + dev-clip indexing
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


_DEV_CLIP_RE = re.compile(r"dia(\d+)_utt(\d+)\.", re.IGNORECASE)


def index_dev_clips(meld_root: str):
    """Map (dialogue_id, utterance_id) -> abs path under dev_splits_complete.
    Recursive + extension-agnostic (.wav/.mp4/.mkv/...)."""
    dev_dir = os.path.join(meld_root, "dev_splits_complete")
    index: Dict[tuple, str] = {}
    if not os.path.isdir(dev_dir):
        return dev_dir, index, False
    for dirpath, _dirs, files in os.walk(dev_dir):
        for fn in files:
            m = _DEV_CLIP_RE.match(fn)
            if m:
                index[(int(m.group(1)), int(m.group(2)))] = os.path.join(dirpath, fn)
    return dev_dir, index, True


# -------------------------------------------------------------------------
# Precomputed context (entity index + emotion counts)
# -------------------------------------------------------------------------
def build_context(train_df, dev_df, meld_root) -> SimpleNamespace:
    main_df = train_df[train_df.Speaker.isin(MAIN_SPEAKERS)]

    # entity token -> {speaker: [evidence rows]} over enrolled (train) main speakers
    ent: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in main_df.itertuples(index=False):
        for tok in set(re.findall(r"\b[A-Z][a-z]{2,}\b", r.Utterance)):
            if tok in MAIN_SPEAKERS or tok in TOPIC_BLOCKLIST or tok in SURNAME_BLOCKLIST:
                continue
            ent[tok][r.Speaker].append({
                "dialogue_id": int(r.Dialogue_ID),
                "utterance_id": int(r.Utterance_ID),
                "speaker": r.Speaker,
                "emotion": r.Emotion,
                "sentiment": r.Sentiment,
                "utterance": r.Utterance,
            })

    # emotion counts (speaker, emotion) -> int
    counts: Dict[tuple, int] = {}
    for (spk, emo), g in main_df.groupby(["Speaker", "Emotion"]):
        counts[(spk, emo)] = int(len(g))

    dev_dir, clip_index, dir_exists = index_dev_clips(meld_root)

    return SimpleNamespace(
        df=train_df, main_df=main_df, dev_df=dev_df, meld_root=meld_root,
        ent=ent, counts=counts,
        clip_index=clip_index, dev_dir=dev_dir, dev_dir_exists=dir_exists,
    )


def emo_ranking(ctx, emotion: str) -> List[tuple]:
    """[(speaker, count), ...] desc for a given emotion over main speakers."""
    pairs = [(s, ctx.counts.get((s, emotion), 0)) for s in MAIN_SPEAKERS]
    return sorted(pairs, key=lambda p: p[1], reverse=True)


def _entity_evidence(spkmap: Dict[str, List[Dict]], per_speaker: int = 2) -> List[Dict]:
    out: List[Dict] = []
    for spk, rows in spkmap.items():
        out.extend(rows[:per_speaker])
    return out


# =========================================================================
# SAMPLERS — each returns candidates carrying a `_payload` (sent to GPT),
# `_evidence`, and any deterministic fields the finisher needs.
# =========================================================================

# ---- identity -----------------------------------------------------------
def s_identity_direct(ctx, k, rng):
    pool = ctx.main_df[ctx.main_df.Utterance.str.len().between(40, 140)]
    sample = pool.sample(min(k, len(pool)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in sample.iterrows():
        out.append({
            "speaker": r.Speaker,
            "utterance": r.Utterance,
            "_payload": {"utterance": r.Utterance, "true_speaker": r.Speaker},
            "_evidence": [_row_to_evidence(r)],
            "_tests": "Speaker NER on the question + semantic match of the quoted "
                      "text against the speaker-filtered Chroma collection.",
        })
    return out


def s_identity_pair(ctx, k, rng):
    items = [(e, m) for e, m in ctx.ent.items() if len(m) == 2]
    rng.shuffle(items)
    out = []
    for ent_name, spkmap in items[:k]:
        speakers = sorted(spkmap.keys())
        out.append({
            "entity": ent_name,
            "ground_truth": speakers,
            "_payload": {"entity": ent_name, "n_speakers": 2},
            "_evidence": _entity_evidence(spkmap, per_speaker=2),
            "_tests": f"Multi-speaker semantic retrieval ($in speaker filter). Only "
                      f"{speakers} mention '{ent_name}'; system must not fabricate others.",
        })
    return out


def s_identity_cross_call(ctx, k, rng):
    pool = ctx.main_df[ctx.main_df.Utterance.str.len().between(40, 160)]
    by_spk = {s: pool[pool.Speaker == s] for s in MAIN_SPEAKERS}
    out, want_yes, attempts = [], True, 0
    while len(out) < k and attempts < k * 40:
        attempts += 1
        if want_yes:
            spk = rng.choice(MAIN_SPEAKERS)
            rows = by_spk[spk]
            dids = list(rows.Dialogue_ID.unique())
            if len(dids) < 2:
                continue
            d1, d2 = rng.sample(dids, 2)
            r1 = rows[rows.Dialogue_ID == d1].sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            r2 = rows[rows.Dialogue_ID == d2].sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            gt = "Yes, same speaker"
        else:
            s1, s2 = rng.sample(MAIN_SPEAKERS, 2)
            if by_spk[s1].empty or by_spk[s2].empty:
                continue
            r1 = by_spk[s1].sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            r2 = by_spk[s2].sample(1, random_state=rng.randint(0, 1 << 30)).iloc[0]
            gt = "No, different speakers"
        out.append({
            "ground_truth": gt,
            "quote_1": r1.Utterance, "quote_2": r2.Utterance,
            "_payload": {"quote_1": r1.Utterance, "quote_2": r2.Utterance},
            "_evidence": [_row_to_evidence(r1), _row_to_evidence(r2)],
            "_tests": "Cross-call speaker linking — union-find clustering of identity "
                      "across separately enrolled conversations.",
        })
        want_yes = not want_yes
    return out


# ---- emotion ------------------------------------------------------------
def s_emotion_direct(ctx, k, rng):
    leak = (r"\b(?:angry|anger|sad|sadness|happy|joyful|afraid|fear|fearful|"
            r"disgust|disgusted|surprised|surprise)\b")
    pool = ctx.main_df[
        (ctx.main_df.Emotion != "neutral")
        & ctx.main_df.Utterance.str.len().between(35, 150)
        & ~ctx.main_df.Utterance.str.contains(leak, regex=True, case=False, na=False)
    ]
    sample = pool.sample(min(k, len(pool)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in sample.iterrows():
        out.append({
            "ground_truth": r.Emotion,
            "utterance": r.Utterance,
            "_payload": {"utterance": r.Utterance, "speaker": r.Speaker,
                         "true_emotion": r.Emotion},
            "_evidence": [_row_to_evidence(r)],
            "_tests": "Retrieval of acoustic_emotion / text_emotion metadata for the "
                      "matching segment.",
        })
    return out


def s_emotion_sarcasm(ctx, k, rng):
    pool = ctx.main_df[
        ctx.main_df.Emotion.isin(["anger", "sadness", "disgust"])
        & ctx.main_df.Utterance.str.contains(POSITIVE_CUES, case=False, regex=True, na=False)
        & ctx.main_df.Utterance.str.len().between(20, 160)
    ]
    sample = pool.sample(min(k, len(pool)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in sample.iterrows():
        out.append({
            "true_emotion": r.Emotion,
            "utterance": r.Utterance,
            "_payload": {"utterance": r.Utterance, "speaker": r.Speaker,
                         "true_emotion": r.Emotion},
            "_evidence": [_row_to_evidence(r)],
            "_tests": "Acoustic vs text emotion DISAGREEMENT — words read positive, "
                      f"annotated emotion is '{r.Emotion}'. Only an audio-grounded "
                      "system has the acoustic channel to catch it.",
        })
    return out


def s_emotion_trajectory(ctx, k, rng):
    dids = list(ctx.df.Dialogue_ID.unique())
    rng.shuffle(dids)
    out = []
    for did in dids:
        dlg = ctx.df[ctx.df.Dialogue_ID == did].sort_values("Utterance_ID")
        for spk in MAIN_SPEAKERS:
            turns = dlg[dlg.Speaker == spk]
            emos = list(turns.Emotion)
            if len(turns) >= 3 and len(set(emos)) >= 2 and any(e != "neutral" for e in emos):
                ev = [_row_to_evidence(r) for _, r in turns.iterrows()]
                # Anchor = the speaker's longest turn in this dialogue: the most
                # distinctive, retrievable line so the system can localise WHICH
                # conversation is meant (the questions otherwise say only "this
                # conversation", which is unanswerable — see eval).
                anchor = max(ev, key=lambda e: len(e["utterance"]))
                out.append({
                    "speaker": spk,
                    "anchor_utterance": anchor["utterance"],
                    "_payload": {"speaker": spk,
                                 "anchor_utterance": anchor["utterance"],
                                 "turns": [
                                     {"utterance_id": e["utterance_id"],
                                      "emotion": e["emotion"],
                                      "utterance": e["utterance"]} for e in ev]},
                    "_evidence": ev,
                    "_tests": "Multi-utterance retrieval ordered by utterance_id within "
                              "ONE dialogue (localised by the anchor line) + per-turn "
                              "emotion metadata.",
                })
                break
        if len(out) >= k:
            break
    return out


# ---- semantic_recall ----------------------------------------------------
def _per_speaker_entities(ctx, min_hits, min_dialogues):
    """Yield (speaker, entity, rows) where one speaker discusses an entity enough."""
    items = list(ctx.ent.items())
    for ent_name, spkmap in items:
        for spk, rows in spkmap.items():
            dids = {r["dialogue_id"] for r in rows}
            if len(rows) >= min_hits and len(dids) >= min_dialogues:
                yield spk, ent_name, rows


def s_semantic_open_summary(ctx, k, rng):
    cands = list(_per_speaker_entities(ctx, min_hits=3, min_dialogues=2))
    rng.shuffle(cands)
    out = []
    for spk, ent_name, rows in cands[:k]:
        hits = rows[:8]
        out.append({
            "speaker": spk, "entity": ent_name,
            "_payload": {"speaker": spk, "topic": ent_name,
                         "hits": [{"emotion": h["emotion"], "utterance": h["utterance"]}
                                  for h in hits]},
            "_evidence": hits,
            "_tests": f"Filtered semantic search (speaker={spk}, topic={ent_name}) + "
                      f"multi-utterance synthesis across {len({h['dialogue_id'] for h in hits})} dialogues.",
        })
    return out


def s_semantic_quote(ctx, k, rng):
    pool = ctx.main_df[ctx.main_df.Utterance.str.len().between(40, 160)]
    idx = list(pool.index)
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = pool.loc[i]
        distinctive = [t for t in re.findall(r"\b[A-Z][a-z]{2,}\b", r.Utterance)
                       if t not in MAIN_SPEAKERS and t not in TOPIC_BLOCKLIST]
        if not distinctive:
            continue
        out.append({
            "speaker": r.Speaker, "utterance": r.Utterance,
            "_payload": {"speaker": r.Speaker, "utterance": r.Utterance,
                         "distinctive_tokens": distinctive[:4]},
            "_evidence": [_row_to_evidence(r)],
            "_tests": "Precise lexical retrieval — embeddings must rank the source "
                      "utterance #1 and the LLM read the exact missing tokens off it.",
        })
        if len(out) >= k:
            break
    return out


# ---- graph --------------------------------------------------------------
def s_graph_co_speakers(ctx, k, rng):
    dids = list(ctx.df.Dialogue_ID.unique())
    rng.shuffle(dids)
    out = []
    for did in dids:
        dlg = ctx.df[ctx.df.Dialogue_ID == did].sort_values("Utterance_ID")
        speakers = set(dlg.Speaker) & set(MAIN_SPEAKERS)
        if len(speakers) < 3:
            continue
        strong = dlg[
            dlg.Speaker.isin(MAIN_SPEAKERS)
            & dlg.Emotion.isin(["anger", "disgust", "sadness"])
            & (dlg.Utterance.str.len() >= 35)
        ]
        if strong.empty:
            continue
        anchor = strong.iloc[0]
        others = sorted(speakers - {anchor.Speaker})
        if not others:
            continue
        out.append({
            "anchor_speaker": anchor.Speaker,
            "anchor_emotion": anchor.Emotion,
            "anchor_utterance": anchor.Utterance,
            "ground_truth": others,
            "_payload": {"anchor_speaker": anchor.Speaker, "anchor_emotion": anchor.Emotion,
                         "anchor_utterance": anchor.Utterance,
                         "dialogue_excerpt": [f"[{r.Speaker}/{r.Emotion}] {r.Utterance}"
                                              for r in dlg.head(20).itertuples()]},
            "_evidence": [_row_to_evidence(r) for _, r in dlg.iterrows()][:20],
            "_tests": "graph_service co-speaker edges anchored by an emotion-rich "
                      f"utterance. Others present (excl. {anchor.Speaker}): {others}.",
        })
        if len(out) >= k:
            break
    return out


def s_graph_entity_cross(ctx, k, rng):
    items = [(e, m) for e, m in ctx.ent.items() if 3 <= len(m) <= 5]
    rng.shuffle(items)
    out = []
    for ent_name, spkmap in items[:k]:
        speakers = sorted(spkmap.keys())
        missing = sorted(set(MAIN_SPEAKERS) - set(speakers))
        out.append({
            "entity": ent_name, "ground_truth": speakers,
            "_payload": {"entity": ent_name},
            "_evidence": _entity_evidence(spkmap, per_speaker=1),
            "_tests": f"Cross-speaker entity merging — EntityResolver must merge "
                      f"'{ent_name}' mentions from {speakers} into one node. The "
                      f"exclusion of {missing} is the hard part.",
        })
    return out


def s_graph_relationship(ctx, k, rng):
    cands = list(_per_speaker_entities(ctx, min_hits=3, min_dialogues=2))
    rng.shuffle(cands)
    out = []
    for spk, ent_name, rows in cands[:k]:
        hits = rows[:8]
        out.append({
            "speaker": spk, "entity": ent_name,
            "_payload": {"speaker": spk, "entity": ent_name,
                         "hits": [{"emotion": h["emotion"], "utterance": h["utterance"]}
                                  for h in hits]},
            "_evidence": hits,
            "_tests": f"Entity description aggregation — accumulated descriptions of "
                      f"'{ent_name}' across {len(hits)} of {spk}'s utterances let the "
                      f"LLM infer the relationship.",
        })
    return out


# ---- aggregate ----------------------------------------------------------
def s_agg_argmax(ctx, k, rng):
    emotions = ["sadness", "anger", "fear", "disgust", "joy", "surprise"]
    rng.shuffle(emotions)
    out = []
    for emo in emotions[:k]:
        ranking = emo_ranking(ctx, emo)
        if ranking[0][1] == 0 or ranking[0][1] == ranking[1][1]:
            continue
        rank_str = ", ".join(f"{s} {c}" for s, c in ranking)
        out.append({
            "emotion": emo, "ground_truth": ranking[0][0],
            "_payload": {"emotion": emo},
            "_evidence": [{"_note": "Aggregation question — answer is a corpus-wide count."}],
            "_tests": f"Metadata aggregation over the enrolled set. Ranking: {rank_str}.",
        })
    return out


def s_agg_count(ctx, k, rng):
    pairs = [(s, e) for s in MAIN_SPEAKERS for e in EMOTIONS_7
             if ctx.counts.get((s, e), 0) > 0]
    rng.shuffle(pairs)
    out = []
    for spk, emo in pairs[:k]:
        n = ctx.counts[(spk, emo)]
        out.append({
            "speaker": spk, "emotion": emo, "ground_truth": int(n),
            "_payload": {"speaker": spk, "emotion": emo},
            "_evidence": [],
            "_tests": f"Exact integer aggregation over (speaker={spk}, emotion={emo}).",
        })
    return out


def s_agg_pairwise(ctx, k, rng):
    emotions = ["disgust", "anger", "fear", "sadness", "joy", "surprise"]
    rng.shuffle(emotions)
    out = []
    for emo in emotions[:k]:
        ranking = [(s, c) for s, c in emo_ranking(ctx, emo) if c > 0]
        if len(ranking) < 2 or ranking[0][1] == ranking[1][1]:
            continue
        a, b = ranking[0], ranking[1]
        choices = [a[0], b[0]]
        rng.shuffle(choices)
        out.append({
            "emotion": emo, "ground_truth": a[0], "choices": choices,
            "_payload": {"emotion": emo, "choices": choices},
            "_evidence": [],
            "_tests": f"Pairwise count comparison — {a[0]} {a[1]} vs {b[0]} {b[1]}.",
        })
    return out


# ---- audio_identity (cross-modal, semantic) -----------------------------
def s_audio_semantic(ctx, k, rng):
    if not ctx.dev_dir_exists:
        print(f"  [diag] dev folder NOT found: {os.path.abspath(ctx.dev_dir)}", file=sys.stderr)
        return []
    print(f"  [diag] indexed {len(ctx.clip_index)} dev clips under "
          f"{os.path.abspath(ctx.dev_dir)}", file=sys.stderr)
    if not ctx.clip_index:
        return []

    enrolled = set(ctx.df.Speaker) & set(MAIN_SPEAKERS)
    probe_speakers = set(ctx.dev_df.Speaker) & set(MAIN_SPEAKERS) & enrolled

    # per-speaker enrolled topics (>=2 mentions)
    spk_topics = defaultdict(list)
    for ent_name, spkmap in ctx.ent.items():
        for spk, rows in spkmap.items():
            if spk in probe_speakers and len(rows) >= 2:
                spk_topics[spk].append((ent_name, rows))

    out = []
    per_speaker_cap = max(2, k // 3)
    for spk in sorted(probe_speakers):
        topics = spk_topics.get(spk, [])
        if not topics:
            continue
        rng.shuffle(topics)
        dev_rows = ctx.dev_df[
            (ctx.dev_df.Speaker == spk)
            & ctx.dev_df.Utterance.str.len().between(40, 200)
        ]
        idx = list(dev_rows.index)
        rng.shuffle(idx)
        clips = []
        for i in idx:
            r = dev_rows.loc[i]
            d, u = int(r.Dialogue_ID), int(r.Utterance_ID)
            apath = ctx.clip_index.get((d, u))
            if apath:
                clips.append((apath,))
            if len(clips) >= per_speaker_cap:
                break
        for (apath,), (ent_name, rows) in zip(clips, topics):
            hits = rows[:6]
            out.append({
                "speaker": spk, "topic": ent_name,
                "audio_path": os.path.abspath(apath),
                "audio_file": os.path.basename(apath),
                "_payload": {"speaker": spk, "topic": ent_name,
                             "hits": [{"emotion": h["emotion"], "utterance": h["utterance"]}
                                      for h in hits]},
                "_evidence": hits,
                "_tests": "Cross-modal: Titanet voice match (dev probe vs enrolled) THEN "
                          f"speaker-filtered semantic recall about '{ent_name}'.",
            })
    rng.shuffle(out)
    return out[:k]


# =========================================================================
# GPT prompts
# =========================================================================
_JSON = "Output STRICT JSON only — no prose, no code fences."

_SYS_RAW = {
    "identity.direct": f"""You phrase a "Who said X?" question for a speaker-ID test.
Given an utterance and its true speaker, decide if it's a GOOD candidate (specific, non-trivial, does NOT contain the speaker's own name). If good, phrase the question keeping the utterance VERBATIM (vary only the wrapper, e.g. 'Who said: "..."?'). Never name the speaker.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "identity.pair_attribution": f"""You phrase a question asking WHICH of the six main friends referred to a given named entity.
Given the entity and the count (n_speakers), reject if the entity is a generic word rather than a name/recurring topic. If good, phrase e.g. "Across the enrolled conversations, {{n}} of the six main friends referred to 'X'. Which {{n}}?" Do NOT reveal which speakers.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "identity.cross_call_linking": f"""You phrase a SAME-SPEAKER yes/no question.
Given two quotes from separate conversations, phrase: "The speaker who said '<quote_1>' — is that the same speaker who said '<quote_2>'?" Keep both quotes verbatim. Reject only if a quote is too generic to be a meaningful retrieval target. Do NOT state whether they are the same.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "emotion.direct_attribution": f"""You phrase a "what emotion?" question.
Given an utterance + speaker + true_emotion, reject if the utterance literally names the emotion (would leak). If good, phrase e.g. "What emotion was <speaker> expressing when they said '<utterance>'?" Keep the utterance verbatim. Do NOT reveal the emotion.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "emotion.sarcasm_detection": f"""You build a sarcasm/incongruity question for an audio-grounded system.
The utterance's WORDS sound positive but the annotated (acoustic) emotion is true_emotion (negative). You MUST quote the utterance VERBATIM inside the question (the system needs the exact line to retrieve it) — note that the surface words sound positive and ask what the speaker was REALLY feeling. A question without the verbatim utterance is INVALID. Then provide exactly 4 short natural-language choices, one of which corresponds to true_emotion, plus 3 plausible distractors; set ground_truth to the choice text matching true_emotion. ground_truth MUST be one of choices. Reject if the words don't actually read as positive.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>", "choices": ["<4 options>"], "ground_truth": "<one choice>"}}""",

    "emotion.trajectory": f"""You phrase an emotional-arc question and write its rubric.
You are given a speaker, their ordered turns (emotion + utterance) within ONE dialogue, and a distinctive `anchor_utterance` from that same conversation. The question MUST make the specific conversation identifiable: quote the `anchor_utterance` VERBATIM (you may add a brief scene description), then ask how <speaker>'s emotional state shifts across their turns in THAT conversation. A question that says only "this conversation" without the anchor is INVALID. Write ground_truth as a faithful description of the arc drawn ONLY from the given turns (cite a couple short phrases), and a judge_rubric for full/half/zero credit. Reject if emotions are flat (all one label).
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<str>", "judge_rubric": "<str>"}}""",

    "semantic_recall.open_summary": f"""You phrase a summary question and write its rubric.
Given a speaker, a topic, and the speaker's hit utterances about it, phrase "Summarize what <speaker> has said about <topic> across the enrolled conversations." Write ground_truth as a faithful 2-4 sentence summary drawn ONLY from the hits (you may quote a few short phrases), and a judge_rubric. Reject if the topic is generic/noise or the hits don't cohere.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<str>", "judge_rubric": "<str>"}}""",

    "semantic_recall.quote_completion": f"""You build a cloze (fill-in-the-blank) from one utterance.
Given the full utterance, choose ONE distinctive contiguous span (a rare noun/phrase, ideally from distinctive_tokens) and replace it with "___" to make the question, prefixed "Complete this <speaker> quote from one of the enrolled conversations: ". Set ground_truth to the EXACT removed span (verbatim, as it appears) and accept_variants to plausible spellings/casings. The removed span MUST be copied verbatim from the utterance. Reject if no distinctive span exists.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<removed span>", "accept_variants": ["<...>"]}}""",

    "graph.co_speakers": f"""You phrase a co-presence question.
Given an anchor (speaker, emotion, utterance) from a multi-speaker dialogue, phrase "In the conversation where <anchor_speaker> got <emotion> and said/insisted '<short paraphrase or quote>', which other main friends were present?" You MAY name the anchor speaker (they are excluded from the answer). Do NOT list the other speakers. Reject if the anchor moment is too generic to identify.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "graph.entity_cross_speaker": f"""You phrase a cross-speaker entity question.
Given an entity, phrase "Across the enrolled corpus, which of the six main friends have referred to 'X' by name in their own speech?" Do NOT list them. Reject if the entity is a generic word rather than a name.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "graph.relationship_inference": f"""You phrase a relationship question and write its rubric.
Given a speaker, an entity (a person), and the speaker's hit utterances, phrase "Based on the enrolled conversations, what is the relationship between <speaker> and <entity>?" Write ground_truth describing the relationship drawn ONLY from the hits (quote a few short phrases), and a judge_rubric. Reject if the hits don't support any relationship inference or <entity> isn't a person.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<str>", "judge_rubric": "<str>"}}""",

    "aggregate.argmax_speaker": f"""You phrase an argmax question.
Given an emotion, phrase "Across the enrolled corpus, which of the six main friends expresses '<emotion>' most often?" Do NOT reveal the answer.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "aggregate.exact_count": f"""You phrase an exact-count question.
Given a speaker and emotion, phrase "How many times does <speaker> express '<emotion>' across all the enrolled conversations?" Do NOT include any number.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "aggregate.pairwise_comparison": f"""You phrase a pairwise comparison.
Given an emotion and two speaker choices, phrase "Across the enrolled corpus, who expresses '<emotion>' more often?" Do NOT reveal the winner.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>"}}""",

    "audio_identity.semantic": f"""You phrase a CROSS-MODAL question and write its rubric.
At test time the system gets an AUDIO CLIP plus your question and must (1) identify the speaker from the voice, then (2) recall what that speaker said about a topic from enrolled data. You receive the true speaker (for ground truth ONLY), a topic, and the speaker's hit utterances about it. You do NOT get the audio.
RULES: the question MUST refer to "the speaker in the attached audio clip" and MUST NOT name the speaker; it MUST name the topic. Write ground_truth as 'Speaker: <name>. About <topic>: <2-3 sentence summary from the hits>.' and a judge_rubric that scores the identity part and the content part separately. Reject if the topic is generic/noise.
{_JSON} {{"keep": <bool>, "reason": "<short>", "question": "<str>", "ground_truth": "<str>", "judge_rubric": "<str>"}}""",
}

# Prepended to every subtype prompt. Forces wording diversity via the rotating
# `phrasing_style` while keeping each subtype's hard constraints intact. Placed
# FIRST so each prompt's JSON-schema line stays last (models weight the final
# instruction most for output format).
VARY_NOTE = """PHRASING VARIETY (important): the input includes a `phrasing_style`. Render THIS question in that style and reword it naturally, so that across many questions the wording is genuinely diverse — vary the sentence structure, opening, and tone. Any example phrasing shown below is ILLUSTRATIVE ONLY; never reproduce a fixed template across questions.
HARD CONSTRAINTS still override variety: keep any quoted utterance/text VERBATIM, never reveal or hint at the answer, and preserve the exact meaning and what is being asked.
"""

SYS = {key: VARY_NOTE + "\n" + body for key, body in _SYS_RAW.items()}


# =========================================================================
# FINISHERS — merge GPT output with deterministic fields; return extra
# fields dict, or None to reject. The driver adds id/q_type/subtype/
# answer_format/question/evidence/_tests.
# =========================================================================
def _leaks_speaker(question: str, speaker: str) -> bool:
    return re.search(rf"\b{re.escape(speaker)}\b", question or "") is not None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower())


def _quote_present(question: str, utterance: str, window: int = 30) -> bool:
    """True if the question embeds the source utterance verbatim. Robust to
    end-trimming: a contiguous `window`-char normalized chunk of the utterance
    must appear in the question. Guards against the phrasing step paraphrasing
    a quote away (which made some questions unanswerable — see eval)."""
    qn, un = _norm(question), _norm(utterance)
    if not un:
        return False
    if len(un) <= window:
        return un in qn
    step = max(1, window // 2)
    return any(un[i:i + window] in qn for i in range(0, len(un) - window + 1, step))


def f_mcq_speakers_single(cand, g):
    return {"choices": list(MAIN_SPEAKERS), "ground_truth": cand["ground_truth"]}


def f_identity_direct(cand, g):
    if _leaks_speaker(g["question"], cand["speaker"]):
        return None
    if not _quote_present(g["question"], cand["utterance"]):
        return None  # the quoted line must survive the phrasing step verbatim
    return {"choices": list(MAIN_SPEAKERS), "ground_truth": cand["speaker"]}


def f_mcq_speakers_multi(cand, g):
    return {"choices": list(MAIN_SPEAKERS), "ground_truth": cand["ground_truth"]}


def f_cross_call(cand, g):
    # both quotes must be present verbatim, else the question is unanswerable
    if not (_quote_present(g["question"], cand["quote_1"])
            and _quote_present(g["question"], cand["quote_2"])):
        return None
    return {"choices": list(CROSSCALL_CHOICES), "ground_truth": cand["ground_truth"]}


def f_emotion_direct(cand, g):
    if not _quote_present(g["question"], cand["utterance"]):
        return None
    return {"choices": list(EMOTIONS_7), "ground_truth": cand["ground_truth"]}


def f_sarcasm(cand, g):
    choices = g.get("choices")
    gt = g.get("ground_truth")
    if not choices or len(choices) < 2 or gt not in choices:
        return None
    if not _quote_present(g["question"], cand["utterance"]):
        return None  # FIX: reject questions that dropped the verbatim utterance
    return {"choices": choices, "ground_truth": gt}


def f_freeform(cand, g):
    if not g.get("ground_truth") or not g.get("judge_rubric"):
        return None
    return {"ground_truth": g["ground_truth"], "judge_rubric": g["judge_rubric"]}


def f_trajectory(cand, g):
    # free-form arc + the anchor line must be present so the dialogue is identifiable
    if not g.get("ground_truth") or not g.get("judge_rubric"):
        return None
    if not _quote_present(g["question"], cand["anchor_utterance"]):
        return None
    return {"ground_truth": g["ground_truth"], "judge_rubric": g["judge_rubric"]}


def f_quote(cand, g):
    gt = (g.get("ground_truth") or "").strip()
    if not gt:
        return None
    if _norm(gt) not in _norm(cand["utterance"]):
        return None  # span must be verbatim from the source utterance
    if "___" not in (g.get("question") or ""):
        return None
    return {"ground_truth": gt, "accept_variants": g.get("accept_variants", [])}


def f_agg_argmax(cand, g):
    return {"choices": list(MAIN_SPEAKERS), "ground_truth": cand["ground_truth"]}


def f_agg_count(cand, g):
    return {"answer_format_extra": {"tolerance": 0}, "ground_truth": cand["ground_truth"]}


def f_agg_pairwise(cand, g):
    return {"choices": cand["choices"], "ground_truth": cand["ground_truth"]}


def f_audio(cand, g):
    if _leaks_speaker(g["question"], cand["speaker"]):
        return None
    if not g.get("ground_truth") or not g.get("judge_rubric"):
        return None
    return {"ground_truth": g["ground_truth"], "judge_rubric": g["judge_rubric"],
            "audio_path": cand["audio_path"], "audio_file": cand["audio_file"]}


# (q_type, subtype, answer_format, sampler, finisher)
REGISTRY = [
    ("identity", "direct",              "mcq_single", s_identity_direct,     f_identity_direct),
    ("identity", "pair_attribution",    "mcq_multi",  s_identity_pair,       f_mcq_speakers_multi),
    ("identity", "cross_call_linking",  "mcq_single", s_identity_cross_call, f_cross_call),

    ("emotion", "direct_attribution",   "mcq_single", s_emotion_direct,      f_emotion_direct),
    ("emotion", "sarcasm_detection",    "mcq_single", s_emotion_sarcasm,     f_sarcasm),
    ("emotion", "trajectory",           "free_form",  s_emotion_trajectory,  f_trajectory),

    ("semantic_recall", "open_summary",     "free_form",   s_semantic_open_summary, f_freeform),
    ("semantic_recall", "quote_completion", "fuzzy_match", s_semantic_quote,        f_quote),

    ("graph", "co_speakers",            "mcq_multi",  s_graph_co_speakers,   f_mcq_speakers_multi),
    ("graph", "entity_cross_speaker",   "mcq_multi",  s_graph_entity_cross,  f_mcq_speakers_multi),
    ("graph", "relationship_inference", "free_form",  s_graph_relationship,  f_freeform),

    ("aggregate", "argmax_speaker",     "mcq_single", s_agg_argmax,          f_agg_argmax),
    ("aggregate", "exact_count",        "numeric",    s_agg_count,           f_agg_count),
    ("aggregate", "pairwise_comparison","mcq_single", s_agg_pairwise,        f_agg_pairwise),

    ("audio_identity", "semantic",      "free_form",  s_audio_semantic,      f_audio),
]


# =========================================================================
# Driver
# =========================================================================
def call_gpt(client, model, system, payload, temperature=0.9, reasoning_effort="low"):
    """One round-trip. Reasoning models (gpt-5*, o-series) use
    max_completion_tokens + reasoning_effort and reject a custom temperature;
    classic chat models (gpt-4o) use max_tokens + temperature."""
    text = ""
    is_reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
    kwargs = dict(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    if is_reasoning:
        kwargs["max_completion_tokens"] = 1024
        kwargs["reasoning_effort"] = reasoning_effort   # keep wording-only work cheap
    else:
        kwargs["max_tokens"] = 1024
        kwargs["temperature"] = temperature
    try:
        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # SDK/model doesn't accept reasoning_effort — drop it and retry.
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[warn] model non-JSON: {e}\n  raw: {text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[warn] model call failed: {e}", file=sys.stderr)
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\nDESIGN")[0])
    p.add_argument("--meld-root", default="/app/MELD.Raw")
    p.add_argument("--csv", default=None, help="train CSV (default <root>/train_sent_emo.csv)")
    p.add_argument("--dev-csv", default=None, help="dev CSV (default <root>/dev_sent_emo.csv)")
    p.add_argument("--out", default="benchmark_v4.json")
    p.add_argument("--n-per-type", type=int, default=2, help="questions PER SUBTYPE")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="sampling temperature for non-reasoning models (gpt-4o); "
                        "ignored for gpt-5/o-series")
    p.add_argument("--reasoning-effort", default="low",
                   help="reasoning effort for gpt-5/o-series (minimal|low|medium|high)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to q_types or 'q_type.subtype' ids")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY in your environment.")

    train_csv = args.csv or os.path.join(args.meld_root, "train_sent_emo.csv")
    dev_csv = args.dev_csv or os.path.join(args.meld_root, "dev_sent_emo.csv")
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)

    rng = random.Random(args.seed)
    client = OpenAI()
    ctx = build_context(train_df, dev_df, args.meld_root)

    def selected(qt, st):
        if not args.only:
            return True
        return qt in args.only or f"{qt}.{st}" in args.only

    benchmark = []
    n = args.n_per_type
    # one shuffled rotation of styles, advanced across ALL questions so wording
    # stays spread out (not reset per subtype).
    styles = VARIATION_STYLES[:]
    rng.shuffle(styles)
    style_i = 0
    for qt, st, afmt, sampler, finisher in REGISTRY:
        if not selected(qt, st):
            continue
        print(f"\n=== {qt}.{st} ({afmt}, target {n}) ===", file=sys.stderr)
        cands = sampler(ctx, n * OVERSAMPLE, rng)
        print(f"  sampled {len(cands)} candidates", file=sys.stderr)
        kept = 0
        for cand in cands:
            if kept >= n:
                break
            payload = dict(cand["_payload"])
            payload["phrasing_style"] = styles[style_i % len(styles)]
            style_i += 1
            g = call_gpt(client, args.model, SYS[f"{qt}.{st}"], payload,
                         temperature=args.temperature,
                         reasoning_effort=args.reasoning_effort)
            if not g or not g.get("keep") or not g.get("question"):
                continue
            extra = finisher(cand, g)
            if extra is None:
                continue
            rec = {
                "id": f"{qt}.{st}_{kept:04d}",
                "q_type": qt,
                "subtype": st,
                "answer_format": afmt,
                "question": g["question"],
            }
            # numeric tolerance bubbles up from the finisher
            afe = extra.pop("answer_format_extra", None)
            if afe:
                rec.update(afe)
            for key in ("choices", "ground_truth", "judge_rubric",
                        "accept_variants", "audio_path", "audio_file"):
                if key in extra:
                    rec[key] = extra[key]
            rec["_tests"] = cand.get("_tests", "")
            rec["evidence"] = cand.get("_evidence", [])
            benchmark.append(rec)
            kept += 1
            print(f"  [{kept}/{n}] {g['question'][:88]}", file=sys.stderr)
        if kept < n:
            print(f"  [warn] only kept {kept}/{n} for {qt}.{st}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(benchmark)} questions to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()