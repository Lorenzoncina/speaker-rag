"""Speaker verification evaluation on MELD dev pairs.

Three experimental conditions:

  1. Direct Qwen2-Audio-7B-Instruct: both audios fed to the audio LLM, which
     generates a free-form response and a 0-100 confidence score → EER.
  2. RAG-without-enrollment + text LLM: each audio is sent through the RAG
     /rag/retrieve endpoint (DB must be empty); the two FactSheets are
     concatenated and shown to a text-only LLM that emits Yes/No → LLR → EER.
  3. RAG-with-MELD-train + text LLM: same as (2) but with MELD train enrolled.

This module is importable (the Gradio UI calls these functions) and also
runnable as a CLI for full benchmark runs.
"""
from __future__ import annotations

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
from sklearn.metrics import roc_curve

from .meld_pairs import (DEFAULT_OUT as DEFAULT_PAIRS_PATH, DEFAULT_OUT_MAIN6,
                         build_pairs, load_pairs, MAIN6)
from .vox1_pairs import (DEFAULT_OUT as DEFAULT_VOX1_PAIRS_PATH,
                          build_pairs as build_vox1_pairs,
                          load_pairs as load_vox1_pairs)


# ---------------------------------------------------------------------------
# Prompts

QWEN_AUDIO_CONFIDENCE_PROMPT = (
    "These are two distinct audios. First, think about the elements that "
    "characterize each speaker, such as their gender, accent, tone, prosody, "
    "and speech rate. Give the characteristics for each audio. Then, from "
    "those characteristics, infer the likelihood that both speakers are the "
    "same. Answer by Yes or No, and give a confidence score between 0 and 100: "
    "0 corresponds to the certainty that they are from different speakers, "
    "100 corresponds to the certainty that they are from the same speaker, "
    "and 50 means you are uncertain."
)

TEXT_VERIFICATION_TEMPLATE = (
    "You are given speaker-identification information for two audio recordings, "
    "produced by an upstream speaker-recognition system.\n\n"
    "=== Audio 1 ===\n{fs_a}\n\n"
    "=== Audio 2 ===\n{fs_b}\n\n"
    "Based on the information above, are the two audios from the same speaker? "
    "Answer with a single word: Yes or No."
)


# Pair-mode template: the backend returns a single FactSheet that already
# contains both audios as labelled tracks ("--- Audio Track 1 (File: ...) ---",
# "--- Audio Track 2 (File: ...) ---"), so we paste it in as-is.
TEXT_VERIFICATION_PAIR_TEMPLATE = (
    "You are given speaker-identification information for two audio recordings, "
    "produced by an upstream speaker-recognition system. The fact sheet below "
    "contains both audios as 'Audio Track 1' and 'Audio Track 2'.\n\n"
    "{combined_fact_sheet}\n\n"
    "Based on the information above, are the two audios from the same speaker? "
    "Answer with a single word: Yes or No."
)


# ---------------------------------------------------------------------------
# EER

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER from continuous scores and binary labels (1 = same speaker)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ---------------------------------------------------------------------------
# Yes/No token resolution

def _yes_no_ids(tokenizer) -> Tuple[int, int]:
    # After apply_chat_template with add_generation_prompt=True the model begins
    # its response directly (no leading space). Qwen2 tokenises "Yes" and "No"
    # as single tokens without a leading-space prefix in that position.
    # We try both surface forms and take whichever is a single token so we don't
    # silently read the wrong ID when the tokeniser splits unexpectedly.
    def _single_token_id(surfaces):
        for s in surfaces:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # fallback: first token of the first surface
        return tokenizer.encode(surfaces[0], add_special_tokens=False)[0]

    yes_id = _single_token_id(["Yes", " Yes"])
    no_id = _single_token_id(["No", " No"])
    return yes_id, no_id


# ---------------------------------------------------------------------------
# Experiment 1: direct Qwen2-Audio scoring (confidence-score protocol)

def _parse_confidence(text: str) -> Optional[float]:
    """Extract the 0-100 confidence integer from model-generated text.

    Search order:
      1. First integer in [0,100] that follows a Yes/No token.
      2. First integer in [0,100] after a 'confidence'/'score'/'certainty' keyword.
      3. Last standalone integer in [0,100] anywhere in the response.
    Returns None when no valid integer is found (parse failure).
    """
    m = re.search(r'\b(?:yes|no)\b[^0-9]*?(\d{1,3})', text, re.IGNORECASE | re.DOTALL)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return float(val)

    m = re.search(r'(?:confidence|score|certainty)[^\d]*?(\d{1,3})', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return float(val)

    for tok in reversed(re.findall(r'\b(\d{1,3})\b', text)):
        val = int(tok)
        if 0 <= val <= 100:
            return float(val)

    return None


def _confidence_qwen_audio(
    audio_model, processor, audio1: str, audio2: str
) -> Tuple[Optional[float], str]:
    """Generate a free-form response and parse a 0-100 confidence score from it."""
    import librosa

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio1},
            {"type": "audio", "audio_url": audio2},
            {"type": "text", "text": QWEN_AUDIO_CONFIDENCE_PROMPT},
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    sr = processor.feature_extractor.sampling_rate
    audios = [librosa.load(p, sr=sr)[0] for p in (audio1, audio2)]
    inputs = processor(text=text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(audio_model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = audio_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response = processor.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    return _parse_confidence(response), response


def run_experiment_direct_audio(
    pairs: List[dict],
    audio_model,
    processor,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    failures = 0
    for i, p in enumerate(pairs):
        try:
            score, response = _confidence_qwen_audio(audio_model, processor, p["audio1"], p["audio2"])
            if score is None:
                print(f"[exp1] pair {i}: parse failure — {response[:120]!r}")
                failures += 1
            else:
                scores.append(score)
                labels.append(p["label"])
        except Exception as e:
            print(f"[exp1] pair {i} exception: {e}")
            failures += 1
        if progress_callback:
            progress_callback(i + 1, len(pairs))
    total = len(pairs)
    print(f"[exp1] failures: {failures}/{total} ({100 * failures / total:.1f}%)")
    return np.array(scores), np.array(labels)


# ---------------------------------------------------------------------------
# Experiments 2 & 3: RAG FactSheets + text LLM

def _get_fact_sheet(audio_path: str, fastapi_url: str, question: str) -> str:
    with open(audio_path, "rb") as f:
        files = [("audios", (os.path.basename(audio_path), f, "audio/wav"))]
        data = {"question": question}
        r = requests.post(f"{fastapi_url}/rag/retrieve", files=files, data=data, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"FastAPI {r.status_code}: {r.text[:300]}")
    return r.json().get("fact_sheet", "")


def _llr_text_llm(text_pipeline, fact_sheet_a: str, fact_sheet_b: str) -> float:
    tokenizer = text_pipeline.tokenizer
    model = text_pipeline.model

    user_msg = TEXT_VERIFICATION_TEMPLATE.format(fs_a=fact_sheet_a, fs_b=fact_sheet_b)
    messages = [
        {"role": "system", "content": "You are a precise speaker-verification assistant. Answer only Yes or No."},
        {"role": "user", "content": user_msg},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # logits[:, -1, :] = distribution over the *first generated token*
        # (last position of the input predicts what comes next)
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    yes_id, no_id = _yes_no_ids(tokenizer)
    llr = float(log_probs[yes_id] - log_probs[no_id])

    # Sanity: log top-5 tokens once during first call to catch ID issues early
    if not hasattr(_llr_text_llm, "_logged"):
        _llr_text_llm._logged = True
        top5 = torch.topk(log_probs, 5)
        top5_decoded = [(tokenizer.decode([i]), round(v, 3))
                        for i, v in zip(top5.indices.tolist(), top5.values.tolist())]
        print(f"[debug] first-token top-5: {top5_decoded}  yes_id={yes_id} no_id={no_id}")

    return llr


def run_experiment_rag_text(
    pairs: List[dict],
    text_pipeline,
    fastapi_url: str,
    rag_question: str = "Who is the speaker in this audio?",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    fs_cache: dict = {}

    def get_fs(path: str) -> str:
        if path not in fs_cache:
            fs_cache[path] = _get_fact_sheet(path, fastapi_url, rag_question)
        return fs_cache[path]

    scores, labels = [], []
    for i, p in enumerate(pairs):
        try:
            fs_a = get_fs(p["audio1"])
            fs_b = get_fs(p["audio2"])
            llr = _llr_text_llm(text_pipeline, fs_a, fs_b)
            scores.append(llr)
            labels.append(p["label"])
        except Exception as e:
            print(f"[exp-rag] pair {i} failed: {e}")
        if progress_callback:
            progress_callback(i + 1, len(pairs))
    return np.array(scores), np.array(labels)


# ---------------------------------------------------------------------------
# Pair-mode variant: both audios in one /rag/retrieve call.
#
# When both probes share a session, the backend's intra-call union-find merges
# them if their TitaNet cosine ≥ 0.45, so same-speaker unenrolled pairs collapse
# to one Unknown_Speaker_<X> identity instead of two random hashes. This trades
# the FactSheet cache for a meaningful no-enrollment baseline; runtime ≈ 8×
# the cached variant because each pair = one retrieval call.

def _get_fact_sheet_pair(audio_a: str, audio_b: str, fastapi_url: str,
                          question: str) -> str:
    """Single /rag/retrieve call for both audios in a pair.

    Returns the combined FactSheet string the backend assembles, which already
    contains both audios as 'Audio Track 1' and 'Audio Track 2' sections —
    suitable for the pair-mode prompt template.
    """
    with open(audio_a, "rb") as fa, open(audio_b, "rb") as fb:
        files = [
            ("audios", (os.path.basename(audio_a), fa, "audio/wav")),
            ("audios", (os.path.basename(audio_b), fb, "audio/wav")),
        ]
        data = {"question": question}
        r = requests.post(f"{fastapi_url}/rag/retrieve", files=files,
                          data=data, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"FastAPI {r.status_code}: {r.text[:300]}")
    return r.json().get("fact_sheet", "")


def _split_pair_fact_sheet(combined: str) -> Tuple[str, str]:
    """Slice a pair-mode combined FactSheet into per-track sections.

    The combined FactSheet has the shape:

        === FACT SHEET: AUDIO FILE RESOLUTION ===
        --- Audio Track 1 (File: ...) ---
        Speaker Profile: ...
        --- Audio Track 2 (File: ...) ---
        Speaker Profile: ...

        === SEMANTIC SEARCH ... ===   ← mixes both speakers' utterances
        === KNOWLEDGE GRAPH ... ===   ← mixes both speakers' facts

    The tail sections are filtered on the union of identified users
    (rag.py:660), so they look identical for same-speaker and different-speaker
    pairs and effectively mask the `Identity:` signal. We drop them and keep
    only the two per-track Speaker Profile blocks.

    Returns (track1, track2). On unexpected layouts, returns the original
    string as track1 and an empty string as track2 (the LLM will still see
    something and the call won't crash).
    """
    t1_start = combined.find("--- Audio Track 1")
    t2_start = combined.find("--- Audio Track 2")
    if t1_start == -1 or t2_start == -1:
        return combined.strip(), ""

    track1 = combined[t1_start:t2_start].rstrip()

    # Track 2 runs from its header to the first trailing "=== " section header
    # (semantic / KG), or to end-of-string if there isn't one.
    rest = combined[t2_start:]
    tail = rest.find("\n=== ", 1)  # skip pos 0 so the "--- Audio Track 2" header itself isn't matched
    track2 = (rest[:tail] if tail != -1 else rest).rstrip()
    return track1, track2


def _llr_text_llm_pair(text_pipeline, combined_fact_sheet: str) -> float:
    """Yes/No → LLR for the pair-mode prompt (single combined FactSheet).

    Reuses the non-pair-mode template (`TEXT_VERIFICATION_TEMPLATE`) with the
    two per-track sections extracted from the combined FactSheet. The mixed
    semantic / KG tail of the combined FactSheet was masking the Identity
    signal — see `_split_pair_fact_sheet` for the rationale.
    """
    tokenizer = text_pipeline.tokenizer
    model = text_pipeline.model

    fs_a, fs_b = _split_pair_fact_sheet(combined_fact_sheet)
    user_msg = TEXT_VERIFICATION_TEMPLATE.format(fs_a=fs_a, fs_b=fs_b)
    messages = [
        {"role": "system",
         "content": "You are a precise speaker-verification assistant. Answer only Yes or No."},
        {"role": "user", "content": user_msg},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    yes_id, no_id = _yes_no_ids(tokenizer)
    llr = float(log_probs[yes_id] - log_probs[no_id])

    if not hasattr(_llr_text_llm_pair, "_logged"):
        _llr_text_llm_pair._logged = True
        top5 = torch.topk(log_probs, 5)
        top5_decoded = [(tokenizer.decode([i]), round(v, 3))
                        for i, v in zip(top5.indices.tolist(), top5.values.tolist())]
        print(f"[debug pair] first-token top-5: {top5_decoded}  yes_id={yes_id} no_id={no_id}")

    return llr


def run_experiment_rag_text_pair(
    pairs: List[dict],
    text_pipeline,
    fastapi_url: str,
    rag_question: str = "Who is the speaker in this audio?",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair-mode RAG experiment: one /rag/retrieve call per pair."""
    scores, labels = [], []
    for i, p in enumerate(pairs):
        try:
            combined = _get_fact_sheet_pair(
                p["audio1"], p["audio2"], fastapi_url, rag_question
            )
            llr = _llr_text_llm_pair(text_pipeline, combined)
            scores.append(llr)
            labels.append(p["label"])
        except Exception as e:
            print(f"[exp-rag-pair] pair {i} failed: {e}")
        if progress_callback:
            progress_callback(i + 1, len(pairs))
    return np.array(scores), np.array(labels)


# ---------------------------------------------------------------------------
# Result persistence

def save_results(out_dir: str, experiment: str, scores: np.ndarray, labels: np.ndarray,
                 model_name: str, extra: Optional[dict] = None) -> Tuple[str, float]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    eer = compute_eer(scores, labels)
    payload = {
        "experiment": experiment,
        "model": model_name,
        "n_pairs": int(len(scores)),
        "eer": eer,
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    fname = f"exp{experiment}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    fpath = str(Path(out_dir) / fname)
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    return fpath, eer


def format_summary(experiment: str, model: str, scores: np.ndarray, labels: np.ndarray,
                   saved_path: str) -> str:
    eer = compute_eer(scores, labels)
    n_same = int((labels == 1).sum())
    n_diff = int((labels == 0).sum())
    return (
        f"Experiment {experiment}\n"
        f"  Model:       {model}\n"
        f"  Pairs:       {len(scores)} (same={n_same}, diff={n_diff})\n"
        f"  EER:         {eer * 100:.2f}%\n"
        f"  Saved to:    {saved_path}"
    )


# ---------------------------------------------------------------------------
# CLI

def _load_qwen_audio(device: str = "cuda"):
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    print("Loading Qwen/Qwen2-Audio-7B-Instruct ...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, processor


def _load_qwen_text(device: str = "cuda"):
    from transformers import pipeline
    print("Loading Qwen/Qwen2-7B-Instruct ...")
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["meld", "vox1"], default="meld",
                        help="Which benchmark to run.")
    parser.add_argument("--pairs", default=None,
                        help="Path to pairs JSON. If missing, generated automatically.")
    parser.add_argument("--mode", choices=["all", "main6"], default="main6",
                        help="MELD only: 'main6' = Friends main cast only; 'all' = every speaker.")
    parser.add_argument("--exp", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--n-pairs", type=int, default=-1,
                        help="Limit pairs (-1 = all). For Vox1 also controls auto-generated subset size.")
    parser.add_argument("--fastapi-url", default="http://localhost:8006")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "eval_results"))
    args = parser.parse_args()

    if args.dataset == "vox1":
        if args.pairs is None:
            args.pairs = DEFAULT_VOX1_PAIRS_PATH
        if not Path(args.pairs).exists():
            n = args.n_pairs if args.n_pairs > 0 else 200
            print(f"Pairs file {args.pairs} not found — generating Vox1-O subset (n={n}).")
            build_vox1_pairs(n_pairs=n, output_path=args.pairs)
        pairs = load_vox1_pairs(args.pairs)
    else:
        if args.pairs is None:
            args.pairs = DEFAULT_OUT_MAIN6 if args.mode == "main6" else DEFAULT_PAIRS_PATH
        if not Path(args.pairs).exists():
            print(f"Pairs file {args.pairs} not found — generating MELD (mode={args.mode}).")
            build_pairs(output_path=args.pairs, mode=args.mode)
        pairs = load_pairs(args.pairs)

    if args.n_pairs > 0:
        pairs = pairs[: args.n_pairs]
    print(f"Evaluating on {len(pairs)} pairs (dataset={args.dataset}).")

    if args.exp in ("1", "all"):
        model, processor = _load_qwen_audio()
        scores, labels = run_experiment_direct_audio(pairs, model, processor)
        path, _ = save_results(args.out_dir, "1", scores, labels, "Qwen/Qwen2-Audio-7B-Instruct",
                               extra={"dataset": args.dataset})
        print(format_summary("1", "Qwen/Qwen2-Audio-7B-Instruct", scores, labels, path))
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.exp in ("2", "3", "all"):
        text_pipe = _load_qwen_text()

        if args.exp in ("2", "all"):
            print(f"\nExperiment 2: ensure DB is EMPTY (no enrollment) for dataset={args.dataset}.")
            scores, labels = run_experiment_rag_text(pairs, text_pipe, args.fastapi_url)
            path, _ = save_results(args.out_dir, "2", scores, labels, "Qwen/Qwen2-7B-Instruct",
                                   extra={"dataset": args.dataset, "rag": "no_enrollment"})
            print(format_summary("2", "Qwen/Qwen2-7B-Instruct + RAG (empty)", scores, labels, path))

        if args.exp in ("3", "all"):
            train_label = "MELD train" if args.dataset == "meld" else "VoxCeleb1 test speakers (Vox1-O)"
            print(f"\nExperiment 3: ensure DB has {train_label} enrolled.")
            scores, labels = run_experiment_rag_text(pairs, text_pipe, args.fastapi_url)
            path, _ = save_results(args.out_dir, "3", scores, labels, "Qwen/Qwen2-7B-Instruct",
                                   extra={"dataset": args.dataset, "rag": f"{args.dataset}_enrolled"})
            print(format_summary("3", f"Qwen/Qwen2-7B-Instruct + RAG ({train_label} enrolled)",
                                  scores, labels, path))


if __name__ == "__main__":
    main()
