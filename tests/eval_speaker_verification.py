"""Speaker verification evaluation on MELD dev pairs.

Three experimental conditions, all scored via LLR(Yes/No) → EER:

  1. Direct Qwen2-Audio-7B-Instruct: both audios fed to the audio LLM.
  2. RAG-without-enrollment + text LLM: each audio is sent through the RAG
     /rag/retrieve endpoint (DB must be empty); the two FactSheets are
     concatenated and shown to a text-only LLM that emits Yes/No.
  3. RAG-with-MELD-train + text LLM: same as (2) but with MELD train enrolled.

This module is importable (the Gradio UI calls these functions) and also
runnable as a CLI for full benchmark runs.
"""
from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
from sklearn.metrics import roc_curve

from .meld_pairs import DEFAULT_OUT as DEFAULT_PAIRS_PATH, build_pairs, load_pairs


# ---------------------------------------------------------------------------
# Prompts

QWEN_AUDIO_PROMPT = (
    "These are two distinct audios. First, think about the elements that "
    "characterize each speaker, such as their gender, accent, tone, prosody, "
    "and speech rate. Give the characteristics for each audio. Then, from "
    "those characteristics, infer the likelihood that both speakers are the "
    "same. Answer by Yes or No."
)

TEXT_VERIFICATION_TEMPLATE = (
    "You are given speaker-identification information for two audio recordings, "
    "produced by an upstream speaker-recognition system.\n\n"
    "=== Audio 1 ===\n{fs_a}\n\n"
    "=== Audio 2 ===\n{fs_b}\n\n"
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
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise RuntimeError("Tokenizer produced no tokens for 'Yes'/'No'.")
    return yes_ids[0], no_ids[0]


# ---------------------------------------------------------------------------
# Experiment 1: direct Qwen2-Audio scoring

def _llr_qwen_audio(audio_model, processor, audio1: str, audio2: str) -> float:
    import librosa

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio1},
            {"type": "audio", "audio_url": audio2},
            {"type": "text", "text": QWEN_AUDIO_PROMPT},
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    sr = processor.feature_extractor.sampling_rate
    audios = [librosa.load(p, sr=sr)[0] for p in (audio1, audio2)]
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = {k: v.to(audio_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = audio_model(**inputs)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    yes_id, no_id = _yes_no_ids(processor.tokenizer)
    return float(log_probs[yes_id] - log_probs[no_id])


def run_experiment_direct_audio(
    pairs: List[dict],
    audio_model,
    processor,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    for i, p in enumerate(pairs):
        try:
            llr = _llr_qwen_audio(audio_model, processor, p["audio1"], p["audio2"])
            scores.append(llr)
            labels.append(p["label"])
        except Exception as e:
            print(f"[exp1] pair {i} failed: {e}")
        if progress_callback:
            progress_callback(i + 1, len(pairs))
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
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    yes_id, no_id = _yes_no_ids(tokenizer)
    return float(log_probs[yes_id] - log_probs[no_id])


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
    parser.add_argument("--pairs", default=DEFAULT_PAIRS_PATH,
                        help="Path to pairs JSON (will be generated if missing).")
    parser.add_argument("--exp", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--n-pairs", type=int, default=-1,
                        help="Limit pairs (-1 = all).")
    parser.add_argument("--fastapi-url", default="http://localhost:8006")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "eval_results"))
    args = parser.parse_args()

    if not Path(args.pairs).exists():
        print(f"Pairs file {args.pairs} not found — generating with defaults.")
        build_pairs(output_path=args.pairs)
    pairs = load_pairs(args.pairs)
    if args.n_pairs > 0:
        pairs = pairs[: args.n_pairs]
    print(f"Evaluating on {len(pairs)} pairs.")

    if args.exp in ("1", "all"):
        model, processor = _load_qwen_audio()
        scores, labels = run_experiment_direct_audio(pairs, model, processor)
        path, _ = save_results(args.out_dir, "1", scores, labels, "Qwen/Qwen2-Audio-7B-Instruct")
        print(format_summary("1", "Qwen/Qwen2-Audio-7B-Instruct", scores, labels, path))
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.exp in ("2", "3", "all"):
        text_pipe = _load_qwen_text()

        if args.exp in ("2", "all"):
            print("\nExperiment 2: ensure DB is EMPTY (no MELD train enrolled).")
            scores, labels = run_experiment_rag_text(pairs, text_pipe, args.fastapi_url)
            path, _ = save_results(args.out_dir, "2", scores, labels, "Qwen/Qwen2-7B-Instruct",
                                   extra={"rag": "no_enrollment"})
            print(format_summary("2", "Qwen/Qwen2-7B-Instruct + RAG (empty)", scores, labels, path))

        if args.exp in ("3", "all"):
            print("\nExperiment 3: ensure DB has MELD train enrolled.")
            scores, labels = run_experiment_rag_text(pairs, text_pipe, args.fastapi_url)
            path, _ = save_results(args.out_dir, "3", scores, labels, "Qwen/Qwen2-7B-Instruct",
                                   extra={"rag": "meld_train_enrolled"})
            print(format_summary("3", "Qwen/Qwen2-7B-Instruct + RAG (enrolled)", scores, labels, path))


if __name__ == "__main__":
    main()
