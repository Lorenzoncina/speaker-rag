"""Diagnostic script to debug the high EER results.

Two hypotheses:
  H1 — Token ID mismatch: _yes_no_ids() resolves to the wrong token IDs
       (e.g. "Yes" vs " Yes" in generation context), so LLR is computed over
       tokens the model never actually emits first.
  H2 — RAG identity mismatch: enrolled MELD-train speakers are not being matched
       correctly to dev-split probes (Titanet cosine search returns wrong name).

Run inside the container:
    python -m tests.debug_eval --fastapi-url http://localhost:8006

Outputs a plain text report — no GPU required for H2, GPU required for H1.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import requests

# ---------------------------------------------------------------------------
# H1: Token ID audit

def audit_token_ids(model_name: str = "Qwen/Qwen2-7B-Instruct") -> None:
    print("\n" + "=" * 60)
    print("H1: Token ID audit")
    print("=" * 60)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"  Could not load tokenizer: {e}")
        return

    for surface in ["Yes", "No", " Yes", " No", "yes", "no", "YES", "NO"]:
        ids = tok.encode(surface, add_special_tokens=False)
        decoded = [tok.decode([i]) for i in ids]
        print(f"  encode({surface!r:8s}) -> ids={ids}  decoded={decoded}")

    # Show the actual first token the model would produce after the chat prompt
    messages = [
        {"role": "system", "content": "You are a precise speaker-verification assistant. Answer only Yes or No."},
        {"role": "user", "content": "Are audio1 and audio2 from the same speaker? Answer Yes or No."},
    ]
    formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n  Chat template tail (last 80 chars): ...{repr(formatted[-80:])}")

    ids_full = tok(formatted, add_special_tokens=False)["input_ids"]
    last_tok = tok.decode([ids_full[-1]])
    print(f"  Last input token: id={ids_full[-1]}  decoded={last_tok!r}")
    print("  => logits[0,-1,:] predicts the token AFTER this last input token.")

    # What token IDs would the model actually start its response with?
    # Typically after <|im_start|>assistant\n the next real token is "Yes" or "No"
    # Let's check both with and without leading space
    import torch
    try:
        from transformers import AutoModelForCausalLM
        print("\n  Loading model to check actual generation (first token)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        # Force the model to generate after a prompt clearly expecting "Yes"
        test_prompt = tok.apply_chat_template(
            [{"role": "system", "content": "Answer only Yes."},
             {"role": "user", "content": "Say Yes."}],
            tokenize=False, add_generation_prompt=True
        )
        enc = tok(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=3, do_sample=False)
        new_tokens = gen[0][enc["input_ids"].shape[1]:]
        decoded_gen = tok.decode(new_tokens, skip_special_tokens=True)
        first_token_id = new_tokens[0].item()
        print(f"  Model generated: {decoded_gen!r}  (first new token id={first_token_id})")
        print(f"  'Yes' token id from encode: {tok.encode('Yes', add_special_tokens=False)}")
        match = first_token_id in tok.encode("Yes", add_special_tokens=False)
        print(f"  ID MATCH: {match}  {'OK' if match else 'BUG — wrong token ID!'}")
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"  Could not run generation check: {e}")


# ---------------------------------------------------------------------------
# H2: RAG identity matching audit

def audit_rag_matching(
    pairs: List[dict],
    fastapi_url: str,
    n_probe: int = 20,
    question: str = "Who is the speaker in this audio?",
) -> None:
    print("\n" + "=" * 60)
    print("H2: RAG identity matching audit")
    print("=" * 60)

    rng = random.Random(0)
    same_pairs = [p for p in pairs if p["label"] == 1]
    diff_pairs = [p for p in pairs if p["label"] == 0]

    sample_same = rng.sample(same_pairs, min(n_probe // 2, len(same_pairs)))
    sample_diff = rng.sample(diff_pairs, min(n_probe // 2, len(diff_pairs)))
    sample = sample_same + sample_diff
    rng.shuffle(sample)

    correct_identity = 0
    total_with_match = 0

    for i, p in enumerate(sample):
        label_str = "SAME" if p["label"] == 1 else "DIFF"
        true_spk1 = p["speaker1"].lower()
        true_spk2 = p["speaker2"].lower()

        try:
            fs_a = _fetch_fact_sheet(p["audio1"], fastapi_url, question)
            fs_b = _fetch_fact_sheet(p["audio2"], fastapi_url, question)
        except Exception as e:
            print(f"  [{i}] {label_str} — API error: {e}")
            continue

        # Extract top matched identity from FactSheet (look for "Matched identity:" line)
        matched_a = _extract_top_match(fs_a)
        matched_b = _extract_top_match(fs_b)

        hit_a = matched_a is not None and true_spk1 in matched_a.lower()
        hit_b = matched_b is not None and true_spk2 in matched_b.lower()

        total_with_match += 1
        if hit_a and hit_b:
            correct_identity += 1

        status_a = "OK" if hit_a else f"WRONG (true={true_spk1}, got={matched_a})"
        status_b = "OK" if hit_b else f"WRONG (true={true_spk2}, got={matched_b})"

        print(f"  [{i:2d}] {label_str}  audio1: {status_a}  |  audio2: {status_b}")

        if i < 4:
            print(f"       --- FactSheet A (first 300 chars) ---")
            print(f"       {fs_a[:300].replace(chr(10), ' | ')}")
            print(f"       --- FactSheet B (first 300 chars) ---")
            print(f"       {fs_b[:300].replace(chr(10), ' | ')}")

    if total_with_match:
        print(f"\n  Identity match accuracy: {correct_identity}/{total_with_match} "
              f"= {correct_identity/total_with_match*100:.1f}%")
        print("  (If this is low, the Titanet retrieval is the bottleneck, not the LLM.)")


def _fetch_fact_sheet(audio_path: str, fastapi_url: str, question: str) -> str:
    with open(audio_path, "rb") as f:
        files = [("audios", (os.path.basename(audio_path), f, "audio/wav"))]
        data = {"question": question}
        r = requests.post(f"{fastapi_url}/rag/retrieve", files=files, data=data, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json().get("fact_sheet", "")


def _extract_top_match(fact_sheet: str) -> str | None:
    # FactSheet format: "Identity: ross" or "| Identity: ross |"
    for line in fact_sheet.split("\n"):
        stripped = line.strip().lstrip("| ").rstrip(" |")
        if stripped.lower().startswith("identity:"):
            return stripped.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# H3: LLR distribution shape from saved results

def audit_llr_distribution(results_dir: str) -> None:
    print("\n" + "=" * 60)
    print("H3: LLR distribution shape from saved results")
    print("=" * 60)
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*.json"))
    if not json_files:
        print(f"  No JSON files found in {results_dir}")
        return

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        scores = np.array(data.get("scores", []))
        labels = np.array(data.get("labels", []))
        if len(scores) == 0:
            continue

        same_scores = scores[labels == 1]
        diff_scores = scores[labels == 0]

        print(f"\n  File: {jf.name}")
        print(f"  EER reported: {data.get('eer', 'N/A')}")
        print(f"  All  — mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"min={scores.min():.4f}  max={scores.max():.4f}")
        if len(same_scores):
            print(f"  Same — mean={same_scores.mean():.4f}  std={same_scores.std():.4f}")
        if len(diff_scores):
            print(f"  Diff — mean={diff_scores.mean():.4f}  std={diff_scores.std():.4f}")

        if len(same_scores) and len(diff_scores):
            direction = "CORRECT" if same_scores.mean() > diff_scores.mean() else "INVERTED (LLR sign bug?)"
            print(f"  Direction: same_mean > diff_mean? {direction}")

        near_zero = np.sum(np.abs(scores) < 0.01)
        print(f"  LLRs near zero (|llr|<0.01): {near_zero}/{len(scores)} "
              f"— if high, model outputs near-uniform Yes/No probs (no signal)")


# ---------------------------------------------------------------------------
# H4: Cosine score distribution for failing speakers

def audit_score_distribution(
    pairs: List[dict],
    fastapi_url: str,
    n_probe: int = 20,
    threshold: float = 0.50,
) -> None:
    """For a sample of pairs, call /admin/probe_similarity and show the actual
    cosine scores between the probe and the correct enrolled speaker vs the
    best enrolled speaker overall.  This reveals whether the threshold is the
    problem (correct speaker at 0.35-0.49) or something deeper (correct speaker
    at < 0.30 or not enrolled at all).
    """
    print("\n" + "=" * 60)
    print("H4: Cosine score distribution (probe vs enrolled)")
    print(f"    threshold = {threshold}")
    print("=" * 60)

    rng = random.Random(1)
    sample = rng.sample(pairs, min(n_probe, len(pairs)))

    correct_scores = []   # score of probe vs its own enrolled speaker
    best_scores = []      # score of probe vs best enrolled speaker overall
    wrong_best_ids = []   # when best != correct, what was the best speaker

    for i, p in enumerate(sample):
        for side, audio_path, true_spk in [
            ("A", p["audio1"], p["speaker1"]),
            ("B", p["audio2"], p["speaker2"]),
        ]:
            try:
                with open(audio_path, "rb") as f:
                    r = requests.post(
                        f"{fastapi_url}/admin/probe_similarity",
                        files=[("audio", (os.path.basename(audio_path), f, "audio/wav"))],
                        params={"top_k": 50},
                        timeout=120,
                    )
                if r.status_code != 200:
                    print(f"  [{i}{side}] API error {r.status_code}: {r.text[:100]}")
                    continue
                matches = r.json().get("matches", [])
            except Exception as e:
                print(f"  [{i}{side}] request failed: {e}")
                continue

            if not matches:
                print(f"  [{i}{side}] no enrolled speakers in DB")
                continue

            # Best overall match
            best = matches[0]
            best_score = best["score"]
            best_id = best["speaker_id"]
            best_scores.append(best_score)

            # Best match for the correct speaker
            correct_matches = [m for m in matches if m["speaker_id"] == true_spk]
            if correct_matches:
                correct_score = max(m["score"] for m in correct_matches)
                correct_scores.append(correct_score)
                above = correct_score >= threshold
                flag = "OK " if above else f"BELOW({correct_score:.3f}<{threshold})"
            else:
                correct_score = None
                flag = "NOT_ENROLLED"

            wrong = (best_id != true_spk)
            if wrong and correct_score is not None:
                wrong_best_ids.append((true_spk, best_id, best_score, correct_score))

            print(
                f"  [{i:2d}{side}] true={true_spk}  "
                f"correct_score={correct_score if correct_score is not None else 'N/A':>6}  "
                f"best={best_id}({best_score:.3f})  {flag}"
                + ("  *** WRONG BEST ***" if wrong else "")
            )

    print()
    if correct_scores:
        arr = np.array(correct_scores)
        print(f"  Correct-speaker scores: n={len(arr)}  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        below = (arr < threshold).sum()
        print(f"  Below threshold ({threshold}): {below}/{len(arr)} = {below/len(arr)*100:.1f}%")
        pct10 = np.percentile(arr, 10)
        pct25 = np.percentile(arr, 25)
        print(f"  10th pct={pct10:.3f}  25th pct={pct25:.3f}")
        print(f"  => If most below-threshold scores are in [0.35, 0.50), "
              f"lowering threshold will help.")
        print(f"  => If most are < 0.35, threshold won't help — deeper Titanet issue.")

    if best_scores:
        barr = np.array(best_scores)
        print(f"  Best-enrolled scores:   n={len(barr)}  mean={barr.mean():.3f}  "
              f"min={barr.min():.3f}  max={barr.max():.3f}")

    if wrong_best_ids:
        print(f"\n  Cases where best match was a WRONG speaker ({len(wrong_best_ids)}):")
        for true_spk, best_id, best_score, correct_score in wrong_best_ids[:10]:
            print(f"    true={true_spk}  got={best_id}({best_score:.3f})  "
                  f"correct={correct_score:.3f}  "
                  f"margin={best_score - correct_score:.3f}")


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastapi-url", default="http://localhost:8006")
    parser.add_argument("--pairs", default=None,
                        help="Path to pairs JSON. Defaults to meld_dev_pairs_main6.json.")
    parser.add_argument("--mode", choices=["all", "main6"], default="main6",
                        help="MELD only: which default file to use when --pairs is not set.")
    parser.add_argument("--dataset", choices=["meld", "vox1"], default="meld",
                        help="Which benchmark's pairs to audit.")
    parser.add_argument("--results-dir",
                        default=str(Path(__file__).parent / "eval_results"))
    parser.add_argument("--check", choices=["token", "rag", "dist", "scores", "all"], default="all")
    parser.add_argument("--n-probe", type=int, default=20,
                        help="Number of audio pairs to probe for RAG / score audit.")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Similarity threshold used in rag.py (for score audit display).")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    args = parser.parse_args()

    if args.check in ("dist", "all"):
        audit_llr_distribution(args.results_dir)

    if args.check in ("token", "all"):
        audit_token_ids(args.model)

    # Load pairs once (needed for rag + scores checks)
    pairs = None
    if args.check in ("rag", "scores", "all"):
        if args.dataset == "vox1":
            from tests.vox1_pairs import load_pairs, DEFAULT_OUT
            pairs_path = args.pairs or DEFAULT_OUT
            hint = "tests.vox1_pairs"
        else:
            from tests.meld_pairs import load_pairs, DEFAULT_OUT, DEFAULT_OUT_MAIN6
            pairs_path = args.pairs or (DEFAULT_OUT_MAIN6 if args.mode == "main6" else DEFAULT_OUT)
            hint = f"tests.meld_pairs --mode {args.mode}"
        if not Path(pairs_path).exists():
            print(f"Pairs file not found: {pairs_path}. Run {hint} first.")
            sys.exit(1)
        pairs = load_pairs(pairs_path)
        print(f"  Using pairs file: {pairs_path} ({len(pairs)} pairs)")

    if args.check in ("rag", "all"):
        audit_rag_matching(pairs, args.fastapi_url, n_probe=args.n_probe)

    if args.check in ("scores",):
        audit_score_distribution(pairs, args.fastapi_url,
                                 n_probe=args.n_probe, threshold=args.threshold)


if __name__ == "__main__":
    main()
