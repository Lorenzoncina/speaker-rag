"""CLI runner for the MLC-SLM Task 2 benchmark — LLM-only baseline.

Thin HTTP client. The backend owns the dataset; this script just orchestrates:

  GET  /admin/mlcslm/qa            — fetch the QA records
  GET  /admin/mlcslm/transcript    — read the .txt next to each .wav
  POST /llm/generate               — pick A/B/C/D with Qwen3-14B

No enrollment, no SpeakerRAG. The model sees the reference transcript and
the question. This mirrors the LongMemEval LLM-only design: feed the raw
conversation as text, ask the multi-choice question, parse the letter.

`--qa-jsonl` and `--audio-root` are paths on the **backend** container, not
on the host where this script runs.

Prerequisites:
  - FastAPI server reachable at --base-url.
  - Backend extractor mode is 'local' (Qwen3-14B-AWQ loaded via vLLM).

Usage:
    tmux new -s mlcslm-llm
    python tests/run_mlcslm_llm_only_cli.py \\
        --qa-jsonl /app/mlc-slm-2nd-workshop-data/questions/mlcslm_2nd_dev_qa_300_subset.jsonl \\
        --audio-root /app/mlc-slm-2nd-workshop-data/dev_data \\
        --n -1 \\
        --output tests/mlcslm_results/run_llm_only.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


_LETTER_RE = re.compile(r"\b([ABCD])\b")


_MCQ_PROMPT_TEMPLATE = """You will answer a multiple-choice question about a recorded conversation.

CONVERSATION TRANSCRIPT:
{transcript}

QUESTION: {question_stem}

OPTIONS:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Output ONLY the single letter (A, B, C, or D) of the best option. \
Do NOT explain. Do NOT repeat the option text.
ANSWER:"""


def assert_backend_local(base_url: str) -> None:
    r = requests.get(f"{base_url}/admin/extractor_mode", timeout=10)
    r.raise_for_status()
    data = r.json()
    if str(data.get("mode", "")).lower() != "local" or not data.get("enabled", False):
        raise RuntimeError(
            f"backend extractor mode is {data.get('mode')!r} (enabled={data.get('enabled')}). "
            f"POST {{\"mode\":\"local\"}} to {base_url}/admin/extractor_mode and wait for it "
            f"to return before running this benchmark."
        )


def fetch_qa(base_url: str, qa_jsonl: str, limit: int) -> list:
    r = requests.get(
        f"{base_url}/admin/mlcslm/qa",
        params={"qa_jsonl": qa_jsonl, "limit": int(limit)},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/qa {r.status_code}: {r.text[:400]}")
    body = r.json()
    return body.get("records", []) or []


def fetch_transcript(base_url: str, audio_path: str, audio_root: str, max_chars: int) -> str:
    r = requests.get(
        f"{base_url}/admin/mlcslm/transcript",
        params={"audio_path": audio_path, "audio_root": audio_root, "max_chars": int(max_chars)},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/transcript {r.status_code}: {r.text[:300]}")
    return r.json().get("transcript", "") or ""


def qwen_generate(base_url: str, prompt: str, max_new_tokens: int) -> str:
    r = requests.post(
        f"{base_url}/llm/generate",
        json={"prompt": prompt, "max_new_tokens": int(max_new_tokens)},
        timeout=600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/llm/generate {r.status_code}: {r.text[:300]}")
    return (r.json().get("text") or "").strip()


def option_text(question: dict, label: str) -> str:
    for opt in question.get("options", []):
        if str(opt.get("label", "")).strip().upper() == label:
            return str(opt.get("text", "")).strip()
    return ""


def build_mcq_prompt(transcript: str, question: dict) -> str:
    return _MCQ_PROMPT_TEMPLATE.format(
        transcript=transcript,
        question_stem=question.get("question_stem", ""),
        option_a=option_text(question, "A"),
        option_b=option_text(question, "B"),
        option_c=option_text(question, "C"),
        option_d=option_text(question, "D"),
    )


def parse_letter(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = _LETTER_RE.search(raw.strip())
    return m.group(1) if m else None


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "mlcslm_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"mlcslm_llm_only_{stamp}.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8006")
    p.add_argument("--qa-jsonl", required=True,
                   help="QA JSONL path on the BACKEND container.")
    p.add_argument("--audio-root", required=True,
                   help="Audio root on the BACKEND container (contains the .wav/.txt tree).")
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-transcript-chars", type=int, default=80000,
                   help="Truncate transcript at this many chars (default: %(default)s).")
    p.add_argument("--output", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out_path = Path(args.output) if args.output else default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_backend_local(args.base_url)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print(f"Fetching QA from {args.base_url} (qa_jsonl={args.qa_jsonl}, limit={args.n}) ...")
    try:
        records = fetch_qa(args.base_url, args.qa_jsonl, args.n)
    except Exception as e:
        print(f"FATAL: failed to fetch QA: {e}", file=sys.stderr)
        return 2

    n_audios = len(records)
    n_questions_total = sum(len(r.get("questions", [])) for r in records)
    print(f"Got {n_audios} audio records, {n_questions_total} questions total")
    print(f"Audio root (resolved on backend): {args.audio_root}")

    done_qids: set = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_qids.add(json.loads(line).get("question_id"))
                except Exception:
                    pass
        print(f"[resume] {len(done_qids)} question_ids already in {out_path}")

    print(f"Writing results to {out_path}\n", flush=True)

    n_ok = 0
    n_fail = 0
    n_correct = 0
    t_start = time.perf_counter()

    with out_path.open("a") as out_f:
        for i, rec in enumerate(records, start=1):
            jsonl_path = rec.get("path", "")
            questions = rec.get("questions", []) or []
            if not questions:
                continue

            qids_here = [q.get("question_id", "") for q in questions]
            if args.resume and all(qid in done_qids for qid in qids_here if qid):
                print(f"[{i}/{n_audios}] {jsonl_path} — all questions already done, skipping",
                      flush=True)
                continue

            try:
                transcript = fetch_transcript(
                    args.base_url, jsonl_path, args.audio_root, args.max_transcript_chars
                )
            except Exception as e:
                n_fail += len(questions)
                print(f"[{i}/{n_audios}] TRANSCRIPT FAIL ({jsonl_path}): {e}", flush=True)
                continue

            print(f"[{i}/{n_audios}] {jsonl_path} transcript={len(transcript)} chars; "
                  f"{len(questions)} questions",
                  flush=True)

            for q in questions:
                qid = q.get("question_id", "")
                if qid in done_qids:
                    continue

                t_q = time.perf_counter()
                try:
                    prompt = build_mcq_prompt(transcript, q)
                    raw = qwen_generate(args.base_url, prompt, args.max_new_tokens)
                    predicted = parse_letter(raw)
                except Exception as e:
                    n_fail += 1
                    print(f"    {qid} FAIL ({time.perf_counter() - t_q:.1f}s): {e}", flush=True)
                    continue

                gold = str(q.get("correct_answer", "")).strip().upper() or None
                is_correct = (predicted is not None and predicted == gold)

                out_f.write(json.dumps({
                    "audio_path":       jsonl_path,
                    "question_id":      qid,
                    "question_stem":    q.get("question_stem", ""),
                    "options":          q.get("options", []),
                    "correct_answer":   gold,
                    "predicted_answer": predicted,
                    "raw_response":     raw,
                    "is_correct":       is_correct,
                }) + "\n")
                out_f.flush()
                n_ok += 1
                if is_correct:
                    n_correct += 1

                acc = n_correct / max(n_ok, 1)
                print(f"    {qid}  pred={predicted!s:>4}  gold={gold!s:>4}  "
                      f"{'OK ' if is_correct else '✗  '} (acc={acc:.3f})",
                      flush=True)

            elapsed_total = time.perf_counter() - t_start
            avg_per_audio = elapsed_total / max(i, 1)
            eta_sec = avg_per_audio * (n_audios - i)
            print(f"  [progress] {i}/{n_audios} audios | "
                  f"{n_ok} answered, {n_correct} correct, {n_fail} fail | "
                  f"eta {eta_sec / 60:.1f}m",
                  flush=True)

    print("", flush=True)
    final_acc = n_correct / max(n_ok, 1)
    print(f"Done. {n_ok} answered, {n_correct} correct, {n_fail} failed. "
          f"Accuracy = {final_acc:.4f}")
    print(f"Results: {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
