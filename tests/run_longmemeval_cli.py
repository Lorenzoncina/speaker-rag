"""CLI runner for the LongMemEval benchmark.

Mirrors the Step 5 "RAG mode" flow from gradio_ui/app_ui.py but as a plain
Python script so it can run detached from the browser — survives SSH drops
when wrapped in `nohup` or `tmux`.

Per question it hits the same FastAPI endpoints the UI uses:
    POST /admin/longmemeval/prepare   — clear DB, enroll the haystack
    POST /rag/retrieve                — build the FACT SHEET prompt
    POST /llm/generate                — answer with the loaded Qwen3-14B

Results are appended to a JSONL file incrementally (one line per question,
flushed immediately) so a kill mid-run still leaves you with partial results.

Prerequisites:
  - The FastAPI server is running and reachable at --base-url.
  - The extractor mode is set to 'local' (vLLM Qwen3-14B loaded) — same as
    clicking "Activate Local Qwen-14B" in the UI.

Usage example:
    nohup python tests/run_longmemeval_cli.py \\
        --split s_cleaned --n 500 --enable-kg \\
        --output tests/longmemeval_results/run.jsonl \\
        > tests/longmemeval_results/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


def fetch_questions(base_url: str, split: str, n: int) -> list:
    r = requests.get(
        f"{base_url}/admin/longmemeval/questions",
        params={"split": split, "limit": int(n)},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("questions", [])


def prepare_haystack(base_url: str, split: str, question_id: str, enable_kg: bool) -> None:
    r = requests.post(
        f"{base_url}/admin/longmemeval/prepare",
        json={"split": split, "question_id": question_id, "enable_kg": bool(enable_kg)},
        timeout=3600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"prepare {r.status_code}: {r.text[:300]}")


def retrieve_prompt(base_url: str, question: str) -> str:
    r = requests.post(
        f"{base_url}/rag/retrieve",
        data={"question": question},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"retrieve {r.status_code}: {r.text[:300]}")
    return r.json().get("recommended_prompt", "") or ""


def qwen_answer(base_url: str, prompt: str, max_new_tokens: int) -> str:
    r = requests.post(
        f"{base_url}/llm/generate",
        json={"prompt": prompt, "max_new_tokens": int(max_new_tokens)},
        timeout=900,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/llm/generate {r.status_code}: {r.text[:300]}")
    return (r.json().get("text") or "").strip()


def default_output_path(split: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "longmemeval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"longmemeval_rag_{split}_{stamp}.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8006",
                   help="FastAPI base URL (default: %(default)s)")
    p.add_argument("--split", default="s_cleaned",
                   help="LongMemEval split: oracle | s_cleaned | m_cleaned (default: %(default)s)")
    p.add_argument("--n", type=int, default=-1,
                   help="Max number of questions to run; -1 = all (default: %(default)s)")
    p.add_argument("--enable-kg", action="store_true",
                   help="Run LLM entity/relation extraction during enrollment.")
    p.add_argument("--max-new-tokens", type=int, default=300,
                   help="Answerer token cap (default: %(default)s)")
    p.add_argument("--output", default=None,
                   help="Output JSONL path. Default: tests/longmemeval_results/longmemeval_rag_<split>_<ts>.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="If --output exists, skip question_ids already present in it.")
    args = p.parse_args()

    out_path = Path(args.output) if args.output else default_output_path(args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("question_id"))
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} question_ids already in {out_path}")

    print(f"Fetching questions from {args.base_url} (split={args.split}, n={args.n})...")
    try:
        questions = fetch_questions(args.base_url, args.split, args.n)
    except Exception as e:
        print(f"FATAL: failed to fetch questions: {e}", file=sys.stderr)
        return 2
    total = len(questions)
    print(f"Got {total} questions. Writing results to {out_path}")
    print(f"  enable_kg={args.enable_kg}  max_new_tokens={args.max_new_tokens}")
    print("", flush=True)

    n_ok = 0
    n_fail = 0
    t_start = time.perf_counter()

    with out_path.open("a") as out_f:
        for i, q in enumerate(questions, start=1):
            qid = q.get("question_id", "")
            if qid in done_ids:
                print(f"[{i}/{total}] {qid} — already done, skipping", flush=True)
                continue

            t_q = time.perf_counter()
            try:
                prepare_haystack(args.base_url, args.split, qid, args.enable_kg)
                prompt = retrieve_prompt(args.base_url, q.get("question", ""))
                if not prompt:
                    raise RuntimeError("empty recommended_prompt")
                hypothesis = qwen_answer(args.base_url, prompt, args.max_new_tokens)
            except Exception as e:
                n_fail += 1
                print(f"[{i}/{total}] {qid} FAIL ({time.perf_counter() - t_q:.1f}s): {e}", flush=True)
                continue

            out_f.write(json.dumps({
                "question_id":   qid,
                "hypothesis":    hypothesis,
                "question":      q.get("question", ""),
                "answer":        q.get("answer", ""),
                "question_type": q.get("question_type", ""),
                "question_date": q.get("question_date", ""),
            }) + "\n")
            out_f.flush()
            n_ok += 1

            elapsed_q = time.perf_counter() - t_q
            elapsed_total = time.perf_counter() - t_start
            avg = elapsed_total / max(n_ok + n_fail, 1)
            eta_sec = avg * (total - i)
            print(
                f"[{i}/{total}] {q.get('question_type', '')} "
                f"({elapsed_q:.1f}s, avg {avg:.1f}s, eta {eta_sec / 60:.1f}m) | "
                f"hyp: {hypothesis[:80].replace(chr(10), ' ')}",
                flush=True,
            )

    print("", flush=True)
    print(f"Done. {n_ok} ok, {n_fail} failed. Results: {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
