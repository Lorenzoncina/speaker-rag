"""CLI runner for the LongMemEval LLM-only baseline.

Mirrors the "Run LLM-only baseline" flow from gradio_ui/app_ui.py, but as a
plain Python script that survives SSH drops when wrapped in `nohup` or
`tmux`. The answerer is the backend's Qwen3-14B-AWQ via vLLM — the SAME
model the RAG path uses — so the two benchmark runs are directly comparable
and the only thing being measured is whether SpeakerRAG's retrieval beats
context-window stuffing.

Per question it hits two endpoints:
    GET  /admin/longmemeval/haystack   — fetch the raw sessions for this q
    POST /llm/generate                 — answer the prompt with Qwen3-14B

No enrollment, no chroma writes, no graph: this baseline DELIBERATELY skips
SpeakerRAG and feeds the model the haystack as one big text blob, so we can
see what's gained by adding the RAG layer.

Prerequisites:
  - FastAPI server reachable at --base-url.
  - Backend extractor mode is 'local' (Qwen3-14B-AWQ loaded via vLLM).
    Activate via:
        curl -X POST <base-url>/admin/extractor_mode \\
             -H 'Content-Type: application/json' -d '{"mode":"local"}'

Usage:
    tmux new -s lme-llm
    python tests/run_longmemeval_llm_only_cli.py \\
        --split s_cleaned --n -1 \\
        --output tests/longmemeval_results/run_llm_only_s_cleaned.jsonl
    # Ctrl+B then D to detach. tmux attach -t lme-llm to come back.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


# Adapted verbatim from the LongMemEval paper Appendix D (Figure 13, non-CoN
# prompt). Same template as the UI's LLM-only path so any score difference
# between this CLI and the UI run isn't due to prompt drift.
_LME_LLM_ONLY_PROMPT = """I will give you several history chats between you and a user. \
Please answer the question based on the relevant chat history.

History Chats:
{chat_history}

Current Date: {question_date}

Question: {question}

Answer:"""


def flatten_haystack(haystack_sessions, haystack_dates, max_chars: int = 80000) -> str:
    """Concatenate all sessions into one text blob, truncating from the end.

    Backend Qwen3-14B-AWQ runs with max_model_len=32k (~96k chars). 80k chars
    keeps room for the prompt scaffolding and the 300-token answer budget.
    """
    lines = []
    for i, session in enumerate(haystack_sessions):
        date = haystack_dates[i] if i < len(haystack_dates) else "?"
        lines.append(f"=== Session {i + 1} ({date}) ===")
        for msg in session:
            role = msg.get("role", "")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("")
    full = "\n".join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[...truncated due to context limit...]"
    return full


def fetch_questions(base_url: str, split: str, n: int) -> list:
    r = requests.get(
        f"{base_url}/admin/longmemeval/questions",
        params={"split": split, "limit": int(n)},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("questions", [])


def fetch_haystack(base_url: str, split: str, question_id: str) -> dict:
    r = requests.get(
        f"{base_url}/admin/longmemeval/haystack",
        params={"split": split, "question_id": question_id},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"haystack {r.status_code}: {r.text[:300]}")
    return r.json()


def qwen_answer(base_url: str, prompt: str, max_new_tokens: int) -> str:
    r = requests.post(
        f"{base_url}/llm/generate",
        json={"prompt": prompt, "max_new_tokens": int(max_new_tokens)},
        timeout=900,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/llm/generate {r.status_code}: {r.text[:300]}")
    return (r.json().get("text") or "").strip()


def assert_backend_local(base_url: str) -> None:
    """Refuse to start if the backend has no LLM loaded — every call would 409."""
    r = requests.get(f"{base_url}/admin/extractor_mode", timeout=10)
    r.raise_for_status()
    data = r.json()
    if str(data.get("mode", "")).lower() != "local" or not data.get("enabled", False):
        raise RuntimeError(
            f"backend extractor mode is {data.get('mode')!r} (enabled={data.get('enabled')}). "
            f"POST {{\"mode\":\"local\"}} to {base_url}/admin/extractor_mode and wait for it "
            f"to return before running this baseline."
        )


def default_output_path(split: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "longmemeval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"longmemeval_llm_only_{split}_{stamp}.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8006",
                   help="FastAPI base URL (default: %(default)s)")
    p.add_argument("--split", default="s_cleaned",
                   help="LongMemEval split: oracle | s_cleaned | m_cleaned (default: %(default)s)")
    p.add_argument("--n", type=int, default=-1,
                   help="Max number of questions to run; -1 = all (default: %(default)s)")
    p.add_argument("--max-new-tokens", type=int, default=300,
                   help="Answerer token cap (default: %(default)s)")
    p.add_argument("--max-haystack-chars", type=int, default=80000,
                   help="Truncate flattened haystack to this many chars (default: %(default)s)")
    p.add_argument("--output", default=None,
                   help="Output JSONL path. Default: tests/longmemeval_results/longmemeval_llm_only_<split>_<ts>.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="If --output exists, skip question_ids already present in it.")
    args = p.parse_args()

    out_path = Path(args.output) if args.output else default_output_path(args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        assert_backend_local(args.base_url)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

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
    print(f"  answerer: backend Qwen3-14B-AWQ via vLLM")
    print(f"  max_new_tokens={args.max_new_tokens}  max_haystack_chars={args.max_haystack_chars}")
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
                hs = fetch_haystack(args.base_url, args.split, qid)
                chat_history = flatten_haystack(
                    hs.get("haystack_sessions", []),
                    hs.get("haystack_dates", []),
                    max_chars=args.max_haystack_chars,
                )
                prompt = _LME_LLM_ONLY_PROMPT.format(
                    chat_history=chat_history,
                    question_date=q.get("question_date", "unknown"),
                    question=q.get("question", ""),
                )
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
