"""Supermemory baseline for the MELD-v3 benchmark — apples-to-apples with SpeakerRAG.

This is the Supermemory counterpart to `MELD_v2_benchmark_evaluation.py`. It
enrolls the MELD train split into Supermemory, answers the same
`core_benchmark_v3.json` questions from Supermemory retrieval, and scores them
with the SAME OpenAI LLM-as-judge protocol (imported directly from the
SpeakerRAG evaluator, so the scoring is byte-for-byte identical).

Two subcommands:

  enroll  -> read the MELD train CSV and add one memory per utterance to
             Supermemory (content = '<Speaker> said: "<utterance>"', plus
             speaker/emotion/sentiment/dialogue metadata), under a container tag.

  eval    -> for each benchmark question: semantic-search Supermemory, build a
             context from the retrieved snippets, generate an answer, then judge
             it against the ground truth. Reports accuracy per q_type / subtype /
             overall, exactly like the SpeakerRAG run.

WHAT IS AND ISN'T COMPARABLE
  - Retrieval differs (Supermemory vs the SpeakerRAG dual-store), which is the
    point of the comparison.
  - The ANSWERER is configurable so control the confound:
      --answerer backend  -> the SAME Qwen3-14B used by the SpeakerRAG run
                             (POST /llm/generate). Best for isolating RETRIEVAL.
      --answerer openai    -> an OpenAI model (--answer-model). Convenient, but
                             then you're comparing two full stacks, not retrieval.
  - audio_identity questions reference an audio clip. Supermemory is text-only
    and cannot identify a voice, so those are expected to lose the identity half.
    They are INCLUDED by default (fair: it shows the modality gap); use
    --skip-audio-identity to drop them.

PREREQUISITES
  pip install supermemory openai
  export SUPERMEMORY_API_KEY=sm_...
  export OPENAI_API_KEY=sk-...
  (and, for --answerer backend, a reachable SpeakerRAG server in local mode)

USAGE
  # 1. enroll the corpus (one-time; async indexing, give it a few minutes)
  python tests/supermemory_benchmark_evaluation.py enroll \\
      --csv /home/.../MELD.Raw/train_sent_emo.csv \\
      --container-tag meld_train --workers 8

  # 2. evaluate, isolating retrieval by reusing the same Qwen answerer
  python tests/supermemory_benchmark_evaluation.py eval \\
      --benchmark tests/core_benchmark_v3.json \\
      --container-tag meld_train \\
      --answerer backend --base-url http://127.0.0.1:8007 \\
      --judge-model gpt-4o --output tests/supermemory_v3_results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the SpeakerRAG evaluator's judge + aggregation so the scoring protocol
# is identical. Both files live in tests/, so add this dir to the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import MELD_v2_benchmark_evaluation as base  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai")


# ---------------------------------------------------------------------------
# Supermemory client + robust add/search wrappers
# ---------------------------------------------------------------------------
def get_supermemory():
    try:
        from supermemory import Supermemory
    except ImportError:
        sys.exit("Missing dependency: pip install supermemory")
    if "SUPERMEMORY_API_KEY" not in os.environ:
        sys.exit("Set SUPERMEMORY_API_KEY in your environment.")
    return Supermemory(api_key=os.environ["SUPERMEMORY_API_KEY"])


def add_memory(client, content: str, container_tags: List[str],
               metadata: Dict[str, Any]) -> None:
    """Add one memory. Tolerant of SDK surface drift (client.add vs memories.add)."""
    kwargs = {"content": content, "container_tags": container_tags, "metadata": metadata}
    last_err: Optional[Exception] = None
    for fn in (
        lambda: client.memories.add(**kwargs),
        lambda: client.add(**kwargs),
    ):
        try:
            return fn()
        except (AttributeError, TypeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"no working add() method on the Supermemory client: {last_err}")


def search_memories(client, query: str, container_tags: List[str], limit: int):
    """Semantic search. Tolerant of search.documents vs search.execute."""
    kwargs = {"q": query, "container_tags": container_tags, "limit": limit}
    last_err: Optional[Exception] = None
    for fn in (
        lambda: client.search.documents(**kwargs),
        lambda: client.search.execute(**kwargs),
    ):
        try:
            return fn()
        except (AttributeError, TypeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"no working search method on the Supermemory client: {last_err}")


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        m = getattr(obj, attr, None)
        if callable(m):
            try:
                return m()
            except Exception:
                pass
    return {k: getattr(obj, k) for k in dir(obj)
            if not k.startswith("_") and not callable(getattr(obj, k, None))}


def result_snippets(response: Any, max_snippets: int) -> List[str]:
    """Flatten a search response into human-readable context snippets.

    Defensive against the exact result shape: pulls top-level content/text, then
    any chunk-level content, and prepends speaker/emotion from metadata when present.
    """
    resp = _as_dict(response)
    results = resp.get("results") or resp.get("documents") or resp.get("memories") or []
    snippets: List[str] = []
    for r in results:
        rd = _as_dict(r)
        meta = _as_dict(rd.get("metadata") or {})
        speaker = meta.get("speaker") or meta.get("user_id") or ""
        emotion = meta.get("emotion") or ""
        tag = ""
        if speaker:
            tag = f"[{speaker}" + (f" | {emotion}]" if emotion else "]") + " "

        texts: List[str] = []
        for key in ("content", "text", "memory", "summary"):
            v = rd.get(key)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())
                break
        for chunk in (rd.get("chunks") or []):
            cd = _as_dict(chunk)
            cv = cd.get("content") or cd.get("text")
            if isinstance(cv, str) and cv.strip():
                texts.append(cv.strip())

        for t in texts:
            snippets.append(f"{tag}{t}")
            if len(snippets) >= max_snippets:
                return snippets
    return snippets


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------
def enroll(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.csv):
        sys.exit(f"MELD train CSV not found: {args.csv}")
    client = get_supermemory()

    rows: List[Dict[str, str]] = []
    with open(args.csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if args.start > 0:
        rows = rows[args.start:]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"Enrolling {total} MELD utterances into Supermemory "
          f"(container_tag={args.container_tag!r}, workers={args.workers}) ...", flush=True)

    counter = {"ok": 0, "fail": 0}
    lock = threading.Lock()
    t0 = time.perf_counter()

    def work(row: Dict[str, str]) -> None:
        utt = (row.get("Utterance") or "").strip()
        speaker = (row.get("Speaker") or "").strip()
        if not utt:
            with lock:
                counter["ok"] += 1
            return
        content = f'{speaker} said: "{utt}"' if speaker else utt
        metadata = {
            "speaker": speaker,
            "emotion": (row.get("Emotion") or "").strip(),
            "sentiment": (row.get("Sentiment") or "").strip(),
            "dialogue_id": int(row.get("Dialogue_ID") or -1),
            "utterance_id": int(row.get("Utterance_ID") or -1),
            "source": "meld_train",
        }
        for attempt in range(3):
            try:
                add_memory(client, content, [args.container_tag], metadata)
                with lock:
                    counter["ok"] += 1
                return
            except Exception as e:
                if attempt == 2:
                    with lock:
                        counter["fail"] += 1
                    print(f"  [fail] dia{metadata['dialogue_id']}_utt{metadata['utterance_id']}: {e}",
                          flush=True)
                else:
                    time.sleep(1.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, r) for r in rows]
        for i, _ in enumerate(as_completed(futures), start=1):
            if i % 250 == 0 or i == total:
                done = counter["ok"] + counter["fail"]
                rate = done / max(time.perf_counter() - t0, 1e-6)
                print(f"  {done}/{total}  ok={counter['ok']} fail={counter['fail']}  "
                      f"{rate:.1f}/s", flush=True)

    print(f"\nEnrollment done: {counter['ok']} ok, {counter['fail']} failed.")
    print("NOTE: Supermemory indexes asynchronously — wait a few minutes before "
          "`eval` so all memories are searchable.")
    return 0 if counter["fail"] == 0 else 1


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
_ANSWER_PROMPT = """You are answering a question using ONLY the retrieved memory snippets below.
Each snippet is one utterance from a TV-show corpus, prefixed with its speaker (and emotion when known).
Answer concisely and directly. If the snippets do not contain the answer, say what they do support; do not invent facts.

RETRIEVED MEMORIES:
{context}

QUESTION: {question}

ANSWER:"""


def build_answer_prompt(question: str, snippets: List[str]) -> str:
    context = "\n".join(f"- {s}" for s in snippets) if snippets else "(no relevant memories found)"
    return _ANSWER_PROMPT.format(context=context, question=question)


def answer_openai(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def evaluate(args: argparse.Namespace) -> int:
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY in your environment (used for the LLM judge).")

    bench_path = Path(args.benchmark)
    if not bench_path.is_file():
        sys.exit(f"Benchmark file not found: {bench_path}")
    with bench_path.open(encoding="utf-8") as f:
        benchmark = json.load(f)

    sm_client = get_supermemory()
    oa_client = OpenAI()

    if args.answerer == "backend":
        try:
            base.assert_backend_local(args.base_url)
        except Exception as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 2

    # Output paths.
    out_arg = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "supermemory_v3_results")
    if out_arg.suffix == ".jsonl":
        out_path = out_arg
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_arg / f"supermemory_v3_eval_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_name(out_path.stem + "_summary.json")

    def selected(item: Dict[str, Any]) -> bool:
        if args.skip_audio_identity and item.get("q_type") == "audio_identity":
            return False
        if not args.only:
            return True
        qt, st = item.get("q_type"), item.get("subtype")
        return qt in args.only or f"{qt}.{st}" in args.only

    items = [it for it in benchmark if selected(it)]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    print(f"Benchmark: {bench_path}  ({len(items)} questions)")
    print(f"Memory:    Supermemory (container_tag={args.container_tag!r}, top_k={args.top_k})")
    print(f"Answerer:  {args.answerer}"
          + (f" ({args.answer_model})" if args.answerer == "openai"
             else f" (backend Qwen @ {args.base_url})"))
    print(f"Judge:     {args.judge_model}")
    print(f"Writing to {out_path}\n", flush=True)

    results: List[Dict[str, Any]] = []
    n_ok = n_fail = 0
    t_start = time.perf_counter()

    with out_path.open("a") as out_f:
        for i, item in enumerate(items, start=1):
            qid = item.get("id", f"q_{i}")
            question = item.get("question", "")
            t_q = time.perf_counter()
            try:
                resp = search_memories(sm_client, question, [args.container_tag], args.top_k)
                snippets = result_snippets(resp, args.max_snippets)
                prompt = build_answer_prompt(question, snippets)
                if args.answerer == "backend":
                    system_answer = base.llm_generate(args.base_url, prompt, args.max_new_tokens)
                else:
                    system_answer = answer_openai(oa_client, args.answer_model, prompt,
                                                  args.max_new_tokens)
                verdict = base.judge_answer(oa_client, args.judge_model, item, system_answer)
            except Exception as e:
                n_fail += 1
                print(f"[{i}/{len(items)}] {qid} FAIL ({time.perf_counter()-t_q:.1f}s): {e}",
                      flush=True)
                continue

            row = {
                "id": qid,
                "q_type": item.get("q_type", ""),
                "subtype": item.get("subtype", ""),
                "answer_format": item.get("answer_format"),
                "question": question,
                "ground_truth": item.get("ground_truth"),
                "n_snippets": len(snippets),
                "system_answer": system_answer,
                "score": verdict["score"],
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            results.append(row)
            n_ok += 1
            running = sum(r["score"] for r in results) / max(len(results), 1)
            print(f"[{i}/{len(items)}] {qid:<34} score={verdict['score']:.2f} "
                  f"({verdict['verdict']:<9}) snips={len(snippets):>2} "
                  f"running_acc={running:.3f}  {time.perf_counter()-t_q:.1f}s", flush=True)

    summary = base.summarize(results)
    summary["meta"] = {
        "memory_backend": "supermemory",
        "container_tag": args.container_tag,
        "benchmark": str(bench_path),
        "answerer": args.answerer,
        "answer_model": args.answer_model if args.answerer == "openai" else "qwen-backend",
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "n_evaluated": len(results),
        "n_failed": n_fail,
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    base.print_summary(summary)
    print(f"\nDone. {n_ok} scored, {n_fail} failed.")
    print(f"Per-question results: {out_path}")
    print(f"Summary:              {summary_path}")
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("enroll", help="Add the MELD train split to Supermemory.")
    pe.add_argument("--csv", default="/home/lorenzo-concina/development/datasets/MELD.Raw/train_sent_emo.csv")
    pe.add_argument("--container-tag", default="meld_train")
    pe.add_argument("--workers", type=int, default=8)
    pe.add_argument("--limit", type=int, default=-1, help="Max utterances to enroll; -1 = all.")
    pe.add_argument("--start", type=int, default=0, help="Skip the first N rows (for resuming).")
    pe.set_defaults(func=enroll)

    pv = sub.add_parser("eval", help="Answer + judge the benchmark from Supermemory.")
    pv.add_argument("--benchmark", default="tests/core_benchmark_v3.json")
    pv.add_argument("--container-tag", default="meld_train")
    pv.add_argument("--top-k", type=int, default=5, help="Memories to retrieve per question.")
    pv.add_argument("--max-snippets", type=int, default=20,
                    help="Cap on context snippets fed to the answerer.")
    pv.add_argument("--answerer", choices=["openai", "backend"], default="openai",
                    help="'backend' reuses the SpeakerRAG Qwen (/llm/generate) to isolate "
                         "retrieval; 'openai' uses --answer-model.")
    pv.add_argument("--answer-model", default="gpt-4o-mini",
                    help="OpenAI answerer model (used when --answerer openai).")
    pv.add_argument("--base-url", default="http://127.0.0.1:8007",
                    help="SpeakerRAG server (used when --answerer backend).")
    pv.add_argument("--judge-model", default="gpt-4o")
    pv.add_argument("--max-new-tokens", type=int, default=400)
    pv.add_argument("--output", default=None)
    pv.add_argument("--only", nargs="*", default=None,
                    help="Restrict to q_types or 'q_type.subtype' ids.")
    pv.add_argument("--skip-audio-identity", action="store_true",
                    help="Drop audio_identity questions (Supermemory has no voice modality).")
    pv.add_argument("--limit", type=int, default=-1)
    pv.set_defaults(func=evaluate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
