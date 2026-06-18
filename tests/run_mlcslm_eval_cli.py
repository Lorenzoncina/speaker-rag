"""CLI runner for the MLC-SLM Task 2 *evaluation* set — SpeakerRAG mode.

Sibling of run_mlcslm_cli.py, but for the released evaluation set, which:
  - has NO ground-truth answers (this script only predicts),
  - stores one question per JSONL line keyed by `session_id` (not grouped by
    audio with a `questions` list),
  - is consumed by the organizers via a plain `hyp.txt`:

        <session_id> <question_id> <answer_letter>

Example input line (task2_phase1_questions_options.jsonl):
  {"session_id": "English_British_1244_003_phone",
   "question_id": "Q001",
   "question_stem": "...",
   "options": [{"label":"A","text":"..."}, ...]}

Like run_mlcslm_cli.py this is a thin HTTP client. The backend owns the dataset
(the eval .wav tree and the questions JSONL); this script orchestrates:

  GET  /admin/mlcslm/qa       — fetch the question records (raw JSONL lines)
  POST /admin/mlcslm/prepare  — clear DB + enroll one audio (by basename)
  POST /rag/retrieve          — get the FACT SHEET for a question
  POST /llm/generate          — pick A/B/C/D with Qwen3-14B

`--qa-jsonl` and `--audio-root` are paths on the **backend** container.

Audio resolution: the backend resolves a wav by *basename* only (it walks the
whole --audio-root). The eval `session_id` is e.g. "English_British_1244_003_phone"
but the on-disk wav is most likely "1244_003_phone.wav" (the dev set dropped the
language prefix). We don't guess: we try several candidate basenames derived
from the session_id and use whichever one the backend can resolve.

Prerequisites:
  - FastAPI server reachable at --base-url.
  - Backend extractor mode is 'local' (Qwen3-14B-AWQ loaded via vLLM):
        curl -X POST <base-url>/admin/extractor_mode \\
             -H 'Content-Type: application/json' -d '{"mode":"local"}'

Usage:
    tmux new -s mlcslm-eval
    python tests/run_mlcslm_eval_cli.py \\
        --qa-jsonl /app/mlc-slm-2nd-workshop-data/questions/task2_phase1_questions_options.jsonl \\
        --audio-root /app/mlc-slm-2nd-workshop-data/evaluation_data \\
        --n -1 \\
        --hyp tests/mlcslm_results/hyp.txt
    # Ctrl+B then D to detach. tmux attach -t mlcslm-eval to come back.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests


_LETTER_RE = re.compile(r"\b([ABCD])\b")
_DIGIT_RE = re.compile(r"^\d+$")


_MCQ_PROMPT_TEMPLATE = """You will answer a multiple-choice question about a recorded conversation.

CONTEXT (retrieved from the conversation memory):
{context}

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


def fetch_questions(base_url: str, qa_jsonl: str, limit: int) -> List[dict]:
    """Fetch raw question records (one per JSONL line) from the backend."""
    r = requests.get(
        f"{base_url}/admin/mlcslm/qa",
        params={"qa_jsonl": qa_jsonl, "limit": int(limit)},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/qa {r.status_code}: {r.text[:400]}")
    return r.json().get("records", []) or []


def group_by_session(records: List[dict]) -> "OrderedDict[str, List[dict]]":
    """Group question records by session_id, preserving first-seen order."""
    sessions: "OrderedDict[str, List[dict]]" = OrderedDict()
    for rec in records:
        sid = str(rec.get("session_id", "")).strip()
        if not sid:
            continue
        sessions.setdefault(sid, []).append(rec)
    return sessions


def candidate_basenames(session_id: str) -> List[str]:
    """Derive candidate wav basenames for a session_id, best guess first.

    session_id e.g. "English_British_1244_003_phone". The dev set named the
    wav after the numeric stem only ("1244_003_phone.wav"), so we try that
    first, then fall back to the full session_id, then the last 3 tokens.
    """
    sid = session_id.strip()
    tokens = sid.split("_")
    cands: List[str] = []

    # 1) From the first all-numeric token onward: "1244_003_phone".
    for i, tok in enumerate(tokens):
        if _DIGIT_RE.match(tok):
            cands.append("_".join(tokens[i:]))
            break

    # 2) The full session_id as-is.
    cands.append(sid)

    # 3) Last three tokens (e.g. "1244_003_phone") as a coarse fallback.
    if len(tokens) >= 3:
        cands.append("_".join(tokens[-3:]))

    # Dedup (preserve order) and append the .wav extension.
    seen = set()
    out: List[str] = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(f"{c}.wav")
    return out


def prepare_audio(base_url: str, audio_path: str, audio_root: str) -> dict:
    """Tell the backend to clear DB and enroll this one audio file.

    Raises FileNotFoundError on a 404 (basename did not resolve under
    audio_root) so the caller can try the next candidate cheaply — the backend
    resolves the wav *before* clearing/enrolling, so a 404 costs nothing.
    """
    r = requests.post(
        f"{base_url}/admin/mlcslm/prepare",
        json={"audio_path": audio_path, "audio_root": audio_root, "clear_first": True},
        timeout=3600,  # diarization + STT + KG on a long phone call is minutes
    )
    if r.status_code == 404:
        raise FileNotFoundError(r.text[:300])
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/prepare {r.status_code}: {r.text[:400]}")
    return r.json()


def prepare_session(base_url: str, session_id: str, audio_root: str) -> dict:
    """Resolve + enroll a session's audio, trying candidate basenames in order."""
    last_err: Optional[Exception] = None
    for cand in candidate_basenames(session_id):
        try:
            return prepare_audio(base_url, cand, audio_root)
        except FileNotFoundError as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"no wav resolved for session {session_id!r} under {audio_root} "
        f"(tried {candidate_basenames(session_id)}); last: {last_err}"
    )


def retrieve_fact_sheet(base_url: str, question: str) -> str:
    r = requests.post(
        f"{base_url}/rag/retrieve",
        data={"question": question},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/rag/retrieve {r.status_code}: {r.text[:300]}")
    return r.json().get("fact_sheet", "") or ""


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


def build_mcq_prompt(context: str, question: dict) -> str:
    return _MCQ_PROMPT_TEMPLATE.format(
        context=context,
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


def default_hyp_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "mlcslm_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"hyp_{stamp}.txt"


def load_done(hyp_path: Path) -> set:
    """Return the set of (session_id, question_id) already present in hyp.txt."""
    done: set = set()
    if not hyp_path.exists():
        return done
    with hyp_path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                done.add((parts[0], parts[1]))
    return done


# Backend errors that corrupt the GPU/process state for *every* subsequent
# request — there is no in-process recovery, so the run must abort and the
# backend container be restarted before resuming. A single bad audio hitting
# one of these would otherwise cascade into "PREPARE FAIL" for all remaining
# sessions (the CUDA context stays poisoned).
_FATAL_BACKEND_PATTERNS = (
    "device-side assert",
    "cuda error",
    "cuda out of memory",
    "illegal memory access",
    "cublas",
    "cudnn",
    "no kernel image is available",
)


def is_fatal_backend_error(msg: str) -> bool:
    low = (msg or "").lower()
    return any(p in low for p in _FATAL_BACKEND_PATTERNS)


def load_skip_file(skip_file: Path) -> set:
    """Read the persistent skip list (one session_id per line)."""
    if not skip_file.exists():
        return set()
    return {ln.strip() for ln in skip_file.read_text().splitlines() if ln.strip()}


def append_skip(skip_file: Path, sid: str) -> None:
    """Add a session_id to the persistent skip list (dedup, create if missing)."""
    if sid in load_skip_file(skip_file):
        return
    skip_file.parent.mkdir(parents=True, exist_ok=True)
    with skip_file.open("a") as f:
        f.write(sid + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8006")
    p.add_argument("--qa-jsonl", required=True,
                   help="Path to the eval questions JSONL on the BACKEND container.")
    p.add_argument("--audio-root", required=True,
                   help="Eval audio root on the BACKEND container (contains language folders).")
    p.add_argument("--n", type=int, default=-1,
                   help="Max number of sessions (audio files) to process; -1 = all.")
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="Token budget for the answer letter (default: %(default)s).")
    p.add_argument("--max-fact-sheet-chars", type=int, default=95000,
                   help="Truncate the retrieved FACT SHEET to this many chars before "
                        "building the MCQ prompt (default: %(default)s).")
    p.add_argument("--default-on-fail", default="A",
                   choices=["A", "B", "C", "D"],
                   help="Letter to emit when retrieval/generation fails or no letter "
                        "could be parsed, so the submission has no blank lines "
                        "(default: %(default)s).")
    p.add_argument("--hyp", default=None,
                   help="Output hyp.txt path (default: tests/mlcslm_results/hyp_<stamp>.txt).")
    p.add_argument("--log", default=None,
                   help="Optional JSONL debug log (per-question prompt/raw_response).")
    p.add_argument("--resume", action="store_true",
                   help="If --hyp exists, skip (session_id, question_id) already present.")
    p.add_argument("--skip-sessions", default="",
                   help="Comma-separated session_ids to NOT enroll/answer (e.g. an audio "
                        "that crashes the backend GPU). They are left unanswered for a "
                        "later --finalize-defaults pass.")
    p.add_argument("--finalize-defaults", action="store_true",
                   help="After processing, append the --default-on-fail letter for every "
                        "(session, question) still missing from --hyp, so the submission "
                        "file is complete. Run this on the FINAL pass once every session "
                        "has been attempted (skipped/crashing sessions get the default).")
    p.add_argument("--skip-file", default=None,
                   help="Persistent skip list (one session_id per line). Auto-detected "
                        "GPU-crashing/poisoning sessions are appended here so you can just "
                        "re-run the SAME command after a restart and they're skipped. "
                        "Default: <hyp dir>/skip_sessions.txt.")
    p.add_argument("--poison-max-sec", type=float, default=0.2,
                   help="If a session answers ALL of its (>=3) questions faster than this, "
                        "treat it as a poisoned/empty-retrieval session: discard its answers, "
                        "add it to --skip-file, and abort (default: %(default)ss). A healthy "
                        "session always has at least one slower answer (real retrieval).")
    args = p.parse_args()

    arg_skips = {s.strip() for s in args.skip_sessions.split(",") if s.strip()}

    hyp_path = Path(args.hyp) if args.hyp else default_hyp_path()
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    skip_file_path = Path(args.skip_file) if args.skip_file else hyp_path.parent / "skip_sessions.txt"
    skip_sessions = arg_skips | load_skip_file(skip_file_path)

    try:
        assert_backend_local(args.base_url)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print(f"Fetching questions from {args.base_url} (qa_jsonl={args.qa_jsonl}) ...")
    try:
        records = fetch_questions(args.base_url, args.qa_jsonl, -1)
    except Exception as e:
        print(f"FATAL: failed to fetch questions: {e}", file=sys.stderr)
        return 2

    sessions = group_by_session(records)
    if args.n and args.n > 0:
        sessions = OrderedDict(list(sessions.items())[: args.n])

    n_sessions = len(sessions)
    n_questions = sum(len(qs) for qs in sessions.values())
    print(f"Got {len(records)} question lines -> {n_sessions} sessions, "
          f"{n_questions} questions to answer")
    print(f"Audio root (resolved on backend): {args.audio_root}")

    done = load_done(hyp_path) if args.resume else set()
    if args.resume:
        print(f"[resume] {len(done)} (session, question) pairs already in {hyp_path}")
    if skip_sessions:
        print(f"[skip] {len(skip_sessions)} session(s) will be skipped "
              f"(from --skip-sessions + {skip_file_path})")

    print(f"Writing hyp to {hyp_path}\n", flush=True)

    n_ok = 0
    n_fail = 0
    failed_sessions: list = []   # enroll failed (non-fatal) — left for retry/finalize
    skipped_sessions: list = []  # explicitly skipped via --skip-sessions
    aborted = False
    t_start = time.perf_counter()

    hyp_f = hyp_path.open("a")
    log_f = log_path.open("a") if log_path else None
    try:
        for i, (sid, questions) in enumerate(sessions.items(), start=1):
            if sid in skip_sessions:
                print(f"[{i}/{n_sessions}] {sid} — in --skip-sessions, leaving unanswered",
                      flush=True)
                skipped_sessions.append(sid)
                continue

            pending = [q for q in questions
                       if (sid, q.get("question_id", "")) not in done]
            if not pending:
                print(f"[{i}/{n_sessions}] {sid} — all questions already done, skipping",
                      flush=True)
                continue

            # Enroll the session audio once for all its questions.
            t_audio = time.perf_counter()
            try:
                prep = prepare_session(args.base_url, sid, args.audio_root)
            except Exception as e:
                msg = str(e)
                # A CUDA/GPU-fatal error poisons the backend process for every
                # subsequent request — stop now so we don't cascade into hundreds
                # of bogus failures. Do NOT write anything for this session, so a
                # post-restart --resume retries it cleanly.
                if is_fatal_backend_error(msg):
                    append_skip(skip_file_path, sid)
                    print(f"[{i}/{n_sessions}] FATAL BACKEND ERROR on {sid}:\n  {msg}",
                          flush=True)
                    print(f"\n*** Aborting: this audio crashed enrollment and the GPU "
                          f"context is now corrupted. Added {sid} to {skip_file_path}. "
                          f"Restart the backend container, then re-run the SAME command "
                          f"— it will be skipped automatically.",
                          flush=True)
                    aborted = True
                    break
                # Non-fatal (e.g. wav not found): record and move on WITHOUT writing
                # defaults, so --resume retries it. --finalize-defaults fills it later.
                print(f"[{i}/{n_sessions}] PREPARE FAIL ({sid}): {msg}", flush=True)
                failed_sessions.append(sid)
                continue
            t_enroll = time.perf_counter() - t_audio
            wav_real = prep.get("wav_path", "")
            print(f"[{i}/{n_sessions}] enrolled {sid} ({Path(wav_real).name}) "
                  f"in {t_enroll:.1f}s; {len(pending)} questions to answer",
                  flush=True)

            # Buffer this session's answers and commit them only after a health
            # check. A poisoned GPU context makes semantic retrieval return empty
            # instantly, so every answer comes back implausibly fast and is a blind
            # guess — we must NOT write those, and the session that caused it must be
            # skipped on the next run. See the poison check after this loop.
            q_times: list = []
            buf_hyp: list = []
            buf_log: list = []
            sess_ok = 0
            sess_fail = 0
            for q in pending:
                qid = q.get("question_id", "")
                t_q = time.perf_counter()
                predicted: Optional[str] = None
                raw = ""
                err = None
                try:
                    fact_sheet = retrieve_fact_sheet(args.base_url, q.get("question_stem", ""))
                    if len(fact_sheet) > args.max_fact_sheet_chars:
                        fact_sheet = (
                            fact_sheet[: args.max_fact_sheet_chars]
                            + "\n\n[...truncated to fit within the model context window...]"
                        )
                    prompt = build_mcq_prompt(fact_sheet, q)
                    raw = qwen_generate(args.base_url, prompt, args.max_new_tokens)
                    predicted = parse_letter(raw)
                except Exception as e:
                    err = e

                dt = time.perf_counter() - t_q
                q_times.append(dt)
                answer = predicted or args.default_on_fail
                buf_hyp.append(f"{sid} {qid} {answer}\n")
                if log_f is not None:
                    buf_log.append(json.dumps({
                        "session_id":       sid,
                        "question_id":      qid,
                        "question_stem":    q.get("question_stem", ""),
                        "options":          q.get("options", []),
                        "predicted_answer": predicted,
                        "emitted_answer":   answer,
                        "raw_response":     raw,
                        "error":            str(err) if err else None,
                    }) + "\n")

                if err is not None or predicted is None:
                    sess_fail += 1
                    tag = f"FAIL: {err}" if err else "no-letter (defaulted)"
                    print(f"    {qid}  -> {answer}  [{tag}] ({dt:.1f}s)", flush=True)
                else:
                    sess_ok += 1
                    print(f"    {qid}  -> {answer} ({dt:.1f}s)", flush=True)

            # Poison auto-detect: a healthy session always has at least one slower
            # answer (real retrieval + generation). If EVERY answer (>=3 of them)
            # came back under --poison-max-sec, the context was already corrupted
            # during this session's enrollment — discard, skip it next time, abort.
            poisoned = (len(q_times) >= 3
                        and all(t < args.poison_max_sec for t in q_times))
            if poisoned:
                append_skip(skip_file_path, sid)
                print(f"\n*** Auto-detected POISONED session {sid}: all {len(q_times)} "
                      f"answers returned in <{args.poison_max_sec}s (empty retrieval — its "
                      f"enrollment corrupted the GPU context). Discarded its answers and "
                      f"added it to {skip_file_path}. Restart the backend container, then "
                      f"re-run the SAME command — it will be skipped automatically.",
                      flush=True)
                aborted = True
                break

            # Healthy — commit the buffered answers.
            for line in buf_hyp:
                hyp_f.write(line)
            hyp_f.flush()
            if log_f is not None:
                for line in buf_log:
                    log_f.write(line)
                log_f.flush()
            n_ok += sess_ok
            n_fail += sess_fail

            elapsed_total = time.perf_counter() - t_start
            avg_per_session = elapsed_total / max(i, 1)
            eta_sec = avg_per_session * (n_sessions - i)
            print(f"  [progress] {i}/{n_sessions} sessions | "
                  f"{n_ok} answered, {n_fail} fail/defaulted | "
                  f"eta {eta_sec / 60:.1f}m",
                  flush=True)
    finally:
        hyp_f.close()
        if log_f:
            log_f.close()

    # Optionally pad the submission so every question has a line. Only do this on
    # a final pass — otherwise defaults would block --resume from retrying.
    n_filled = 0
    if args.finalize_defaults and not aborted:
        have = load_done(hyp_path)
        with hyp_path.open("a") as out_f:
            for sid, questions in sessions.items():
                for q in questions:
                    qid = q.get("question_id", "")
                    if qid and (sid, qid) not in have:
                        out_f.write(f"{sid} {qid} {args.default_on_fail}\n")
                        have.add((sid, qid))
                        n_filled += 1
        print(f"[finalize] wrote {n_filled} default '{args.default_on_fail}' answers "
              f"for unanswered questions", flush=True)

    print("", flush=True)
    print(f"Done. {n_ok} answered cleanly, {n_fail} per-question fail/defaulted.")
    if failed_sessions:
        print(f"  {len(failed_sessions)} sessions failed to enroll (left for retry/finalize): "
              f"{', '.join(failed_sessions[:10])}{' ...' if len(failed_sessions) > 10 else ''}")
    if skipped_sessions:
        print(f"  {len(skipped_sessions)} sessions skipped via --skip-sessions: "
              f"{', '.join(skipped_sessions)}")
    if args.finalize_defaults and not aborted:
        print(f"  finalize: filled {n_filled} missing questions with '{args.default_on_fail}'")
    total_lines = len(load_done(hyp_path))
    print(f"  hyp.txt now has {total_lines} / {n_questions} questions")
    print(f"Submission file: {hyp_path}")

    if aborted:
        return 3  # backend restart required, then re-run with --resume
    return 0 if (n_fail == 0 and not failed_sessions) else 1


if __name__ == "__main__":
    sys.exit(main())
