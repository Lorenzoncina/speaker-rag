"""Phase-2 runner for the MLC-SLM Task 2 eval — reuses prior phase-1 answers.

Phase 2 ships a superset question file (task2_phase2_questions_options.jsonl,
~9k questions) that *includes* every phase-1 question plus new ones. Re-running
the whole thing would re-enroll every audio (minutes each) for nothing — the
phase-1 answers are already computed.

This script is a thin wrapper around run_mlcslm_eval_cli.py:

  1. Copy answers from one or more prior hyp files (e.g. hyp_voxtral.txt) into
     the output hyp, but ONLY for (session_id, question_id) pairs that actually
     appear in the phase-2 question set.
  2. Then run the normal enroll → retrieve → generate loop, which (thanks to
     the resume logic) skips every already-present pair and only answers the
     NEW questions. A session whose questions are all copied is never enrolled.

Result: the output hyp has the old answers verbatim + freshly computed answers
for the new questions, with audio enrollment paid only for sessions that have
new questions.

All the heavy lifting (HTTP calls, audio resolution, poison auto-detect,
persistent skip list, fatal-abort) is imported unchanged from
run_mlcslm_eval_cli.py — this file only adds prior-answer seeding.

Usage:
    tmux new -s vox-phase2
    python tests/run_mlcslm_eval_phase2_cli.py \\
        --base-url http://127.0.0.1:8007 \\
        --qa-jsonl /app/mlc-slm-2nd-workshop-data/questions/task2_phase2_questions_options.jsonl \\
        --audio-root /app/mlc-slm-2nd-workshop-data/evaluation_data \\
        --prior-hyp tests/mlcslm_results/hyp_voxtral.txt \\
        --hyp tests/mlcslm_results/hyp_voxtral_phase2.txt \\
        --log tests/mlcslm_results/eval_voxtral_phase2.jsonl \\
        --resume --poison-max-sec 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

# Reuse every helper from the phase-1 runner so behaviour stays identical.
import run_mlcslm_eval_cli as base


def load_prior_answers(paths: str) -> Dict[Tuple[str, str], str]:
    """Merge prior hyp files into {(session_id, question_id): answer}.

    `paths` is a comma-separated list; later files override earlier ones for
    the same (session, question), so you can layer (e.g. a more-trusted run
    listed last).
    """
    answers: Dict[Tuple[str, str], str] = {}
    for raw in paths.split(","):
        p = raw.strip()
        if not p:
            continue
        fp = Path(p)
        if not fp.exists():
            print(f"WARNING: prior hyp not found, skipping: {fp}", file=sys.stderr)
            continue
        n = 0
        with fp.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    answers[(parts[0], parts[1])] = parts[2]
                    n += 1
        print(f"[prior] loaded {n} answers from {fp}")
    return answers


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8006")
    p.add_argument("--qa-jsonl", required=True,
                   help="Phase-2 questions JSONL on the BACKEND container (superset).")
    p.add_argument("--audio-root", required=True,
                   help="Eval audio root on the BACKEND container.")
    p.add_argument("--prior-hyp", required=True,
                   help="Comma-separated prior hyp file(s) to copy answers from "
                        "(e.g. tests/mlcslm_results/hyp_voxtral.txt). Only pairs that "
                        "also appear in --qa-jsonl are copied; later files win.")
    p.add_argument("--n", type=int, default=-1,
                   help="Max number of sessions to process; -1 = all.")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-fact-sheet-chars", type=int, default=95000)
    p.add_argument("--default-on-fail", default="A", choices=["A", "B", "C", "D"])
    p.add_argument("--hyp", default=None,
                   help="Output hyp path (default: tests/mlcslm_results/hyp_phase2_<stamp>.txt).")
    p.add_argument("--log", default=None,
                   help="Optional JSONL debug log (per NEW question only).")
    p.add_argument("--resume", action="store_true",
                   help="If --hyp already exists, keep its lines and skip those pairs.")
    p.add_argument("--skip-file", default=None,
                   help="Persistent skip list (default: <hyp dir>/skip_sessions.txt).")
    p.add_argument("--poison-max-sec", type=float, default=0.2,
                   help="Timing poison-detect threshold; set 0 to disable.")
    p.add_argument("--finalize-defaults", action="store_true",
                   help="After processing, write --default-on-fail for every phase-2 "
                        "(session, question) still missing, so the file is complete.")
    args = p.parse_args()

    hyp_path = Path(args.hyp) if args.hyp else base.default_hyp_path()
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    skip_file_path = Path(args.skip_file) if args.skip_file else hyp_path.parent / "skip_sessions.txt"
    skip_sessions = base.load_skip_file(skip_file_path)

    try:
        base.assert_backend_local(args.base_url)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print(f"Fetching phase-2 questions from {args.base_url} (qa_jsonl={args.qa_jsonl}) ...")
    try:
        records = base.fetch_questions(args.base_url, args.qa_jsonl, -1)
    except Exception as e:
        print(f"FATAL: failed to fetch questions: {e}", file=sys.stderr)
        return 2

    sessions = base.group_by_session(records)
    if args.n and args.n > 0:
        sessions = OrderedDict(list(sessions.items())[: args.n])

    n_sessions = len(sessions)
    n_questions = sum(len(qs) for qs in sessions.values())
    print(f"Got {len(records)} question lines -> {n_sessions} sessions, "
          f"{n_questions} questions total")

    # --- Seed the output hyp with prior answers (only pairs that exist in phase 2). ---
    prior = load_prior_answers(args.prior_hyp)
    done = base.load_done(hyp_path) if args.resume else set()
    n_copied = 0
    with hyp_path.open("a") as seed_f:
        for sid, questions in sessions.items():
            for q in questions:
                qid = q.get("question_id", "")
                key = (sid, qid)
                if not qid or key in done:
                    continue
                if key in prior:
                    seed_f.write(f"{sid} {qid} {prior[key]}\n")
                    done.add(key)
                    n_copied += 1
    print(f"[seed] copied {n_copied} prior answers into {hyp_path}")

    n_new_total = n_questions - len(done)
    print(f"[plan] {len(done)} already answered (copied/resumed); "
          f"{n_new_total} NEW questions to compute")
    if skip_sessions:
        print(f"[skip] {len(skip_sessions)} session(s) will be skipped "
              f"(from {skip_file_path})")
    print(f"Writing hyp to {hyp_path}\n", flush=True)

    # --- Same enroll/answer loop as the phase-1 runner, computing only NEW questions. ---
    n_ok = 0
    n_fail = 0
    skipped_sessions: list = []
    failed_sessions: list = []
    aborted = False
    t_start = time.perf_counter()

    hyp_f = hyp_path.open("a")
    log_f = log_path.open("a") if log_path else None
    try:
        for i, (sid, questions) in enumerate(sessions.items(), start=1):
            if sid in skip_sessions:
                print(f"[{i}/{n_sessions}] {sid} — in skip list, leaving unanswered",
                      flush=True)
                skipped_sessions.append(sid)
                continue

            pending = [q for q in questions
                       if (sid, q.get("question_id", "")) not in done]
            if not pending:
                # Every question for this session was copied from prior — no enrollment.
                continue

            t_audio = time.perf_counter()
            try:
                prep = base.prepare_session(args.base_url, sid, args.audio_root)
            except Exception as e:
                msg = str(e)
                if base.is_fatal_backend_error(msg):
                    base.append_skip(skip_file_path, sid)
                    print(f"[{i}/{n_sessions}] FATAL BACKEND ERROR on {sid}:\n  {msg}",
                          flush=True)
                    print(f"\n*** Aborting: GPU context corrupted. Added {sid} to "
                          f"{skip_file_path}. Restart the backend, then re-run the SAME "
                          f"command — it skips automatically.", flush=True)
                    aborted = True
                    break
                print(f"[{i}/{n_sessions}] PREPARE FAIL ({sid}): {msg}", flush=True)
                failed_sessions.append(sid)
                continue
            t_enroll = time.perf_counter() - t_audio
            print(f"[{i}/{n_sessions}] enrolled {sid} "
                  f"({Path(prep.get('wav_path','')).name}) in {t_enroll:.1f}s; "
                  f"{len(pending)} NEW questions", flush=True)

            q_times: list = []
            buf_hyp: list = []
            buf_log: list = []
            sess_ok = 0
            sess_fail = 0
            for q in pending:
                qid = q.get("question_id", "")
                t_q = time.perf_counter()
                predicted = None
                raw = ""
                err = None
                try:
                    fact_sheet = base.retrieve_fact_sheet(args.base_url, q.get("question_stem", ""))
                    if len(fact_sheet) > args.max_fact_sheet_chars:
                        fact_sheet = (fact_sheet[: args.max_fact_sheet_chars]
                                      + "\n\n[...truncated...]")
                    prompt = base.build_mcq_prompt(fact_sheet, q)
                    raw = base.qwen_generate(args.base_url, prompt, args.max_new_tokens)
                    predicted = base.parse_letter(raw)
                except Exception as e:
                    err = e

                dt = time.perf_counter() - t_q
                q_times.append(dt)
                answer = predicted or args.default_on_fail
                buf_hyp.append(f"{sid} {qid} {answer}\n")
                if log_f is not None:
                    buf_log.append(json.dumps({
                        "session_id": sid, "question_id": qid,
                        "question_stem": q.get("question_stem", ""),
                        "options": q.get("options", []),
                        "predicted_answer": predicted, "emitted_answer": answer,
                        "raw_response": raw, "error": str(err) if err else None,
                    }) + "\n")
                if err is not None or predicted is None:
                    sess_fail += 1
                    tag = f"FAIL: {err}" if err else "no-letter (defaulted)"
                    print(f"    {qid}  -> {answer}  [{tag}] ({dt:.1f}s)", flush=True)
                else:
                    sess_ok += 1
                    print(f"    {qid}  -> {answer} ({dt:.1f}s)", flush=True)

            poisoned = (args.poison_max_sec > 0 and len(q_times) >= 3
                        and all(t < args.poison_max_sec for t in q_times))
            if poisoned:
                base.append_skip(skip_file_path, sid)
                print(f"\n*** Auto-detected POISONED session {sid}: all {len(q_times)} "
                      f"answers <{args.poison_max_sec}s (empty retrieval). Discarded, "
                      f"added to {skip_file_path}. Restart the backend and re-run the "
                      f"SAME command.", flush=True)
                aborted = True
                break

            for line in buf_hyp:
                hyp_f.write(line)
            hyp_f.flush()
            if log_f is not None:
                for line in buf_log:
                    log_f.write(line)
                log_f.flush()
            n_ok += sess_ok
            n_fail += sess_fail

            elapsed = time.perf_counter() - t_start
            done_new = n_ok + n_fail
            rate = done_new / elapsed if elapsed > 0 else 0
            eta = (n_new_total - done_new) / rate / 60 if rate > 0 else 0
            print(f"  [progress] {i}/{n_sessions} sessions | {n_ok} new answered, "
                  f"{n_fail} fail/defaulted | eta {eta:.1f}m", flush=True)
    finally:
        hyp_f.close()
        if log_f:
            log_f.close()

    n_filled = 0
    if args.finalize_defaults and not aborted:
        have = base.load_done(hyp_path)
        with hyp_path.open("a") as out_f:
            for sid, questions in sessions.items():
                for q in questions:
                    qid = q.get("question_id", "")
                    if qid and (sid, qid) not in have:
                        out_f.write(f"{sid} {qid} {args.default_on_fail}\n")
                        have.add((sid, qid))
                        n_filled += 1
        print(f"[finalize] wrote {n_filled} default '{args.default_on_fail}' answers")

    print("", flush=True)
    print(f"Done. {n_ok} NEW answered cleanly, {n_fail} fail/defaulted, "
          f"{n_copied} copied from prior.")
    if failed_sessions:
        print(f"  {len(failed_sessions)} sessions failed to enroll: "
              f"{', '.join(failed_sessions[:10])}"
              f"{' ...' if len(failed_sessions) > 10 else ''}")
    if skipped_sessions:
        print(f"  {len(skipped_sessions)} sessions skipped: {', '.join(skipped_sessions)}")
    total_lines = len(base.load_done(hyp_path))
    print(f"  hyp now has {total_lines} / {n_questions} questions")
    print(f"Submission file: {hyp_path}")

    if aborted:
        return 3
    return 0 if (n_fail == 0 and not failed_sessions) else 1


if __name__ == "__main__":
    sys.exit(main())
