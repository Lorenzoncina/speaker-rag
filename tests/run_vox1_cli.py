"""Run the three VoxCeleb1 Vox1-O experiments end-to-end from the CLI.

Designed to be launched inside the backend container and detached (nohup /
tmux / screen) so the long evaluation survives an SSH disconnect.

Experiments (mirrors the Gradio Step 4 buttons):
  1. Qwen2-Audio direct — both pair audios fed to Qwen2-Audio-7B-Instruct.
  2. RAG (empty DB) + text LLM — FactSheets are retrieved with the DB cleared,
     then Qwen2-7B-Instruct emits Yes/No → LLR. Lower bound for RAG quality.
  3. RAG (Vox1 enrolled) + text LLM — same as (2), but the 40 Vox1-O test
     speakers are first enrolled via /admin/ingest_voxceleb (probe files
     supplied to the backend are excluded from enrollment).

The script drives the running backend through HTTP — it does NOT spin up its
own FastAPI. Make sure the gunicorn server in the container is up before
running. Default URL is http://127.0.0.1:8007 (the in-container port set by
the Dockerfile).

Example:
    # Run all three with a 200-pair balanced subset, default URL, log to file
    nohup python -m tests.run_vox1_cli --n-pairs 200 \
        > /app/tests/vox1_run.log 2>&1 &

    # Skip Exp 1 if Qwen2-Audio doesn't fit alongside Voxtral + Qwen3-14B
    python -m tests.run_vox1_cli --exp 2,3 --n-pairs 500

    # Re-use existing enrollment (skip the /admin/ingest_voxceleb step)
    python -m tests.run_vox1_cli --exp 3 --skip-ingest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import torch

# Make `tests.*` importable when invoked as a script (python tests/run_vox1_cli.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.vox1_pairs import (
    DEFAULT_OUT as DEFAULT_VOX1_PAIRS_PATH,
    build_pairs as build_vox1_pairs,
    load_pairs as load_vox1_pairs,
)
from tests.eval_speaker_verification import (
    _load_qwen_audio,
    _load_qwen_text,
    format_summary,
    run_experiment_direct_audio,
    run_experiment_rag_text,
    run_experiment_rag_text_pair,
    save_results,
)


DEFAULT_FASTAPI_URL = "http://127.0.0.1:8007"
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "eval_results")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_pairs(pairs_path: str, n_pairs: int, regenerate: bool):
    if regenerate or not Path(pairs_path).exists():
        # build_vox1_pairs treats n_pairs <= 0 as "use the full 37,720" — pass
        # the raw value through so --n-pairs -1 actually gives us the full list.
        _log(f"Building Vox1-O pairs (n_pairs={n_pairs}) → {pairs_path}")
        build_vox1_pairs(n_pairs=n_pairs, output_path=pairs_path)
    pairs = load_vox1_pairs(pairs_path)
    if n_pairs > 0:
        pairs = pairs[:n_pairs]
    n_same = sum(1 for p in pairs if p["label"] == 1)
    n_diff = sum(1 for p in pairs if p["label"] == 0)
    _log(f"Loaded {len(pairs)} pairs (same={n_same}, diff={n_diff})")
    return pairs


def _progress_logger(prefix: str, every: int = 25):
    """Build a (done, total) callback that logs every `every` steps + at end."""
    def _cb(done: int, total: int) -> None:
        if done == 1 or done == total or done % every == 0:
            _log(f"{prefix}: {done}/{total} pairs")
    return _cb


def _collect_probe_rel_paths(pairs) -> list:
    seen, out = set(), []
    for p in pairs:
        for key in ("audio1_rel", "audio2_rel"):
            v = p.get(key, "")
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def _clear_db(fastapi_url: str) -> None:
    _log("Clearing backend DB (DELETE /rag/clear)...")
    r = requests.delete(f"{fastapi_url}/rag/clear", timeout=120)
    r.raise_for_status()
    _log(f"  → {r.json()}")


def _ingest_voxceleb(fastapi_url: str, per_speaker: int, probe_rel_paths) -> None:
    n = len(probe_rel_paths)
    if n == 1 and probe_rel_paths[0].startswith("__no_probe_exclusion"):
        _log(f"Ingesting VoxCeleb1 (per_speaker={per_speaker}, probe exclusion DISABLED)...")
    else:
        _log(f"Ingesting VoxCeleb1 (per_speaker={per_speaker}, probe_excluded={n})...")
    payload = {"per_speaker": per_speaker, "probe_rel_paths": probe_rel_paths}
    r = requests.post(
        f"{fastapi_url}/admin/ingest_voxceleb",
        json=payload,
        timeout=60 * 60 * 6,  # up to 6h for the long ingestion
    )
    r.raise_for_status()
    body = r.json()
    _log(f"  → speakers={body.get('n_speakers')} "
         f"per_speaker={body.get('per_speaker')} "
         f"enrolled={body.get('enrolled_count')}")


# ---------------------------------------------------------------------------
# Experiments

def _exp1(pairs, out_dir: str) -> str:
    _log("=== Experiment 1: Qwen2-Audio direct ===")
    model, processor = _load_qwen_audio()
    try:
        scores, labels = run_experiment_direct_audio(
            pairs, model, processor,
            progress_callback=_progress_logger("Exp 1"),
        )
        path, _ = save_results(out_dir, "1", scores, labels,
                                "Qwen/Qwen2-Audio-7B-Instruct",
                                extra={"dataset": "vox1"})
        summary = format_summary("1 (Vox1)", "Qwen/Qwen2-Audio-7B-Instruct",
                                 scores, labels, path)
        _log(summary)
        return summary
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _exp_rag(pairs, fastapi_url: str, exp_id: str, rag_tag: str,
             label: str, out_dir: str, text_pipe, pair_mode: bool = False) -> str:
    _log(f"=== Experiment {exp_id}: {label}{' [pair-mode]' if pair_mode else ''} ===")
    runner = run_experiment_rag_text_pair if pair_mode else run_experiment_rag_text
    scores, labels = runner(
        pairs, text_pipe, fastapi_url,
        progress_callback=_progress_logger(f"Exp {exp_id}"),
    )
    extra = {"dataset": "vox1", "rag": rag_tag}
    if pair_mode:
        extra["mode"] = "pair"
    path, _ = save_results(out_dir, exp_id, scores, labels,
                            "Qwen/Qwen2-7B-Instruct",
                            extra=extra)
    summary = format_summary(f"{exp_id} (Vox1)", label, scores, labels, path)
    _log(summary)
    return summary


# ---------------------------------------------------------------------------
# Main

def _parse_exps(arg: str) -> list:
    if arg.strip().lower() == "all":
        return ["1", "2", "3"]
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok not in {"1", "2", "3"}:
            raise argparse.ArgumentTypeError(f"Invalid experiment id: {tok!r}")
        if tok not in out:
            out.append(tok)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fastapi-url", default=DEFAULT_FASTAPI_URL,
                        help="Base URL of the SpeakerRAG backend (default: %(default)s).")
    parser.add_argument("--pairs", default=DEFAULT_VOX1_PAIRS_PATH,
                        help="Path to vox1o_pairs.json (default: %(default)s).")
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Subset size, -1 for full Vox1-O (default: %(default)s).")
    parser.add_argument("--regenerate-pairs", action="store_true",
                        help="Force re-building the pairs file even if it exists.")
    parser.add_argument("--exp", type=_parse_exps, default=_parse_exps("all"),
                        help='Comma-separated subset of "1,2,3", or "all" (default: all).')
    parser.add_argument("--per-speaker", type=int, default=10,
                        help="Utterances per speaker for VoxCeleb1 enrollment (default: %(default)s).")
    parser.add_argument("--skip-clear", action="store_true",
                        help="Skip DB clear before Exp 2 (use current DB state).")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip /admin/ingest_voxceleb before Exp 3 (assume already enrolled).")
    parser.add_argument("--no-probe-exclusion", action="store_true",
                        help=("Disable probe-file exclusion at enrollment. Lets all 40 Vox1-O "
                              "speakers get 10 utterances enrolled, accepting that some enrolled "
                              "files also appear as probes in trial pairs. The 0.99 cosine cap "
                              "at retrieval filters exact self-matches, so the probe is verified "
                              "against the 9 *other* enrolled utterances of that speaker. "
                              "Required when running Exp 3 on the full 37,611-pair Vox1-O — "
                              "otherwise 14 speakers fail to enroll (100%% probe coverage)."))
    parser.add_argument("--pair-mode", action="store_true",
                        help=("Pair-mode retrieval for Exp 2 and Exp 3: both pair audios are "
                              "sent in a single /rag/retrieve call so they share a session_id, "
                              "letting the backend's intra-call union-find cluster them when "
                              "TitaNet cosine >= 0.45. Collapses the random Unknown_Speaker hash "
                              "to a shared identifier for same-speaker unenrolled pairs. Cost: "
                              "no FactSheet cache, ~8x retrieval calls (~16-24h on full Vox1-O)."))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Where to drop the per-experiment JSON results.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Backend reachability check up-front so the user doesn't wait through
    # model loads only to fail at the first HTTP call.
    try:
        h = requests.get(f"{args.fastapi_url}/health", timeout=10).json()
        _log(f"Backend health: {h}")
        if not h.get("models_loaded"):
            _log("WARNING: backend says models are not loaded yet.")
    except Exception as e:
        _log(f"FATAL: backend unreachable at {args.fastapi_url}: {e}")
        sys.exit(1)

    pairs = _ensure_pairs(args.pairs, args.n_pairs, args.regenerate_pairs)
    if args.no_probe_exclusion:
        # Endpoint falls back to the FULL Vox1-O probe list when probe_rel_paths is
        # empty/None, which is the broken-by-default behaviour. A non-empty list with
        # a sentinel path that can't match any real file disables exclusion cleanly.
        probe_rel_paths = ["__no_probe_exclusion__/sentinel.wav"]
        _log("Probe-file exclusion DISABLED (--no-probe-exclusion).")
    else:
        probe_rel_paths = _collect_probe_rel_paths(pairs)
        _log(f"Probe files to exclude from enrollment: {len(probe_rel_paths)}")

    summaries = []
    text_pipe = None  # lazy-loaded if any RAG experiment runs

    # Experiment 1 — independent of DB state.
    if "1" in args.exp:
        summaries.append(_exp1(pairs, args.out_dir))

    if "2" in args.exp or "3" in args.exp:
        _log("Loading text LLM (Qwen/Qwen2-7B-Instruct)...")
        text_pipe = _load_qwen_text()

    # Experiment 2 — RAG over an empty DB.
    if "2" in args.exp:
        if not args.skip_clear:
            _clear_db(args.fastapi_url)
        else:
            _log("Skipping DB clear (--skip-clear).")
        summaries.append(_exp_rag(
            pairs, args.fastapi_url, "2", "no_enrollment",
            "Qwen/Qwen2-7B-Instruct + RAG (empty DB)", args.out_dir, text_pipe,
            pair_mode=args.pair_mode,
        ))

    # Experiment 3 — RAG with VoxCeleb1 enrolled.
    if "3" in args.exp:
        if not args.skip_ingest:
            _ingest_voxceleb(args.fastapi_url, args.per_speaker, probe_rel_paths)
        else:
            _log("Skipping VoxCeleb1 ingest (--skip-ingest).")
        summaries.append(_exp_rag(
            pairs, args.fastapi_url, "3", "vox1_enrolled",
            "Qwen/Qwen2-7B-Instruct + RAG (VoxCeleb1 enrolled)", args.out_dir, text_pipe,
            pair_mode=args.pair_mode,
        ))

    # Final recap so the tail of the log is easy to skim.
    _log("=" * 60)
    _log("All requested experiments finished.")
    for s in summaries:
        print(s, flush=True)


if __name__ == "__main__":
    main()
