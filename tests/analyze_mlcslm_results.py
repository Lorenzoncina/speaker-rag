"""Analyze MLC-SLM Task 2 benchmark result JSONLs.

Reads one or two result files produced by run_mlcslm_cli.py or
run_mlcslm_llm_only_cli.py and prints:

  - Overall accuracy and count.
  - Per-language breakdown (fine: e.g. English_Australian, English_British;
    coarse: e.g. English) so language-specific weaknesses pop out.
  - Top failed audios (by miss count).
  - Optional sample of individual misses (--show-failures N).
  - Side-by-side comparison when --compare <other.jsonl> is given.

Language grouping rules (derived from the audio_path):
  mlc-slm-2nd-dev/<lang_dir>/<file>.wav  →  fine = <lang_dir>
                                           coarse = lang_dir.split('_')[0]
  except:
    French_Canada → coarse 'French'
    Spanish_Mexico/Spanish_Spain → coarse 'Spanish'
    Portuguese_Brazil/Portuguese_Portugal → coarse 'Portuguese'
    Tagalog/Korean/Russian/Japanese/etc. → coarse same as fine

Usage:
    # single-file summary:
    python tests/analyze_mlcslm_results.py tests/mlcslm_results/run_rag.jsonl

    # show first 20 failures verbatim:
    python tests/analyze_mlcslm_results.py tests/mlcslm_results/run_rag.jsonl --show-failures 20

    # RAG vs LLM-only side-by-side:
    python tests/analyze_mlcslm_results.py \\
        tests/mlcslm_results/run_rag.jsonl \\
        --compare tests/mlcslm_results/run_llm_only.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def coarse_lang(fine: str) -> str:
    """Collapse 'English_Australian' → 'English' etc.

    Some directories (Tagalog, Korean, Japanese, Russian, Thai, Turkish, Urdu,
    Vietnamese, German, Italian) are already their own coarse bucket — for
    those we just return them as-is.
    """
    if not fine:
        return "unknown"
    if "_" not in fine:
        return fine
    head, _, _tail = fine.partition("_")
    return head


def extract_lang(audio_path: str) -> Tuple[str, str]:
    """Return (fine, coarse) language buckets from a JSONL-style audio path."""
    parts = audio_path.split("/")
    fine = parts[1] if len(parts) >= 2 else "unknown"
    return fine, coarse_lang(fine)


def load_results(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def acc(rows: List[dict]) -> Tuple[int, int, float]:
    """Return (n_correct, n_total, fraction)."""
    if not rows:
        return 0, 0, 0.0
    n_total = len(rows)
    n_correct = sum(1 for r in rows if r.get("is_correct"))
    return n_correct, n_total, n_correct / n_total


def group_by_lang(rows: List[dict]) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Bucket rows by (fine, coarse) language. Returns two dicts: fine, coarse."""
    by_fine: Dict[str, list] = defaultdict(list)
    by_coarse: Dict[str, list] = defaultdict(list)
    for r in rows:
        fine, cs = extract_lang(r.get("audio_path", ""))
        by_fine[fine].append(r)
        by_coarse[cs].append(r)
    return by_fine, by_coarse


def fmt_bar(frac: float, width: int = 20) -> str:
    """Cheap ASCII bar for a 0..1 fraction."""
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def print_lang_table(rows: List[dict], heading: str) -> None:
    by_fine, by_coarse = group_by_lang(rows)

    print(f"\n=== {heading} ===")
    print(f"{'language':<24} {'n':>5} {'correct':>8} {'acc':>7}   bar")

    def emit(buckets: Dict[str, list], indent: str = "  ") -> None:
        rows_sorted = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for lang, lang_rows in rows_sorted:
            n_correct, n_total, frac = acc(lang_rows)
            print(f"{indent}{lang:<22} {n_total:>5} {n_correct:>8} {frac:>7.3f}   {fmt_bar(frac)}")

    print("\n[coarse — primary language]")
    emit(by_coarse, indent="  ")

    print("\n[fine — dialect / variant]")
    emit(by_fine, indent="  ")


def print_overall(rows: List[dict], heading: str) -> None:
    n_correct, n_total, frac = acc(rows)
    n_failed_parse = sum(1 for r in rows if r.get("predicted_answer") is None)
    answer_dist = Counter((r.get("predicted_answer") or "?") for r in rows)
    gold_dist = Counter((r.get("correct_answer") or "?") for r in rows)
    print(f"\n=== {heading} ===")
    print(f"  total questions:        {n_total}")
    print(f"  correct:                {n_correct}")
    print(f"  accuracy:               {frac:.4f}")
    print(f"  predicted=None (parse): {n_failed_parse}")
    print(f"  predicted letter dist:  " + " ".join(
        f"{k}={answer_dist.get(k, 0)}" for k in ["A", "B", "C", "D", "?"]
    ))
    print(f"  gold letter dist:       " + " ".join(
        f"{k}={gold_dist.get(k, 0)}" for k in ["A", "B", "C", "D", "?"]
    ))


def print_top_failed_audios(rows: List[dict], k: int = 10) -> None:
    """Audios with the most wrong answers, useful for spotting bad-audio outliers."""
    misses_by_audio: Counter = Counter()
    totals_by_audio: Counter = Counter()
    for r in rows:
        ap = r.get("audio_path", "?")
        totals_by_audio[ap] += 1
        if not r.get("is_correct"):
            misses_by_audio[ap] += 1
    if not misses_by_audio:
        return
    print(f"\n=== Top {k} audios by miss count ===")
    for ap, n_miss in misses_by_audio.most_common(k):
        n_tot = totals_by_audio[ap]
        print(f"  {n_miss}/{n_tot}   {ap}")


def print_failures(rows: List[dict], k: int) -> None:
    fails = [r for r in rows if not r.get("is_correct")]
    print(f"\n=== Sample of {min(k, len(fails))} failures (of {len(fails)} total) ===")
    for i, r in enumerate(fails[:k]):
        ap = r.get("audio_path", "")
        qid = r.get("question_id", "")
        pred = r.get("predicted_answer")
        gold = r.get("correct_answer")
        stem = (r.get("question_stem", "") or "").strip().replace("\n", " ")
        if len(stem) > 140:
            stem = stem[:137] + "..."
        print(f"  [{i+1}] {ap}  {qid}  pred={pred}  gold={gold}")
        print(f"       {stem}")


def print_comparison(rows_a: List[dict], rows_b: List[dict],
                     label_a: str, label_b: str) -> None:
    """Side-by-side accuracy table, grouped on the intersection of question_ids."""
    by_qid_a = {r.get("question_id"): r for r in rows_a if r.get("question_id")}
    by_qid_b = {r.get("question_id"): r for r in rows_b if r.get("question_id")}
    common = set(by_qid_a) & set(by_qid_b)
    only_a = set(by_qid_a) - common
    only_b = set(by_qid_b) - common

    print(f"\n=== Comparison: {label_a} vs {label_b} ===")
    print(f"  intersect (questions in both):  {len(common)}")
    print(f"  only in {label_a:<14}: {len(only_a)}")
    print(f"  only in {label_b:<14}: {len(only_b)}")

    if not common:
        return

    # Overall on the intersection.
    correct_a = sum(1 for qid in common if by_qid_a[qid].get("is_correct"))
    correct_b = sum(1 for qid in common if by_qid_b[qid].get("is_correct"))
    n = len(common)
    print(f"\n  [overall on intersection]")
    print(f"    {label_a:<24} {correct_a}/{n} = {correct_a / n:.4f}")
    print(f"    {label_b:<24} {correct_b}/{n} = {correct_b / n:.4f}")
    print(f"    delta ({label_a} - {label_b}): {(correct_a - correct_b) / n:+.4f}")

    # Confusion: who got which right.
    a_only_right = sum(1 for qid in common
                       if by_qid_a[qid].get("is_correct") and not by_qid_b[qid].get("is_correct"))
    b_only_right = sum(1 for qid in common
                       if by_qid_b[qid].get("is_correct") and not by_qid_a[qid].get("is_correct"))
    both_right = sum(1 for qid in common
                     if by_qid_a[qid].get("is_correct") and by_qid_b[qid].get("is_correct"))
    both_wrong = sum(1 for qid in common
                     if not by_qid_a[qid].get("is_correct") and not by_qid_b[qid].get("is_correct"))
    print(f"\n  [agreement matrix on {n} shared questions]")
    print(f"    both correct:                {both_right:>5} ({both_right / n:.3f})")
    print(f"    {label_a} only correct:      {a_only_right:>5} ({a_only_right / n:.3f})")
    print(f"    {label_b} only correct:      {b_only_right:>5} ({b_only_right / n:.3f})")
    print(f"    both wrong:                  {both_wrong:>5} ({both_wrong / n:.3f})")

    # Per-coarse-language deltas — where does A beat B and vice versa?
    print(f"\n  [per-coarse-language accuracy on intersection]")
    print(f"    {'language':<20} {'n':>5}  {label_a:>8}  {label_b:>8}  {'Δ':>8}")
    lang_buckets: Dict[str, List[str]] = defaultdict(list)
    for qid in common:
        _fine, cs = extract_lang(by_qid_a[qid].get("audio_path", ""))
        lang_buckets[cs].append(qid)
    for lang, qids in sorted(lang_buckets.items(), key=lambda kv: -len(kv[1])):
        if not qids:
            continue
        n_l = len(qids)
        ca = sum(1 for qid in qids if by_qid_a[qid].get("is_correct"))
        cb = sum(1 for qid in qids if by_qid_b[qid].get("is_correct"))
        delta = (ca - cb) / n_l
        print(f"    {lang:<20} {n_l:>5}  {ca / n_l:>8.3f}  {cb / n_l:>8.3f}  {delta:>+8.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="Primary results JSONL (e.g. RAG mode).")
    p.add_argument("--compare", default=None,
                   help="Optional second JSONL to compare against (e.g. LLM-only baseline).")
    p.add_argument("--label-a", default=None, help="Label for primary file in comparison.")
    p.add_argument("--label-b", default=None, help="Label for --compare file in comparison.")
    p.add_argument("--show-failures", type=int, default=0,
                   help="Print this many individual failure cases from the primary file (default 0).")
    p.add_argument("--top-bad-audios", type=int, default=10,
                   help="Number of worst audios to list (default 10).")
    args = p.parse_args()

    primary_path = Path(args.results)
    if not primary_path.is_file():
        print(f"FATAL: results file not found: {primary_path}", file=sys.stderr)
        return 2

    rows = load_results(primary_path)
    label_a = args.label_a or primary_path.name

    print(f"Loaded {len(rows)} rows from {primary_path}")
    print_overall(rows, heading=f"{label_a} — overall")
    print_lang_table(rows, heading=f"{label_a} — by language")
    print_top_failed_audios(rows, k=args.top_bad_audios)
    if args.show_failures > 0:
        print_failures(rows, k=args.show_failures)

    if args.compare:
        cmp_path = Path(args.compare)
        if not cmp_path.is_file():
            print(f"FATAL: --compare file not found: {cmp_path}", file=sys.stderr)
            return 2
        rows_b = load_results(cmp_path)
        label_b = args.label_b or cmp_path.name

        print(f"\nLoaded {len(rows_b)} rows from {cmp_path}")
        print_overall(rows_b, heading=f"{label_b} — overall")
        print_lang_table(rows_b, heading=f"{label_b} — by language")
        print_comparison(rows, rows_b, label_a, label_b)

    return 0


if __name__ == "__main__":
    sys.exit(main())
