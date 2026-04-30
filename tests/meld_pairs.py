"""Build a balanced speaker-verification trial list from the MELD dev split.

Produces a JSON file with same-speaker + different-speaker pairs sampled
deterministically. Multi-speaker labels (e.g. "All", "Monica and Rachel")
are excluded — they are not valid single-speaker audios.

Two modes:
  --mode all     (default) All single speakers with >=2 utterances.
  --mode main6   Only the 6 main Friends characters. Use this when the
                 enrollment DB covers MELD train, because minor characters
                 are typically absent from train and degrade EER measurement.
"""
import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_CSV = "/app/MELD.Raw/dev_sent_emo.csv"
DEFAULT_AUDIO_DIR = "/app/MELD.Raw/dev_splits_complete"
DEFAULT_OUT = str(Path(__file__).resolve().parent / "meld_dev_pairs.json")
DEFAULT_OUT_MAIN6 = str(Path(__file__).resolve().parent / "meld_dev_pairs_main6.json")

MAIN6 = {"monica", "rachel", "phoebe", "joey", "chandler", "ross"}


def _is_single_speaker(name: str) -> bool:
    n = str(name).strip().lower()
    if not n or n == "all":
        return False
    if " and " in n:
        return False
    if "," in n:
        return False
    return True


def _collect_utterances(
    csv_path: str,
    audio_dir: str,
    allowed_speakers: Optional[set] = None,
) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path)
    by_speaker: Dict[str, List[str]] = {}
    missing = 0
    for _, row in df.iterrows():
        speaker = str(row.get("Speaker", "")).strip()
        if not _is_single_speaker(speaker):
            continue
        speaker_lc = speaker.lower()
        if allowed_speakers is not None and speaker_lc not in allowed_speakers:
            continue
        try:
            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])
        except (KeyError, ValueError):
            continue
        path = os.path.join(audio_dir, f"dia{dialogue_id}_utt{utterance_id}.wav")
        if not os.path.exists(path):
            missing += 1
            continue
        by_speaker.setdefault(speaker_lc, []).append(path)
    label = f"(restricted to {sorted(allowed_speakers)})" if allowed_speakers else ""
    print(f"Collected {sum(len(v) for v in by_speaker.values())} utterances "
          f"across {len(by_speaker)} speakers {label} "
          f"({missing} missing audio files skipped).")
    return by_speaker


def build_pairs(
    csv_path: str = DEFAULT_CSV,
    audio_dir: str = DEFAULT_AUDIO_DIR,
    n_same: int = 1000,
    n_diff: int = 1000,
    seed: int = 42,
    output_path: str = DEFAULT_OUT,
    mode: str = "all",
) -> Tuple[List[dict], dict]:
    allowed = MAIN6 if mode == "main6" else None
    by_speaker = _collect_utterances(csv_path, audio_dir, allowed_speakers=allowed)
    eligible = sorted([s for s, paths in by_speaker.items() if len(paths) >= 2])
    if len(eligible) < 2:
        raise RuntimeError(
            f"Need at least 2 speakers with >=2 utterances; found {len(eligible)}."
        )

    rng = random.Random(seed)

    same_pool = [s for s in eligible if len(by_speaker[s]) >= 2]
    same_pairs = []
    seen_same = set()
    attempts = 0
    while len(same_pairs) < n_same and attempts < n_same * 50:
        attempts += 1
        spk = rng.choice(same_pool)
        a, b = rng.sample(by_speaker[spk], 2)
        key = (a, b) if a < b else (b, a)
        if key in seen_same:
            continue
        seen_same.add(key)
        same_pairs.append({"audio1": a, "audio2": b, "speaker1": spk, "speaker2": spk, "label": 1})

    diff_pairs = []
    seen_diff = set()
    attempts = 0
    while len(diff_pairs) < n_diff and attempts < n_diff * 50:
        attempts += 1
        s1, s2 = rng.sample(eligible, 2)
        a = rng.choice(by_speaker[s1])
        b = rng.choice(by_speaker[s2])
        key = (a, b) if a < b else (b, a)
        if key in seen_diff:
            continue
        seen_diff.add(key)
        diff_pairs.append({"audio1": a, "audio2": b, "speaker1": s1, "speaker2": s2, "label": 0})

    pairs = same_pairs + diff_pairs
    rng.shuffle(pairs)

    meta = {
        "csv_path": csv_path,
        "audio_dir": audio_dir,
        "mode": mode,
        "n_same_requested": n_same,
        "n_diff_requested": n_diff,
        "n_same_actual": len(same_pairs),
        "n_diff_actual": len(diff_pairs),
        "seed": seed,
        "n_speakers": len(eligible),
        "speakers": eligible,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"meta": meta, "pairs": pairs}, f, indent=2)
    print(f"Wrote {len(pairs)} pairs to {output_path} "
          f"({len(same_pairs)} same, {len(diff_pairs)} different, mode={mode}).")
    return pairs, meta


def load_pairs(path: str = DEFAULT_OUT) -> List[dict]:
    with open(path) as f:
        return json.load(f)["pairs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--n-same", type=int, default=1000)
    parser.add_argument("--n-diff", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["all", "main6"], default="all",
                        help="'all' = every single speaker; 'main6' = Friends main cast only")
    parser.add_argument("--out", default=None,
                        help="Output path (defaults to meld_dev_pairs.json or meld_dev_pairs_main6.json)")
    args = parser.parse_args()

    out = args.out or (DEFAULT_OUT_MAIN6 if args.mode == "main6" else DEFAULT_OUT)
    build_pairs(args.csv, args.audio_dir, args.n_same, args.n_diff, args.seed, out, args.mode)
