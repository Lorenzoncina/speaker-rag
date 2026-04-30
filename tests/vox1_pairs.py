"""Build a speaker-verification trial list from VoxCeleb1 Vox1-O.

Vox1-O.txt format (one trial per line):
    <label> <audio1_relpath> <audio2_relpath>

where label is 1 (same speaker) or 0 (different), and paths are relative to
/app/VoxCeleb1/wav/test/ (e.g. id10270/x6uYqmx31kE/00001.wav).

The full Vox1-O has 37,720 pairs across 40 test speakers. By default this
script subsamples a balanced subset for faster iteration; use --n-pairs -1
to keep the full list.
"""
from __future__ import annotations

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Set, Tuple


DEFAULT_TRIALS = "/app/VoxCeleb1/Vox1-O/Vox1-O.txt"
DEFAULT_WAV_ROOT = "/app/VoxCeleb1/wav/test"
DEFAULT_OUT = str(Path(__file__).resolve().parent / "vox1o_pairs.json")


def _parse_line(line: str) -> Tuple[int, str, str]:
    parts = line.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Bad trial line: {line!r}")
    label, a1, a2 = parts
    return int(label), a1, a2


def parse_vox1o(trials_path: str = DEFAULT_TRIALS,
                wav_root: str = DEFAULT_WAV_ROOT) -> List[dict]:
    pairs = []
    with open(trials_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, a1, a2 = _parse_line(line)
            spk1 = a1.split("/", 1)[0]
            spk2 = a2.split("/", 1)[0]
            pairs.append({
                "audio1": os.path.join(wav_root, a1),
                "audio2": os.path.join(wav_root, a2),
                "speaker1": spk1,
                "speaker2": spk2,
                "label": label,
                "audio1_rel": a1,
                "audio2_rel": a2,
            })
    return pairs


def get_probe_files_relative(pairs: List[dict]) -> Set[str]:
    """Set of all rel-paths used as probes in trial pairs (for enrollment exclusion)."""
    files = set()
    for p in pairs:
        files.add(p["audio1_rel"])
        files.add(p["audio2_rel"])
    return files


def build_pairs(
    trials_path: str = DEFAULT_TRIALS,
    wav_root: str = DEFAULT_WAV_ROOT,
    n_pairs: int = 200,
    seed: int = 42,
    output_path: str = DEFAULT_OUT,
) -> Tuple[List[dict], dict]:
    all_pairs = parse_vox1o(trials_path, wav_root)
    print(f"Parsed {len(all_pairs)} trials from {trials_path}")

    if n_pairs > 0 and n_pairs < len(all_pairs):
        same = [p for p in all_pairs if p["label"] == 1]
        diff = [p for p in all_pairs if p["label"] == 0]
        rng = random.Random(seed)
        rng.shuffle(same)
        rng.shuffle(diff)
        n_same = n_pairs // 2
        n_diff = n_pairs - n_same
        pairs = same[:n_same] + diff[:n_diff]
        rng.shuffle(pairs)
    else:
        pairs = all_pairs

    speakers = sorted({p["speaker1"] for p in pairs} | {p["speaker2"] for p in pairs})
    meta = {
        "trials_path": trials_path,
        "wav_root": wav_root,
        "n_total_in_trials": len(all_pairs),
        "n_pairs_selected": len(pairs),
        "n_same": sum(1 for p in pairs if p["label"] == 1),
        "n_diff": sum(1 for p in pairs if p["label"] == 0),
        "seed": seed,
        "n_speakers": len(speakers),
        "speakers": speakers,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"meta": meta, "pairs": pairs}, f, indent=2)
    print(f"Wrote {len(pairs)} pairs ({meta['n_same']} same, {meta['n_diff']} diff) "
          f"to {output_path}")
    return pairs, meta


def load_pairs(path: str = DEFAULT_OUT) -> List[dict]:
    with open(path) as f:
        return json.load(f)["pairs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", default=DEFAULT_TRIALS)
    parser.add_argument("--wav-root", default=DEFAULT_WAV_ROOT)
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Subsample size; -1 = use all 37,720 pairs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    build_pairs(args.trials, args.wav_root, args.n_pairs, args.seed, args.out)
