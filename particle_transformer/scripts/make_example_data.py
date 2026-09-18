"""Generate independent synthetic ROOT splits for an executable demonstration."""
import argparse
from pathlib import Path

import numpy as np
import uproot


def make_candidates(rng, size):
    labels = np.tile([0, 1], (size + 1) // 2)[:size].astype(np.int64)
    rng.shuffle(labels)
    shift = labels.astype(np.float32)
    return {
        "event_id": np.arange(size, dtype=np.int64),
        "isSignal": labels,
        "fPt": rng.uniform(1, 10, size).astype(np.float32),
        "decay_length": rng.lognormal(0.5 * shift, 0.4).astype(np.float32),
        "pointing_cos": (0.9 + 0.1 * rng.beta(1 + 3 * shift, 2)).astype(np.float32),
        "daughter_a_pt": rng.lognormal(0.4 * shift, 0.5).astype(np.float32),
        "daughter_a_ip": rng.normal(0.6 * shift, 0.8).astype(np.float32),
        "daughter_b_pt": rng.lognormal(0.3 * shift, 0.5).astype(np.float32),
        "daughter_b_ip": rng.normal(0.5 * shift, 0.8).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    targets = [args.output_dir / name for name in (
        "train.root", "val.root", "test.root", "input_1.root")]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        parser.error(f"Refusing to overwrite existing files: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for path, size in zip(targets[:3], [512, 128, 128]):
        candidates = make_candidates(rng, size)
        with uproot.recreate(path) as root_file:
            root_file.mktree("DecayCandidates", candidates)
        print(f"Wrote {size} synthetic candidates to {path}")
    # Batch inference uses the same test candidates, nested in a DF directory.
    with uproot.recreate(targets[3]) as root_file:
        root_file.mktree("DF_0/DecayCandidates", candidates)


if __name__ == "__main__":
    main()
