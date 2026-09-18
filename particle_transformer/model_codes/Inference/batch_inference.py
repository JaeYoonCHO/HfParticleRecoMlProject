"""
Bulk AO2D inference.

In AO2D the trees are not at the top level but one per DF_* directory, and a
single file holds hundreds of millions of candidates, so it cannot be loaded at
once. Instead this streams DF by DF and, inside each DF, chunk by chunk of
`run.chunk_rows` rows, splits into pT bins by fPt, runs the per-bin model, and
writes the results straight into part files.

The chunking matters for the pT-pruned input written by prune_pt_bins.py: there
every file is a single DF_0 holding up to 67.4M rows, so a whole-tree read would
be ~8 GB of numpy per file (x3 concurrent shards). It also keeps progress
visible, which a per-DF print cannot do when there is one DF.

uproot cannot append to an existing TTree (on reopen it is not visible in the
WritableDirectory), so one part file is written per input file x pT bin, and
merge_scores.py merges them at the end. This also means an interrupted run can
resume, skipping the input files that already finished.

Usage:
    python batch_inference.py batch_config_bachelor_pion.yaml
    python batch_inference.py batch_config_bachelor_pion.yaml --files 1-1 --max-rows 2000000
    python batch_inference.py batch_config_bachelor_pion.yaml --shard 1/3
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import uproot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CODES_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODEL_CODES_DIR)
sys.path.insert(0, MODEL_CODES_DIR)

from utils import load_config, get_device
from model import ParticleTransformerClassifier
from data_preprocessing import get_particle_feature_map, load_standardization_stats


def resolve(path):
    """Relative paths in the config are resolved against the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


# ============================================================
# Part-file writer
# ============================================================
class PartWriter:
    """Buffer rows and flush once flush_rows accumulate, to avoid basket
    fragmentation."""

    def __init__(self, path, tree_name, columns, flush_rows=1_000_000):
        self.path = path
        self.tree_name = tree_name
        self.columns = list(columns)
        self.flush_rows = flush_rows

        self.buffer = {c: [] for c in self.columns}
        self.buffered = 0
        self.total = 0

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = uproot.recreate(path)
        self.file.mktree(tree_name, {c: np.float32 for c in self.columns})

    def append(self, data):
        n = len(data[self.columns[0]])
        if n == 0:
            return

        for c in self.columns:
            self.buffer[c].append(np.asarray(data[c], dtype=np.float32))

        self.buffered += n
        self.total += n

        if self.buffered >= self.flush_rows:
            self.flush()

    def flush(self):
        if self.buffered == 0:
            return

        self.file[self.tree_name].extend(
            {c: np.concatenate(self.buffer[c]) for c in self.columns}
        )

        self.buffer = {c: [] for c in self.columns}
        self.buffered = 0

    def close(self):
        self.flush()
        self.file.close()


# ============================================================
# Prepare the per-pT-bin models
# ============================================================
def build_bins(config, feature_map, device, dtype):
    model_cfg = config["model"]

    example_particle_dict = {
        name: torch.zeros(2, len(features))
        for name, features in feature_map.items()
    }

    bins = []

    for spec in config["pt_bins"]:
        stats = load_standardization_stats(resolve(spec["stats_path"]))

        model = ParticleTransformerClassifier(
            particle_dict=example_particle_dict,
            embed_dim=model_cfg["embed_dim"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            ff_dim=model_cfg["ff_dim"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"],
            head_hidden_dim=model_cfg["head_hidden_dim"],
            use_final_norm=model_cfg["use_final_norm"],
            use_positional_encoding=model_cfg.get("use_positional_encoding", False),
        )

        model_path = resolve(spec["model_path"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device=device, dtype=dtype).eval()

        missing = [g for g in feature_map if g not in stats]
        if missing:
            raise KeyError(
                f"[{spec['name']}] particle groups missing from stats: {missing}\n"
                f"  stats file: {spec['stats_path']}"
            )

        bins.append({
            "name": spec["name"],
            "pt_min": float(spec["pt_min"]),
            "pt_max": float(spec["pt_max"]),
            "model": model,
            "mean": {g: stats[g]["mean"].to(device=device, dtype=dtype) for g in feature_map},
            "std": {g: stats[g]["std"].to(device=device, dtype=dtype) for g in feature_map},
        })

        print(f"[INFO] bin {spec['name']}: pT [{spec['pt_min']}, {spec['pt_max']}) "
              f"model={spec['model_path']}")

    return bins


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def score_matrix(bin_spec, features, group_index, batch_size):
    """features: (N, n_feature) GPU tensor, before standardisation
    -> numpy array of prob(class=1)"""
    out = []

    for start in range(0, features.shape[0], batch_size):
        chunk = features[start:start + batch_size]

        particle_dict = {}
        for group, idx in group_index.items():
            x = chunk[:, idx]
            particle_dict[group] = (x - bin_spec["mean"][group]) / (bin_spec["std"][group] + 1e-8)

        logits = bin_spec["model"](particle_dict)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        out.append(probs.cpu().numpy().astype(np.float32))

    if not out:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(out)


def process_tree(tree, read_columns, feature_cols, copy_cols, bins, group_index,
                 writers, score_branch, device, dtype, batch_size, counters,
                 chunk_rows, row_budget=None, report=None):
    """Score one tree in blocks of chunk_rows rows.

    row_budget caps the rows read from this tree (for --max-rows); report, if
    given, is called after every chunk as report(rows_done, rows_total).
    Returns the number of rows read.
    """
    n_entries = tree.num_entries
    if row_budget is not None:
        n_entries = min(n_entries, row_budget)
    if n_entries == 0:
        return 0

    for start in range(0, n_entries, chunk_rows):
        stop = min(start + chunk_rows, n_entries)
        arrays = tree.arrays(read_columns, library="np",
                             entry_start=start, entry_stop=stop)
        process_chunk(arrays, read_columns, feature_cols, copy_cols, bins,
                      group_index, writers, score_branch, device, dtype,
                      batch_size, counters)
        if report is not None:
            report(stop, n_entries)

    return n_entries


def process_chunk(arrays, read_columns, feature_cols, copy_cols, bins, group_index,
                  writers, score_branch, device, dtype, batch_size, counters):
    n_rows = len(arrays[read_columns[0]])
    if n_rows == 0:
        return

    features = np.empty((n_rows, len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        features[:, j] = arrays[col]

    copies = {c: np.asarray(arrays[c], dtype=np.float32) for c in copy_cols}

    finite = np.isfinite(features).all(axis=1)
    for arr in copies.values():
        finite &= np.isfinite(arr)

    counters["read"] += n_rows
    counters["dropped_nonfinite"] += int((~finite).sum())

    pt = copies["fPt"] if "fPt" in copies else np.asarray(arrays["fPt"], dtype=np.float32)

    for bin_spec in bins:
        selected = finite & (pt >= bin_spec["pt_min"]) & (pt < bin_spec["pt_max"])
        n_sel = int(selected.sum())
        if n_sel == 0:
            continue

        gpu_features = torch.from_numpy(features[selected]).to(device=device, dtype=dtype)
        scores = score_matrix(bin_spec, gpu_features, group_index, batch_size)

        payload = {c: arr[selected] for c, arr in copies.items()}
        payload[score_branch] = scores
        writers[bin_spec["name"]].append(payload)

        counters["scored"] += n_sel
        counters["per_bin"][bin_spec["name"]] += n_sel


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="bulk AO2D inference")
    parser.add_argument("config", nargs="?", default="batch_config.yaml")
    parser.add_argument("--files", default=None,
                        help="index range to process (e.g. 1-5 or 7); overrides the config")
    parser.add_argument("--max-dfs", type=int, default=None,
                        help="cap on the DFs processed per file (for quick checks). "
                             "No use on the pruned input, which has one DF per "
                             "file; use --max-rows there.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap on the rows read per file (for quick checks)")
    parser.add_argument("--chunk-rows", type=int, default=None,
                        help="rows read per step; overrides run.chunk_rows")
    parser.add_argument("--restart", action="store_true",
                        help="ignore progress.json and start over")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="process shard I of N (1-based), balanced by input file "
                             "size rather than by file number. Input files differ by "
                             "30x in size, so splitting the index range into equal "
                             "counts leaves one shard running long after the others. "
                             "Each shard keeps its own progress file.")
    return parser.parse_args()


def split_by_size(indices, path_template, n_shards):
    """Split `indices` into n_shards CONTIGUOUS runs with roughly equal total bytes.

    Contiguity matters more than perfect balance on a spinning disk. A greedy
    size-first split gives 0% byte imbalance but interleaves the indices across
    shards, so every shard seeks over the whole platter; measured 1.94 GB/min
    versus 3.49 GB/min for contiguous ranges on the same data. Contiguous runs
    keep each reader in a narrower block range (files 22-30 span 154M blocks,
    while files 1-10 span 834M) and leave ~11% byte imbalance, which costs far
    less than the extra seeking. See PERFORMANCE_NOTES.md section 8.
    """
    indices = list(indices)
    sizes = []
    for index in indices:
        try:
            sizes.append(os.path.getsize(path_template.format(i=index)))
        except OSError:
            sizes.append(0)

    total = sum(sizes)
    n = len(indices)
    if n_shards >= n:
        cuts = list(range(1, n))
    else:
        # Walk the sequence and cut when the accumulated bytes cross the next
        # k/n_shards fraction of the total. O(n) and good enough for 30 files.
        cuts, acc, k = [], 0, 1
        for i, size in enumerate(sizes):
            acc += size
            if k < n_shards and acc >= total * k / n_shards:
                cuts.append(i + 1)
                k += 1
        cuts = cuts[:n_shards - 1]

    bounds = [0] + cuts + [n]
    shards = [indices[bounds[i]:bounds[i + 1]] for i in range(n_shards)]

    for i, shard in enumerate(shards, start=1):
        load = sum(sizes[indices.index(x)] for x in shard)
        span = f"{shard[0]}-{shard[-1]}" if shard else "empty"
        print(f"[INFO] shard {i}/{n_shards}: files {span} "
              f"({len(shard)} file(s), {load / 1024**3:.0f} GB)")
    return shards


def probe_columns(input_paths, tree_name, df_prefix, feature_cols, copy_cols):
    """Inspect the first readable DF tree and fix the columns to read, once.

    This must happen before the writer is created, so a missing branch cannot
    disagree with the writer schema.
    """
    for path in input_paths:
        if not os.path.exists(path):
            continue

        with uproot.open(path) as root_file:
            for df_key in root_file.keys(recursive=False):
                if not df_key.startswith(df_prefix):
                    continue

                directory = root_file[df_key]
                if tree_name not in [k.split(";")[0] for k in directory.keys(recursive=False)]:
                    continue

                available = set(k.split(";")[0] for k in directory[tree_name].keys())

                missing_features = [c for c in feature_cols if c not in available]
                if missing_features:
                    raise ValueError(f"model feature branches missing from the input tree: {missing_features}")

                kept = []
                for c in copy_cols:
                    if c not in available:
                        print(f"[WARNING] Branch not found in input tree, skipping: {c}")
                        continue
                    kept.append(c)

                if "fPt" not in available:
                    raise ValueError("no fPt in the input tree, cannot split into pT bins")

                read_columns = list(dict.fromkeys(feature_cols + kept + ["fPt"]))
                print(f"[INFO] probed branches from {os.path.basename(path)}/{df_key}: "
                      f"copy={kept}, read {len(read_columns)} columns")
                return kept, read_columns

    raise RuntimeError("no input file with a usable DF tree was found")


def resolve_index_range(args, input_cfg):
    if args.files is None:
        return int(input_cfg["index_start"]), int(input_cfg["index_end"])

    if "-" in args.files:
        lo, hi = args.files.split("-", 1)
        return int(lo), int(hi)

    value = int(args.files)
    return value, value


def main():
    args = parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)
    print(f"[INFO] Loaded config: {config_path}")

    input_cfg = config["input"]
    output_cfg = config["output"]
    run_cfg = config["run"]

    device = get_device(run_cfg["device"])
    dtype = torch.float16 if str(run_cfg.get("precision", "fp32")).lower() == "fp16" else torch.float32
    batch_size = int(run_cfg["batch_size"])
    flush_rows = int(run_cfg.get("flush_rows", 1_000_000))
    max_dfs = args.max_dfs if args.max_dfs is not None else run_cfg.get("max_dfs_per_file")
    max_rows = args.max_rows if args.max_rows is not None else run_cfg.get("max_rows_per_file")
    chunk_rows = int(args.chunk_rows if args.chunk_rows is not None
                     else run_cfg.get("chunk_rows", 2_000_000))

    print(f"[INFO] device={device} dtype={dtype} batch_size={batch_size} "
          f"chunk_rows={chunk_rows}")

    particle_file = config["paths"].get("particle_columns_file")
    feature_map = get_particle_feature_map(
        resolve(particle_file) if particle_file else None, config.get("particles")
    )

    feature_cols = []
    for features in feature_map.values():
        feature_cols.extend(features)
    feature_cols = list(dict.fromkeys(feature_cols))

    group_index = {
        group: torch.tensor([feature_cols.index(c) for c in features],
                            dtype=torch.long, device=device)
        for group, features in feature_map.items()
    }

    copy_cols = list(output_cfg["branches"])
    score_branch = output_cfg["score_branch"]

    if "fPt" not in copy_cols and "fPt" not in feature_cols:
        raise ValueError("fPt must be read to split into pT bins; add fPt to output.branches")

    bins = build_bins(config, feature_map, device, dtype)

    out_dir = resolve(output_cfg["dir"])
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(out_dir, exist_ok=True)

    index_start, index_end = resolve_index_range(args, input_cfg)
    tree_name = input_cfg["tree_name"]
    df_prefix = input_cfg["df_prefix"]

    indices = list(range(index_start, index_end + 1))

    # Shards get their own progress file: concurrent shards writing one
    # progress.json would each save only their own completed set, so the last
    # writer wins and the others' records are lost.
    if args.shard:
        shard_i, shard_n = (int(v) for v in args.shard.split("/", 1))
        if not 1 <= shard_i <= shard_n:
            raise SystemExit(f"--shard must be 1..N/N, got {args.shard}")
        indices = sorted(split_by_size(indices, input_cfg["path_template"], shard_n)[shard_i - 1])
        progress_path = os.path.join(out_dir, f"progress_shard{shard_i}of{shard_n}.json")
        print(f"[INFO] this process runs shard {shard_i}/{shard_n}: {indices}")
    else:
        progress_path = os.path.join(out_dir, "progress.json")

    done = set()
    if os.path.exists(progress_path) and not args.restart:
        with open(progress_path, "r", encoding="utf-8") as f:
            done = set(json.load(f).get("completed", []))
        print(f"[INFO] resume: {len(done)} file(s) already completed "
              f"({os.path.basename(progress_path)})")

    input_paths = [input_cfg["path_template"].format(i=index) for index in indices]

    copy_cols, read_columns = probe_columns(
        input_paths=input_paths,
        tree_name=tree_name,
        df_prefix=df_prefix,
        feature_cols=feature_cols,
        copy_cols=copy_cols,
    )

    grand_total = {b["name"]: 0 for b in bins}
    run_started = time.time()

    for position, index in enumerate(indices, start=1):
        input_path = input_cfg["path_template"].format(i=index)
        stem = os.path.splitext(os.path.basename(input_path))[0]

        if input_path in done:
            print(f"[SKIP] already done: {stem}")
            continue

        if not os.path.exists(input_path):
            print(f"[WARNING] input not found, skipping: {input_path}")
            continue

        print(f"\n[FILE {position}/{len(indices)} | index {index}] {stem}")
        file_started = time.time()

        writers = {
            b["name"]: PartWriter(
                path=os.path.join(parts_dir, b["name"], f"{stem}.root"),
                tree_name=tree_name,
                columns=copy_cols + [score_branch],
                flush_rows=flush_rows,
            )
            for b in bins
        }

        counters = {
            "read": 0,
            "dropped_nonfinite": 0,
            "scored": 0,
            "per_bin": {b["name"]: 0 for b in bins},
        }

        try:
            with uproot.open(input_path) as root_file:
                df_keys = [k for k in root_file.keys(recursive=False) if k.startswith(df_prefix)]
                if max_dfs is not None:
                    df_keys = df_keys[:int(max_dfs)]

                print(f"  {len(df_keys)} DF directories")

                used_dfs = 0
                rows_left = int(max_rows) if max_rows is not None else None

                # A single DF_0 of tens of millions of rows is the normal case
                # for the pruned input, so progress is reported per chunk and
                # the DF counter only when there are several DFs.
                def report(df_position, df_key, rows_done, rows_total):
                    elapsed = time.time() - file_started
                    rate = counters["read"] / elapsed if elapsed > 0 else 0
                    where = (f"{df_key} {rows_done:,}/{rows_total:,} "
                             f"({100 * rows_done / max(1, rows_total):.1f}%)"
                             if len(df_keys) == 1
                             else f"DF {df_position}/{len(df_keys)} {df_key}")
                    print(f"  {where} | read {counters['read']:,} "
                          f"| scored {counters['scored']:,} | {rate/1e6:.2f} M cand/s "
                          f"| {elapsed/60:.1f} min")

                for df_position, df_key in enumerate(df_keys, start=1):
                    if rows_left is not None and rows_left <= 0:
                        break

                    directory = root_file[df_key]

                    if tree_name not in [k.split(";")[0] for k in directory.keys(recursive=False)]:
                        print(f"  [WARNING] {df_key}: no tree '{tree_name}', skipping")
                        continue

                    tree = directory[tree_name]
                    used_dfs += 1

                    # With many small DFs a line per chunk would flood the log,
                    # so there it stays at one line per 50 DFs as before.
                    if len(df_keys) == 1:
                        chunk_report = lambda done, total: report(df_position, df_key, done, total)
                    else:
                        chunk_report = None

                    rows_read = process_tree(
                        tree=tree,
                        read_columns=read_columns,
                        feature_cols=feature_cols,
                        copy_cols=copy_cols,
                        bins=bins,
                        group_index=group_index,
                        writers=writers,
                        score_branch=score_branch,
                        device=device,
                        dtype=dtype,
                        batch_size=batch_size,
                        counters=counters,
                        chunk_rows=chunk_rows,
                        row_budget=rows_left,
                        report=chunk_report,
                    )

                    if rows_left is not None:
                        rows_left -= rows_read

                    if chunk_report is None and (df_position % 50 == 0 or df_position == len(df_keys)):
                        report(df_position, df_key, rows_read, rows_read)

                if used_dfs == 0:
                    print("  [WARNING] no usable DF directory found in this file")

        finally:
            for writer in writers.values():
                writer.close()

        for name, count in counters["per_bin"].items():
            grand_total[name] += count

        elapsed = time.time() - file_started
        print(f"  [DONE] {stem} in {elapsed/60:.1f} min | read {counters['read']:,} "
              f"| non-finite dropped {counters['dropped_nonfinite']:,} "
              f"| scored {counters['scored']:,}")
        print(f"         per bin: {counters['per_bin']}")

        done.add(input_path)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed": sorted(done)}, f, indent=2)

    total_elapsed = time.time() - run_started
    print(f"\n[ALL DONE] {total_elapsed/60:.1f} min")
    print(f"  scored per bin: {grand_total}")
    print(f"  parts: {parts_dir}")
    print(f"  next step: python merge_scores.py {os.path.basename(config_path)}")


if __name__ == "__main__":
    main()
