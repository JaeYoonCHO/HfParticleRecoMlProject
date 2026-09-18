import pandas as pd
import torch
import os
import numpy as np
import uproot


def get_particle_feature_map(particle_columns_file=None, particles=None):
    """Resolve ordered token features from config lists or CSV Particle values.

    `particles` maps token names to feature lists or {particle: CSV value}.
    Without it, group CSV rows by Particle in their first-occurrence order.
    Explicit empty lists retain legacy tokens with zero input features.
    """
    table = None

    def read_table():
        nonlocal table
        if table is None:
            if particle_columns_file is None:
                raise ValueError("particle_columns_file is required for CSV feature selection")
            table = pd.read_csv(particle_columns_file)
            if not {"Particle", "Branch"}.issubset(table.columns):
                raise ValueError("Particle CSV must contain Particle and Branch columns")
            if table[["Particle", "Branch"]].isna().any().any():
                raise ValueError("Particle and Branch values must not be missing")
        return table

    if particles is None:
        table = read_table()
        particles = {
            name: {"particle": name} for name in table["Particle"].drop_duplicates()
        }
    if not isinstance(particles, dict) or not particles:
        raise ValueError("particles must be a non-empty mapping of token names to features")

    feature_map = {}
    for name, selection in particles.items():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(f"Invalid token name: {name!r}; use a non-empty name without dots")
        if isinstance(selection, list):
            features = list(selection)
        elif isinstance(selection, dict) and set(selection) == {"particle"}:
            table = read_table()
            features = table.loc[table["Particle"] == selection["particle"], "Branch"].tolist()
            if not features:
                raise ValueError(f"No CSV features match token {name!r}: {selection['particle']!r}")
        else:
            raise ValueError(f"Token {name!r} needs a feature list or {{particle: CSV value}}")
        if any(not isinstance(f, str) or not f for f in features):
            raise ValueError(f"Invalid feature name for token {name!r}")
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicate features for token {name!r}")
        feature_map[name] = features
    return feature_map


def get_raw_data(data_file_path, particle_columns_file, label_column="isSignal", assign_label=None,
                 pt_range=None, pt_column="fPt", particles=None, tree_name="O2hfxicxi2pifull"):
    """
    If assign_label is None the label is read from label_column of the root file
    (for mixed files). If assign_label is an integer, that value is assigned to
    every row. Pure signal (=1) / background (=0) files have no label_column, so
    they use this path.

    pt_range: (pt_min, pt_max) keeps only rows with pt_min <= pt_column < pt_max.
    Used to restrict the standardisation stats to the pT window under analysis;
    pt_column is read from the file but never becomes a feature.
    """
    particle_feature_map = get_particle_feature_map(particle_columns_file, particles)

    # collect every column that will be used
    all_feature_cols = []
    for features in particle_feature_map.values():
        all_feature_cols.extend(features)

    if assign_label is None:
        all_cols = list(dict.fromkeys(all_feature_cols + [label_column]))
    else:
        all_cols = list(dict.fromkeys(all_feature_cols))

    # pt_column is needed only for the pT-window mask, so it is read but kept
    # out of all_feature_cols (and therefore out of the particle_dict).
    if pt_range is not None and pt_column not in all_cols:
        all_cols = all_cols + [pt_column]

    print(f"[INFO] Open ROOT file via uproot: {data_file_path}")
    with uproot.open(data_file_path) as root_file:
        tree = root_file[tree_name]
        available_cols = tree.keys()

        missing_root_cols = [col for col in all_cols if col not in available_cols]
        if len(missing_root_cols) > 0:
            raise ValueError(
                f"In ROOT file, the following required columns are missing:\n"
                f"{missing_root_cols}"
            )

        # load only the needed columns into a DataFrame (key for memory use)
        array = tree.arrays(all_cols, library="np")
        df = pd.DataFrame({col: array[col] for col in all_cols})
    # force numeric conversion
    df[all_cols] = df[all_cols].apply(pd.to_numeric, errors="coerce")

    # drop rows with any non-finite value
    finite_mask = np.isfinite(df[all_cols].to_numpy()).all(axis=1)

    if (~finite_mask).any():
        removed_count = int((~finite_mask).sum())
        print(f"[INFO] Removing {removed_count} non-finite rows from {data_file_path}")

        bad_df = df.loc[~finite_mask, all_cols]
        bad_mask = ~np.isfinite(bad_df.to_numpy())
        bad_cols = bad_df.columns[bad_mask.any(axis=0)].tolist()

        print(f"[INFO] Non-finite columns in {data_file_path}: {bad_cols[:30]}")

        df = df.loc[finite_mask].reset_index(drop=True)

    # keep only the pT window under analysis (stats-only path)
    if pt_range is not None:
        pt_min, pt_max = pt_range
        pt_values = df[pt_column].to_numpy()
        window_mask = (pt_values >= pt_min) & (pt_values < pt_max)
        kept = int(window_mask.sum())
        print(f"[INFO] pT window [{pt_min}, {pt_max}) on {pt_column}: "
              f"keeping {kept} / {len(df)} rows from {data_file_path}")
        if kept == 0:
            raise ValueError(
                f"No rows left after applying pT window [{pt_min}, {pt_max}) "
                f"on '{pt_column}' in {data_file_path}"
            )
        df = df.loc[window_mask].reset_index(drop=True)

    if assign_label is None:
        labels = torch.tensor(df[label_column].to_numpy(), dtype=torch.long)
    else:
        labels = torch.full((len(df),), int(assign_label), dtype=torch.long)

    particle_dict = {}

    for name, features in particle_feature_map.items():
        missing_features = [col for col in features if col not in df.columns]

        if len(missing_features) > 0:
            raise ValueError(
                f"In file '{data_file_path}', the following columns are missing for particle '{name}':\n"
                f"{missing_features}"
            )

        raw_features = df[features].to_numpy(dtype=np.float32)
        particle_dict[name] = torch.from_numpy(raw_features)

    return particle_dict, labels


def get_combined_raw_data(data_file_paths, particle_columns_file, label_column="isSignal", assign_labels=None, particles=None, tree_name="O2hfxicxi2pifull"):
    """
    Read several root files (e.g. signal_train, background_train) and return a
    single particle_dict with the per-particle tensors concatenated, plus the
    combined labels.

    Used when signal and background are read separately and merged, e.g. to
    compute stats.

    assign_labels: list of the same length as data_file_paths, giving the label
        for each file (signal=1, background=0). If None, label_column is read
        from every file. Pure signal/background root files have no label_column,
        so this argument is required for them.
    """
    if isinstance(data_file_paths, str):
        data_file_paths = [data_file_paths]

    if len(data_file_paths) == 0:
        raise ValueError("data_file_paths is empty")

    if assign_labels is None:
        assign_labels = [None] * len(data_file_paths)
    elif len(assign_labels) != len(data_file_paths):
        raise ValueError(
            f"assign_labels length ({len(assign_labels)}) "
            f"!= data_file_paths length ({len(data_file_paths)})"
        )

    per_file_dicts = []
    per_file_labels = []
    for path, assign_label in zip(data_file_paths, assign_labels):
        pdict, labels = get_raw_data(
            data_file_path=path,
            particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name,
            label_column=label_column,
            assign_label=assign_label,
        )
        per_file_dicts.append(pdict)
        per_file_labels.append(labels)

    # check that every file has the same particle keys
    ref_keys = set(per_file_dicts[0].keys())
    for path, pdict in zip(data_file_paths[1:], per_file_dicts[1:]):
        if set(pdict.keys()) != ref_keys:
            raise ValueError(
                f"Particle keys mismatch between files.\n"
                f"  {data_file_paths[0]}: {sorted(ref_keys)}\n"
                f"  {path}: {sorted(pdict.keys())}"
            )

    combined_particle_dict = {
        name: torch.cat([pdict[name] for pdict in per_file_dicts], dim=0)
        for name in per_file_dicts[0].keys()
    }
    combined_labels = torch.cat(per_file_labels, dim=0)

    total = combined_labels.shape[0]
    print(
        f"[INFO] Combined {len(data_file_paths)} file(s) -> {total} rows "
        f"(per-file rows: {[lb.shape[0] for lb in per_file_labels]})"
    )

    return combined_particle_dict, combined_labels


def compute_standardization_stats(particle_dict):
    """
    Compute per-particle feature mean/std from the train particle_dict.
    Computed in float64 to avoid overflow, then stored as float32.
    """
    stats = {}

    for name, tensor in particle_dict.items():
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN found in raw tensor before stats computation for particle '{name}'")

        if torch.isinf(tensor).any():
            raise ValueError(f"Inf found in raw tensor before stats computation for particle '{name}'")

        # convert to float64 to compute the statistics
        tensor64 = tensor.double()

        mean = tensor64.mean(dim=0, keepdim=True)
        std = tensor64.std(dim=0, keepdim=True, unbiased=False)

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            bad_cols = (torch.isnan(mean) | torch.isinf(mean)).nonzero(as_tuple=False)[:, 1].tolist()

            # also print min/max of each problematic column
            debug_info = []
            for col_idx in bad_cols[:10]:
                col = tensor64[:, col_idx]
                debug_info.append(
                    {
                        "col_idx": int(col_idx),
                        "min": float(col.min().item()),
                        "max": float(col.max().item()),
                        "abs_max": float(col.abs().max().item()),
                    }
                )

            raise ValueError(
                f"Invalid mean found for particle '{name}'. "
                f"Problem column indices: {bad_cols[:20]}. "
                f"Debug info: {debug_info}"
            )

        if torch.isnan(std).any() or torch.isinf(std).any():
            bad_cols = (torch.isnan(std) | torch.isinf(std)).nonzero(as_tuple=False)[:, 1].tolist()

            debug_info = []
            for col_idx in bad_cols[:10]:
                col = tensor64[:, col_idx]
                debug_info.append(
                    {
                        "col_idx": int(col_idx),
                        "min": float(col.min().item()),
                        "max": float(col.max().item()),
                        "abs_max": float(col.abs().max().item()),
                    }
                )

            raise ValueError(
                f"Invalid std found for particle '{name}'. "
                f"Problem column indices: {bad_cols[:20]}. "
                f"Debug info: {debug_info}"
            )

        # Either float32 or double would work here, but float32 matches what
        # the standardisation and the model input expect.
        stats[name] = {
            "mean": mean.float(),
            "std": std.float(),
        }

    return stats

def apply_standardization(particle_dict, stats, eps=1e-8):
    """
    Standardise particle_dict with the given stats (mean/std).
    Used for train, val and test alike.
    """
    standardized_particle_dict = {}

    for name, tensor in particle_dict.items():
        if name not in stats:
            raise ValueError(f"Stats for particle '{name}' not found.")

        mean = stats[name]["mean"]
        std = stats[name]["std"]

        if torch.isnan(tensor).any():
            raise ValueError(f"NaN found before standardization for particle '{name}'")

        if torch.isinf(tensor).any():
            raise ValueError(f"Inf found before standardization for particle '{name}'")

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            raise ValueError(f"Invalid mean found before standardization for particle '{name}'")

        if torch.isnan(std).any() or torch.isinf(std).any():
            raise ValueError(f"Invalid std found before standardization for particle '{name}'")

        standardized_tensor = (tensor - mean) / (std + eps)

        if torch.isnan(standardized_tensor).any():
            bad_cols = torch.isnan(standardized_tensor).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"NaN found after standardization for particle '{name}'. "
                f"Problem column indices: {bad_cols[:20]}"
            )

        if torch.isinf(standardized_tensor).any():
            bad_cols = torch.isinf(standardized_tensor).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Inf found after standardization for particle '{name}'. "
                f"Problem column indices: {bad_cols[:20]}"
            )

        standardized_particle_dict[name] = standardized_tensor.float()

    return standardized_particle_dict

def save_standardization_stats(stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(stats, save_path)


def load_standardization_stats(load_path):

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Stats file not found: {load_path}")

    stats = torch.load(load_path)
    return stats

# Returns the standardised tensors and labels per particle and per feature,
# together with the statistics used for the standardisation.
def build_split_particle_dict(data_file_path, particle_columns_file, stats=None, stats_save_path=None, label_column="isSignal", particles=None, tree_name="O2hfxicxi2pifull",):

    particle_dict, labels = get_raw_data(data_file_path=data_file_path, particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name, label_column=label_column,)

    if stats is None:
        stats = compute_standardization_stats(particle_dict)
        if stats_save_path is not None:
            save_standardization_stats(stats, stats_save_path)

    standardized_particle_dict = apply_standardization(particle_dict, stats)

    return standardized_particle_dict, labels, stats


def build_all_splits(train_path, val_path, test_path, particle_columns_file, stats_save_path, stats_signal_path=None, stats_background_path=None, label_column="isSignal", stats_pt_range=None, pt_column="fPt", particles=None, tree_name="O2hfxicxi2pifull",):
    """
    train/val/test are single files mixing signal and background, as before.

    Three ways to obtain the standardisation stats (mean/std), in priority order:

    1. stats_signal_path + stats_background_path  (both set)
       Combined distribution of those two files.
    2. stats_pt_range = (pt_min, pt_max)
       train_path restricted to that pT window. Use this when the train file
       mixes a wide-pT signal sample with window-restricted background: the
       signal in train_*.root spans pT > 3 (or > 4) while the background is
       already cut to the analysis window, so the unrestricted distribution is
       not the one the model sees at inference time.
    3. neither
       Whole train_path distribution (original behaviour).

    Only the stats are affected. The train/val/test tensors returned for
    training are never filtered by pT.
    """
    if stats_signal_path is not None and stats_background_path is not None:
        # Compute mean/std from the combined stats-only signal/background
        # distribution. (Labels do not affect the stats, but are assigned
        # signal=1 / background=0 anyway.)
        stats_particle_dict_raw, _ = get_combined_raw_data(
            data_file_paths=[stats_signal_path, stats_background_path],
            particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name,
            label_column=label_column,
            assign_labels=[1, 0],
        )
        stats = compute_standardization_stats(stats_particle_dict_raw)
        if stats_save_path is not None:
            save_standardization_stats(stats, stats_save_path)
    elif stats_pt_range is not None:
        # Compute mean/std from train_path restricted to the analysis pT window.
        # The rows dropped here are only excluded from the stats, not from training.
        print(f"[INFO] Computing stats from {train_path} "
              f"restricted to pT {tuple(stats_pt_range)}")
        stats_particle_dict_raw, _ = get_raw_data(
            data_file_path=train_path,
            particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name,
            label_column=label_column,
            pt_range=stats_pt_range,
            pt_column=pt_column,
        )
        stats = compute_standardization_stats(stats_particle_dict_raw)
        if stats_save_path is not None:
            save_standardization_stats(stats, stats_save_path)
    else:
        stats = None  # computed from train_path in the train step below

    # 1. train: load the mixed train file -> compute and save stats if needed
    #    -> standardise
    train_particle_dict, train_labels, stats = build_split_particle_dict(
        data_file_path=train_path,
        particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name,
        stats=stats,
        stats_save_path=stats_save_path,
        label_column=label_column,
    )

    # 2. val: apply the train stats
    val_particle_dict, val_labels, _ = build_split_particle_dict(
        data_file_path=val_path,
        particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name,
        stats=stats,
        stats_save_path=None,
        label_column=label_column,
    )

    # 3. test: apply the train stats
    test_particle_dict, test_labels, _ = build_split_particle_dict(
        data_file_path=test_path,
        particle_columns_file=particle_columns_file, particles=particles, tree_name=tree_name,
        stats=stats,
        stats_save_path=None,
        label_column=label_column,
    )

    return (
        train_particle_dict, train_labels,
        val_particle_dict, val_labels,
        test_particle_dict, test_labels,
        stats
    )

if __name__ == "__main__":
    import argparse
    from utils import load_config

    parser = argparse.ArgumentParser(
        description="read paths from config.yaml, standardise train/val/test and save stats"
    )
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    (
        train_particle_dict, train_labels,
        val_particle_dict, val_labels,
        test_particle_dict, test_labels,
        stats
    ) = build_all_splits(
        train_path=paths["train_csv"],
        val_path=paths["val_csv"],
        test_path=paths["test_csv"],
        particle_columns_file=paths.get("particle_columns_file"),
        particles=config.get("particles"),
        tree_name=config["data"].get("tree_name", "O2hfxicxi2pifull"),
        stats_save_path=paths["stats_save_path"],
        stats_signal_path=paths.get("stats_signal_path"),
        stats_background_path=paths.get("stats_background_path"),
        label_column=config["data"]["label_column"],
    )

    print("=== Train ===")
    for name, tensor in train_particle_dict.items():
        print(f"{name}: {tensor.shape}")
    print("train_labels:", train_labels.shape)

    print("\n=== Validation ===")
    for name, tensor in val_particle_dict.items():
        print(f"{name}: {tensor.shape}")
    print("val_labels:", val_labels.shape)

    print("\n=== Test ===")
    for name, tensor in test_particle_dict.items():
        print(f"{name}: {tensor.shape}")
    print("test_labels:", test_labels.shape)

    n_sig = int((train_labels == 1).sum())
    n_bkg = int((train_labels == 0).sum())
    print(f"\ntrain label counts -> signal: {n_sig} | background: {n_bkg}")
    print(f"Saved train stats to: {paths['stats_save_path']}")
