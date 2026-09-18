import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, get_device, move_particle_dict_to_device
from model import ParticleTransformerClassifier
from data_preprocessing import get_particle_feature_map, load_standardization_stats


# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, particle_dict):
        self.particle_dict = particle_dict
        self.particle_names = list(particle_dict.keys())
        self.num_samples = next(iter(particle_dict.values())).shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {
            name: self.particle_dict[name][idx]
            for name in self.particle_names
        }
        return sample, idx


# ============================================================
# ROOT → DataFrame → standardized particle_dict
# ============================================================
def build_particle_dict_from_root(root_path, tree_name, particle_columns_file, stats, particles=None):
    particle_feature_map = get_particle_feature_map(particle_columns_file, particles)

    feature_cols = []
    for features in particle_feature_map.values():
        feature_cols.extend(features)
    feature_cols = list(dict.fromkeys(feature_cols))

    print(f"[INFO] Reading ROOT: {root_path}")

    with uproot.open(root_path) as f:
        if tree_name not in f:
            print("[ERROR] Available keys:", f.keys())
            raise KeyError(f"Tree '{tree_name}' not found")

        tree = f[tree_name]

        missing = [c for c in feature_cols if c not in tree.keys()]
        if missing:
            raise ValueError(f"Missing branches in ROOT:\n{missing}")

        arrays = tree.arrays(feature_cols, library="np")
        df = pd.DataFrame({c: arrays[c] for c in feature_cols})

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    finite_mask = np.isfinite(df[feature_cols].to_numpy()).all(axis=1)
    if (~finite_mask).any():
        print(f"[INFO] Removing {(~finite_mask).sum()} non-finite rows")
        df = df.loc[finite_mask].reset_index(drop=True)

    particle_dict = {}

    for name, features in particle_feature_map.items():
        x = df[features].to_numpy(dtype=np.float32)
        x = torch.from_numpy(x)

        mean = stats[name]["mean"]
        std = stats[name]["std"]

        x = (x - mean) / (std + 1e-8)
        particle_dict[name] = x.float()

    return df, particle_dict, finite_mask


# ============================================================
# Predict prob_1
# ============================================================
@torch.no_grad()
def predict_prob1(model, loader, device, pe_scale=0.2):
    model.eval()

    all_idx = []
    all_prob1 = []

    for batch_particle_dict, batch_idx in loader:
        batch_particle_dict = move_particle_dict_to_device(batch_particle_dict, device)

        logits = model(batch_particle_dict,)
        probs = torch.softmax(logits, dim=1)

        all_idx.append(batch_idx.cpu())
        all_prob1.append(probs[:, 1].cpu())

    all_idx = torch.cat(all_idx).numpy()
    all_prob1 = torch.cat(all_prob1).numpy().astype(np.float32)

    pred_df = pd.DataFrame({
        "row_index": all_idx,
        "prob_1": all_prob1,
    })

    pred_df = pred_df.sort_values("row_index").reset_index(drop=True)
    return pred_df["prob_1"].to_numpy(dtype=np.float32)


# ============================================================
# Save selected branches + score branch
# ============================================================
def save_root_with_prob(input_root, output_root, tree_name, prob_1, finite_mask,
                        output_branches, score_branch="Transformer_score"):
    with uproot.open(input_root) as f:
        tree = f[tree_name]
        available = set(tree.keys())

        keep = []
        for branch in output_branches:
            if branch not in available:
                print(f"[WARNING] Branch not found in input tree, skipping: {branch}")
                continue
            keep.append(branch)

        arrays = tree.arrays(keep, library="np") if keep else {}

    out_dict = {}

    for branch in keep:
        arr = arrays[branch][finite_mask]

        if arr.dtype == object:
            print(f"[WARNING] Skip object branch: {branch}")
            continue

        out_dict[branch] = arr

    if not out_dict:
        print("[WARNING] No input branches kept; writing score only")

    out_dict[score_branch] = prob_1

    out_dir = os.path.dirname(output_root)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(output_root) as f:
        f[tree_name] = out_dict

    print(f"[DONE] Saved ROOT with {list(out_dict.keys())}: {output_root}")


# ============================================================
# Main
# ============================================================
def main(config_path):
    config = load_config(config_path)
    print(f"[INFO] Loaded config: {config_path}")
    device = get_device(config["train"]["device"])
    input_root =config["dataset"]["input_root"]
    output_root =config["dataset"]["output_root"]
    tree_name =config["dataset"]["tree_name"]
    output_branches =config["dataset"]["output_branches"]
    score_branch =config["dataset"].get("score_branch", "Transformer_score")

    particle_columns_file = config["paths"].get("particle_columns_file")
    stats_path = config["paths"]["stats_save_path"]

    model_path =config["model"]["model_path"]

    stats = load_standardization_stats(stats_path)

    df, particle_dict, finite_mask = build_particle_dict_from_root(
        root_path=input_root,
        tree_name=tree_name,
        particle_columns_file=particle_columns_file,
        stats=stats,
        particles=config.get("particles"),
    )

    example_particle_dict = {
        name: tensor[:2]
        for name, tensor in particle_dict.items()
    }

    dataset = InferenceDataset(particle_dict)
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = ParticleTransformerClassifier(
        particle_dict=example_particle_dict,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        ff_dim=config["model"]["ff_dim"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"],
        head_hidden_dim=config["model"]["head_hidden_dim"],
        use_final_norm=config["model"]["use_final_norm"],
        use_positional_encoding=config["model"].get("use_positional_encoding", False),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model: {model_path}")

    prob_1 = predict_prob1(model, loader, device)

    print(f"[INFO] valid events: {len(prob_1)}")
    print(f"[INFO] prob_1 mean: {prob_1.mean():.6f}")
    print(f"[INFO] prob_1 max : {prob_1.max():.6f}")
    print(f"[INFO] prob_1 >= 0.85: {(prob_1 >= 0.85).sum()}")

    save_root_with_prob(
        input_root=input_root,
        output_root=output_root,
        tree_name=tree_name,
        prob_1=prob_1,
        finite_mask=finite_mask,
        output_branches=output_branches,
        score_branch=score_branch,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a ROOT tree with a trained classifier")
    parser.add_argument("--config", required=True, help="YAML inference config")
    main(parser.parse_args().config)
