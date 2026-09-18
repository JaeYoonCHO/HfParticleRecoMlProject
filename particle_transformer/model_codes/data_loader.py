import torch
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import build_all_splits


class ParticleDataset(Dataset):
    """
    Dataset that takes particle_dict and labels and returns per-event samples.
    """

    def __init__(self, particle_dict, labels):
        """
        particle_dict:
            {
                "Xicplus": tensor of shape (N, d1),
                "Ximinus": tensor of shape (N, d2),
                ...
            }

        labels:
            tensor of shape (N,)
        """
        self.particle_dict = particle_dict
        self.labels = labels
        self.particle_names = list(particle_dict.keys())

        self.num_samples = labels.shape[0]

        for name in self.particle_names:
            if particle_dict[name].shape[0] != self.num_samples:
                raise ValueError(
                    f"Number of samples mismatch for particle '{name}': "
                    f"{particle_dict[name].shape[0]} != {self.num_samples}"
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_particle_dict = {
            name: self.particle_dict[name][idx]
            for name in self.particle_names
        }

        label = self.labels[idx]

        return sample_particle_dict, label, idx


def get_input_example(particle_dict, batch_size=2):
    """
    Build an example particle_dict for initialising ParticleEmbedding.
    The batch size is irrelevant; only the feature dimensions matter.
    """
    example_particle_dict = {
        name: tensor[:batch_size]
        for name, tensor in particle_dict.items()
    }
    return example_particle_dict


def get_dataloaders(train_path, val_path, test_path, particle_columns_file, stats_save_path, stats_signal_path=None, stats_background_path=None, batch_size=128, num_workers=0, pin_memory=False, label_column="isSignal", drop_last=False, stats_pt_range=None, pt_column="fPt", particles=None, tree_name="O2hfxicxi2pifull",):
    """
    Full data pipeline:
    1) load the train/val/test split from preprocess.py
    2) compute stats on train
    3) standardise val/test with the train stats
    4) build the Datasets
    5) build the DataLoaders

    Returns:
    - train_loader
    - val_loader
    - test_loader
    - example_particle_dict  (for initialising ParticleEmbedding)
    - stats
    """

    (
        train_particle_dict, train_labels,
        val_particle_dict, val_labels,
        test_particle_dict, test_labels,
        stats
    ) = build_all_splits(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        particle_columns_file=particle_columns_file,
        particles=particles,
        tree_name=tree_name,
        stats_save_path=stats_save_path,
        stats_signal_path=stats_signal_path,
        stats_background_path=stats_background_path,
        label_column=label_column,
        stats_pt_range=stats_pt_range,
        pt_column=pt_column,
    )

    train_dataset = ParticleDataset(train_particle_dict, train_labels)
    val_dataset = ParticleDataset(val_particle_dict, val_labels)
    test_dataset = ParticleDataset(test_particle_dict, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    example_particle_dict = get_input_example(train_particle_dict, batch_size=2)

    return train_loader, val_loader, test_loader, example_particle_dict, stats


if __name__ == "__main__":
    import argparse
    from utils import load_config

    parser = argparse.ArgumentParser(description="check the dataloader using config.yaml")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    train_loader, val_loader, test_loader, example_particle_dict, stats = get_dataloaders(
        train_path=paths["train_csv"],
        val_path=paths["val_csv"],
        test_path=paths["test_csv"],
        particle_columns_file=paths.get("particle_columns_file"),
        particles=config.get("particles"),
        tree_name=config["data"].get("tree_name", "O2hfxicxi2pifull"),
        stats_save_path=paths["stats_save_path"],
        stats_signal_path=paths.get("stats_signal_path"),
        stats_background_path=paths.get("stats_background_path"),
        batch_size=config["data"]["batch_size"],
        num_workers=0,
        pin_memory=False,
        label_column=config["data"]["label_column"],
        drop_last=False,
    )

    print("=== Example particle dict for ParticleEmbedding ===")
    for name, tensor in example_particle_dict.items():
        print(f"{name}: {tensor.shape}")

    print("\n=== One batch from train_loader ===")
    batch_particle_dict, batch_labels, batch_idx = next(iter(train_loader))

    for name, tensor in batch_particle_dict.items():
        print(f"{name}: {tensor.shape}")
    print("labels:", batch_labels.shape)
    print("indices:", batch_idx.shape)