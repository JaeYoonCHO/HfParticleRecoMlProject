import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from tqdm import tqdm

from model import ParticleTransformerClassifier
from data_loader import get_dataloaders
from utils import (
    load_config,
    ensure_dir,
    set_seed,
    get_device,
    move_particle_dict_to_device,
    EpochTimer,
    EpochMetricTracker,
    save_checkpoint,
    save_model_state,
    save_dataframe,
    format_metric_value,
    make_history_row,
)


def check_finite(tensor, name, context):
    """Fail before invalid inputs, logits or losses reach optimization/metrics."""
    if not torch.isfinite(tensor).all():
        raise ValueError(f"NaN or Inf detected in {name} ({context})")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    score_threshold=0.85):
    model.train()

    tracker = EpochMetricTracker(score_threshold)
    timer = EpochTimer()
    timer.start()

    progress_bar = tqdm(
        loader,
        desc=f"Train [{epoch}/{total_epochs}]",
        leave=False,
    )

    for batch_idx, (batch_particle_dict, batch_labels, batch_indices) in enumerate(progress_bar):
        context = f"train, epoch {epoch}, batch {batch_idx}"
        for name, tensor in batch_particle_dict.items():
            check_finite(tensor, f"input '{name}'", context)
        check_finite(batch_labels, "labels", context)

        batch_particle_dict = move_particle_dict_to_device(batch_particle_dict, device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        logits = model(batch_particle_dict)

        check_finite(logits, "logits", context)

        loss = criterion(logits, batch_labels)

        check_finite(loss, "loss", context)

        loss.backward()

        # guard against gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        tracker.update(loss, logits, batch_labels, batch_indices)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    timer.stop()

    summary, raw_pred_df, roc_curve_df = tracker.compute_epoch_metrics(
        epoch=epoch,
        split="train",
        epoch_time_sec=timer.elapsed(),
    )

    return summary, raw_pred_df, roc_curve_df


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, epoch, total_epochs, split,
                       score_threshold=0.85):
    model.eval()

    tracker = EpochMetricTracker(score_threshold)
    timer = EpochTimer()
    timer.start()

    progress_bar = tqdm(
        loader,
        desc=f"{split.capitalize()} [{epoch}/{total_epochs}]",
        leave=False,
    )

    for batch_idx, (batch_particle_dict, batch_labels, batch_indices) in enumerate(progress_bar):
        context = f"{split}, epoch {epoch}, batch {batch_idx}"
        for name, tensor in batch_particle_dict.items():
            check_finite(tensor, f"input '{name}'", context)
        check_finite(batch_labels, "labels", context)

        batch_particle_dict = move_particle_dict_to_device(batch_particle_dict, device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_particle_dict)

        check_finite(logits, "logits", context)

        loss = criterion(logits, batch_labels)

        check_finite(loss, "loss", context)

        tracker.update(loss, logits, batch_labels, batch_indices)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    timer.stop()

    summary, raw_pred_df, roc_curve_df = tracker.compute_epoch_metrics(
        epoch=epoch,
        split=split,
        epoch_time_sec=timer.elapsed(),
    )

    print(
        f"\n{split} | score >= {score_threshold:g} | "
        f"purity: {format_metric_value(summary['purity'])} | "
        f"signal efficiency: {format_metric_value(summary['signal_efficiency'])} | "
        f"background efficiency: {format_metric_value(summary['background_efficiency'])}"
    )

    return summary, raw_pred_df, roc_curve_df


def print_epoch_summary(train_summary, val_summary, total_epochs):
    epoch = train_summary["epoch"]

    print(f"\nEpoch [{epoch}/{total_epochs}]")
    print(
        "Train | "
        f"loss: {format_metric_value(train_summary['loss'])} | "
        f"acc: {format_metric_value(train_summary['acc'])} | "
        f"auc: {format_metric_value(train_summary['auc'])} | "
        f"time: {format_metric_value(train_summary['epoch_time_sec'])} sec"
    )
    print(
        "Val   | "
        f"loss: {format_metric_value(val_summary['loss'])} | "
        f"acc: {format_metric_value(val_summary['acc'])} | "
        f"auc: {format_metric_value(val_summary['auc'])} | "
        f"time: {format_metric_value(val_summary['epoch_time_sec'])} sec"
    )


def main(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_3-4GeV.yaml")
    config = load_config(config_path)
    score_threshold = config.get("metrics", {}).get("score_threshold", 0.85)
    EpochMetricTracker(score_threshold)  # Validate before reading large datasets.

    seed = config["train"]["seed"]
    device = get_device(config["train"]["device"])

    set_seed(seed)

    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    save_dir = config["save"]["save_dir"]
    ensure_dir(save_dir)

    history_dir = os.path.join(save_dir, "history")
    roc_dir = os.path.join(save_dir, "roc_tables")
    raw_pred_dir = os.path.join(save_dir, "raw_predictions")

    ensure_dir(history_dir)
    ensure_dir(roc_dir)
    ensure_dir(raw_pred_dir)

    best_model_path = os.path.join(save_dir, config["save"]["best_model_name"])
    final_model_path = os.path.join(save_dir, config["save"]["final_model_name"])
    best_checkpoint_path = os.path.join(save_dir, "best_checkpoint.pt")
    final_checkpoint_path = os.path.join(save_dir, "final_checkpoint.pt")

    # Restrict the standardisation stats to the analysis pT window when the
    # config provides one. Only the stats are affected, not the training data.
    stats_pt_min = config["paths"].get("stats_pt_min")
    stats_pt_max = config["paths"].get("stats_pt_max")
    if (stats_pt_min is None) != (stats_pt_max is None):
        raise ValueError("stats_pt_min and stats_pt_max must be set together")
    stats_pt_range = None if stats_pt_min is None else (float(stats_pt_min), float(stats_pt_max))

    train_loader, val_loader, test_loader, example_particle_dict, stats = get_dataloaders(
        train_path=config["paths"]["train_csv"],
        val_path=config["paths"]["val_csv"],
        test_path=config["paths"]["test_csv"],
        particle_columns_file=config["paths"].get("particle_columns_file"),
        particles=config.get("particles"),
        tree_name=config["data"].get("tree_name", "O2hfxicxi2pifull"),
        stats_save_path=config["paths"]["stats_save_path"],
        stats_signal_path=config["paths"].get("stats_signal_path"),
        stats_background_path=config["paths"].get("stats_background_path"),
        stats_pt_range=stats_pt_range,
        pt_column=config["paths"].get("pt_column", "fPt"),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        label_column=config["data"]["label_column"],
        drop_last=config["data"]["drop_last"],
    )

    # A dedicated generator keeps evaluation from advancing training's RNG state.
    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    scheduler = None
    if config["scheduler"]["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"],
        )

    epochs = config["train"]["epochs"]
    print_every = config["log"]["print_every"]

    history_rows = []

    best_val_auc = float("-inf")
    best_val_acc = float("-inf")
    best_epoch = -1
    best_model_state = None

    best_val_raw_pred_df = None
    best_val_roc_curve_df = None

    start_epoch = 1
    resume_cfg = config.get("resume", {})
    if resume_cfg.get("enabled", False):
        ckpt_path = resume_cfg["checkpoint_path"]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint.get("best_val_auc", float("-inf"))
        best_val_acc = checkpoint.get("best_val_acc", float("-inf"))
        print(f"Resumed from checkpoint: {ckpt_path}")
        print(f"Starting from epoch {start_epoch} / {epochs}")

        history_csv_path = os.path.join(history_dir, "training_history.csv")
        if os.path.exists(history_csv_path):
            prev_history_df = pd.read_csv(history_csv_path)
            history_rows = prev_history_df.to_dict(orient="records")
            print(f"Loaded {len(history_rows)} rows from existing training_history.csv")

    for epoch in range(start_epoch, epochs + 1):
        train_summary, train_raw_pred_df, train_roc_curve_df = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            score_threshold=score_threshold,
        )

        val_summary, val_raw_pred_df, val_roc_curve_df = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            split="val",
            score_threshold=score_threshold,
        )

        train_eval_summary, _, _ = evaluate_one_epoch(
            model=model,
            loader=train_eval_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            split="train_eval",
            score_threshold=score_threshold,
        )

        if scheduler is not None:
            scheduler.step()

        history_rows.append(make_history_row(train_summary))
        history_rows.append(make_history_row(val_summary))
        history_rows.append(make_history_row(train_eval_summary))

        if epoch % print_every == 0:
            print_epoch_summary(train_summary, val_summary, epochs)
            print(f"Train eval | auc: {format_metric_value(train_eval_summary['auc'])}")
        # Select by validation AUC; use accuracy only when AUC is undefined.
        current_val_auc = val_summary["auc"]
        current_val_acc = val_summary["acc"]

        is_best = False

        if current_val_auc is not None:
            if current_val_auc > best_val_auc:
                is_best = True
        else:
            if best_val_auc == float("-inf") and current_val_acc > best_val_acc:
                is_best = True

        every_model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        save_model_state(model, every_model_path)

        if is_best:
            best_epoch = epoch
            best_val_auc = current_val_auc if current_val_auc is not None else best_val_auc
            best_val_acc = current_val_acc

            best_model_state = copy.deepcopy(model.state_dict())
            best_val_raw_pred_df = val_raw_pred_df.copy()
            best_val_roc_curve_df = None if val_roc_curve_df is None else val_roc_curve_df.copy()

            save_model_state(model, best_model_path)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                    "best_val_auc": best_val_auc,
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
                best_checkpoint_path,
            )

            save_dataframe(
                best_val_raw_pred_df,
                os.path.join(raw_pred_dir, "best_val_raw_predictions.csv"),
            )

            save_dataframe(
                best_val_roc_curve_df,
                os.path.join(roc_dir, "best_val_roc_curve.csv"),
            )

            print("Best model updated.")

    history_df = pd.DataFrame(history_rows)
    save_dataframe(history_df, os.path.join(history_dir, "training_history.csv"))

    save_model_state(model, final_model_path)
    save_checkpoint(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "best_val_auc": best_val_auc,
            "best_val_acc": best_val_acc,
            "config": config,
        },
        final_checkpoint_path,
    )

    # also save the validation result of the last epoch
    save_dataframe(
        val_raw_pred_df,
        os.path.join(raw_pred_dir, "final_val_raw_predictions.csv"),
    )
    save_dataframe(
        val_roc_curve_df,
        os.path.join(roc_dir, "final_val_roc_curve.csv"),
    )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model from epoch {best_epoch} for test evaluation.")
    else:
        print("\nNo best model state was recorded separately. Using current final model for test.")

    test_summary, test_raw_pred_df, test_roc_curve_df = evaluate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        epoch=best_epoch if best_epoch != -1 else epochs,
        total_epochs=epochs,
        split="test",
        score_threshold=score_threshold,
    )

    save_dataframe(
        test_raw_pred_df,
        os.path.join(raw_pred_dir, "test_raw_predictions.csv"),
    )
    save_dataframe(
        test_roc_curve_df,
        os.path.join(roc_dir, "test_roc_curve.csv"),
    )

    test_history_df = pd.DataFrame([make_history_row(test_summary)])
    save_dataframe(test_history_df, os.path.join(history_dir, "test_summary.csv"))

    print("\n==============================")
    print("Training finished")
    print("==============================")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best val acc    : {format_metric_value(best_val_acc)}")
    print(f"Best val auc    : {format_metric_value(best_val_auc)}")
    print("------------------------------")
    print(f"Test loss       : {format_metric_value(test_summary['loss'])}")
    print(f"Test acc        : {format_metric_value(test_summary['acc'])}")
    print(f"Test auc        : {format_metric_value(test_summary['auc'])}")
    print(f"Test time (sec) : {format_metric_value(test_summary['epoch_time_sec'])}")
    print("------------------------------")
    print(f"Best model path : {best_model_path}")
    print(f"Final model path: {final_model_path}")
    print(f"History path    : {os.path.join(history_dir, 'training_history.csv')}")
    print(f"Test summary    : {os.path.join(history_dir, 'test_summary.csv')}")
    print(f"Raw preds dir   : {raw_pred_dir}")
    print(f"ROC dir         : {roc_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="train the Xi_c^+ particle transformer classifier"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to the config yaml (default: config_3-4GeV.yaml next to this script)",
    )
    args = parser.parse_args()
    main(args.config)
