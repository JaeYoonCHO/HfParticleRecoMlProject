import os
import time
import random
import yaml
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, roc_curve


def load_config(config_path):

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path):

    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42, deterministic=True):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(device_str="cuda"):

    device_str = str(device_str).lower()

    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device("cpu")


def move_particle_dict_to_device(particle_dict, device):
    """
    Move every tensor inside batch_particle_dict to the given device.
    """
    return {name: tensor.to(device) for name, tensor in particle_dict.items()}


def extract_batch_outputs(logits, labels):
    """
    Take per-batch logits and labels and return the following.

    Returns:
    - probs: softmax probability, shape (batch, num_classes)
    - preds: predicted class index, shape (batch,)
    - num_correct: number of correct predictions in the batch (int)
    """
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    num_correct = (preds == labels).sum().item()

    return probs, preds, num_correct


def safe_compute_auc(labels, positive_probs):
    """
    Compute the binary-classification AUC safely.
    Returns None if only one class is present, since sklearn would raise.
    Also raises on NaN / Inf so the problem is immediately visible.
    """
    labels = np.asarray(labels)
    positive_probs = np.asarray(positive_probs)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    if np.isnan(positive_probs).any():
        nan_idx = np.where(np.isnan(positive_probs))[0][:10]
        raise ValueError(
            f"NaN detected in positive_probs before AUC computation. "
            f"Example positions: {nan_idx.tolist()}"
        )

    if np.isinf(positive_probs).any():
        inf_idx = np.where(np.isinf(positive_probs))[0][:10]
        raise ValueError(
            f"Inf detected in positive_probs before AUC computation. "
            f"Example positions: {inf_idx.tolist()}"
        )

    auc = roc_auc_score(labels, positive_probs)
    return float(auc)


def build_raw_prediction_table(
    labels,
    preds,
    probs,
    indices,
    epoch,
    split,
):
    """
    Build a raw prediction table (DataFrame) so the ROC can be drawn later.

    For binary classification:
    - true_label
    - pred_label
    - prob_0
    - prob_1
    - correct
    - epoch
    - split
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, but got shape {probs.shape}")

    num_classes = probs.shape[1]
    if num_classes < 2:
        raise ValueError("For ROC/AUC, probs must have at least 2 classes.")

    df = pd.DataFrame({
        "epoch": epoch,
        "split": split,
        "true_label": labels.astype(int),
        "pred_label": preds.astype(int),
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "index": indices.astype(int),
    })

    df["correct"] = (df["true_label"] == df["pred_label"]).astype(int)

    return df


def compute_selection_metrics(labels, positive_probs, score_threshold=0.85):
    """Unweighted metrics for signal=1 and background=0; undefined fractions are None."""
    labels = np.asarray(labels)
    positive_probs = np.asarray(positive_probs)
    if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be finite and in [0, 1]")
    if labels.ndim != 1 or positive_probs.shape != labels.shape:
        raise ValueError("labels and positive_probs must be matching 1D arrays")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Selection metrics require signal=1 and background=0")
    if not np.isfinite(positive_probs).all() or ((positive_probs < 0) | (positive_probs > 1)).any():
        raise ValueError("positive_probs must be finite probabilities in [0, 1]")

    selected = positive_probs >= score_threshold
    signal = labels == 1
    background = labels == 0
    num_signal = int(signal.sum())
    num_background = int(background.sum())
    selected_signal = int((selected & signal).sum())
    selected_background = int((selected & background).sum())
    num_selected = selected_signal + selected_background
    background_efficiency = selected_background / num_background if num_background else None
    return {
        "score_threshold": float(score_threshold),
        "num_signal": num_signal,
        "num_background": num_background,
        "selected_signal": selected_signal,
        "selected_background": selected_background,
        "purity": selected_signal / num_selected if num_selected else None,
        "signal_efficiency": selected_signal / num_signal if num_signal else None,
        "background_efficiency": background_efficiency,
        "background_rejection": 1 - background_efficiency if num_background else None,
    }


def build_roc_curve_table(labels, positive_probs, epoch, split):
    """
    Build a DataFrame holding the ROC curve coordinates.
    Returns None if only one class is present, since no ROC can be built.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    fpr, tpr, thresholds = roc_curve(labels, positive_probs)

    df = pd.DataFrame({
        "epoch": epoch,
        "split": split,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })

    return df


def save_dataframe(df, save_path):
 
    if df is None:
        return

    save_dir = os.path.dirname(save_path)
    ensure_dir(save_dir)
    df.to_csv(save_path, index=False)


def save_checkpoint(state, save_path):

    save_dir = os.path.dirname(save_path)
    ensure_dir(save_dir)
    torch.save(state, save_path)


def save_model_state(model, save_path):

    save_dir = os.path.dirname(save_path)
    ensure_dir(save_dir)
    torch.save(model.state_dict(), save_path)


class EpochTimer:


    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        end_time = self.end_time if self.end_time is not None else time.time()
        return float(end_time - self.start_time)


class EpochMetricTracker:
    """
    Tracks loss / acc / auc / raw predictions over one epoch.

    Usage:
    1) tracker = EpochMetricTracker()
    2) tracker.update(...) per batch
    3) tracker.compute_epoch_metrics(...) at the end of the epoch
    """

    def __init__(self, score_threshold=0.85):
        if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be finite and in [0, 1]")
        self.score_threshold = float(score_threshold)
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_correct = 0

        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        self.all_indices = []

    def update(self, loss, logits, labels, indices):
        """
        Accumulate the results of one batch.

        parameters
        ----------
        loss : torch.Tensor or float
            typically the result of criterion(logits, labels)
        logits : torch.Tensor
            shape (batch, num_classes)
        labels : torch.Tensor
            shape (batch,)
        """
        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = float(loss)

        batch_size = labels.size(0)

        probs, preds, num_correct = extract_batch_outputs(logits, labels)

        self.total_loss += loss_value * batch_size
        self.total_samples += batch_size
        self.total_correct += num_correct

        self.all_labels.append(labels.detach().cpu())
        self.all_preds.append(preds.detach().cpu())
        self.all_probs.append(probs.detach().cpu())
        self.all_indices.append(indices.detach().cpu())
        
    def compute_epoch_metrics(self, epoch, split, epoch_time_sec=None):
        """
        Return the summary dict and prediction tables from the accumulated epoch.

        Returns:
        summary_dict, raw_pred_df, roc_curve_df
        """
        if self.total_samples == 0:
            raise ValueError("No samples were accumulated in this epoch.")

        labels = torch.cat(self.all_labels, dim=0).numpy()
        preds = torch.cat(self.all_preds, dim=0).numpy()
        probs = torch.cat(self.all_probs, dim=0).numpy()
        indices = torch.cat(self.all_indices, dim=0).numpy()

        avg_loss = self.total_loss / self.total_samples
        acc = self.total_correct / self.total_samples

        positive_probs = probs[:, 1]
        auc = safe_compute_auc(labels, positive_probs)

        raw_pred_df = build_raw_prediction_table(
            labels=labels,
            preds=preds,
            probs=probs,
            indices=indices,
            epoch=epoch,
            split=split,
        )

        roc_curve_df = build_roc_curve_table(
            labels=labels,
            positive_probs=positive_probs,
            epoch=epoch,
            split=split,
        )

        summary = {
            "epoch": epoch,
            "split": split,
            "loss": float(avg_loss),
            "acc": float(acc),
            "auc": auc,
            "num_samples": int(self.total_samples),
            "epoch_time_sec": None if epoch_time_sec is None else float(epoch_time_sec),
        }

        summary.update(compute_selection_metrics(labels, positive_probs, self.score_threshold))
        return summary, raw_pred_df, roc_curve_df


def format_metric_value(value, digits=4):
    """
    Format a metric for printing; returns 'N/A' for None.
    """
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def make_history_row(summary_dict):
    """
    Convert a summary dict into a shape that maps onto a single DataFrame row.
    """
    return {
        "epoch": summary_dict["epoch"],
        "split": summary_dict["split"],
        "loss": summary_dict["loss"],
        "acc": summary_dict["acc"],
        "auc": summary_dict["auc"],
        "num_samples": summary_dict["num_samples"],
        "epoch_time_sec": summary_dict["epoch_time_sec"],
        **{key: summary_dict[key] for key in (
            "score_threshold", "num_signal", "num_background",
            "selected_signal", "selected_background", "purity",
            "signal_efficiency", "background_efficiency", "background_rejection",
        )},
    }


if __name__ == "__main__":
    # Small smoke test
    set_seed(42)

    logits = torch.tensor([
        [2.0, 1.0],
        [0.5, 1.5],
        [1.2, 0.8],
        [0.2, 2.3],
    ])
    labels = torch.tensor([0, 1, 0, 1])
    loss = torch.tensor(0.35)

    tracker = EpochMetricTracker()
    timer = EpochTimer()

    timer.start()
    tracker.update(loss, logits, labels)
    timer.stop()

    summary, raw_pred_df, roc_curve_df = tracker.compute_epoch_metrics(
        epoch=1,
        split="val",
        epoch_time_sec=timer.elapsed(),
    )

    print("=== Summary ===")
    print(summary)

    print("\n=== Raw Prediction Table ===")
    print(raw_pred_df.head())

    print("\n=== ROC Curve Table ===")
    print(roc_curve_df.head() if roc_curve_df is not None else None)
