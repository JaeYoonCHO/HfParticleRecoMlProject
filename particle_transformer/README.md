# Particle Transformer

This folder is a self-contained example for sharing the particle-token Transformer with other researchers. It supports configurable particle tokens, CSV feature selection, optional sinusoidal positional encoding, ROOT tree names, training-time evaluation, and reproducible selection metrics.

## Quick start

From this folder, after installing `requirements.txt`:

```bash
python scripts/make_example_data.py
python model_codes/train.py --config configs/train_example.yaml
python model_codes/Inference/Inference.py --config configs/inference_example.yaml
```

The example uses generated ROOT files and writes ignored results under `outputs/example/`. It does not require the original analysis dataset.

The optional batch workflow uses ROOT files containing `DF_*` directories:

```bash
python model_codes/Inference/batch_inference.py configs/batch_inference_example.yaml
```

## Configuration

The `particles` mapping defines token order and feature order. A value can be an explicit feature list:

```yaml
particles:
  mother: [decay_length, pointing_cos]
```

or a selector into a CSV file:

```yaml
particles:
  mother: {particle: Parent}
```

The CSV requires only `Branch,Particle`; `Description` is optional metadata. Feature names and token order must be identical during training and inference. The example CSV is `examples/Particle_name.csv`.

Set `model.use_positional_encoding: true` to add sinusoidal encoding to the CLS and particle tokens. This changes model outputs and must match the setting used to train the checkpoint. `data.tree_name` selects the training ROOT tree.

## Metrics in history

Set `metrics.score_threshold` to choose the inclusive signal-score cut (default `0.85`). New `training_history.csv` and `test_summary.csv` files contain:

- `purity = selected signal / selected candidates`
- `signal_efficiency = selected signal / all signal`
- `background_efficiency = selected background / all background`
- `background_rejection = 1 - background_efficiency`

The counts and threshold are also stored. These are unweighted metrics on the evaluated sample; purity depends on its signal/background mixture. The old hard-coded `ratio = signal / (150 * background)` is not written by new runs.

Each epoch records `train`, `val`, and `train_eval`. The last one evaluates all training rows with fixed weights and dropout disabled.

## Tests

```bash
python -m unittest discover -s model_codes/tests -v
```

Tests use temporary synthetic data and do not modify production datasets.
