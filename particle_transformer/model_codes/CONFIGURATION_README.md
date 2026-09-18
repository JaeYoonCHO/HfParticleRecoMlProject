# Configurable particle tokens

Training (`train.py`), single-file inference (`Inference/Inference.py`) and batch
inference (`Inference/batch_inference.py`) share the top-level `particles` mapping.
Token order and each feature list's order are significant: use the same mapping
for training, standardization, resumed runs and inference.

## Explicit features

```yaml
particles:
  parent: [parent_decay_length, parent_pointing_angle]
  daughter: [daughter_pt, daughter_ip]

model:
  use_positional_encoding: false

data:
  tree_name: O2hfxicxi2pifull
```

These snippets supplement the existing full configs. With explicit feature lists,
`paths.particle_columns_file` can be omitted. Token names must be non-empty strings
without dots (they become PyTorch module names).

## Select features from CSV

```yaml
paths:
  particle_columns_file: model_codes/particle_file/my_particles.csv

particles:
  parent_token:
    particle: Parent
  daughter_token:
    particle: Daughter
```

```csv
Branch,Particle
parent_decay_length,Parent
parent_pointing_angle,Parent
daughter_pt,Daughter
```

Only `Particle` and `Branch` are read; `Description` is optional metadata. Matching
is exact. Features follow CSV row order. Explicit lists and CSV selectors may be
mixed. A selector without matching rows raises an error.

If `particles` is omitted, each distinct `Particle` value becomes a token, in
first-occurrence order. Before using this mode, distinguish the different pions
in the CSV's `Particle` column. Existing CSV files have not been rewritten.

## Existing experiments

Configs whose CSVs exist now contain explicit feature lists preserving the old
Description-based selection, including zero-feature tokens (`[]`). Editing a CSV
does not change these explicit lists; switch the relevant entry to a CSV selector
when ready. The old selection omitted some normalized pion-IP rows and retained
an empty bachelor-pion token for some feature sets. These choices remain unchanged.

The six `23feat` configs reference the absent `Particle_name_23feat.csv`; they
still need the correct CSV and an explicit mapping before reproducing old runs.
No replacement feature set was inferred.

Existing checkpoints remain compatible when token names, order, feature order,
feature counts and model settings match. Changing the mapping requires matching
standardization statistics and generally retraining. The positional buffer is
retained even when encoding is disabled. Enable `model.use_positional_encoding`
for sinusoidal encoding of all positions, including CLS, before layer normalization
and dropout. This changes model outputs and must match at training and inference.

`data.tree_name` selects the ROOT tree for training and statistics files; its
default is `O2hfxicxi2pifull`. Single-file inference continues to use
`dataset.tree_name`; batch inference uses `input.tree_name`.

## History

`history/training_history.csv` contains three rows per epoch:

- `train`: predictions made while training, with dropout and changing weights.
- `val`: validation predictions from the epoch's final weights in evaluation mode.
- `train_eval`: all training examples evaluated with those same final weights,
  dropout disabled and no gradient tracking, including any incomplete last batch.

`train_eval` includes AUC, loss, accuracy and the selection metrics below. Its loader
has a separate RNG generator, preserving the training shuffle/dropout RNG state.
It adds one full training-data forward pass per epoch. Best-model selection still
uses validation metrics. Old resumed-history rows are not retroactively evaluated.

### Selection metrics

Set `metrics.score_threshold` in YAML (default: `0.85`). Candidates are selected
when their signal probability is greater than or equal to this threshold.
Labels must be signal=1 and background=0; counts are unweighted.

- `purity`: selected signal / (selected signal + selected background).
- `signal_efficiency`: selected signal / all signal in the evaluated split.
- `background_efficiency`: selected background / all background in that split.
- `background_rejection`: 1 - background efficiency (not its reciprocal).

History also records `score_threshold`, `num_signal`, `num_background`,
`selected_signal`, and `selected_background`. Undefined fractions are empty CSV
cells. Purity depends on the sample's class mixture; these efficiencies are
conditional on the input sample and preprocessing, not full detector/acceptance
efficiencies. Training metrics cover the batches consumed by training; use
`train_eval` for all training examples at fixed weights.

New runs no longer write `ratio`. Existing history files are not converted.
When resuming an old run, old ratio values remain as a legacy column and new
metrics are only populated for new epochs. Old ratios alone cannot reconstruct
these efficiencies. No factor of 150 or assumed class-prior correction is used.

## Validation

```bash
python -m unittest discover -s model_codes/tests -v
```

Tests use temporary synthetic ROOT files; no production training data are modified.
