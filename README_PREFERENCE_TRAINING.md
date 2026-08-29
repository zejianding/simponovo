# Windows PrimeNovo + SimPO preference training

This workflow trains on Windows without `ctcdecode` or CuPy.  Run Linux
beam-search and PMC inference only after copying `final.ckpt`.

## Environment

The project-local environment is `simponovo` and uses Python 3.10.

```powershell
.\simponovo\Scripts\activate
```

The CUDA PyTorch wheel is installed separately from the PyTorch CUDA index.
The remaining Windows-only training dependencies are listed in
`requirements-windows-training.txt`. Do not install `ctcdecode` or CuPy in
this Windows environment.

## Configure data

Copy `pi-PrimeNovo\PrimeNovo\preference_config.yaml` and fill the six paths:

```text
train_mgf, train_parquet, train_lmdb
val_mgf, val_parquet, val_lmdb
```

The MGF and Parquet must be row-aligned. Every Parquet row must contain the
same `scan_id` and `positive_peptide` as the MGF `TITLE` and `SEQ`, and have
at least `num_negatives` ordered peptides in `negative_peptides`.

## Run order

```powershell
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py build-cache --config <config.yaml>
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py compact-cache --config <config.yaml>
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py calibrate --config <config.yaml>
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py train --config <config.yaml> --fresh
```

`compact-cache` preserves the original LMDBs and validates deterministic
records before publishing `*.compact.lmdb`; update the config to those files
after the command succeeds. Calibration writes `reward_calibration.json` once.
Training automatically resumes from `last.ckpt`; `--fresh` archives the old
`best.ckpt`, `last.ckpt`, `final.ckpt`, metric state, and fingerprint. If
training finished its train epoch but validation failed, run:

```powershell
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py validate-last --config <config.yaml>
```

Training writes `final.ckpt` plus `training_metadata.json` only after complete
validation succeeds.

The training schedule uses a linear warm-up for the first 10% of optimizer
steps and cosine decay for the remaining 90%, ending at `min_learning_rate`.
The fixed validation subset is evaluated every configured interval and saves
`best.ckpt` whenever `monitor/val_subset_total_loss` reaches a new minimum.
Each completed run keeps `best.ckpt`, rolling `last.ckpt`, and `final.ckpt`.

Training loss is logged once per optimizer update. With gradient accumulation,
each `train/*_step` value is the sum of the micro-batch losses after applying
the same accumulation scaling used by Lightning for backward.

Before a long run, the two optional checks are:

```powershell
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py workers-preflight --config .\preference_run.yaml
.\simponovo\Scripts\python.exe .\scripts\run_preference_training.py checkpoint-smoke --config .\preference_run.yaml
```

## Loss

```text
total_loss = simpo_loss + 0.1 * positive_ctc_loss
```

The first run uses `K=2`. Pairwise SimPO losses are averaged across the two
negative peptides.

## Linux inference

Copy `final.ckpt`, the preference config, and `reward_calibration.json` to a
Linux environment with `ctcdecode` and CuPy installed. Load the checkpoint
with `enable_inference_decoder=true` and retain the usual PrimeNovo PMC
settings.
