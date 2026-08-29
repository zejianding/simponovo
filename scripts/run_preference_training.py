"""Build preference caches, calibrate rewards once, and train PreferenceSpec2Pep."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pi-PrimeNovo"))

from PrimeNovo.denovo.preference_data import (
    PreferenceDataModule,
    build_preference_lmdb,
    compact_preference_lmdb,
    preference_collate,
    read_preference_lmdb_metadata,
)
from PrimeNovo.denovo.preference_model import PreferenceSpec2Pep


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_config_ref = config.get("base_config")
    if base_config_ref:
        base_path = Path(base_config_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        with base_path.open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        inherited = {
            key: base[key]
            for key in [
                "dim_model",
                "n_head",
                "dim_feedforward",
                "n_layers",
                "dropout",
                "dim_intensity",
                "max_length",
                "max_charge",
                "precursor_mass_tol",
                "isotope_error_range",
                "n_beams",
                "residues",
            ]
        }
        inherited.update({key: value for key, value in config.get("model", {}).items() if value is not None})
        config["model"] = inherited
    config["model"]["residues"] = {
        str(token): float(mass) for token, mass in config["model"]["residues"].items()
    }
    return config


def model_kwargs(config: dict, *, warmup_steps: int = 0, total_steps: int = 1) -> dict:
    model = config["model"]
    training = config["training"]
    preference = config["preference"]
    learning_rate = float(training["learning_rate"])
    min_learning_rate = float(training.get("min_learning_rate", 0.0))
    if training.get("scheduler", "cosine") != "cosine":
        raise ValueError("Preference training currently requires scheduler: cosine")
    if learning_rate <= 0 or not 0.0 <= min_learning_rate <= learning_rate:
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    return {
        "dim_model": model["dim_model"],
        "n_head": model["n_head"],
        "dim_feedforward": model["dim_feedforward"],
        "n_layers": model["n_layers"],
        "dropout": model["dropout"],
        "dim_intensity": model.get("dim_intensity"),
        "max_length": model["max_length"],
        "residues": model["residues"],
        "max_charge": model["max_charge"],
        "precursor_mass_tol": model.get("precursor_mass_tol", 50),
        "isotope_error_range": tuple(model.get("isotope_error_range", [0, 1])),
        "n_beams": model.get("n_beams", 0),
        "PMC_enable": False,
        "enable_inference_decoder": False,
        "num_negatives": preference["num_negatives"],
        "beta": preference["beta"],
        "target_margin": preference["target_margin"],
        "positive_ctc_weight": preference["positive_ctc_weight"],
        "warmup_steps": warmup_steps,
        "total_optimizer_steps": total_steps,
        "cosine_min_lr_ratio": min_learning_rate / learning_rate,
        "lr": learning_rate,
        "weight_decay": training["weight_decay"],
    }


def load_base_model(config: dict, warmup_steps: int = 0, total_steps: int = 1) -> PreferenceSpec2Pep:
    checkpoint_path = Path(config["training"]["base_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PreferenceSpec2Pep(**model_kwargs(config, warmup_steps=warmup_steps, total_steps=total_steps))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def make_datamodule(config: dict, batch_size: int) -> PreferenceDataModule:
    data = config["data"]
    return PreferenceDataModule(
        train_lmdb=data["train_lmdb"],
        val_lmdb=data["val_lmdb"],
        num_negatives=config["preference"]["num_negatives"],
        batch_size=batch_size,
        num_workers=data.get("num_workers", 0),
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(path) + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(str(temporary_path), str(path))
    finally:
        temporary_path.unlink(missing_ok=True)


class PeriodicValidationSubsetCallback(Callback):
    """Evaluate one fixed validation subset without re-entering Trainer.validate."""

    def __init__(
        self,
        subset_loader: DataLoader,
        interval_optimizer_steps: int,
        best_path: str | Path,
    ) -> None:
        self.subset_loader = subset_loader
        self.interval_optimizer_steps = interval_optimizer_steps
        self.best_path = Path(best_path)
        self.best_state_path = Path(str(self.best_path) + ".json")
        self.last_evaluated_step = -1
        self.best_loss = float("inf")
        self.best_step = -1
        best_exists = self.best_path.is_file()
        state_exists = self.best_state_path.is_file()
        if best_exists != state_exists:
            raise RuntimeError(
                "best.ckpt and its metric state are inconsistent; use --fresh to archive them"
            )
        if state_exists:
            state = json.loads(self.best_state_path.read_text(encoding="utf-8"))
            self.best_loss = float(state["best_val_subset_total_loss"])
            self.best_step = int(state["best_step"])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if (
            step <= 0
            or step == self.last_evaluated_step
            or step % self.interval_optimizer_steps != 0
        ):
            return
        self.last_evaluated_step = step
        was_training = pl_module.training
        totals = {"total_loss": 0.0, "simpo_loss": 0.0, "positive_ctc_loss": 0.0}
        count = 0
        pl_module.eval()
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for subset_batch in self.subset_loader:
                    spectra = subset_batch["spectra"].to(pl_module.device, non_blocking=True)
                    precursors = subset_batch["precursors"].to(pl_module.device, non_blocking=True)
                    logits, _, _ = pl_module._forward_step(
                        spectra, precursors, subset_batch["positive_peptides"]
                    )
                    losses = pl_module.compute_preference_losses(
                        logits, subset_batch["positive_peptides"], subset_batch["negative_peptides"]
                    )
                    batch_size = spectra.shape[0]
                    count += batch_size
                    for key in totals:
                        totals[key] += float(losses[key].detach()) * batch_size
        finally:
            if was_training:
                pl_module.train()
        if count == 0:
            raise RuntimeError("Validation subset is empty")
        metrics = {key: value / count for key, value in totals.items()}
        writer = trainer.logger.experiment
        writer.add_scalar("monitor/val_subset_total_loss", metrics["total_loss"], step)
        writer.add_scalar("monitor/val_subset_simpo_loss", metrics["simpo_loss"], step)
        writer.add_scalar(
            "monitor/val_subset_positive_ctc_loss", metrics["positive_ctc_loss"], step
        )
        if metrics["total_loss"] < self.best_loss:
            previous_loss, previous_step = self.best_loss, self.best_step
            self.best_loss = float(metrics["total_loss"])
            self.best_step = int(step)
            try:
                _atomic_save_checkpoint(trainer, self.best_path)
                _atomic_write_json(
                    self.best_state_path,
                    {
                        "best_val_subset_total_loss": self.best_loss,
                        "best_step": self.best_step,
                        "best_checkpoint_path": str(self.best_path),
                    },
                )
            except Exception:
                self.best_loss, self.best_step = previous_loss, previous_step
                raise
            writer.add_scalar("checkpoint/best_val_subset_total_loss", self.best_loss, step)
            writer.add_scalar("checkpoint/best_saved_step", step, step)
            print(
                f"\nSaved best checkpoint: {self.best_path} "
                f"(global_step={step}, val_subset_total_loss={self.best_loss:.6f})",
                flush=True,
            )
        writer.flush()

    def state_dict(self) -> dict:
        return {
            "interval_optimizer_steps": self.interval_optimizer_steps,
            "last_evaluated_step": self.last_evaluated_step,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        saved_interval = int(state_dict.get("interval_optimizer_steps", self.interval_optimizer_steps))
        if saved_interval != self.interval_optimizer_steps:
            raise ValueError(
                "Validation subset interval does not match the checkpoint: "
                f"{saved_interval} != {self.interval_optimizer_steps}"
            )
        self.last_evaluated_step = int(state_dict.get("last_evaluated_step", -1))
        saved_loss = float(state_dict.get("best_loss", float("inf")))
        if saved_loss < self.best_loss:
            self.best_loss = saved_loss
            self.best_step = int(state_dict.get("best_step", -1))


def _atomic_save_checkpoint(trainer: pl.Trainer, path: Path) -> None:
    """Save a complete Lightning checkpoint and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(path) + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        # ``weights_only`` is available in current Lightning; the fallback keeps
        # this runner usable with the older PrimeNovo environment as well.
        try:
            trainer.save_checkpoint(str(temporary_path), weights_only=False)
        except TypeError:
            trainer.save_checkpoint(str(temporary_path))
        with temporary_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        temporary_path.unlink(missing_ok=True)


class RollingCheckpointCallback(Callback):
    """Keep one atomically replaced, fully resumable rolling checkpoint."""

    def __init__(self, last_path: str | Path, interval_optimizer_steps: int) -> None:
        super().__init__()
        if interval_optimizer_steps < 1:
            raise ValueError("rolling checkpoint interval must be at least one optimizer step")
        self.last_path = Path(last_path)
        self.interval_optimizer_steps = int(interval_optimizer_steps)
        self.last_saved_step = -1

    def _publish(self, trainer: pl.Trainer, step: int, reason: str) -> None:
        previous_step = self.last_saved_step
        self.last_saved_step = int(step)
        try:
            _atomic_save_checkpoint(trainer, self.last_path)
        except Exception:
            self.last_saved_step = previous_step
            raise
        logger = getattr(trainer, "logger", None)
        if logger is not None:
            experiment = logger.experiment
            if hasattr(experiment, "add_scalar"):
                experiment.add_scalar("checkpoint/last_saved_step", step, step)
                experiment.flush()
        next_step = ((step // self.interval_optimizer_steps) + 1) * self.interval_optimizer_steps
        print(
            f"\nSaved rolling checkpoint: {self.last_path} "
            f"(global_step={step}, reason={reason}); "
            f"next rolling checkpoint at step={next_step}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step > 0 and step % self.interval_optimizer_steps == 0 and step != self.last_saved_step:
            self._publish(trainer, step, "rolling_interval")
        num_batches = trainer.num_training_batches
        if math.isfinite(float(num_batches)) and batch_idx + 1 == int(num_batches):
            # This save happens after the final train batch and before Lightning
            # enters the epoch-end validation loop.
            self._publish(trainer, step, "train_epoch_end")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Fallback for custom data fetchers where num_training_batches is not a
        # finite integer.  In the normal path this is a no-op because the final
        # batch hook has already published the same step.
        step = int(trainer.global_step)
        if step != self.last_saved_step:
            self._publish(trainer, step, "train_epoch_end_fallback")

    def state_dict(self) -> dict:
        return {
            "last_saved_step": self.last_saved_step,
            "interval_optimizer_steps": self.interval_optimizer_steps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        saved_interval = int(state_dict.get("interval_optimizer_steps", self.interval_optimizer_steps))
        if saved_interval != self.interval_optimizer_steps:
            raise ValueError(
                "Rolling checkpoint interval does not match the current run: "
                f"{saved_interval} != {self.interval_optimizer_steps}"
            )
        self.last_saved_step = int(state_dict.get("last_saved_step", -1))


def make_tensorboard_logger(config: dict) -> TensorBoardLogger:
    logging_config = config["logging"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{logging_config['run_name_prefix']}_{timestamp}"
    return TensorBoardLogger(
        save_dir=logging_config["save_dir"],
        name=run_name,
        version="",
        default_hp_metric=False,
        flush_secs=logging_config["tensorboard_flush_secs"],
    )


def log_resume_info(logger: TensorBoardLogger, resume_info: dict) -> None:
    """Write resume provenance without relying on Lightning's logger internals."""
    experiment = logger.experiment
    start_step = int(resume_info.get("start_global_step", 0))
    if hasattr(experiment, "add_scalar"):
        experiment.add_scalar("resume/start_global_step", start_step, start_step)
        source = resume_info.get("source")
        if source and hasattr(experiment, "add_text"):
            experiment.add_text("resume/source_checkpoint", str(source), start_step)
        experiment.flush()


def make_validation_subset_callback(
    config: dict,
    module: PreferenceDataModule,
    batch_size: int,
    output_dir: Path,
    best_path: Path,
) -> PeriodicValidationSubsetCallback:
    logging_config = config["logging"]
    subset_size = min(logging_config["val_subset_size"], len(module.val_dataset))
    generator = torch.Generator().manual_seed(logging_config["val_subset_seed"])
    indices = torch.randperm(len(module.val_dataset), generator=generator)[:subset_size].tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "val_subset_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    subset_loader = DataLoader(
        Subset(module.val_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=module.num_workers,
        pin_memory=True,
        persistent_workers=module.num_workers > 0,
        collate_fn=preference_collate,
    )
    return PeriodicValidationSubsetCallback(
        subset_loader,
        logging_config["val_subset_interval_optimizer_steps"],
        best_path,
    )


def compact_caches(config: dict, overwrite: bool) -> None:
    """Compact both configured file-style LMDBs and validate their records."""
    data = config["data"]
    results = []
    for split in ("train", "val"):
        source = Path(data[f"{split}_lmdb"])
        if source.name.endswith(".compact.lmdb"):
            metadata = read_preference_lmdb_metadata(source)
            results.append(
                {
                    "source": str(source),
                    "compact": str(source),
                    "already_compact": True,
                    "n_spectra": metadata["n_spectra"],
                    "source_spectra": metadata["source_spectra"],
                    "skipped_invalid_ctc": metadata["skipped_invalid_ctc"],
                    "compact_bytes": source.stat().st_size,
                }
            )
            continue
        destination = source.with_name(f"{source.stem}.compact{source.suffix}")
        # The original cache was created with a 2 TB map.  Supplying a bounded
        # read map based on the source MGF avoids reproducing that mapping while
        # copying it on Windows.
        source_map_size = max(1024**3, math.ceil(Path(data[f"{split}_mgf"]).stat().st_size * 1.5))
        result = compact_preference_lmdb(
            source,
            destination,
            overwrite=overwrite,
            source_map_size=source_map_size,
        )
        results.append(result)
    print(json.dumps(results, indent=2), flush=True)


def _file_fingerprint(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    result = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if resolved.name.endswith(".lmdb"):
        result["metadata"] = read_preference_lmdb_metadata(resolved)
    return result


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_full_checkpoint(path: str | Path) -> dict:
    """Load a Lightning checkpoint across torch versions with/without weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_run_fingerprint(
    config: dict,
    *,
    micro_batch: int,
    accumulation: int,
    warmup_steps: int,
    optimizer_steps: int,
) -> dict:
    """Build the immutable configuration identity used for safe resume."""
    training = config["training"]
    preference = config["preference"]
    values = {
        "train_lmdb": _file_fingerprint(config["data"]["train_lmdb"]),
        "val_lmdb": _file_fingerprint(config["data"]["val_lmdb"]),
        "base_checkpoint": _file_fingerprint(training["base_checkpoint"]),
        "num_negatives": int(preference["num_negatives"]),
        "beta": float(preference["beta"]),
        "target_margin": float(preference["target_margin"]),
        "positive_ctc_weight": float(preference["positive_ctc_weight"]),
        "micro_batch_size": int(micro_batch),
        "gradient_accumulation_steps": int(accumulation),
        "learning_rate": float(training["learning_rate"]),
        "scheduler": str(training.get("scheduler", "cosine")),
        "min_learning_rate": float(training.get("min_learning_rate", 0.0)),
        "weight_decay": float(training["weight_decay"]),
        "warmup_steps": int(warmup_steps),
        "total_optimizer_steps": int(optimizer_steps),
        "model_architecture": _jsonable(model_kwargs(config)),
        "residue_vocabulary": _jsonable(config["model"]["residues"]),
    }
    values = _jsonable(values)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"schema_version": 1, "sha256": hashlib.sha256(encoded).hexdigest(), "values": values}


def _checkpoint_paths(config: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    checkpointing = config.get("checkpointing", {})
    return (
        Path(checkpointing.get("last_path", output_dir / "last.ckpt")),
        Path(checkpointing.get("best_path", output_dir / "best.ckpt")),
        Path(checkpointing.get("final_path", output_dir / "final.ckpt")),
    )


def _archive_resume_files(output_dir: Path, paths: list[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / "checkpoint_archive" / timestamp
    suffix = 1
    while archive_dir.exists():
        archive_dir = output_dir / "checkpoint_archive" / f"{timestamp}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.is_file():
            shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def prepare_resume(
    config: dict,
    *,
    output_dir: Path,
    fingerprint: dict,
    fresh: bool,
) -> tuple[Path | None, Path, dict]:
    checkpointing = config.get("checkpointing", {})
    auto_resume = bool(checkpointing.get("auto_resume", True))
    last_path, best_path, final_path = _checkpoint_paths(config, output_dir)
    fingerprint_path = output_dir / "run_fingerprint.json"
    best_state_path = Path(str(best_path) + ".json")
    if fresh:
        archive_paths = [last_path, best_path, best_state_path, final_path, fingerprint_path]
        if any(path.is_file() for path in archive_paths):
            archive_dir = _archive_resume_files(output_dir, archive_paths)
            print(f"Archived previous resume state to {archive_dir}", flush=True)
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        return None, last_path, {"resumed": False, "source": None}
    if not auto_resume:
        stale_paths = [last_path, best_path, best_state_path, final_path, fingerprint_path]
        if any(path.is_file() for path in stale_paths):
            raise RuntimeError(
                "Checkpoint artifacts exist while auto_resume is disabled; use --fresh to archive them "
                "before starting a new run"
            )
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        return None, last_path, {"resumed": False, "source": None, "auto_resume": False}
    if last_path.is_file():
        if not fingerprint_path.is_file():
            raise RuntimeError(
                f"{last_path} exists but {fingerprint_path} is missing; "
                "use --fresh to start a new compatible run"
            )
        saved = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if saved.get("sha256") != fingerprint.get("sha256"):
            raise RuntimeError(
                "Resume fingerprint mismatch; refusing to load an incompatible checkpoint. "
                "Use --fresh to start a new run."
            )
        checkpoint = _load_full_checkpoint(last_path)
        loops = checkpoint.get("loops", {})
        fit_loop = loops.get("fit_loop", {}) if isinstance(loops, dict) else {}
        epoch_loop = fit_loop.get("epoch_loop", {}) if isinstance(fit_loop, dict) else {}
        saved_step = checkpoint.get("global_step", epoch_loop.get("_batches_that_stepped", 0))
        step = int(saved_step)
        print(f"Resuming from {last_path} at global_step={step}", flush=True)
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        return last_path, last_path, {"resumed": True, "source": str(last_path), "start_global_step": step}
    stale_paths = [fingerprint_path, best_path, best_state_path, final_path]
    if any(path.is_file() for path in stale_paths):
        raise RuntimeError(
            "Checkpoint artifacts exist without last.ckpt; use --fresh to archive them before "
            "starting a new run"
        )
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return None, last_path, {"resumed": False, "source": None}


def build_caches(config: dict, overwrite: bool) -> None:
    data, model = config["data"], config["model"]
    preprocessing = data["preprocessing"]
    for split in ("train", "val"):
        build_preference_lmdb(
            data[f"{split}_mgf"],
            data[f"{split}_parquet"],
            data[f"{split}_lmdb"],
            num_negatives=config["preference"]["num_negatives"],
            residues=model["residues"],
            n_peaks=preprocessing["n_peaks"],
            min_mz=preprocessing["min_mz"],
            max_mz=preprocessing["max_mz"],
            min_intensity=preprocessing["min_intensity"],
            remove_precursor_tol=preprocessing["remove_precursor_tol"],
            overwrite=overwrite,
        )


def calibrate(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Reward calibration requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 training requires BF16-capable CUDA hardware")
    torch.set_float32_matmul_precision("high")
    sample_count = config["calibration"]["sample_count"]
    batch_size = config["calibration"].get("batch_size", 8)
    module = make_datamodule(config, batch_size)
    module.setup("fit")
    model = load_base_model(config).cuda().eval()
    margins = []
    sampled_spectra = 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in module.train_dataloader():
            spectra = batch["spectra"].cuda(non_blocking=True)
            precursors = batch["precursors"].cuda(non_blocking=True)
            logits, _, _ = model._forward_step(spectra, precursors, batch["positive_peptides"])
            losses = model.compute_preference_losses(
                logits, batch["positive_peptides"], batch["negative_peptides"]
            )
            margins.append(losses["margins"].detach().float().cpu().reshape(-1))
            sampled_spectra += spectra.shape[0]
            if sampled_spectra >= sample_count:
                break
    values = torch.cat(margins)
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    iqr = float(quantiles[3] - quantiles[1])
    if iqr < 1e-3:
        raise RuntimeError(f"Calibration IQR is too small: {iqr}")
    result = {
        "sample_count_spectra": sampled_spectra,
        "sample_count_pair_margins": int(values.numel()),
        "seed": config["calibration"]["seed"],
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "iqr": iqr,
        "initial_preference_accuracy": float((values > 0).float().mean()),
        "beta": 1.0 / iqr,
        "target_margin": 0.25 * iqr,
    }
    output = Path(config["calibration"]["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def choose_batch_size(config: dict) -> tuple[int, int]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Preference training requires BF16-capable CUDA hardware")
    torch.set_float32_matmul_precision("high")
    training = config["training"]
    candidates = [8, 16]
    selected = 4
    for batch_size in candidates:
        torch.cuda.empty_cache()
        try:
            module = make_datamodule(config, batch_size)
            module.setup("fit")
            batch = next(iter(module.train_dataloader()))
            model = load_base_model(config).cuda().train()
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _ = model._forward_step(
                    batch["spectra"].cuda(), batch["precursors"].cuda(), batch["positive_peptides"]
                )
                loss = model.compute_preference_losses(
                    logits, batch["positive_peptides"], batch["negative_peptides"]
                )["total_loss"]
            loss.backward()
            peak_ratio = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            selected = batch_size
            del model, loss, logits
            torch.cuda.empty_cache()
            if peak_ratio >= 0.8:
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    accumulation = math.ceil(training["effective_batch_size"] / selected)
    return selected, accumulation


def train(config: dict, *, fresh: bool = False) -> None:
    calibration_path = Path(config["calibration"]["output_json"])
    if not calibration_path.is_file():
        raise FileNotFoundError("Run calibration once before training")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    torch.set_float32_matmul_precision("high")
    config["preference"]["beta"] = calibration["beta"]
    config["preference"]["target_margin"] = calibration["target_margin"]
    micro_batch, accumulation = choose_batch_size(config)
    module = make_datamodule(config, micro_batch)
    module.setup("fit")
    optimizer_steps = math.ceil(len(module.train_dataloader()) / accumulation)
    warmup_steps = math.ceil(optimizer_steps * config["training"]["warmup_ratio"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = make_run_fingerprint(
        config,
        micro_batch=micro_batch,
        accumulation=accumulation,
        warmup_steps=warmup_steps,
        optimizer_steps=optimizer_steps,
    )
    resume_path, last_path, resume_info = prepare_resume(
        config,
        output_dir=output_dir,
        fingerprint=fingerprint,
        fresh=fresh,
    )
    model = load_base_model(config, warmup_steps=warmup_steps, total_steps=optimizer_steps)
    tensorboard_logger = make_tensorboard_logger(config)
    log_resume_info(tensorboard_logger, resume_info)
    checkpointing = config.get("checkpointing", {})
    _, best_path, final_path = _checkpoint_paths(config, output_dir)
    subset_callback = make_validation_subset_callback(
        config,
        module,
        micro_batch,
        Path(tensorboard_logger.log_dir),
        best_path,
    )
    rolling_callback = RollingCheckpointCallback(
        last_path=last_path,
        interval_optimizer_steps=int(checkpointing.get("rolling_interval_optimizer_steps", 2000)),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        accumulate_grad_batches=accumulation,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        logger=tensorboard_logger,
        log_every_n_steps=config["logging"]["train_log_every_n_steps"],
        # Save the train state before the periodic subset callback can run; if
        # subset validation fails on the same step, the latest train weights
        # are still recoverable.
        callbacks=[rolling_callback, subset_callback],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=module, ckpt_path=str(resume_path) if resume_path else None)
    if not best_path.is_file():
        raise RuntimeError("Training completed without producing best.ckpt")
    _atomic_save_checkpoint(trainer, final_path)
    metadata = {
        "num_negatives": config["preference"]["num_negatives"],
        "beta": calibration["beta"],
        "target_margin": calibration["target_margin"],
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "warmup_steps": warmup_steps,
        "total_optimizer_steps": optimizer_steps,
        "scheduler": config["training"].get("scheduler", "cosine"),
        "min_learning_rate": config["training"].get("min_learning_rate", 0.0),
        "tensorboard_run_dir": tensorboard_logger.log_dir,
        "val_subset_indices_path": str(Path(tensorboard_logger.log_dir) / "val_subset_indices.json"),
        "train_log_every_n_steps": config["logging"]["train_log_every_n_steps"],
        "val_subset_size": config["logging"]["val_subset_size"],
        "val_subset_seed": config["logging"]["val_subset_seed"],
        "val_subset_interval_optimizer_steps": config["logging"]["val_subset_interval_optimizer_steps"],
        "rolling_interval_optimizer_steps": rolling_callback.interval_optimizer_steps,
        "auto_resume": bool(checkpointing.get("auto_resume", True)),
        "last_checkpoint_path": str(last_path),
        "best_checkpoint_path": str(best_path),
        "best_val_subset_total_loss": subset_callback.best_loss,
        "best_optimizer_step": subset_callback.best_step,
        "final_checkpoint_path": str(final_path),
        "resume": resume_info,
        "run_fingerprint_sha256": fingerprint["sha256"],
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_last(config: dict) -> None:
    """Run full validation from the most recent rolling checkpoint."""
    torch.set_float32_matmul_precision("high")
    checkpointing = config.get("checkpointing", {})
    output_dir = Path(config["training"]["output_dir"])
    last_path, _, final_path = _checkpoint_paths(config, output_dir)
    if not last_path.is_file():
        raise FileNotFoundError(f"Rolling checkpoint not found: {last_path}")
    calibration_path = Path(config["calibration"]["output_json"])
    if not calibration_path.is_file():
        raise FileNotFoundError("Run calibration once before validation")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["preference"]["beta"] = calibration["beta"]
    config["preference"]["target_margin"] = calibration["target_margin"]
    training_metadata_path = output_dir / "training_metadata.json"
    if training_metadata_path.is_file():
        training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
        batch_size = int(training_metadata.get("micro_batch_size", config["calibration"].get("batch_size", 8)))
        warmup_steps = int(training_metadata.get("warmup_steps", 0))
        total_steps = int(training_metadata.get("total_optimizer_steps", 1))
    else:
        batch_size = int(config["calibration"].get("batch_size", 8))
        warmup_steps, total_steps = 0, 1
    checkpoint = _load_full_checkpoint(last_path)
    module = make_datamodule(config, batch_size)
    module.setup("validate")
    model = load_base_model(config, warmup_steps=warmup_steps, total_steps=total_steps)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    tensorboard_logger = make_tensorboard_logger(config)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        logger=tensorboard_logger,
        enable_checkpointing=False,
    )
    print(f"Validating rolling checkpoint: {last_path}", flush=True)
    trainer.validate(model, datamodule=module, verbose=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_temporary = Path(str(final_path) + ".tmp")
    final_temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(last_path, final_temporary)
        os.replace(str(final_temporary), str(final_path))
    finally:
        final_temporary.unlink(missing_ok=True)
    print(f"Validation succeeded; copied {last_path} to {final_path}", flush=True)


def smoke_train(config: dict) -> None:
    """Run two optimizer batches and one validation batch on the real cache."""
    module = make_datamodule(config, batch_size=2)
    module.setup("fit")
    model = load_base_model(config, warmup_steps=1, total_steps=2)
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = make_tensorboard_logger(config)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accumulate_grad_batches=1,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        logger=tensorboard_logger,
        log_every_n_steps=config["logging"]["train_log_every_n_steps"],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=module)
    trainer.save_checkpoint(output_dir / "smoke_test.ckpt")


def checkpoint_smoke_train(config: dict) -> None:
    """Verify rolling checkpoint contents with three short train batches."""
    config = copy.deepcopy(config)
    output_dir = Path(config["training"]["output_dir"]) / "checkpoint_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.ckpt"
    last_path.unlink(missing_ok=True)
    config["checkpointing"] = {
        "rolling_interval_optimizer_steps": 1,
        "last_path": str(last_path),
    }
    module = make_datamodule(config, batch_size=2)
    module.setup("fit")
    model = load_base_model(config, warmup_steps=1, total_steps=3)
    tensorboard_logger = make_tensorboard_logger(config)
    rolling_callback = RollingCheckpointCallback(last_path, interval_optimizer_steps=1)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=1,
        accumulate_grad_batches=1,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        callbacks=[rolling_callback],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=module)
    if not last_path.is_file():
        raise RuntimeError(f"Checkpoint smoke test did not create {last_path}")
    checkpoint = _load_full_checkpoint(last_path)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops"):
        if key not in checkpoint:
            raise RuntimeError(f"Rolling checkpoint is missing {key}")
    print(f"Checkpoint smoke test passed: {last_path}", flush=True)


def workers_preflight(config: dict) -> None:
    """Keep four train/val/subset workers alive while reading ten batches."""
    batch_size = int(config["calibration"].get("batch_size", 8))
    module = make_datamodule(config, batch_size)
    module.setup("fit")
    generator = torch.Generator().manual_seed(config["logging"]["val_subset_seed"])
    indices = torch.randperm(len(module.val_dataset), generator=generator)[: min(2048, len(module.val_dataset))].tolist()
    subset_loader = DataLoader(
        Subset(module.val_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=module.num_workers,
        pin_memory=True,
        persistent_workers=module.num_workers > 0,
        collate_fn=preference_collate,
    )
    train_iterator = iter(module.train_dataloader())
    val_iterator = iter(module.val_dataloader())
    subset_iterator = iter(subset_loader)
    try:
        for index in range(10):
            train_batch = next(train_iterator)
            val_batch = next(val_iterator)
            subset_batch = next(subset_iterator)
            if not (
                len(train_batch["negative_peptides"]) > 0
                and len(val_batch["negative_peptides"]) > 0
                and len(subset_batch["negative_peptides"]) > 0
            ):
                raise RuntimeError("Worker preflight returned an empty batch")
            print(f"Worker preflight batch {index + 1}/10 ok", flush=True)
    finally:
        del train_iterator, val_iterator, subset_iterator
    print("Worker preflight passed: train/val/subset workers read concurrently", flush=True)


def subset_smoke_train(config: dict) -> None:
    """Exercise the periodic validation callback with a tiny fixed subset."""
    config = copy.deepcopy(config)
    config["logging"]["val_subset_size"] = 16
    config["logging"]["val_subset_interval_optimizer_steps"] = 1
    config["logging"]["run_name_prefix"] = f"{config['logging']['run_name_prefix']}_subset_smoke"
    config["training"]["output_dir"] = str(Path(config["training"]["output_dir"]) / "logging_smoke")
    module = make_datamodule(config, batch_size=2)
    module.setup("fit")
    model = load_base_model(config, warmup_steps=1, total_steps=2)
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = make_tensorboard_logger(config)
    subset_callback = make_validation_subset_callback(
        config,
        module,
        2,
        Path(tensorboard_logger.log_dir),
        output_dir / "best.ckpt",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accumulate_grad_batches=1,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        callbacks=[subset_callback],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "build-cache",
            "compact-cache",
            "smoke-train",
            "checkpoint-smoke",
            "workers-preflight",
            "subset-smoke",
            "calibrate",
            "train",
            "validate-last",
        ],
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and archive an existing rolling checkpoint before training",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    torch.manual_seed(config["calibration"]["seed"])
    if args.command == "build-cache":
        build_caches(config, args.overwrite)
    elif args.command == "compact-cache":
        compact_caches(config, args.overwrite)
    elif args.command == "smoke-train":
        smoke_train(config)
    elif args.command == "checkpoint-smoke":
        checkpoint_smoke_train(config)
    elif args.command == "workers-preflight":
        workers_preflight(config)
    elif args.command == "subset-smoke":
        subset_smoke_train(config)
    elif args.command == "calibrate":
        print(json.dumps(calibrate(config), indent=2))
    elif args.command == "validate-last":
        validate_last(config)
    else:
        train(config, fresh=args.fresh)


if __name__ == "__main__":
    main()
