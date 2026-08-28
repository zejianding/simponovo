"""Preference MGF/Parquet validation, LMDB caching, and DataLoader support."""

from __future__ import annotations

import pickle
import re
import time
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List

import lmdb
import numpy as np
import pyarrow.parquet as pq
import pytorch_lightning as pl
import spectrum_utils.spectrum as sus
import torch
from torch.utils.data import DataLoader, Dataset


PROTON_MASS = 1.007276
TOKEN_SPLIT = re.compile(r"(?<=.)(?=[A-Z])")


def peptide_tokens(sequence: str) -> List[str]:
    tokens = TOKEN_SPLIT.split(
        sequence.replace("[", "").replace("]", "").replace("I", "L")
    )
    return list(reversed(tokens))  # Matches PrimeNovo PeptideDecoder(reverse=True).


def ctc_required_length(tokens: List[str]) -> int:
    return len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))


def iter_mgf(path: Path) -> Iterator[Dict[str, object]]:
    title = precursor_mz = precursor_charge = peptide = None
    peaks: List[List[float]] = []
    in_block = False
    with path.open("r", encoding="utf-8", errors="strict", newline=None) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                if in_block:
                    raise ValueError(f"Nested BEGIN IONS in {path}")
                title = precursor_mz = precursor_charge = peptide = None
                peaks = []
                in_block = True
            elif line == "END IONS":
                if not in_block or None in (title, precursor_mz, precursor_charge, peptide):
                    raise ValueError(f"Incomplete spectrum in {path}")
                yield {
                    "scan_id": title,
                    "precursor_mz": precursor_mz,
                    "precursor_charge": precursor_charge,
                    "positive_peptide": peptide,
                    "mz_array": np.asarray([peak[0] for peak in peaks], dtype=np.float64),
                    "intensity_array": np.asarray([peak[1] for peak in peaks], dtype=np.float32),
                }
                in_block = False
            elif in_block and line.startswith("TITLE="):
                title = line[6:]
            elif in_block and line.startswith("PEPMASS="):
                precursor_mz = float(line[8:].split()[0])
            elif in_block and line.startswith("CHARGE="):
                match = re.match(r"(\d+)", line[7:])
                if match is None:
                    raise ValueError(f"Invalid CHARGE line: {line}")
                precursor_charge = int(match.group(1))
            elif in_block and line.startswith("SEQ="):
                peptide = line[4:]
            elif in_block and line and line[0].isdigit():
                values = line.split()
                if len(values) >= 2:
                    peaks.append([float(values[0]), float(values[1])])
    if in_block:
        raise ValueError(f"Unclosed MGF spectrum in {path}")


def process_spectrum(
    record: Dict[str, object],
    n_peaks: int,
    min_mz: float,
    max_mz: float,
    min_intensity: float,
    remove_precursor_tol: float,
) -> torch.Tensor:
    spectrum = sus.MsmsSpectrum(
        "",
        float(record["precursor_mz"]),
        int(record["precursor_charge"]),
        record["mz_array"],
        record["intensity_array"],
    )
    try:
        spectrum.set_mz_range(min_mz, max_mz)
        spectrum.remove_precursor_peak(remove_precursor_tol, "Da")
        spectrum.filter_intensity(min_intensity, n_peaks)
        if len(spectrum.mz) == 0:
            raise ValueError
        intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
        return torch.tensor(np.stack([spectrum.mz, intensities], axis=1), dtype=torch.float32)
    except ValueError:
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)


def _remove_lmdb_files(path: Path) -> None:
    """Remove a file-style LMDB and its lock file, if present.

    The preference cache is opened with ``subdir=False``.  On Windows LMDB
    creates a sibling ``*-lock`` file, which must be removed together with the
    data file when rebuilding a cache.  This helper deliberately only touches
    the exact requested path and its sibling lock file.
    """
    path.unlink(missing_ok=True)
    Path(str(path) + "-lock").unlink(missing_ok=True)


def read_preference_lmdb_metadata(
    lmdb_path: str | Path, *, map_size: int | None = None
) -> Dict[str, int]:
    """Read and validate scalar metadata stored in a preference LMDB."""
    path = Path(lmdb_path)
    open_kwargs = {"subdir": False, "readonly": True, "lock": False}
    if map_size is not None:
        open_kwargs["map_size"] = int(map_size)
    with lmdb.open(str(path), **open_kwargs) as environment:
        with environment.begin() as transaction:
            required = ("complete", "n_spectra", "source_spectra", "skipped_invalid_ctc")
            values = {key: transaction.get(key.encode()) for key in required}
            if values["complete"] != b"1":
                raise ValueError(f"Incomplete preference LMDB: {path}")
            metadata = {key: int(values[key].decode()) for key in required[1:]}
            for optional in ("initial_map_size", "final_map_size", "actual_used_bytes"):
                value = transaction.get(optional.encode())
                if value is not None:
                    metadata[optional] = int(value.decode())
            metadata["entries"] = int(environment.stat()["entries"])
            metadata["map_size"] = int(environment.info()["map_size"])
            metadata["last_pgno"] = int(environment.info()["last_pgno"])
            metadata["page_size"] = int(environment.stat()["psize"])
    return metadata


def _record_digest(record: Dict[str, object]) -> str:
    """Create a stable digest for the tensor fields of one cache record."""
    digest = hashlib.sha256()
    digest.update(str(record["scan_id"]).encode("utf-8"))
    digest.update(str(record["positive_peptide"]).encode("utf-8"))
    digest.update(repr(record["negative_peptides"]).encode("utf-8"))
    for key in ("precursor", "spectrum"):
        tensor = record[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"LMDB record field {key} is not a tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_compacted_preference_lmdb(
    source_path: str | Path,
    compact_path: str | Path,
    *,
    sample_count: int = 1000,
    seed: int = 6558,
    source_map_size: int | None = None,
) -> Dict[str, object]:
    """Compare a compact LMDB with its source, including deterministic samples."""
    source_path, compact_path = Path(source_path), Path(compact_path)
    source_metadata = read_preference_lmdb_metadata(source_path, map_size=source_map_size)
    compact_metadata = read_preference_lmdb_metadata(compact_path)
    for key in ("n_spectra", "source_spectra", "skipped_invalid_ctc", "entries"):
        if source_metadata[key] != compact_metadata[key]:
            raise ValueError(
                f"Compact LMDB metadata mismatch for {key}: "
                f"{source_metadata[key]} != {compact_metadata[key]}"
            )
    n_spectra = source_metadata["n_spectra"]
    if n_spectra <= 0:
        raise ValueError("Cannot validate an empty preference LMDB")
    indices = {0, n_spectra - 1}
    generator = np.random.default_rng(seed)
    indices.update(
        int(index)
        for index in generator.choice(
            n_spectra, size=min(sample_count, n_spectra), replace=False
        ).tolist()
    )
    with (
        lmdb.open(
            str(source_path),
            subdir=False,
            readonly=True,
            lock=False,
            **({"map_size": int(source_map_size)} if source_map_size is not None else {}),
        ) as source_env,
        lmdb.open(str(compact_path), subdir=False, readonly=True, lock=False) as compact_env,
    ):
        for index in sorted(indices):
            key = str(index).encode()
            with source_env.begin() as source_txn, compact_env.begin() as compact_txn:
                source_payload = source_txn.get(key)
                compact_payload = compact_txn.get(key)
            if source_payload is None or compact_payload is None:
                raise ValueError(f"Missing compact LMDB record at index {index}")
            source_record = pickle.loads(source_payload)
            compact_record = pickle.loads(compact_payload)
            if _record_digest(source_record) != _record_digest(compact_record):
                raise ValueError(f"Compact LMDB record mismatch at index {index}")
    result = {
        "source": str(source_path),
        "compact": str(compact_path),
        "n_spectra": n_spectra,
        "source_spectra": source_metadata["source_spectra"],
        "skipped_invalid_ctc": source_metadata["skipped_invalid_ctc"],
        "entries": compact_metadata["entries"],
        "source_bytes": source_path.stat().st_size,
        "compact_bytes": compact_path.stat().st_size,
        "source_used_bytes": (source_metadata["last_pgno"] + 1) * source_metadata["page_size"],
        "compact_used_bytes": (compact_metadata["last_pgno"] + 1) * compact_metadata["page_size"],
        "sampled_records": len(indices),
    }
    if compact_path.stat().st_size >= source_path.stat().st_size:
        raise ValueError(
            f"Compact LMDB is not smaller than source: "
            f"{result['compact_bytes']} >= {result['source_bytes']}"
        )
    return result


def compact_preference_lmdb(
    source_path: str | Path,
    compact_path: str | Path,
    *,
    overwrite: bool = False,
    sample_count: int = 1000,
    source_map_size: int | None = None,
) -> Dict[str, object]:
    """Create and validate a compact copy of a file-style preference LMDB."""
    source_path, compact_path = Path(source_path), Path(compact_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Preference LMDB does not exist: {source_path}")
    if compact_path.exists() and not overwrite:
        raise FileExistsError(f"Compact preference LMDB exists: {compact_path}")
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(compact_path) + ".tmp")
    _remove_lmdb_files(temporary_path)
    source_environment = lmdb.open(
        str(source_path),
        subdir=False,
        readonly=True,
        lock=False,
        **({"map_size": int(source_map_size)} if source_map_size is not None else {}),
    )
    try:
        source_environment.copy(str(temporary_path), compact=True)
    finally:
        source_environment.close()
    try:
        result = validate_compacted_preference_lmdb(
            source_path,
            temporary_path,
            sample_count=sample_count,
            source_map_size=source_map_size,
        )
        os.replace(str(temporary_path), str(compact_path))
        Path(str(temporary_path) + "-lock").unlink(missing_ok=True)
        result["compact"] = str(compact_path)
        result["compact_bytes"] = compact_path.stat().st_size
        return result
    except Exception:
        _remove_lmdb_files(temporary_path)
        raise


def build_preference_lmdb(
    mgf_path: str | Path,
    parquet_path: str | Path,
    lmdb_path: str | Path,
    *,
    num_negatives: int,
    residues: Dict[str, float],
    n_peaks: int,
    min_mz: float,
    max_mz: float,
    min_intensity: float,
    remove_precursor_tol: float,
    overwrite: bool = False,
    map_size: int | None = None,
) -> None:
    """Build a validated, random-access cache from one preference MGF/Parquet pair."""
    mgf_path, parquet_path, lmdb_path = Path(mgf_path), Path(parquet_path), Path(lmdb_path)
    if lmdb_path.exists() and not overwrite:
        raise FileExistsError(f"Preference LMDB exists: {lmdb_path}")
    if lmdb_path.exists():
        _remove_lmdb_files(lmdb_path)
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    initial_map_size = (
        max(1024**3, math.ceil(mgf_path.stat().st_size * 1.5))
        if map_size is None
        else int(map_size)
    )
    current_map_size = initial_map_size
    environment = lmdb.open(str(lmdb_path), subdir=False, map_size=current_map_size, lock=True)
    mgf_records = iter_mgf(mgf_path)
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    started_at = time.monotonic()
    source_index = 0
    stored_index = 0
    skipped_invalid_ctc = 0
    try:
        pending: List[tuple[bytes, bytes]] = []

        def commit_pending() -> None:
            nonlocal current_map_size
            if not pending:
                return
            while True:
                transaction = environment.begin(write=True)
                try:
                    for key, value in pending:
                        transaction.put(key, value)
                    transaction.commit()
                    pending.clear()
                    return
                except lmdb.MapFullError:
                    transaction.abort()
                    current_map_size *= 2
                    environment.set_mapsize(current_map_size)

        for batch in parquet_file.iter_batches(
            batch_size=1024, columns=["scan_id", "positive_peptide", "negative_peptides"]
        ):
            for row in batch.to_pylist():
                try:
                    mgf_record = next(mgf_records)
                except StopIteration as exc:
                    raise ValueError("MGF ended before preference Parquet") from exc
                if row["scan_id"] != mgf_record["scan_id"]:
                    raise ValueError(f"scan_id mismatch at row {source_index}")
                if row["positive_peptide"] != mgf_record["positive_peptide"]:
                    raise ValueError(f"positive peptide mismatch at {row['scan_id']}")
                negatives = list(row["negative_peptides"])
                if len(negatives) < num_negatives:
                    raise ValueError(f"Insufficient negatives at {row['scan_id']}")
                invalid_ctc = False
                for peptide in [row["positive_peptide"], *negatives[:num_negatives]]:
                    tokens = peptide_tokens(peptide)
                    if not tokens or any(token not in residues for token in tokens):
                        raise ValueError(f"Invalid peptide token at {row['scan_id']}: {peptide}")
                    if ctc_required_length(tokens) > 40:
                        invalid_ctc = True
                        break
                source_index += 1
                if invalid_ctc:
                    skipped_invalid_ctc += 1
                    continue
                cache_record = {
                    "scan_id": row["scan_id"],
                    "spectrum": process_spectrum(
                        mgf_record, n_peaks, min_mz, max_mz, min_intensity, remove_precursor_tol
                    ),
                    "precursor": torch.tensor(
                        [
                            (float(mgf_record["precursor_mz"]) - PROTON_MASS)
                            * int(mgf_record["precursor_charge"]),
                            int(mgf_record["precursor_charge"]),
                            float(mgf_record["precursor_mz"]),
                        ],
                        dtype=torch.float32,
                    ),
                    "positive_peptide": row["positive_peptide"],
                    "negative_peptides": negatives,
                }
                pending.append(
                    (
                        str(stored_index).encode(),
                        pickle.dumps(cache_record, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                )
                stored_index += 1
                if source_index % 10_000 == 0 or source_index == total_rows:
                    elapsed = max(time.monotonic() - started_at, 1e-9)
                    rate = source_index / elapsed
                    eta = (total_rows - source_index) / rate if rate > 0 else float("inf")
                    print(
                        f"\rBuilding {lmdb_path.name}: {source_index:,}/{total_rows:,} "
                        f"({source_index / total_rows:.2%}) {rate:,.0f} spectra/s "
                        f"skipped={skipped_invalid_ctc:,} ETA {eta / 60:.1f} min",
                        end="\n" if source_index == total_rows else "",
                        flush=True,
                    )
                if len(pending) >= 1000:
                    commit_pending()
        commit_pending()
        pending.extend(
            [
                (b"n_spectra", str(stored_index).encode()),
                (b"source_spectra", str(source_index).encode()),
                (b"skipped_invalid_ctc", str(skipped_invalid_ctc).encode()),
                (b"initial_map_size", str(initial_map_size).encode()),
                (b"complete", b"1"),
            ]
        )
        commit_pending()
        actual_info = environment.info()
        actual_used_bytes = (int(actual_info["last_pgno"]) + 1) * int(environment.stat()["psize"])
        pending.extend(
            [
                (b"final_map_size", str(int(actual_info["map_size"])).encode()),
                (b"actual_used_bytes", str(actual_used_bytes).encode()),
            ]
        )
        commit_pending()
        try:
            next(mgf_records)
        except StopIteration:
            pass
        else:
            raise ValueError("MGF contains extra rows after preference Parquet")
    except Exception:
        environment.close()
        _remove_lmdb_files(lmdb_path)
        raise
    environment.close()


class PreferenceLmdbDataset(Dataset):
    def __init__(self, lmdb_path: str | Path, num_negatives: int) -> None:
        self.lmdb_path = str(lmdb_path)
        self.num_negatives = num_negatives
        self._environment = None
        with lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False) as environment:
            with environment.begin() as transaction:
                if transaction.get(b"complete") != b"1":
                    raise ValueError(f"Incomplete preference LMDB: {self.lmdb_path}")
                self.n_spectra = int(transaction.get(b"n_spectra").decode())

    def _env(self):
        if self._environment is None:
            self._environment = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False)
        return self._environment

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_environment"] = None
        return state

    def __len__(self) -> int:
        return self.n_spectra

    def __getitem__(self, index: int) -> Dict[str, object]:
        with self._env().begin() as transaction:
            payload = transaction.get(str(index).encode())
        if payload is None:
            raise IndexError(index)
        record = pickle.loads(payload)
        negatives = record["negative_peptides"][: self.num_negatives]
        if len(negatives) != self.num_negatives:
            raise ValueError(f"LMDB row {index} has insufficient negatives")
        return {
            "spectrum": record["spectrum"],
            "precursor": record["precursor"],
            "positive_peptide": record["positive_peptide"],
            "negative_peptides": negatives,
            "scan_id": record["scan_id"],
        }


def preference_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    spectra = torch.nn.utils.rnn.pad_sequence([item["spectrum"] for item in batch], batch_first=True)
    return {
        "spectra": spectra,
        "precursors": torch.stack([item["precursor"] for item in batch]),
        "positive_peptides": [item["positive_peptide"] for item in batch],
        "negative_peptides": [item["negative_peptides"] for item in batch],
        "scan_ids": [item["scan_id"] for item in batch],
    }


class PreferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_lmdb: str | Path,
        val_lmdb: str | Path,
        num_negatives: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_lmdb = train_lmdb
        self.val_lmdb = val_lmdb
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = PreferenceLmdbDataset(self.train_lmdb, self.num_negatives)
            self.val_dataset = PreferenceLmdbDataset(self.val_lmdb, self.num_negatives)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=preference_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=preference_collate,
        )
