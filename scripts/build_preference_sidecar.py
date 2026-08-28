"""Build a versioned PrimeNovo preference sidecar from MGF labels and candidates.

The raw MGF files and candidate Parquet files are never modified.  Each output
row remains aligned with its source spectrum and holds every filtered,
PrimeNovo-metric-incorrect candidate in original score order.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


PROTON_MASS = 1.007276
WATER_MASS = 18.01
MASS_TOLERANCE_DA = 0.1
MAX_CTC_LENGTH = 40
INDIVIDUAL_MATCH_TOLERANCE_DA = 0.1
CUMULATIVE_MATCH_TOLERANCE_DA = 0.5
DATASET_VERSION = "massive_kb_preference_v1"
TOKEN_SPLIT = re.compile(r"(?<=.)(?=[A-Z])")
ACTIVE_MASSES: dict[str, float] = {}


@dataclass(frozen=True)
class SpectrumRecord:
    scan_id: str
    precursor_mz: float
    precursor_charge: int
    peptide: str


@dataclass
class ShardStats:
    split: str
    shard_id: int
    source_mgf: str
    source_candidate_parquet: str
    total_rows: int = 0
    negative_rows: int = 0
    two_negative_rows: int = 0
    raw_candidate_count: int = 0
    kept_negative_count: int = 0
    filtered_metric_match_count: int = 0
    filtered_mass_count: int = 0
    filtered_duplicate_count: int = 0
    filtered_invalid_count: int = 0
    error_type_counts: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "shard_id": self.shard_id,
            "source_mgf": self.source_mgf,
            "source_candidate_parquet": self.source_candidate_parquet,
            "total_rows": self.total_rows,
            "negative_rows": self.negative_rows,
            "two_negative_rows": self.two_negative_rows,
            "raw_candidate_count": self.raw_candidate_count,
            "kept_negative_count": self.kept_negative_count,
            "filtered_metric_match_count": self.filtered_metric_match_count,
            "filtered_mass_count": self.filtered_mass_count,
            "filtered_duplicate_count": self.filtered_duplicate_count,
            "filtered_invalid_count": self.filtered_invalid_count,
            "same_composition_reorder_count": self.error_type_counts[
                "same_composition_reorder"
            ],
            "same_length_multi_residue_count": self.error_type_counts[
                "same_length_multi_residue"
            ],
            "different_length_count": self.error_type_counts["different_length"],
            "other_count": self.error_type_counts["other"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\data_backup\massivekb"),
        help="Directory containing the split MGF files and candidate directory.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Defaults to <data-root>/massive_kb_parquets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <data-root>/massive_kb_preference_v1.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "pi-PrimeNovo"
        / "PrimeNovo"
        / "config.yaml",
        help="PrimeNovo config that supplies the residue vocabulary.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Candidate Parquet batch size."
    )
    parser.add_argument(
        "--row-group-size", type=int, default=50_000, help="Output Parquet row group size."
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional per-shard limit for smoke tests."
    )
    parser.add_argument(
        "--train-shards",
        type=str,
        default="1-20",
        help="Inclusive train shard range, e.g. 1-20 or 1-2.",
    )
    parser.add_argument("--skip-val", action="store_true", help="Do not build validation.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output directory."
    )
    parser.add_argument(
        "--skip-output-scan",
        action="store_true",
        help="Skip the post-write serialization validation scan.",
    )
    parser.add_argument(
        "--skip-global-duplicate-check",
        action="store_true",
        help="Skip the final disk-backed cross-shard scan_id uniqueness check.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Refresh the terminal progress display every N rows (default: 10000).",
    )
    return parser.parse_args()


def parse_shard_range(value: str) -> range:
    match = re.fullmatch(r"(\d+)-(\d+)", value)
    if not match:
        raise ValueError("--train-shards must use an inclusive range such as 1-20")
    start, end = map(int, match.groups())
    if start < 1 or end < start:
        raise ValueError("invalid --train-shards range")
    return range(start, end + 1)


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def show_progress(label: str, completed: int, total: int, started_at: float, *, done: bool = False) -> None:
    total = max(total, 1)
    ratio = min(completed / total, 1.0)
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = completed / elapsed
    remaining = (total - completed) / rate if rate > 0 else float("inf")
    width = 28
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    message = (
        f"\r{label:<28} [{bar}] {ratio:6.2%} "
        f"{completed:,}/{total:,}  {rate:,.0f} rows/s  "
        f"elapsed {format_duration(elapsed)}  ETA {format_duration(remaining)}"
    )
    print(message, end="\n" if done else "", flush=True)


def load_residues(config_path: Path) -> dict[str, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    residues = config["residues"]
    if not isinstance(residues, dict):
        raise ValueError(f"Expected a residue mapping in {config_path}")
    return {str(token): float(mass) for token, mass in residues.items()}


def tokenize(sequence: str) -> list[str]:
    normalized = sequence.replace("[", "").replace("]", "")
    return TOKEN_SPLIT.split(normalized) if normalized else []


def canonical_tokens(tokens: list[str]) -> tuple[str, ...]:
    aliases = {"I": "L", "N+0.984": "D", "Q+0.984": "E"}
    return tuple(aliases.get(token, token) for token in tokens)


def required_ctc_length(tokens: list[str]) -> int:
    return len(tokens) + sum(
        left == right for left, right in zip(tokens, tokens[1:])
    )


def residue_mass(tokens: list[str], masses: dict[str, float]) -> float:
    return sum(masses[token] for token in tokens)


@functools.lru_cache(maxsize=20_000)
def cached_candidate(sequence: str) -> tuple[tuple[str, ...], tuple[str, ...], float, int] | None:
    """Parse a candidate once; the same peptide appears across many spectra."""
    tokens = tuple(tokenize(sequence))
    if (
        not tokens
        or any(token not in ACTIVE_MASSES for token in tokens)
        or required_ctc_length(list(tokens)) > MAX_CTC_LENGTH
    ):
        return None
    return (
        tokens,
        canonical_tokens(list(tokens)),
        residue_mass(list(tokens), ACTIVE_MASSES),
        required_ctc_length(list(tokens)),
    )


def peptide_match(
    peptide1: list[str], peptide2: list[str], masses: dict[str, float]
) -> bool:
    """Exact implementation of PrimeNovo evaluate.aa_match(..., mode='best')."""
    aa_matches = [False] * max(len(peptide1), len(peptide2))
    i1 = i2 = 0
    cumulative1 = cumulative2 = 0.0

    while i1 < len(peptide1) and i2 < len(peptide2):
        mass1, mass2 = masses[peptide1[i1]], masses[peptide2[i2]]
        if abs((cumulative1 + mass1) - (cumulative2 + mass2)) < CUMULATIVE_MATCH_TOLERANCE_DA:
            aa_matches[max(i1, i2)] = abs(mass1 - mass2) < INDIVIDUAL_MATCH_TOLERANCE_DA
            i1 += 1
            i2 += 1
            cumulative1 += mass1
            cumulative2 += mass2
        elif cumulative2 + mass2 > cumulative1 + mass1:
            cumulative1 += mass1
            i1 += 1
        else:
            cumulative2 += mass2
            i2 += 1

    if all(aa_matches):
        return True

    first_unmatched = next(index for index, matched in enumerate(aa_matches) if not matched)
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    cumulative1 = cumulative2 = 0.0
    while i1 >= first_unmatched and i2 >= first_unmatched:
        mass1, mass2 = masses[peptide1[i1]], masses[peptide2[i2]]
        if abs((cumulative1 + mass1) - (cumulative2 + mass2)) < CUMULATIVE_MATCH_TOLERANCE_DA:
            aa_matches[max(i1, i2)] = abs(mass1 - mass2) < INDIVIDUAL_MATCH_TOLERANCE_DA
            i1 -= 1
            i2 -= 1
            cumulative1 += mass1
            cumulative2 += mass2
        elif cumulative2 + mass2 > cumulative1 + mass1:
            cumulative1 += mass1
            i1 -= 1
        else:
            cumulative2 += mass2
            i2 -= 1
    return all(aa_matches)


@functools.lru_cache(maxsize=50_000)
def cached_peptide_match(
    peptide1: tuple[str, ...], peptide2: tuple[str, ...]
) -> bool:
    return peptide_match(list(peptide1), list(peptide2), ACTIVE_MASSES)


def error_type(positive: tuple[str, ...], negative: tuple[str, ...]) -> str:
    if Counter(positive) == Counter(negative):
        return "same_composition_reorder"
    if len(positive) == len(negative):
        return "same_length_multi_residue"
    if len(positive) != len(negative):
        return "different_length"
    return "other"


def iter_mgf(path: Path) -> Iterator[SpectrumRecord]:
    title: str | None = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None
    peptide: str | None = None
    in_spectrum = False

    with path.open("r", encoding="utf-8", errors="strict", newline=None) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                if in_spectrum:
                    raise ValueError(f"Nested BEGIN IONS in {path}")
                in_spectrum = True
                title = precursor_mz = precursor_charge = peptide = None
            elif line == "END IONS":
                if not in_spectrum:
                    raise ValueError(f"END IONS without BEGIN IONS in {path}")
                if title is None or precursor_mz is None or precursor_charge is None or peptide is None:
                    raise ValueError(f"Incomplete annotated spectrum in {path}: {title!r}")
                yield SpectrumRecord(title, precursor_mz, precursor_charge, peptide)
                in_spectrum = False
            elif in_spectrum and line.startswith("TITLE="):
                title = line[6:]
            elif in_spectrum and line.startswith("PEPMASS="):
                precursor_mz = float(line[8:].split()[0])
            elif in_spectrum and line.startswith("CHARGE="):
                charge_match = re.match(r"(\d+)", line[7:])
                if charge_match is None:
                    raise ValueError(f"Invalid CHARGE line in {path}: {line}")
                precursor_charge = int(charge_match.group(1))
            elif in_spectrum and line.startswith("SEQ="):
                peptide = line[4:]
    if in_spectrum:
        raise ValueError(f"Unclosed spectrum in {path}")


def output_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("split", pa.string()),
            pa.field("shard_id", pa.int16()),
            pa.field("source_spectrum_index", pa.int32()),
            pa.field("scan_id", pa.string()),
            pa.field("source_mgf", pa.string()),
            pa.field("source_candidate_parquet", pa.string()),
            pa.field("positive_peptide", pa.string()),
            pa.field("positive_ctc_length", pa.int16()),
            pa.field("row_status", pa.string()),
            pa.field("precursor_mz", pa.float64()),
            pa.field("precursor_charge", pa.int8()),
            pa.field("precursor_residue_mass", pa.float64()),
            pa.field("positive_mass_error_da", pa.float32()),
            pa.field("negative_peptides", pa.list_(pa.string())),
            pa.field("negative_scores_raw", pa.list_(pa.float64())),
            pa.field("negative_scores", pa.list_(pa.float64())),
            pa.field("negative_source_ranks", pa.list_(pa.int16())),
            pa.field("negative_mass_errors_da", pa.list_(pa.float32())),
            pa.field("negative_error_types", pa.list_(pa.string())),
            pa.field("negative_ctc_lengths", pa.list_(pa.int16())),
            pa.field("candidate_count_raw", pa.int16()),
            pa.field("negative_count", pa.int16()),
            pa.field("has_negative", pa.bool_()),
            pa.field("has_two_negatives", pa.bool_()),
            pa.field("positive_candidate_rank", pa.int16()),
            pa.field("filtered_metric_match_count", pa.int16()),
            pa.field("filtered_mass_count", pa.int16()),
            pa.field("filtered_duplicate_count", pa.int16()),
            pa.field("filtered_invalid_count", pa.int16()),
        ]
    )


SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("split", pa.string()),
        pa.field("shard_id", pa.int16()),
        pa.field("source_mgf", pa.string()),
        pa.field("source_candidate_parquet", pa.string()),
        *[
            pa.field(name, pa.int64())
            for name in [
                "total_rows",
                "negative_rows",
                "two_negative_rows",
                "raw_candidate_count",
                "kept_negative_count",
                "filtered_metric_match_count",
                "filtered_mass_count",
                "filtered_duplicate_count",
                "filtered_invalid_count",
                "same_composition_reorder_count",
                "same_length_multi_residue_count",
                "different_length_count",
                "other_count",
            ]
        ],
    ]
)


def append_row(buffer: dict[str, list], row: dict) -> None:
    for key in buffer:
        buffer[key].append(row[key])


def flush_rows(writer: pq.ParquetWriter, buffer: dict[str, list], schema: pa.Schema) -> None:
    if not next(iter(buffer.values())):
        return
    writer.write_table(pa.Table.from_pydict(buffer, schema=schema))
    for values in buffer.values():
        values.clear()


def process_shard(
    *,
    split: str,
    shard_id: int,
    mgf_path: Path,
    candidate_path: Path,
    output_path: Path,
    masses: dict[str, float],
    batch_size: int,
    row_group_size: int,
    max_rows: int | None,
    progress_every: int,
) -> ShardStats:
    schema = output_schema()
    temporary_path = output_path.with_name(f"{output_path.name}.{uuid.uuid4().hex}.tmp")
    stats = ShardStats(split, shard_id, str(mgf_path), str(candidate_path))
    buffer = {field.name: [] for field in schema}
    mgf_records = iter_mgf(mgf_path)
    candidate_file = pq.ParquetFile(candidate_path)
    source_index = 0
    expected_rows = min(candidate_file.metadata.num_rows, max_rows) if max_rows is not None else candidate_file.metadata.num_rows
    started_at = time.monotonic()
    show_progress(f"Build {split}_{shard_id}", 0, expected_rows, started_at)

    try:
        with pq.ParquetWriter(
            temporary_path,
            schema=schema,
            compression="zstd",
            use_dictionary=["split", "scan_id", "negative_error_types"],
            write_statistics=True,
        ) as writer:
            for batch in candidate_file.iter_batches(
                batch_size=batch_size,
                columns=["scan_id", "candidates", "scores_raw", "scores"],
            ):
                for candidate_row in batch.to_pylist():
                    if max_rows is not None and source_index >= max_rows:
                        break
                    try:
                        spectrum = next(mgf_records)
                    except StopIteration as exc:
                        raise ValueError(
                            f"MGF ended before candidate parquet at row {source_index}: {candidate_path}"
                        ) from exc
                    if candidate_row["scan_id"] != spectrum.scan_id:
                        raise ValueError(
                            "scan_id mismatch at row "
                            f"{source_index}: candidate={candidate_row['scan_id']!r}, "
                            f"mgf={spectrum.scan_id!r}"
                        )

                    candidates = candidate_row["candidates"]
                    scores_raw = candidate_row["scores_raw"]
                    scores = candidate_row["scores"]
                    if not (len(candidates) == len(scores_raw) == len(scores)):
                        raise ValueError(f"Unequal candidate list lengths at {spectrum.scan_id}")
                    if any(
                        left > right
                        for left, right in zip(scores_raw, scores_raw[1:])
                    ):
                        raise ValueError(f"scores_raw is not non-decreasing at {spectrum.scan_id}")

                    positive_parsed = cached_candidate(spectrum.peptide)
                    precursor_residue_mass = (
                        (spectrum.precursor_mz - PROTON_MASS) * spectrum.precursor_charge
                        - WATER_MASS
                    )
                    if positive_parsed is None:
                        raw_positive_tokens = tokenize(spectrum.peptide)
                        invalid_tokens = any(token not in masses for token in raw_positive_tokens)
                        positive_ctc_length = (
                            required_ctc_length(raw_positive_tokens)
                            if not invalid_tokens
                            else 0
                        )
                        row_status = (
                            "positive_invalid_token"
                            if invalid_tokens
                            else "positive_ctc_too_long"
                        )
                        positive_token_tuple = tuple(raw_positive_tokens)
                        positive_canonical = canonical_tokens(raw_positive_tokens)
                        positive_tokens = raw_positive_tokens
                        positive_mass_error = None
                    else:
                        positive_token_tuple, positive_canonical, positive_mass, positive_ctc_length = positive_parsed
                        positive_tokens = list(positive_token_tuple)
                        positive_mass_error = abs(positive_mass - precursor_residue_mass)
                        row_status = "eligible"
                        if positive_mass_error > MASS_TOLERANCE_DA:
                            row_status = "positive_mass_out_of_tolerance"

                    negatives: list[str] = []
                    negative_scores_raw: list[float] = []
                    negative_scores: list[float] = []
                    negative_ranks: list[int] = []
                    negative_mass_errors: list[float] = []
                    negative_types: list[str] = []
                    negative_ctc_lengths: list[int] = []
                    seen_negatives: set[tuple[str, ...]] = set()
                    positive_candidate_rank = 0
                    filtered_metric = filtered_mass = filtered_duplicate = filtered_invalid = 0

                    for rank, (candidate, score_raw, score) in enumerate(
                        zip(candidates, scores_raw, scores), start=1
                    ):
                        if row_status != "eligible":
                            break
                        candidate_parsed = cached_candidate(candidate)
                        if candidate_parsed is None:
                            filtered_invalid += 1
                            continue
                        candidate_token_tuple, candidate_canonical, candidate_mass, candidate_ctc_length = candidate_parsed
                        candidate_mass_error = abs(candidate_mass - precursor_residue_mass)
                        if candidate_mass_error > MASS_TOLERANCE_DA:
                            filtered_mass += 1
                            continue
                        if cached_peptide_match(positive_token_tuple, candidate_token_tuple):
                            filtered_metric += 1
                            if positive_candidate_rank == 0:
                                positive_candidate_rank = rank
                            continue
                        if candidate_canonical in seen_negatives:
                            filtered_duplicate += 1
                            continue
                        seen_negatives.add(candidate_canonical)
                        category = error_type(positive_canonical, candidate_canonical)
                        negatives.append(candidate)
                        negative_scores_raw.append(float(score_raw))
                        negative_scores.append(float(score))
                        negative_ranks.append(rank)
                        negative_mass_errors.append(float(candidate_mass_error))
                        negative_types.append(category)
                        negative_ctc_lengths.append(candidate_ctc_length)
                        stats.error_type_counts[category] += 1

                    count = len(negatives)
                    if count:
                        stats.negative_rows += 1
                    if count >= 2:
                        stats.two_negative_rows += 1
                    stats.total_rows += 1
                    stats.raw_candidate_count += len(candidates)
                    stats.kept_negative_count += count
                    stats.filtered_metric_match_count += filtered_metric
                    stats.filtered_mass_count += filtered_mass
                    stats.filtered_duplicate_count += filtered_duplicate
                    stats.filtered_invalid_count += filtered_invalid

                    row = {
                        "split": split,
                        "shard_id": shard_id,
                        "source_spectrum_index": source_index,
                        "scan_id": spectrum.scan_id,
                        "source_mgf": str(mgf_path),
                        "source_candidate_parquet": str(candidate_path),
                        "positive_peptide": spectrum.peptide,
                        "positive_ctc_length": positive_ctc_length,
                        "row_status": row_status,
                        "precursor_mz": spectrum.precursor_mz,
                        "precursor_charge": spectrum.precursor_charge,
                        "precursor_residue_mass": precursor_residue_mass,
                        "positive_mass_error_da": (
                            float(positive_mass_error)
                            if positive_mass_error is not None
                            else None
                        ),
                        "negative_peptides": negatives,
                        "negative_scores_raw": negative_scores_raw,
                        "negative_scores": negative_scores,
                        "negative_source_ranks": negative_ranks,
                        "negative_mass_errors_da": negative_mass_errors,
                        "negative_error_types": negative_types,
                        "negative_ctc_lengths": negative_ctc_lengths,
                        "candidate_count_raw": len(candidates),
                        "negative_count": count,
                        "has_negative": count >= 1,
                        "has_two_negatives": count >= 2,
                        "positive_candidate_rank": positive_candidate_rank or None,
                        "filtered_metric_match_count": filtered_metric,
                        "filtered_mass_count": filtered_mass,
                        "filtered_duplicate_count": filtered_duplicate,
                        "filtered_invalid_count": filtered_invalid,
                    }
                    lengths = {
                        len(row[column])
                        for column in [
                            "negative_peptides",
                            "negative_scores_raw",
                            "negative_scores",
                            "negative_source_ranks",
                            "negative_mass_errors_da",
                            "negative_error_types",
                            "negative_ctc_lengths",
                        ]
                    }
                    if lengths != {count}:
                        raise AssertionError(f"Negative list alignment failure at {spectrum.scan_id}")
                    append_row(buffer, row)
                    if len(buffer["scan_id"]) >= row_group_size:
                        flush_rows(writer, buffer, schema)
                    source_index += 1
                    if source_index % progress_every == 0 or source_index == expected_rows:
                        show_progress(
                            f"Build {split}_{shard_id}",
                            source_index,
                            expected_rows,
                            started_at,
                            done=source_index == expected_rows,
                        )
                if max_rows is not None and source_index >= max_rows:
                    break
            flush_rows(writer, buffer, schema)

        if max_rows is None:
            try:
                extra_spectrum = next(mgf_records)
            except StopIteration:
                extra_spectrum = None
            if extra_spectrum is not None:
                raise ValueError(
                    f"MGF contains extra spectrum after candidate parquet: {extra_spectrum.scan_id}"
                )
            if source_index != candidate_file.metadata.num_rows:
                raise ValueError(
                    f"Candidate parquet not fully consumed: {source_index} != {candidate_file.metadata.num_rows}"
                )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return stats


def validate_output(path: Path, expected_rows: int, progress_every: int) -> None:
    rows = 0
    started_at = time.monotonic()
    show_progress(f"Verify {path.stem}", 0, expected_rows, started_at)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        data = batch.to_pydict()
        for index, count in enumerate(data["negative_count"]):
            list_columns = [
                "negative_peptides",
                "negative_scores_raw",
                "negative_scores",
                "negative_source_ranks",
                "negative_mass_errors_da",
                "negative_error_types",
                "negative_ctc_lengths",
            ]
            if any(len(data[column][index]) != count for column in list_columns):
                raise ValueError(f"Serialized list length mismatch in {path} at row {rows + index}")
            if data["has_negative"][index] != (count >= 1):
                raise ValueError(f"Serialized has_negative mismatch in {path} at row {rows + index}")
            if data["has_two_negatives"][index] != (count >= 2):
                raise ValueError(f"Serialized has_two_negatives mismatch in {path} at row {rows + index}")
            raw = data["negative_scores_raw"][index]
            if any(left > right for left, right in zip(raw, raw[1:])):
                raise ValueError(f"Serialized score order mismatch in {path} at row {rows + index}")
        rows += batch.num_rows
        if rows % progress_every < batch.num_rows or rows == expected_rows:
            show_progress(
                f"Verify {path.stem}", rows, expected_rows, started_at, done=rows == expected_rows
            )
    if rows != expected_rows:
        raise ValueError(f"Output row count mismatch for {path}: {rows} != {expected_rows}")


def source_metadata(path: Path) -> dict:
    info = path.stat()
    return {
        "path": str(path),
        "size_bytes": info.st_size,
        "mtime_utc": datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat(),
    }


def audit_and_check_outputs(
    output_paths: list[Path], output_dir: Path, progress_every: int
) -> dict:
    """Verify global scan_id uniqueness and retain deterministic audit samples."""
    rng = random.Random(6558)
    sample_limit = 100
    samples: dict[str, list[dict]] = {
        "has_negative": [],
        "no_negative": [],
        "metric_filtered": [],
    }
    seen_per_bucket = Counter()
    sqlite_path = output_dir / f".scan_id_check_{uuid.uuid4().hex}.sqlite"
    connection = sqlite3.connect(sqlite_path)
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in output_paths)
    processed_rows = 0
    started_at = time.monotonic()
    show_progress("Global audit", 0, total_rows, started_at)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE scan_ids (scan_id TEXT PRIMARY KEY)")
        for path in output_paths:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=8192,
                columns=[
                    "scan_id",
                    "source_spectrum_index",
                    "positive_peptide",
                    "negative_count",
                    "negative_peptides",
                    "negative_error_types",
                    "filtered_metric_match_count",
                ],
            ):
                data = batch.to_pydict()
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO scan_ids(scan_id) VALUES (?)",
                    [(scan_id,) for scan_id in data["scan_id"]],
                )
                if connection.total_changes - before != batch.num_rows:
                    raise ValueError(f"Duplicate scan_id detected while checking {path}")

                for index, scan_id in enumerate(data["scan_id"]):
                    record = {
                        "scan_id": scan_id,
                        "source_spectrum_index": data["source_spectrum_index"][index],
                        "positive_peptide": data["positive_peptide"][index],
                        "negative_count": data["negative_count"][index],
                        "negative_peptides": data["negative_peptides"][index][:3],
                        "negative_error_types": data["negative_error_types"][index][:3],
                    }
                    buckets = []
                    if data["negative_count"][index] >= 1:
                        buckets.append("has_negative")
                    else:
                        buckets.append("no_negative")
                    if data["filtered_metric_match_count"][index] > 0:
                        buckets.append("metric_filtered")
                    for bucket in buckets:
                        seen_per_bucket[bucket] += 1
                        reservoir = samples[bucket]
                        if len(reservoir) < sample_limit:
                            reservoir.append(record)
                        else:
                            replace_index = rng.randrange(seen_per_bucket[bucket])
                            if replace_index < sample_limit:
                                reservoir[replace_index] = record
            connection.commit()
            processed_rows += batch.num_rows
            if processed_rows % progress_every < batch.num_rows or processed_rows == total_rows:
                show_progress(
                    "Global audit",
                    processed_rows,
                    total_rows,
                    started_at,
                    done=processed_rows == total_rows,
                )
    finally:
        connection.close()
        sqlite_path.unlink(missing_ok=True)

    audit = {
        "seed": 6558,
        "sample_limit_per_bucket": sample_limit,
        "population_sizes": dict(seen_per_bucket),
        "samples": samples,
    }
    with (output_dir / "audit_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return audit


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    candidate_dir = (args.candidate_dir or data_root / "massive_kb_parquets").resolve()
    output_dir = (args.output_dir or data_root / DATASET_VERSION).resolve()
    train_shards = parse_shard_range(args.train_shards)
    masses = load_residues(args.config)
    ACTIVE_MASSES.clear()
    ACTIVE_MASSES.update(masses)
    cached_candidate.cache_clear()
    cached_peptide_match.cache_clear()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "train").mkdir()
    if not args.skip_val:
        (output_dir / "val").mkdir()

    shard_stats: list[ShardStats] = []
    source_files: list[Path] = []
    output_paths: list[Path] = []
    try:
        for shard_id in train_shards:
            mgf_path = data_root / f"massivekb_82c0124b_train_{shard_id}.mgf"
            candidate_path = candidate_dir / f"massivekb_82c0124b_train_{shard_id}.parquet"
            output_path = output_dir / "train" / f"train_{shard_id}.preference.parquet"
            if not mgf_path.is_file() or not candidate_path.is_file():
                raise FileNotFoundError(f"Missing paired train inputs for shard {shard_id}")
            print(f"Building train shard {shard_id}: {candidate_path.name}", flush=True)
            stats = process_shard(
                split="train",
                shard_id=shard_id,
                mgf_path=mgf_path,
                candidate_path=candidate_path,
                output_path=output_path,
                masses=masses,
                batch_size=args.batch_size,
                row_group_size=args.row_group_size,
                max_rows=args.max_rows,
                progress_every=args.progress_every,
            )
            if not args.skip_output_scan:
                validate_output(output_path, stats.total_rows, args.progress_every)
            shard_stats.append(stats)
            output_paths.append(output_path)
            source_files.extend([mgf_path, candidate_path])
            print(
                f"Completed train shard {shard_id}: rows={stats.total_rows}, "
                f"has_negative={stats.negative_rows}, has_two={stats.two_negative_rows}",
                flush=True,
            )

        if not args.skip_val:
            mgf_path = data_root / "massivekb_82c0124b_val.mgf"
            candidate_path = candidate_dir / "massivekb_82c0124b_val.parquet"
            output_path = output_dir / "val" / "val.preference.parquet"
            print("Building validation shard", flush=True)
            stats = process_shard(
                split="val",
                shard_id=0,
                mgf_path=mgf_path,
                candidate_path=candidate_path,
                output_path=output_path,
                masses=masses,
                batch_size=args.batch_size,
                row_group_size=args.row_group_size,
                max_rows=args.max_rows,
                progress_every=args.progress_every,
            )
            if not args.skip_output_scan:
                validate_output(output_path, stats.total_rows, args.progress_every)
            shard_stats.append(stats)
            output_paths.append(output_path)
            source_files.extend([mgf_path, candidate_path])
            print(
                f"Completed validation: rows={stats.total_rows}, "
                f"has_negative={stats.negative_rows}, has_two={stats.two_negative_rows}",
                flush=True,
            )

        summary_rows = [stats.as_dict() for stats in shard_stats]
        pq.write_table(
            pa.Table.from_pylist(summary_rows, schema=SUMMARY_SCHEMA),
            output_dir / "shard_summary.parquet",
            compression="zstd",
        )
        total_rows = sum(item.total_rows for item in shard_stats)
        total_negative_rows = sum(item.negative_rows for item in shard_stats)
        total_two_negative_rows = sum(item.two_negative_rows for item in shard_stats)
        audit = None
        if not args.skip_global_duplicate_check:
            print("Running global scan_id uniqueness check and audit sampling", flush=True)
            audit = audit_and_check_outputs(output_paths, output_dir, args.progress_every)
        manifest = {
            "dataset_version": DATASET_VERSION,
            "created_utc": datetime.now(tz=timezone.utc).isoformat(),
            "schema": str(output_schema()),
            "configuration": {
                "mass_tolerance_da": MASS_TOLERANCE_DA,
                "individual_match_tolerance_da": INDIVIDUAL_MATCH_TOLERANCE_DA,
                "cumulative_match_tolerance_da": CUMULATIVE_MATCH_TOLERANCE_DA,
                "max_ctc_length": MAX_CTC_LENGTH,
                "candidate_order": "source scores_raw ascending / scores descending",
                "prime_novo_config": str(args.config.resolve()),
                "residues": masses,
            },
            "inputs": [source_metadata(path) for path in source_files],
            "totals": {
                "rows": total_rows,
                "rows_with_negative": total_negative_rows,
                "rows_with_two_negatives": total_two_negative_rows,
                "coverage_at_least_one": total_negative_rows / total_rows if total_rows else 0.0,
                "coverage_at_least_two": total_two_negative_rows / total_rows if total_rows else 0.0,
            },
            "shards": summary_rows,
            "audit": {
                "path": "audit_samples.json" if audit is not None else None,
                "population_sizes": audit["population_sizes"] if audit is not None else None,
            },
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        print(json.dumps(manifest["totals"], indent=2), flush=True)
    except Exception:
        # Preserve completed shard files for forensic inspection, but make failure explicit.
        raise


if __name__ == "__main__":
    main()
