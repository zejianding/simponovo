"""Extract train/val MGF+Parquet datasets with exactly the top-K negatives."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Iterator, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def iter_mgf_blocks(path: Path) -> Iterator[Tuple[str, str, str]]:
    """Yield the unmodified MGF block plus TITLE and SEQ."""
    lines = []
    title = sequence = None
    in_block = False
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line == "BEGIN IONS":
                if in_block:
                    raise ValueError(f"Nested BEGIN IONS in {path}")
                lines, title, sequence, in_block = [raw_line], None, None, True
            elif in_block:
                lines.append(raw_line)
                if line.startswith("TITLE="):
                    title = line[6:]
                elif line.startswith("SEQ="):
                    sequence = line[4:]
                elif line == "END IONS":
                    if title is None or sequence is None:
                        raise ValueError(f"Incomplete annotated MGF block in {path}")
                    yield "".join(lines), title, sequence
                    in_block = False
    if in_block:
        raise ValueError(f"Unclosed MGF block in {path}")


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("scan_id", pa.string()),
        pa.field("positive_peptide", pa.string()),
        pa.field("negative_peptides", pa.list_(pa.string())),
        pa.field("negative_scores_raw", pa.list_(pa.float64())),
        pa.field("negative_source_ranks", pa.list_(pa.int16())),
        pa.field("negative_error_types", pa.list_(pa.string())),
    ]
)


def write_rows(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
        rows.clear()


def extract_shard(source_sidecar: Path, output_mgf: Path, output_parquet: Path, k: int) -> dict:
    source = pq.ParquetFile(source_sidecar)
    first_row = next(
        source.iter_batches(batch_size=1, columns=["source_mgf"])
    ).to_pylist()[0]
    source_mgf = Path(first_row["source_mgf"])
    if not source_mgf.is_file():
        raise FileNotFoundError(f"Source MGF does not exist: {source_mgf}")

    mgf_temp = output_mgf.with_name(f"{output_mgf.name}.{uuid.uuid4().hex}.tmp")
    parquet_temp = output_parquet.with_name(f"{output_parquet.name}.{uuid.uuid4().hex}.tmp")
    selected_rows = 0
    source_rows = 0
    output_rows: list[dict] = []
    mgf_blocks = iter_mgf_blocks(source_mgf)
    columns = [
        "scan_id",
        "positive_peptide",
        "negative_peptides",
        "negative_scores_raw",
        "negative_source_ranks",
        "negative_error_types",
    ]
    try:
        with mgf_temp.open("w", encoding="utf-8", newline="") as mgf_handle, pq.ParquetWriter(
            parquet_temp, OUTPUT_SCHEMA, compression="zstd"
        ) as writer:
            for batch in source.iter_batches(batch_size=4096, columns=columns):
                for row in batch.to_pylist():
                    try:
                        block, scan_id, positive = next(mgf_blocks)
                    except StopIteration as exc:
                        raise ValueError(f"MGF ended before sidecar at row {source_rows}") from exc
                    if row["scan_id"] != scan_id or row["positive_peptide"] != positive:
                        raise ValueError(f"MGF/sidecar mismatch at source row {source_rows}")
                    source_rows += 1
                    negatives = row["negative_peptides"]
                    if len(negatives) < k:
                        continue
                    selected = {
                        "scan_id": scan_id,
                        "positive_peptide": positive,
                        "negative_peptides": negatives[:k],
                        "negative_scores_raw": row["negative_scores_raw"][:k],
                        "negative_source_ranks": row["negative_source_ranks"][:k],
                        "negative_error_types": row["negative_error_types"][:k],
                    }
                    if any(len(selected[field]) != k for field in selected if field.startswith("negative_")):
                        raise ValueError(f"Negative list mismatch at {scan_id}")
                    mgf_handle.write(block)
                    output_rows.append(selected)
                    selected_rows += 1
                    if len(output_rows) >= 50_000:
                        write_rows(writer, output_rows)
            write_rows(writer, output_rows)
        try:
            next(mgf_blocks)
        except StopIteration:
            pass
        else:
            raise ValueError("MGF contains extra spectra after sidecar")
        if pq.ParquetFile(parquet_temp).metadata.num_rows != selected_rows:
            raise ValueError("Output MGF/Parquet row count validation failed")
        os.replace(mgf_temp, output_mgf)
        os.replace(parquet_temp, output_parquet)
    except Exception:
        mgf_temp.unlink(missing_ok=True)
        parquet_temp.unlink(missing_ok=True)
        raise
    return {"source_rows": source_rows, "selected_rows": selected_rows}


def append_sidecar(
    source_sidecar: Path,
    mgf_handle,
    parquet_writer: pq.ParquetWriter,
    k: int,
    output_rows: list[dict],
) -> dict:
    """Append one sidecar shard to already-open merged train outputs."""
    source = pq.ParquetFile(source_sidecar)
    first_row = next(source.iter_batches(batch_size=1, columns=["source_mgf"])).to_pylist()[0]
    mgf_blocks = iter_mgf_blocks(Path(first_row["source_mgf"]))
    source_rows = selected_rows = 0
    columns = [
        "scan_id",
        "positive_peptide",
        "negative_peptides",
        "negative_scores_raw",
        "negative_source_ranks",
        "negative_error_types",
    ]
    for batch in source.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            try:
                block, scan_id, positive = next(mgf_blocks)
            except StopIteration as exc:
                raise ValueError(f"MGF ended before sidecar at row {source_rows}") from exc
            if row["scan_id"] != scan_id or row["positive_peptide"] != positive:
                raise ValueError(f"MGF/sidecar mismatch at source row {source_rows}")
            source_rows += 1
            if len(row["negative_peptides"]) < k:
                continue
            selected = {
                "scan_id": scan_id,
                "positive_peptide": positive,
                "negative_peptides": row["negative_peptides"][:k],
                "negative_scores_raw": row["negative_scores_raw"][:k],
                "negative_source_ranks": row["negative_source_ranks"][:k],
                "negative_error_types": row["negative_error_types"][:k],
            }
            if any(len(selected[field]) != k for field in selected if field.startswith("negative_")):
                raise ValueError(f"Negative list mismatch at {scan_id}")
            mgf_handle.write(block)
            output_rows.append(selected)
            selected_rows += 1
            if len(output_rows) >= 50_000:
                write_rows(parquet_writer, output_rows)
    try:
        next(mgf_blocks)
    except StopIteration:
        pass
    else:
        raise ValueError("MGF contains extra spectra after sidecar")
    return {"source_rows": source_rows, "selected_rows": selected_rows}


def extract_merged_train(train_inputs: list[Path], output_mgf: Path, output_parquet: Path, k: int) -> list[dict]:
    mgf_temp = output_mgf.with_name(f"{output_mgf.name}.{uuid.uuid4().hex}.tmp")
    parquet_temp = output_parquet.with_name(f"{output_parquet.name}.{uuid.uuid4().hex}.tmp")
    summaries = []
    rows: list[dict] = []
    try:
        with mgf_temp.open("w", encoding="utf-8", newline="") as mgf_handle, pq.ParquetWriter(
            parquet_temp, OUTPUT_SCHEMA, compression="zstd"
        ) as writer:
            for input_path in train_inputs:
                shard = input_path.name.replace(".preference.parquet", "")
                print(f"Extracting train {shard}", flush=True)
                summary = append_sidecar(input_path, mgf_handle, writer, k, rows)
                summary.update({"split": "train", "shard": shard})
                summaries.append(summary)
                print(summary, flush=True)
            write_rows(writer, rows)
        if pq.ParquetFile(parquet_temp).metadata.num_rows != sum(item["selected_rows"] for item in summaries):
            raise ValueError("Merged train MGF/Parquet row count validation failed")
        os.replace(mgf_temp, output_mgf)
        os.replace(parquet_temp, output_parquet)
    except Exception:
        mgf_temp.unlink(missing_ok=True)
        parquet_temp.unlink(missing_ok=True)
        raise
    return summaries


def shard_number(path: Path) -> int:
    match = re.fullmatch(r"train_(\d+)\.preference\.parquet", path.name)
    if match is None:
        raise ValueError(f"Unexpected train sidecar name: {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-negatives", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--separate-train-shards",
        action="store_true",
        help="Write one MGF/Parquet pair per train shard instead of merged train files.",
    )
    args = parser.parse_args()
    if args.num_negatives < 1:
        raise ValueError("num-negatives must be positive")
    train_inputs = sorted(
        (args.sidecar_dir / "train").glob("*.preference.parquet"), key=shard_number
    )
    val_input = args.sidecar_dir / "val" / "val.preference.parquet"
    if not train_inputs or not val_input.is_file():
        raise FileNotFoundError("Expected completed train and val preference sidecars")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    summaries = []
    if args.separate_train_shards:
        (args.output_dir / "train").mkdir(parents=True)
        for input_path in train_inputs:
            shard = input_path.name.replace(".preference.parquet", "")
            summary = extract_shard(
                input_path,
                args.output_dir / "train" / f"{shard}.mgf",
                args.output_dir / "train" / f"{shard}.parquet",
                args.num_negatives,
            )
            summary.update({"split": "train", "shard": shard})
            summaries.append(summary)
    else:
        summaries.extend(
            extract_merged_train(
                train_inputs,
                args.output_dir / "train.mgf",
                args.output_dir / "train.parquet",
                args.num_negatives,
            )
        )
    print("Extracting validation", flush=True)
    summary = extract_shard(
        val_input,
        args.output_dir / "val.mgf",
        args.output_dir / "val.parquet",
        args.num_negatives,
    )
    summary.update({"split": "val", "shard": "val"})
    summaries.append(summary)
    pq.write_table(pa.Table.from_pylist(summaries), args.output_dir / "summary.parquet", compression="zstd")


if __name__ == "__main__":
    main()
