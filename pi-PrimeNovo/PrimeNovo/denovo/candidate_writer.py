"""Streaming Parquet export for de novo CTC beam candidates."""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl


CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("scan_id", pa.string()),
        pa.field("candidates", pa.list_(pa.string())),
        pa.field("scores_raw", pa.list_(pa.float64())),
        pa.field("scores", pa.list_(pa.float64())),
    ]
)


class CandidateParquetWriter(pl.Callback):
    """Write prediction candidate batches without retaining all results in RAM."""

    def __init__(self, output_path: str | Path, row_group_size: int = 10_000):
        super().__init__()
        if row_group_size < 1:
            raise ValueError("row_group_size must be positive")

        self.output_path = Path(output_path)
        self.temp_path = self.output_path.with_name(
            f".{self.output_path.name}.{uuid.uuid4().hex}.tmp"
        )
        self.row_group_size = row_group_size
        self.rows_written = 0
        self._writer: pq.ParquetWriter | None = None
        self._buffer = {field.name: [] for field in CANDIDATE_SCHEMA}

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Buffer one prediction batch and flush complete row groups."""
        if outputs is None:
            return
        if not isinstance(outputs, dict):
            raise TypeError("Candidate export requires predict_step to return a dict")

        required = tuple(self._buffer)
        missing = [name for name in required if name not in outputs]
        if missing:
            raise KeyError(f"Candidate export output is missing fields: {missing}")

        rows = {name: list(outputs[name]) for name in required}
        row_count = len(rows["scan_id"])
        if any(len(rows[name]) != row_count for name in required):
            raise ValueError("Candidate export batch has unequal column lengths")

        for scan_id, candidates, scores_raw, scores in zip(
            rows["scan_id"],
            rows["candidates"],
            rows["scores_raw"],
            rows["scores"],
        ):
            if not isinstance(scan_id, str):
                raise TypeError("Candidate export scan_id must be a string")
            if not (len(candidates) == len(scores_raw) == len(scores)):
                raise ValueError(f"Unequal candidate list lengths for scan_id {scan_id!r}")
            if any(left > right for left, right in zip(scores_raw, scores_raw[1:])):
                raise ValueError(f"scores_raw is not non-decreasing for scan_id {scan_id!r}")
            for score_raw, score in zip(scores_raw, scores):
                if not math.isfinite(score_raw) or not math.isfinite(score):
                    raise ValueError(f"Non-finite candidate score for scan_id {scan_id!r}")
                if not math.isclose(score, math.exp(-score_raw), rel_tol=1e-6, abs_tol=1e-12):
                    raise ValueError(f"Candidate score transform mismatch for scan_id {scan_id!r}")

        for name in required:
            self._buffer[name].extend(rows[name])
        if len(self._buffer["scan_id"]) >= self.row_group_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer["scan_id"]:
            return
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temp_path,
                CANDIDATE_SCHEMA,
                compression="snappy",
            )
        table = pa.Table.from_pydict(self._buffer, schema=CANDIDATE_SCHEMA)
        self._writer.write_table(table, row_group_size=self.row_group_size)
        self.rows_written += table.num_rows
        self._buffer = {field.name: [] for field in CANDIDATE_SCHEMA}

    def finalize(self, expected_rows: int) -> None:
        """Close and atomically publish a complete candidate file."""
        try:
            self._flush()
            if self.rows_written != expected_rows:
                raise ValueError(
                    "Candidate Parquet row count mismatch: "
                    f"wrote {self.rows_written}, expected {expected_rows}"
                )
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self.temp_path,
                    CANDIDATE_SCHEMA,
                    compression="snappy",
                )
            self._writer.close()
            self._writer = None
            os.replace(self.temp_path, self.output_path)
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Discard incomplete output while preserving a prior final file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.temp_path.unlink(missing_ok=True)
