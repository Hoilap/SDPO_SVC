"""Audit continual-training parquet files and remove logical duplicates.

The DAPO-Math-17k parquet distributed with this project contains 100 complete
copies of the same 17,917 examples.  This utility scans one logical training
dataset (which may span several parquet shards), reports duplicate statistics,
and writes a deduplicated parquet only when duplicates are present.

Rows are identified by the fields that determine the learning example rather
than bookkeeping fields such as ``extra_info.index``.  In particular, two rows
with the same prompt but different reward targets are kept as distinct rows.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import decimal
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


AUDIT_VERSION = 2
DEFAULT_IDENTITY_COLUMNS = ("data_source", "ability", "prompt", "reward_model")


def _json_value(value: Any) -> Any:
    """Convert Arrow/Python values to a deterministic JSON representation."""
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, decimal.Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
    return value


def row_fingerprint(row: dict[str, Any], identity_columns: Sequence[str]) -> bytes:
    identity = {column: _json_value(row.get(column)) for column in identity_columns}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _source_metadata(paths: Sequence[Path]) -> list[dict[str, Any]]:
    metadata = []
    for path in paths:
        stat = path.stat()
        parquet = pq.ParquetFile(path)
        metadata.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "rows": parquet.metadata.num_rows,
            }
        )
    return metadata


def _iter_rows(paths: Sequence[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def audit_rows(
    paths: Sequence[Path], identity_columns: Sequence[str], batch_size: int = 8192
) -> tuple[int, int, int | None, bool]:
    seen: set[bytes] = set()
    total_rows = 0
    first_duplicate_index: int | None = None
    found_new_unique_after_duplicate = False
    for row in _iter_rows(paths, batch_size):
        fingerprint = row_fingerprint(row, identity_columns)
        if fingerprint in seen:
            if first_duplicate_index is None:
                first_duplicate_index = total_rows
        else:
            if first_duplicate_index is not None:
                found_new_unique_after_duplicate = True
            seen.add(fingerprint)
        total_rows += 1
    return total_rows, len(seen), first_duplicate_index, found_new_unique_after_duplicate


def _validate_schemas(paths: Sequence[Path]) -> pa.Schema:
    schema = pq.ParquetFile(paths[0]).schema_arrow
    for path in paths[1:]:
        candidate = pq.ParquetFile(path).schema_arrow
        if not candidate.equals(schema, check_metadata=False):
            raise ValueError(f"Parquet schema differs across training shards: {path}")
    return schema


def write_unique_rows(
    paths: Sequence[Path],
    output_path: Path,
    identity_columns: Sequence[str],
    batch_size: int = 8192,
) -> int:
    """Write first occurrences in source order and return the output row count."""
    schema = _validate_schemas(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary_path.unlink(missing_ok=True)

    seen: set[bytes] = set()
    written = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                rows = batch.to_pylist()
                keep_indices: list[int] = []
                for index, row in enumerate(rows):
                    fingerprint = row_fingerprint(row, identity_columns)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    keep_indices.append(index)
                if keep_indices:
                    unique_batch = batch.take(pa.array(keep_indices, type=pa.int64()))
                    writer.write_batch(unique_batch)
                    written += len(keep_indices)
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
    return written


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _write_paths_atomic(path: Path, paths: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text("".join(f"{item.resolve()}\n" for item in paths), encoding="utf-8")
    temporary_path.replace(path)


def _cached_report_is_valid(
    report: dict[str, Any],
    source_metadata: list[dict[str, Any]],
    identity_columns: Sequence[str],
    mode: str,
) -> bool:
    if report.get("audit_version") != AUDIT_VERSION:
        return False
    if report.get("sources") != source_metadata:
        return False
    if report.get("identity_columns") != list(identity_columns):
        return False
    if report.get("mode") != mode:
        return False
    selected_paths = [Path(path) for path in report.get("selected_paths", [])]
    return bool(selected_paths) and all(path.is_file() for path in selected_paths)


def audit_training_files(
    paths: Sequence[Path],
    output_path: Path,
    report_path: Path,
    paths_file: Path | None,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    batch_size: int = 8192,
    check_only: bool = False,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one training parquet is required")
    paths = [path.resolve() for path in paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training parquet does not exist: {', '.join(missing)}")

    available_columns = set(pq.ParquetFile(paths[0]).schema_arrow.names)
    missing_identity = [column for column in identity_columns if column not in available_columns]
    if missing_identity:
        raise ValueError(f"Training parquet lacks identity columns: {', '.join(missing_identity)}")

    sources = _source_metadata(paths)
    if report_path.is_file():
        cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        mode = "check-only" if check_only else "deduplicate"
        if _cached_report_is_valid(cached_report, sources, identity_columns, mode):
            selected_paths = [Path(path) for path in cached_report["selected_paths"]]
            if paths_file is not None:
                _write_paths_atomic(paths_file, selected_paths)
            print(
                f"Data audit cache hit: {cached_report['total_rows']} rows, "
                f"{cached_report['unique_rows']} unique"
            )
            if check_only:
                print(f"Duplicate status: {'YES' if cached_report['duplicate_rows'] else 'NO'}")
            return cached_report

    total_rows, unique_rows, first_duplicate_index, found_new_unique_after_duplicate = audit_rows(
        paths, identity_columns, batch_size=batch_size
    )
    duplicate_rows = total_rows - unique_rows
    unique_prefix = duplicate_rows == 0 or (
        first_duplicate_index == unique_rows and not found_new_unique_after_duplicate
    )
    if duplicate_rows and not check_only:
        written = write_unique_rows(paths, output_path, identity_columns, batch_size=batch_size)
        if written != unique_rows:
            raise RuntimeError(f"Audit counted {unique_rows} unique rows but wrote {written}")
        selected_paths = [output_path.resolve()]
    else:
        selected_paths = paths

    report = {
        "audit_version": AUDIT_VERSION,
        "mode": "check-only" if check_only else "deduplicate",
        "sources": sources,
        "identity_columns": list(identity_columns),
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_fraction": duplicate_rows / total_rows if total_rows else 0.0,
        "first_duplicate_index": first_duplicate_index,
        "unique_prefix": unique_prefix,
        "suggested_max_samples": unique_rows if duplicate_rows and unique_prefix else -1 if not duplicate_rows else None,
        "suggested_shuffle": False if duplicate_rows and unique_prefix else True if not duplicate_rows else None,
        "selected_paths": [str(path) for path in selected_paths],
    }
    _write_json_atomic(report_path, report)
    if paths_file is not None:
        _write_paths_atomic(paths_file, selected_paths)
    print(
        f"Data audit complete: {total_rows} rows, {unique_rows} unique, "
        f"{duplicate_rows} duplicates ({report['duplicate_fraction']:.2%})"
    )
    if check_only:
        print(f"Duplicate status: {'YES' if duplicate_rows else 'NO'}")
    elif duplicate_rows:
        print(f"Using deduplicated training parquet: {output_path.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path, help="Training parquet file(s), in shard order")
    parser.add_argument("--output", type=Path, help="Deduplicated parquet output")
    parser.add_argument("--report", type=Path, required=True, help="JSON audit report/cache")
    parser.add_argument("--paths-file", type=Path, help="Selected training paths, one per line")
    parser.add_argument("--check-only", action="store_true", help="Audit without writing a deduplicated parquet")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.check_only and (args.output is None or args.paths_file is None):
        parser.error("--output and --paths-file are required unless --check-only is used")
    output_path = args.output or args.report.with_suffix(".unused.parquet")
    audit_training_files(
        args.parquet,
        output_path=output_path,
        report_path=args.report,
        paths_file=args.paths_file,
        batch_size=args.batch_size,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()
