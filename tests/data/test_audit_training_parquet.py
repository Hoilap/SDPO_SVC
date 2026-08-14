from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.audit_training_parquet import audit_training_files


def _row(index: int, question: str, answer: str) -> dict:
    return {
        "data_source": "math_dapo",
        "ability": "math",
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"index": index},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _audit(tmp_path: Path, sources: list[Path], check_only: bool = False):
    return audit_training_files(
        sources,
        output_path=tmp_path / "audit" / "deduplicated.parquet",
        report_path=tmp_path / "audit" / "report.json",
        paths_file=tmp_path / "audit" / "paths.txt",
        batch_size=2,
        check_only=check_only,
    )


def test_duplicate_free_shards_are_used_without_copying(tmp_path: Path) -> None:
    first = tmp_path / "train-0.parquet"
    second = tmp_path / "train-1.parquet"
    _write(first, [_row(0, "q0", "a0")])
    _write(second, [_row(1, "q1", "a1")])

    report = _audit(tmp_path, [first, second])

    assert report["total_rows"] == 2
    assert report["unique_rows"] == 2
    assert report["duplicate_rows"] == 0
    assert report["suggested_max_samples"] == -1
    assert report["suggested_shuffle"] is True
    assert report["selected_paths"] == [str(first.resolve()), str(second.resolve())]
    assert not (tmp_path / "audit" / "deduplicated.parquet").exists()


def test_repeated_dataset_blocks_are_deduplicated_across_shards(tmp_path: Path) -> None:
    first = tmp_path / "train-0.parquet"
    second = tmp_path / "train-1.parquet"
    unique_rows = [_row(0, "q0", "a0"), _row(1, "q1", "a1")]
    _write(first, unique_rows + unique_rows)
    _write(second, unique_rows)

    report = _audit(tmp_path, [first, second])

    output = tmp_path / "audit" / "deduplicated.parquet"
    assert report["total_rows"] == 6
    assert report["unique_rows"] == 2
    assert report["duplicate_rows"] == 4
    assert report["unique_prefix"] is True
    assert report["suggested_max_samples"] == 2
    assert report["suggested_shuffle"] is False
    assert report["selected_paths"] == [str(output.resolve())]
    assert pq.read_table(output).to_pylist() == unique_rows


def test_same_prompt_with_different_reward_target_is_not_removed(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    rows = [_row(0, "same question", "answer a"), _row(1, "same question", "answer b")]
    _write(source, rows)

    report = _audit(tmp_path, [source])

    assert report["unique_rows"] == 2
    assert report["duplicate_rows"] == 0


def test_interleaved_duplicates_cannot_be_handled_by_loader_settings(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    rows = [_row(0, "q0", "a0"), _row(0, "q0", "a0"), _row(1, "q1", "a1")]
    _write(source, rows)

    report = _audit(tmp_path, [source], check_only=True)

    assert report["unique_prefix"] is False
    assert report["suggested_max_samples"] is None
    assert report["suggested_shuffle"] is None
    assert not (tmp_path / "audit" / "deduplicated.parquet").exists()


def test_audit_cache_is_reused_for_unchanged_sources(tmp_path: Path, capsys) -> None:
    source = tmp_path / "train.parquet"
    _write(source, [_row(0, "q0", "a0"), _row(0, "q0", "a0")])
    first_report = _audit(tmp_path, [source])
    capsys.readouterr()

    second_report = _audit(tmp_path, [source])

    assert second_report == first_report
    assert "Data audit cache hit" in capsys.readouterr().out
