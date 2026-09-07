from pathlib import Path

import datasets
import pytest

from data.preflight_rl_dataset import validate_group


def _verl_row(elo):
    return {
        "prompt": [{"role": "user", "content": "question"}],
        "data_source": "test",
        "reward_model": {"style": "rule", "ground_truth": "answer"},
        "extra_info": {"elo": elo},
    }


def _write(path: Path, rows: list[dict]) -> None:
    datasets.Dataset.from_list(rows).to_parquet(path)


def test_null_metadata_aligns_with_integer_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write(first, [_verl_row(None)])
    _write(second, [_verl_row(1500)])

    assert validate_group("validation", [first, second]) == 2


def test_nested_feature_type_conflict_is_rejected(tmp_path: Path) -> None:
    string_elo = tmp_path / "string.parquet"
    integer_elo = tmp_path / "integer.parquet"
    _write(string_elo, [_verl_row("-")])
    _write(integer_elo, [_verl_row(1000)])

    with pytest.raises(ValueError, match="cannot concatenate"):
        validate_group("validation", [string_elo, integer_elo])
