import json

import pytest
import torch

from verl.model_merger import svc
from verl.model_merger.svc import calibrate_matrix


def test_first_task_boundary_is_noop() -> None:
    torch.manual_seed(0)
    base = torch.zeros(8, 6)
    raw = torch.randn(8, 6)

    calibrated, metrics = calibrate_matrix(
        base,
        base,
        raw,
        rank=6,
        oversample=0,
        strength=1.0,
    )

    torch.testing.assert_close(calibrated, raw, atol=1e-5, rtol=1e-5)
    assert abs(metrics["gamma_min"] - 1.0) < 1e-6
    assert abs(metrics["gamma_max"] - 1.0) < 1e-6


def test_shared_update_is_suppressed_to_single_copy() -> None:
    base = torch.zeros(8, 6)
    shared = torch.zeros(8, 6)
    shared[0, 0] = 3.0
    previous = shared
    raw = 2.0 * shared

    calibrated, metrics = calibrate_matrix(
        base,
        previous,
        raw,
        rank=1,
        oversample=0,
        strength=1.0,
    )

    torch.testing.assert_close(calibrated, shared, atol=1e-5, rtol=1e-5)
    assert abs(metrics["gamma_mean"] - 0.5) < 1e-6


def test_zero_strength_preserves_raw_model() -> None:
    torch.manual_seed(1)
    base = torch.zeros(8, 6)
    previous = torch.randn(8, 6)
    raw = previous + torch.randn(8, 6)

    calibrated, _ = calibrate_matrix(
        base,
        previous,
        raw,
        rank=4,
        strength=0.0,
    )

    torch.testing.assert_close(calibrated, raw, atol=1e-5, rtol=1e-5)


def _checkpoints(tmp_path, checkpoint_format="pytorch"):
    paths = [tmp_path / role for role in ("base", "previous", "raw")]
    for scale, path in enumerate(paths):
        path.mkdir()
        weight_map = {}
        for index in range(5):
            name = f"layer.{index}.q_proj.weight"
            suffix = "safetensors" if checkpoint_format == "safetensors" else "bin"
            shard = f"model-{index:05d}.{suffix}"
            matrix = torch.arange(48, dtype=torch.float32).reshape(8, 6) * scale / 48
            weights = {name: matrix, f"layer.{index}.norm.weight": torch.ones(6) * scale}
            svc._save_shard(weights, path / shard, checkpoint_format)
            weight_map.update({key: shard for key in weights})
        index_name = (
            "model.safetensors.index.json" if checkpoint_format == "safetensors" else "pytorch_model.bin.index.json"
        )
        (path / index_name).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        (path / "config.json").write_text('{"model_type": "test"}')
    return dict(base_model=str(paths[0]), previous_model=str(paths[1]), raw_model=str(paths[2]), rank=2, seed=17)


@pytest.mark.parametrize("checkpoint_format", ["pytorch", "safetensors"])
def test_spawned_workers_match_serial_and_preserve_metadata(tmp_path, monkeypatch, checkpoint_format):
    options = _checkpoints(tmp_path, checkpoint_format)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    parallel.mkdir()  # Empty output directories remain supported.
    expected = svc.calibrate_checkpoints(**options, output_dir=str(serial))
    # Exercise real spawn/IPC and shared staging on CPU-only CI. CUDA validation
    # is separately tested; production --devices accepts only distinct GPUs.
    monkeypatch.setattr(svc, "_validate_devices", lambda _: ["cpu"] * 4)
    actual = svc.calibrate_checkpoints(**options, output_dir=str(parallel), devices=["test"])
    assert [item.name for item in actual] == [item.name for item in expected]
    assert len(actual) == 5
    expected_source, actual_source = svc.TensorSource(serial), svc.TensorSource(parallel)
    assert actual_source.keys == expected_source.keys
    for name in actual_source.keys:
        torch.testing.assert_close(actual_source.get_tensor(name), expected_source.get_tensor(name))
    assert (parallel / "config.json").read_text() == (serial / "config.json").read_text()
    assert (parallel / actual_source.index_name).read_text() == (serial / expected_source.index_name).read_text()
    report = json.loads((parallel / "svc_report.json").read_text())
    assert report["num_calibrated_layers"] == 5
    assert not list(tmp_path.glob(".parallel.svc-*"))


@pytest.mark.parametrize("existing_empty", [False, True])
def test_parallel_failure_does_not_publish_output(tmp_path, monkeypatch, existing_empty):
    options = _checkpoints(tmp_path)
    # A worker encounters a shape mismatch after other shards can have finished.
    bad = tmp_path / "previous/model-00004.bin"
    torch.save({"layer.4.q_proj.weight": torch.zeros(2, 2)}, bad)
    monkeypatch.setattr(svc, "_validate_devices", lambda _: ["cpu"] * 4)
    output = tmp_path / "failed"
    if existing_empty:
        output.mkdir()
    with pytest.raises(ValueError, match="Shape mismatch"):
        svc.calibrate_checkpoints(**options, output_dir=str(output), devices=["test"])
    if existing_empty:
        assert list(output.iterdir()) == []
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".failed.svc-*"))


def test_nonempty_output_is_preserved(tmp_path):
    options = _checkpoints(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user-file").write_text("keep")
    with pytest.raises(FileExistsError):
        svc.calibrate_checkpoints(**options, output_dir=str(output))
    assert (output / "user-file").read_text() == "keep"


def test_device_validation_and_shard_assignment(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert svc._validate_devices(["cuda:0", "cuda:3"]) == ["cuda:0", "cuda:3"]
    for devices in ([], ["cuda:0", "cuda:0"], ["cpu"], ["cuda"], ["cuda:4"]):
        with pytest.raises(ValueError):
            svc._validate_devices(devices)
    _checkpoints(tmp_path)
    raw = svc.TensorSource(tmp_path / "raw")
    assignments = svc._assign_shards(tmp_path / "raw", raw.shard_names, 4)
    assert sorted(name for group in assignments for name in group) == raw.shard_names
    assert sorted(map(len, assignments)) == [1, 1, 1, 2]


def test_single_file_uses_one_worker_and_preserves_unselected_tensors(tmp_path, monkeypatch, capsys):
    options = {}
    for scale, role in enumerate(("base", "previous", "raw")):
        path = tmp_path / role
        path.mkdir()
        torch.save(
            {"q_proj.weight": torch.eye(6) * scale, "norm.weight": torch.ones(6) * scale},
            path / "pytorch_model.bin",
        )
        options[f"{role}_model"] = str(path)
    monkeypatch.setattr(svc, "_validate_devices", lambda _: ["cpu"] * 4)
    output = tmp_path / "single-file-output"
    stats = svc.calibrate_checkpoints(**options, output_dir=str(output), devices=["test"], rank=2)
    assert len(stats) == 1
    assert "only 1 shards/workers" in capsys.readouterr().out
    torch.testing.assert_close(svc.TensorSource(output).get_tensor("norm.weight"), torch.ones(6) * 2)


def test_output_inside_input_is_rejected(tmp_path):
    options = _checkpoints(tmp_path)
    output = tmp_path / "raw/nested"
    with pytest.raises(ValueError, match="outside all input"):
        svc.calibrate_checkpoints(**options, output_dir=str(output))
    assert not output.exists()
    assert not list((tmp_path / "raw").glob(".nested.svc-*"))


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four visible CUDA GPUs")
def test_four_cuda_workers_match_single_cuda(tmp_path):
    options = _checkpoints(tmp_path, "safetensors")
    svc.calibrate_checkpoints(**options, output_dir=str(tmp_path / "single"), device="cuda:0")
    svc.calibrate_checkpoints(
        **options, output_dir=str(tmp_path / "multi"), devices=[f"cuda:{index}" for index in range(4)]
    )
    single = svc.TensorSource(tmp_path / "single")
    multi = svc.TensorSource(tmp_path / "multi")
    for name in single.keys:
        torch.testing.assert_close(multi.get_tensor(name), single.get_tensor(name), atol=1e-5, rtol=1e-5)
