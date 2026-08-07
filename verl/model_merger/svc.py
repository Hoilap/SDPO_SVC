# Copyright 2026 SDPO-CL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Singular Value Calibration for continual-learning checkpoints.

This module adapts the SVC rule from "When Shared Knowledge Hurts:
Spectral Over-Accumulation in Model Merging" to a task boundary in
continual learning.  For every selected 2-D weight matrix it treats

    previous - base
    raw       - previous

as the old-task and new-task updates, respectively.  Only the singular
values of ``raw - base`` are calibrated; its residual outside the computed
low-rank subspace is preserved exactly.

The CLI reads Hugging Face safetensors or PyTorch checkpoint shards lazily so
that three complete language models do not need to be resident in memory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


DEFAULT_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)

_WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def _resolve_model_path(model: str, cache_dir: str | None = None) -> Path:
    """Resolve a local model directory or download an HF repository snapshot."""

    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise FileNotFoundError(
            f"{model!r} is not a local directory and huggingface_hub is unavailable. "
            "Pass a local Hugging Face checkpoint path instead."
        ) from exc

    downloaded = snapshot_download(
        repo_id=model,
        cache_dir=cache_dir,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.py",
            "*.txt",
            "*.tiktoken",
            "*.safetensors",
            "*.bin",
        ],
    )
    return Path(downloaded).resolve()


class TensorSource:
    """Lazy access to tensors in a sharded Hugging Face checkpoint."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.format: str
        self.weight_map: dict[str, str]

        safe_index = model_dir / "model.safetensors.index.json"
        bin_index = model_dir / "pytorch_model.bin.index.json"
        safe_single = model_dir / "model.safetensors"
        bin_single = model_dir / "pytorch_model.bin"

        if safe_index.is_file():
            self.format = "safetensors"
            self.weight_map = json.loads(safe_index.read_text())["weight_map"]
            self.index_name = safe_index.name
        elif safe_single.is_file():
            self.format = "safetensors"
            try:
                from safetensors import safe_open
            except ImportError as exc:  # pragma: no cover - depends on runtime image
                raise ImportError("Reading safetensors checkpoints requires safetensors.") from exc
            with safe_open(safe_single, framework="pt", device="cpu") as handle:
                self.weight_map = {key: safe_single.name for key in handle.keys()}
            self.index_name = None
        elif bin_index.is_file():
            self.format = "pytorch"
            self.weight_map = json.loads(bin_index.read_text())["weight_map"]
            self.index_name = bin_index.name
        elif bin_single.is_file():
            self.format = "pytorch"
            state = torch.load(bin_single, map_location="cpu", weights_only=True)
            self.weight_map = {key: bin_single.name for key in state}
            del state
            self.index_name = None
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin checkpoint found in {model_dir}"
            )

        self._bin_cache_name: str | None = None
        self._bin_cache: dict[str, torch.Tensor] | None = None

    @property
    def keys(self) -> set[str]:
        return set(self.weight_map)

    @property
    def shard_names(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def keys_in_shard(self, shard_name: str) -> list[str]:
        return sorted(key for key, filename in self.weight_map.items() if filename == shard_name)

    def get_tensor(self, key: str) -> torch.Tensor:
        try:
            shard_name = self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from {self.model_dir}") from exc

        shard_path = self.model_dir / shard_name
        if self.format == "safetensors":
            from safetensors import safe_open

            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)

        if self._bin_cache_name != shard_name:
            self._bin_cache = torch.load(shard_path, map_location="cpu", weights_only=True)
            self._bin_cache_name = shard_name
        assert self._bin_cache is not None
        return self._bin_cache[key]


@dataclass
class LayerCalibrationStats:
    name: str
    shape: list[int]
    rank: int
    gamma_min: float
    gamma_mean: float
    gamma_max: float
    old_projection_mean: float
    new_projection_mean: float
    relative_correction_norm: float


def calibrate_matrix(
    base: torch.Tensor,
    previous: torch.Tensor,
    raw: torch.Tensor,
    *,
    rank: int,
    oversample: int = 8,
    niter: int = 2,
    alpha: float = 1.0,
    strength: float = 0.5,
    eps: float = 1e-8,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Calibrate one weight matrix and preserve the non-SVD residual.

    ``alpha=1`` implements the paper's suppression-only calibration.  The
    returned tensor has the same dtype and device as ``raw``.
    """

    if base.ndim != 2 or previous.ndim != 2 or raw.ndim != 2:
        raise ValueError("SVC expects three rank-2 weight matrices.")
    if base.shape != previous.shape or base.shape != raw.shape:
        raise ValueError(
            f"Shape mismatch: base={tuple(base.shape)}, previous={tuple(previous.shape)}, raw={tuple(raw.shape)}"
        )
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    output_dtype = raw.dtype
    compute_device = torch.device(device)
    base_f = base.to(device=compute_device, dtype=torch.float32)
    previous_f = previous.to(device=compute_device, dtype=torch.float32)
    raw_f = raw.to(device=compute_device, dtype=torch.float32)

    old_delta = previous_f - base_f
    new_delta = raw_f - previous_f
    merged_delta = raw_f - base_f

    max_rank = min(merged_delta.shape)
    effective_rank = min(rank, max_rank)
    q = min(effective_rank + max(0, oversample), max_rank)

    # svd_lowrank returns V rather than Vh.  q may exceed the desired
    # calibration rank to improve the randomized approximation.
    # SVC is a checkpoint transformation, so it should be reproducible.  Isolate
    # the randomized SVD RNG from callers that import this function in-process.
    fork_devices = []
    if compute_device.type == "cuda":
        fork_devices = [compute_device.index if compute_device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        u, singular_values, v = torch.svd_lowrank(merged_delta, q=q, niter=niter)
    order = torch.argsort(singular_values, descending=True)[:effective_rank]
    u = u[:, order]
    singular_values = singular_values[order]
    vh = v[:, order].T

    old_response = u.T @ old_delta
    new_response = u.T @ new_delta
    merged_response = u.T @ merged_delta

    def projection_scale(task_response: torch.Tensor) -> torch.Tensor:
        numerator = (merged_response * task_response).sum(dim=-1)
        denominator = task_response.square().sum(dim=-1).clamp_min(eps)
        scale = numerator / denominator
        # A zero task update should be neutral rather than causing a NaN or an
        # artificial correction at the first continual-learning boundary.
        return torch.where(task_response.square().sum(dim=-1) > eps, scale, torch.ones_like(scale))

    old_scale = projection_scale(old_response)
    new_scale = projection_scale(new_response)
    gamma = 2.0 / (old_scale.clamp_min(alpha) + new_scale.clamp_min(alpha))
    gamma = 1.0 + strength * (gamma - 1.0)

    correction = (u * ((gamma - 1.0) * singular_values).unsqueeze(0)) @ vh
    calibrated = raw_f + correction

    correction_norm = torch.linalg.vector_norm(correction)
    merged_norm = torch.linalg.vector_norm(merged_delta).clamp_min(eps)
    metrics: dict[str, float | int] = {
        "rank": effective_rank,
        "gamma_min": gamma.min().item(),
        "gamma_mean": gamma.mean().item(),
        "gamma_max": gamma.max().item(),
        "old_projection_mean": old_scale.mean().item(),
        "new_projection_mean": new_scale.mean().item(),
        "relative_correction_norm": (correction_norm / merged_norm).item(),
    }
    return calibrated.to(device=raw.device, dtype=output_dtype), metrics


def _copy_auxiliary_files(source_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer/config files but never copy source model weights."""

    for entry in source_dir.iterdir():
        if entry.name in _WEIGHT_FILENAMES or entry.suffix in {".safetensors", ".bin"}:
            continue
        target = output_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        elif entry.is_file():
            shutil.copy2(entry, target)


def _save_shard(tensors: dict[str, torch.Tensor], path: Path, checkpoint_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_format == "safetensors":
        from safetensors.torch import save_file

        contiguous = {key: tensor.contiguous() for key, tensor in tensors.items()}
        save_file(contiguous, path, metadata={"format": "pt"})
    else:
        torch.save(tensors, path)


def calibrate_checkpoints(
    *,
    base_model: str,
    previous_model: str,
    raw_model: str,
    output_dir: str,
    rank: int = 64,
    oversample: int = 8,
    niter: int = 2,
    alpha: float = 1.0,
    strength: float = 0.5,
    eps: float = 1e-8,
    device: str = "cpu",
    include_suffixes: Iterable[str] = DEFAULT_WEIGHT_SUFFIXES,
    include_regex: str | None = None,
    cache_dir: str | None = None,
    seed: int = 0,
) -> list[LayerCalibrationStats]:
    """Calibrate selected matrices from three Hugging Face checkpoints."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base_path = _resolve_model_path(base_model, cache_dir)
    previous_path = _resolve_model_path(previous_model, cache_dir)
    raw_path = _resolve_model_path(raw_model, cache_dir)
    base_source = TensorSource(base_path)
    previous_source = TensorSource(previous_path)
    raw_source = TensorSource(raw_path)

    # HF versions can differ in whether tied weights are materialized.  Only
    # selected matrices need to exist in all three checkpoints, so do not fail
    # globally on harmless unselected-key differences.
    if base_source.keys != raw_source.keys:
        print(
            "Warning: base/raw checkpoint keys differ "
            f"({len(base_source.keys)} versus {len(raw_source.keys)}); validating selected tensors lazily."
        )
    if previous_source.keys != raw_source.keys:
        print(
            "Warning: previous/raw checkpoint keys differ "
            f"({len(previous_source.keys)} versus {len(raw_source.keys)}); validating selected tensors lazily."
        )

    suffixes = tuple(suffix.strip() for suffix in include_suffixes if suffix.strip())
    pattern = re.compile(include_regex) if include_regex else None

    def selected(name: str, tensor: torch.Tensor) -> bool:
        name_selected = name.endswith(suffixes) or (
            pattern is not None and pattern.search(name) is not None
        )
        return name_selected and tensor.ndim == 2 and tensor.is_floating_point()

    _copy_auxiliary_files(raw_path, output)
    stats: list[LayerCalibrationStats] = []
    print(f"SVC base model:     {base_path}")
    print(f"SVC previous model: {previous_path}")
    print(f"SVC raw model:      {raw_path}")
    print(f"SVC output:         {output}")

    for shard_index, shard_name in enumerate(raw_source.shard_names, start=1):
        output_tensors: dict[str, torch.Tensor] = {}
        shard_keys = raw_source.keys_in_shard(shard_name)
        print(
            f"[{shard_index}/{len(raw_source.shard_names)}] "
            f"Processing {shard_name} ({len(shard_keys)} tensors)"
        )
        for name in shard_keys:
            raw_tensor = raw_source.get_tensor(name)
            if selected(name, raw_tensor):
                base_tensor = base_source.get_tensor(name)
                previous_tensor = previous_source.get_tensor(name)
                calibrated, layer_metrics = calibrate_matrix(
                    base_tensor,
                    previous_tensor,
                    raw_tensor,
                    rank=rank,
                    oversample=oversample,
                    niter=niter,
                    alpha=alpha,
                    strength=strength,
                    eps=eps,
                    device=device,
                    seed=seed,
                )
                stats.append(
                    LayerCalibrationStats(
                        name=name,
                        shape=list(raw_tensor.shape),
                        **layer_metrics,
                    )
                )
                print(
                    f"  calibrated {name}: rank={layer_metrics['rank']} "
                    f"gamma={layer_metrics['gamma_mean']:.4f} "
                    f"relative_correction={layer_metrics['relative_correction_norm']:.4e}"
                )
                output_tensors[name] = calibrated.cpu()
            else:
                output_tensors[name] = raw_tensor

        _save_shard(output_tensors, output / shard_name, raw_source.format)
        del output_tensors

    if raw_source.index_name is not None:
        shutil.copy2(raw_path / raw_source.index_name, output / raw_source.index_name)

    report = {
        "base_model": str(base_path),
        "previous_model": str(previous_path),
        "raw_model": str(raw_path),
        "rank": rank,
        "oversample": oversample,
        "niter": niter,
        "alpha": alpha,
        "strength": strength,
        "device": device,
        "seed": seed,
        "include_suffixes": list(suffixes),
        "include_regex": include_regex,
        "num_calibrated_layers": len(stats),
        "layers": [asdict(item) for item in stats],
    }
    (output / "svc_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Calibrated {len(stats)} matrices. Report: {output / 'svc_report.json'}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply task-boundary Singular Value Calibration to Hugging Face checkpoints."
    )
    parser.add_argument("--base-model", required=True, help="Initial/common-anchor HF model path or repository ID.")
    parser.add_argument("--previous-model", required=True, help="Calibrated checkpoint before the current task.")
    parser.add_argument("--raw-model", required=True, help="Raw checkpoint produced by SDPO on the current task.")
    parser.add_argument("--output-dir", required=True, help="Empty directory for the calibrated HF checkpoint.")
    parser.add_argument("--rank", type=int, default=64, help="Number of leading spectral directions to calibrate.")
    parser.add_argument("--oversample", type=int, default=8, help="Randomized SVD oversampling dimension.")
    parser.add_argument("--niter", type=int, default=2, help="Number of randomized SVD power iterations.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Projection floor; 1.0 is suppression-only SVC.")
    parser.add_argument("--strength", type=float, default=0.5, help="Interpolation from raw (0) to full SVC (1).")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu", help="SVD compute device, e.g. cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=0, help="Randomized SVD seed.")
    parser.add_argument(
        "--include-suffixes",
        default=",".join(DEFAULT_WEIGHT_SUFFIXES),
        help="Comma-separated parameter-name suffixes to calibrate.",
    )
    parser.add_argument("--include-regex", default=None, help="Optional additional parameter-name regex.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face snapshot cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrate_checkpoints(
        base_model=args.base_model,
        previous_model=args.previous_model,
        raw_model=args.raw_model,
        output_dir=args.output_dir,
        rank=args.rank,
        oversample=args.oversample,
        niter=args.niter,
        alpha=args.alpha,
        strength=args.strength,
        eps=args.eps,
        device=args.device,
        include_suffixes=args.include_suffixes.split(","),
        include_regex=args.include_regex,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
