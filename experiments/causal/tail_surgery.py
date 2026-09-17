#!/usr/bin/env python3
"""Build the tail-ablation controls described in ``experiments/causal/PLAN.md``.

The two trained inputs and the base must be Hugging Face checkpoints of the
same architecture.  Original full checkpoints are referenced in the emitted
manifest; six derived checkpoints are written shard by shard:

* GRPO-no-tail / GRPO-random-tail / GRPO-principal+SDPO-tail
* SDPO-no-tail / SDPO-random-tail / SDPO-principal+GRPO-tail

Only the configured two-dimensional projection matrices are modified.  All
other parameters follow the host checkpoint exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from verl.model_merger.svc import (
    DEFAULT_WEIGHT_SUFFIXES,
    TensorSource,
    _copy_auxiliary_files,
    _resolve_model_path,
    _save_shard,
)


DERIVED_ARMS = {
    "GRPO-no-tail": "grpo_no_tail",
    "GRPO-random-tail": "grpo_random_tail",
    "GRPO-principal+SDPO-tail": "grpo_principal_sdpo_tail",
    "SDPO-no-tail": "sdpo_no_tail",
    "SDPO-random-tail": "sdpo_random_tail",
    "SDPO-principal+GRPO-tail": "sdpo_principal_grpo_tail",
}


@dataclass
class Decomposition:
    principal: torch.Tensor
    tail: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    rank: int
    delta_norm: float
    principal_norm: float
    tail_norm: float
    principal_energy_ratio: float
    relative_reconstruction_error: float


@dataclass
class MatrixReport:
    name: str
    shape: list[int]
    rank: int
    grpo_delta_norm: float
    grpo_principal_norm: float
    grpo_tail_norm: float
    grpo_principal_energy_ratio: float
    grpo_reconstruction_error: float
    sdpo_delta_norm: float
    sdpo_principal_norm: float
    sdpo_tail_norm: float
    sdpo_principal_energy_ratio: float
    sdpo_reconstruction_error: float
    grpo_random_tail_norm: float
    sdpo_random_tail_norm: float
    grpo_to_sdpo_tail_norm: float
    sdpo_to_grpo_tail_norm: float


def _tensor_seed(seed: int, name: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def decompose_update(
    base: torch.Tensor,
    trained: torch.Tensor,
    *,
    rank_fraction: float,
    device: torch.device | str,
    seed: int,
    oversample: int,
    niter: int,
    exact_svd: bool = False,
) -> Decomposition:
    """Return a top-rank principal update and its exact arithmetic residual."""

    if base.ndim != 2 or trained.ndim != 2 or base.shape != trained.shape:
        raise ValueError("base and trained must be same-shaped rank-2 tensors")
    if not 0.0 < rank_fraction < 1.0:
        raise ValueError("rank_fraction must be in (0, 1)")

    compute_device = torch.device(device)
    delta = trained.to(compute_device, torch.float32) - base.to(compute_device, torch.float32)
    max_rank = min(delta.shape)
    rank = max(1, min(max_rank, math.ceil(rank_fraction * max_rank)))

    if exact_svd:
        u_all, singular_values, vh_all = torch.linalg.svd(delta, full_matrices=False)
        u = u_all[:, :rank]
        singular_values = singular_values[:rank]
        v = vh_all[:rank].T
    else:
        q = min(max_rank, rank + max(0, oversample))
        fork_devices: list[int] = []
        if compute_device.type == "cuda":
            fork_devices = [compute_device.index if compute_device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            u_all, singular_values, v_all = torch.svd_lowrank(delta, q=q, niter=niter)
        order = torch.argsort(singular_values, descending=True)[:rank]
        u = u_all[:, order]
        singular_values = singular_values[order]
        v = v_all[:, order]

    principal = (u * singular_values.unsqueeze(0)) @ v.T
    tail = delta - principal
    reconstructed = principal + tail
    delta_norm_t = torch.linalg.vector_norm(delta)
    principal_norm_t = torch.linalg.vector_norm(principal)
    tail_norm_t = torch.linalg.vector_norm(tail)
    denominator = delta_norm_t.clamp_min(torch.finfo(torch.float32).eps)
    reconstruction_error = torch.linalg.vector_norm(delta - reconstructed) / denominator
    energy_ratio = principal_norm_t.square() / denominator.square()
    return Decomposition(
        principal=principal,
        tail=tail,
        u=u,
        v=v,
        rank=rank,
        delta_norm=delta_norm_t.item(),
        principal_norm=principal_norm_t.item(),
        tail_norm=tail_norm_t.item(),
        principal_energy_ratio=energy_ratio.item(),
        relative_reconstruction_error=reconstruction_error.item(),
    )


def _orthogonalize(matrix: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project a matrix outside a host principal's left and right subspaces."""

    result = matrix - u @ (u.T @ matrix)
    result = result - (result @ v) @ v.T
    return result


def _match_norm(matrix: torch.Tensor, target_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(matrix)
    if target_norm == 0.0:
        return torch.zeros_like(matrix)
    if norm.item() == 0.0:
        raise ValueError("Cannot norm-match a zero control matrix to a nonzero tail")
    return matrix * (target_norm / norm)


def make_random_tail(
    tail: torch.Tensor,
    host_u: torch.Tensor,
    host_v: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Randomize tail coordinates, project off principal, and restore its norm.

    Signed row/column permutations preserve the input singular values before
    projection.  The final projection enforces separation from the host
    principal; norm matching is exact while spectrum matching is approximate.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows = torch.randperm(tail.shape[0], generator=generator, device="cpu").to(tail.device)
    cols = torch.randperm(tail.shape[1], generator=generator, device="cpu").to(tail.device)
    row_sign = (torch.randint(0, 2, (tail.shape[0],), generator=generator) * 2 - 1).to(
        device=tail.device, dtype=tail.dtype
    )
    col_sign = (torch.randint(0, 2, (tail.shape[1],), generator=generator) * 2 - 1).to(
        device=tail.device, dtype=tail.dtype
    )
    randomized = tail.index_select(0, rows).index_select(1, cols)
    randomized = randomized * row_sign[:, None] * col_sign[None, :]
    randomized = _orthogonalize(randomized, host_u, host_v)
    return _match_norm(randomized, torch.linalg.vector_norm(tail).item())


def transplant_tail(donor_tail: torch.Tensor, host: Decomposition) -> torch.Tensor:
    """Move donor directions outside the host principal and match host-tail norm."""

    transplanted = _orthogonalize(donor_tail, host.u, host.v)
    return _match_norm(transplanted, host.tail_norm)


def build_matrix_arms(
    base: torch.Tensor,
    grpo: torch.Tensor,
    sdpo: torch.Tensor,
    *,
    rank_fraction: float,
    device: torch.device | str,
    seed: int,
    oversample: int,
    niter: int,
    exact_svd: bool = False,
) -> tuple[dict[str, torch.Tensor], Decomposition, Decomposition]:
    """Construct all six derived tensors for one selected matrix."""

    compute_device = torch.device(device)
    base_f = base.to(compute_device, torch.float32)
    grpo_dec = decompose_update(
        base,
        grpo,
        rank_fraction=rank_fraction,
        device=compute_device,
        seed=seed,
        oversample=oversample,
        niter=niter,
        exact_svd=exact_svd,
    )
    sdpo_dec = decompose_update(
        base,
        sdpo,
        rank_fraction=rank_fraction,
        device=compute_device,
        seed=seed + 1,
        oversample=oversample,
        niter=niter,
        exact_svd=exact_svd,
    )
    grpo_random = make_random_tail(grpo_dec.tail, grpo_dec.u, grpo_dec.v, seed=seed + 2)
    sdpo_random = make_random_tail(sdpo_dec.tail, sdpo_dec.u, sdpo_dec.v, seed=seed + 3)
    sdpo_to_grpo = transplant_tail(sdpo_dec.tail, grpo_dec)
    grpo_to_sdpo = transplant_tail(grpo_dec.tail, sdpo_dec)

    tensors = {
        "GRPO-no-tail": base_f + grpo_dec.principal,
        "GRPO-random-tail": base_f + grpo_dec.principal + grpo_random,
        "GRPO-principal+SDPO-tail": base_f + grpo_dec.principal + sdpo_to_grpo,
        "SDPO-no-tail": base_f + sdpo_dec.principal,
        "SDPO-random-tail": base_f + sdpo_dec.principal + sdpo_random,
        "SDPO-principal+GRPO-tail": base_f + sdpo_dec.principal + grpo_to_sdpo,
    }
    return tensors, grpo_dec, sdpo_dec


def _prepare_outputs(staging: Path, grpo_path: Path, sdpo_path: Path) -> dict[str, Path]:
    outputs = {label: staging / dirname for label, dirname in DERIVED_ARMS.items()}
    for label, path in outputs.items():
        path.mkdir(parents=True)
        host = grpo_path if label.startswith("GRPO") else sdpo_path
        _copy_auxiliary_files(host, path)
    return outputs


def build_checkpoint_arms(
    *,
    base_model: str,
    grpo_model: str,
    sdpo_model: str,
    output_dir: str,
    rank_fraction: float = 0.1,
    oversample: int = 8,
    niter: int = 2,
    device: str = "cpu",
    seed: int = 0,
    include_suffixes: tuple[str, ...] = DEFAULT_WEIGHT_SUFFIXES,
    include_regex: str | None = None,
    cache_dir: str | None = None,
    exact_svd: bool = False,
) -> dict:
    """Build checkpoint arms atomically and return their manifest."""

    started_at = time.perf_counter()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    base_path = _resolve_model_path(base_model, cache_dir)
    grpo_path = _resolve_model_path(grpo_model, cache_dir)
    sdpo_path = _resolve_model_path(sdpo_model, cache_dir)
    sources = [TensorSource(path) for path in (base_path, grpo_path, sdpo_path)]
    base_source, grpo_source, sdpo_source = sources
    if grpo_source.keys != sdpo_source.keys:
        raise ValueError("GRPO and SDPO checkpoints must contain identical tensor keys")
    if base_source.keys != grpo_source.keys:
        print(
            "Warning: base/trained checkpoint keys differ; selected matrices are validated lazily. "
            "This is normally caused by tied weights being materialized only in exported checkpoints."
        )
    if grpo_source.format != sdpo_source.format:
        raise ValueError("GRPO and SDPO checkpoints must use the same weight format")

    suffixes = tuple(value.strip() for value in include_suffixes if value.strip())
    pattern = re.compile(include_regex) if include_regex else None
    reports: list[MatrixReport] = []

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.tail-", dir=output.parent) as temp:
        staging = Path(temp)
        outputs = _prepare_outputs(staging, grpo_path, sdpo_path)
        for shard_name in grpo_source.shard_names:
            print(f"Tail surgery [{device}] processing {shard_name}", flush=True)
            shard_outputs: dict[str, dict[str, torch.Tensor]] = {label: {} for label in outputs}
            for name in grpo_source.keys_in_shard(shard_name):
                grpo_tensor = grpo_source.get_tensor(name)
                sdpo_tensor = sdpo_source.get_tensor(name)
                selected = name.endswith(suffixes) or (pattern is not None and pattern.search(name) is not None)
                if selected and grpo_tensor.ndim == 2 and grpo_tensor.is_floating_point():
                    base_tensor = base_source.get_tensor(name)
                    tensor_seed = _tensor_seed(seed, name, "decomposition")
                    arms, grpo_dec, sdpo_dec = build_matrix_arms(
                        base_tensor,
                        grpo_tensor,
                        sdpo_tensor,
                        rank_fraction=rank_fraction,
                        device=device,
                        seed=tensor_seed,
                        oversample=oversample,
                        niter=niter,
                        exact_svd=exact_svd,
                    )
                    if max(grpo_dec.relative_reconstruction_error, sdpo_dec.relative_reconstruction_error) >= 1e-5:
                        raise RuntimeError(f"Reconstruction error exceeded 1e-5 for {name}")
                    for label, tensor in arms.items():
                        host_dtype = grpo_tensor.dtype if label.startswith("GRPO") else sdpo_tensor.dtype
                        shard_outputs[label][name] = tensor.to("cpu", host_dtype)
                    reports.append(
                        MatrixReport(
                            name=name,
                            shape=list(grpo_tensor.shape),
                            rank=grpo_dec.rank,
                            grpo_delta_norm=grpo_dec.delta_norm,
                            grpo_principal_norm=grpo_dec.principal_norm,
                            grpo_tail_norm=grpo_dec.tail_norm,
                            grpo_principal_energy_ratio=grpo_dec.principal_energy_ratio,
                            grpo_reconstruction_error=grpo_dec.relative_reconstruction_error,
                            sdpo_delta_norm=sdpo_dec.delta_norm,
                            sdpo_principal_norm=sdpo_dec.principal_norm,
                            sdpo_tail_norm=sdpo_dec.tail_norm,
                            sdpo_principal_energy_ratio=sdpo_dec.principal_energy_ratio,
                            sdpo_reconstruction_error=sdpo_dec.relative_reconstruction_error,
                            grpo_random_tail_norm=torch.linalg.vector_norm(
                                arms["GRPO-random-tail"].float()
                                - arms["GRPO-no-tail"].float()
                            ).item(),
                            sdpo_random_tail_norm=torch.linalg.vector_norm(
                                arms["SDPO-random-tail"].float()
                                - arms["SDPO-no-tail"].float()
                            ).item(),
                            grpo_to_sdpo_tail_norm=torch.linalg.vector_norm(
                                arms["SDPO-principal+GRPO-tail"].float()
                                - arms["SDPO-no-tail"].float()
                            ).item(),
                            sdpo_to_grpo_tail_norm=torch.linalg.vector_norm(
                                arms["GRPO-principal+SDPO-tail"].float()
                                - arms["GRPO-no-tail"].float()
                            ).item(),
                        )
                    )
                    del arms, grpo_dec, sdpo_dec
                else:
                    for label in outputs:
                        host_tensor = grpo_tensor if label.startswith("GRPO") else sdpo_tensor
                        shard_outputs[label][name] = host_tensor

            for label, destination in outputs.items():
                _save_shard(shard_outputs[label], destination / shard_name, grpo_source.format)
            del shard_outputs

        if grpo_source.index_name is not None:
            for destination in outputs.values():
                shutil.copy2(grpo_path / grpo_source.index_name, destination / grpo_source.index_name)

        manifest = {
            "base_model": str(base_path),
            "grpo_model": str(grpo_path),
            "sdpo_model": str(sdpo_path),
            "rank_fraction": rank_fraction,
            "oversample": oversample,
            "niter": niter,
            "exact_svd": exact_svd,
            "device": device,
            "seed": seed,
            "include_suffixes": list(suffixes),
            "include_regex": include_regex,
            "num_selected_matrices": len(reports),
            "elapsed_seconds": time.perf_counter() - started_at,
            "arms": {
                "GRPO-full": str(grpo_path),
                "SDPO-full": str(sdpo_path),
                **{label: str(output / dirname) for label, dirname in DERIVED_ARMS.items()},
            },
            "matrices": [asdict(item) for item in reports],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        Path(temp).rename(output)

    print(f"Tail surgery complete: {output}")
    print(f"Manifest: {output / 'manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--grpo-model", required=True)
    parser.add_argument("--sdpo-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank-fraction", type=float, default=0.1)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-suffixes", default=",".join(DEFAULT_WEIGHT_SUFFIXES))
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--exact-svd", action="store_true", help="Use exact SVD; intended for tests/small models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_checkpoint_arms(
        base_model=args.base_model,
        grpo_model=args.grpo_model,
        sdpo_model=args.sdpo_model,
        output_dir=args.output_dir,
        rank_fraction=args.rank_fraction,
        oversample=args.oversample,
        niter=args.niter,
        device=args.device,
        seed=args.seed,
        include_suffixes=tuple(args.include_suffixes.split(",")),
        include_regex=args.include_regex,
        cache_dir=args.cache_dir,
        exact_svd=args.exact_svd,
    )


if __name__ == "__main__":
    main()
