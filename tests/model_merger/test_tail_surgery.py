import json
from pathlib import Path

import pytest
import torch

from experiments.causal.tail_surgery import (
    DERIVED_ARMS,
    build_checkpoint_arms,
    build_matrix_arms,
    decompose_update,
)


def test_decompose_update_reconstructs_and_uses_fractional_rank():
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(10, 8, generator=generator)
    trained = base + torch.randn(10, 8, generator=generator)

    result = decompose_update(
        base,
        trained,
        rank_fraction=0.25,
        device="cpu",
        seed=3,
        oversample=2,
        niter=1,
        exact_svd=True,
    )

    assert result.rank == 2
    torch.testing.assert_close(result.principal + result.tail, trained - base)
    assert result.relative_reconstruction_error < 1e-7


def test_matrix_arms_remove_randomize_and_transplant_tail():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(9, 7, generator=generator)
    grpo = base + torch.randn(9, 7, generator=generator)
    sdpo = base + torch.randn(9, 7, generator=generator)

    arms, grpo_dec, sdpo_dec = build_matrix_arms(
        base,
        grpo,
        sdpo,
        rank_fraction=0.3,
        device="cpu",
        seed=5,
        oversample=2,
        niter=1,
        exact_svd=True,
    )

    torch.testing.assert_close(arms["GRPO-no-tail"] - base, grpo_dec.principal)
    torch.testing.assert_close(arms["SDPO-no-tail"] - base, sdpo_dec.principal)
    assert torch.linalg.vector_norm(
        arms["GRPO-random-tail"] - arms["GRPO-no-tail"]
    ).item() == pytest.approx(grpo_dec.tail_norm, rel=1e-5)
    assert torch.linalg.vector_norm(
        arms["SDPO-principal+GRPO-tail"] - arms["SDPO-no-tail"]
    ).item() == pytest.approx(sdpo_dec.tail_norm, rel=1e-5)
    assert not torch.allclose(arms["GRPO-random-tail"], grpo)


def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    path.mkdir()
    safetensors.save_file({key: value.contiguous() for key, value in tensors.items()}, path / "model.safetensors")
    (path / "config.json").write_text('{"architectures": ["ToyModel"]}\n')


def test_build_checkpoint_arms_preserves_nonselected_host_parameters(tmp_path):
    generator = torch.Generator().manual_seed(17)
    base_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 6, generator=generator),
        "model.layers.0.input_layernorm.weight": torch.randn(6, generator=generator),
    }
    grpo_tensors = {key: value + 0.1 for key, value in base_tensors.items()}
    sdpo_tensors = {key: value - 0.2 for key, value in base_tensors.items()}
    base = tmp_path / "base"
    grpo = tmp_path / "grpo"
    sdpo = tmp_path / "sdpo"
    _write_checkpoint(base, base_tensors)
    _write_checkpoint(grpo, grpo_tensors)
    _write_checkpoint(sdpo, sdpo_tensors)

    output = tmp_path / "arms"
    manifest = build_checkpoint_arms(
        base_model=str(base),
        grpo_model=str(grpo),
        sdpo_model=str(sdpo),
        output_dir=str(output),
        rank_fraction=0.25,
        device="cpu",
        seed=19,
        exact_svd=True,
    )

    assert set(manifest["arms"]) == {"GRPO-full", "SDPO-full", *DERIVED_ARMS}
    assert manifest["num_selected_matrices"] == 1
    assert json.loads((output / "manifest.json").read_text())["rank_fraction"] == 0.25

    from safetensors import safe_open

    with safe_open(output / "grpo_no_tail" / "model.safetensors", framework="pt", device="cpu") as handle:
        torch.testing.assert_close(
            handle.get_tensor("model.layers.0.input_layernorm.weight"),
            grpo_tensors["model.layers.0.input_layernorm.weight"],
        )
    with safe_open(output / "sdpo_no_tail" / "model.safetensors", framework="pt", device="cpu") as handle:
        torch.testing.assert_close(
            handle.get_tensor("model.layers.0.input_layernorm.weight"),
            sdpo_tensors["model.layers.0.input_layernorm.weight"],
        )
