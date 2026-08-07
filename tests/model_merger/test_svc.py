import torch

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
