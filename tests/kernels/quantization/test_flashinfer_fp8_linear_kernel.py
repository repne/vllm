# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)


def test_flashinfer_fp8_linear_kernel_restores_output_shape(monkeypatch) -> None:
    """Checks that the FlashInfer kernel restores the caller-provided output shape."""
    expected_shape = (2, 4, 32)

    def fake_flashinfer_scaled_fp8_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del scale_a, scale_b, bias
        return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype)

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.scaled_mm.flashinfer.flashinfer_scaled_fp8_mm",
        fake_flashinfer_scaled_fp8_mm,
    )

    out = FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm(
        object(),
        A=torch.empty((8, 16), dtype=torch.float8_e4m3fn),
        B=torch.empty((16, 32), dtype=torch.float8_e4m3fn),
        out_dtype=torch.bfloat16,
        As=torch.ones(1, dtype=torch.float32),
        Bs=torch.ones(1, dtype=torch.float32),
        bias=None,
        output_shape=list(expected_shape),
    )

    assert out.shape == expected_shape
