# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import flashinfer_b12x_moe
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_quant_config,
)


class _FakeB12xMoEWrapper:
    instances: list["_FakeB12xMoEWrapper"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)


def _install_fake_flashinfer(monkeypatch) -> None:
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.B12xMoEWrapper = _FakeB12xMoEWrapper
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)


def _install_fake_w4a16_prepare(monkeypatch):
    prepared = object()
    calls = []
    module_name = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare"
    for parent_name in (
        "flashinfer",
        "flashinfer.fused_moe",
        "flashinfer.fused_moe.cute_dsl",
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x",
    ):
        module = ModuleType(parent_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, parent_name, module)

    prepare_module = ModuleType(module_name)

    def prepare_w4a16_packed_weights(*args, **kwargs):
        calls.append((args, kwargs))
        return prepared

    prepare_module.prepare_w4a16_packed_weights = prepare_w4a16_packed_weights
    monkeypatch.setitem(sys.modules, module_name, prepare_module)
    return prepared, calls


def _make_experts(
    *,
    w4a16: bool,
    num_experts: int = 2,
    experts_per_token: int = 2,
    max_num_tokens: int = 64,
) -> FlashInferB12xExperts:
    ones = torch.ones(num_experts, dtype=torch.float32)
    if w4a16:
        quant_config = nvfp4_w4a16_moe_quant_config(
            g1_alphas=ones,
            g2_alphas=ones,
            w1_scale=torch.ones(1),
            w2_scale=torch.ones(1),
            source_format="compressed_tensors",
        )
    else:
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=ones,
            g2_alphas=ones,
            a1_gscale=ones,
            a2_gscale=ones,
            w1_scale=torch.ones(1),
            w2_scale=torch.ones(1),
            source_format="modelopt",
        )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max_num_tokens,
    )
    return FlashInferB12xExperts(moe_config=moe_config, quant_config=quant_config)


def _make_layer() -> torch.nn.Module:
    num_experts = 2
    hidden_dim = 128
    intermediate_size = 256
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.empty(
            num_experts,
            2 * intermediate_size,
            hidden_dim // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.empty(
            num_experts,
            hidden_dim,
            intermediate_size // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(num_experts, 2 * intermediate_size, hidden_dim // 16),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones(num_experts, hidden_dim, intermediate_size // 16),
        requires_grad=False,
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        torch.ones(num_experts), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        torch.ones(num_experts), requires_grad=False
    )
    return layer


def _reset_cache() -> None:
    _FakeB12xMoEWrapper.instances.clear()
    flashinfer_b12x_moe._B12X_WRAPPERS.clear()


def test_w4a16_b12x_wrapper_is_shared_for_matching_shape(monkeypatch) -> None:
    _reset_cache()
    _install_fake_flashinfer(monkeypatch)
    monkeypatch.setattr(flashinfer_b12x_moe, "dbo_current_ubatch_id", lambda: 0)

    experts_0 = _make_experts(w4a16=True)
    experts_1 = _make_experts(w4a16=True)

    wrapper_0 = experts_0._ensure_wrapper(torch.device("cuda:0"))
    wrapper_1 = experts_1._ensure_wrapper(torch.device("cuda:0"))

    assert wrapper_0 is wrapper_1
    assert len(_FakeB12xMoEWrapper.instances) == 1
    assert wrapper_0.kwargs["use_cuda_graph"] is True
    assert wrapper_0.kwargs["activation_precision"] == "bf16"
    assert wrapper_0.kwargs["max_num_tokens"] == 64


def test_w4a16_b12x_wrapper_is_separate_per_ubatch(monkeypatch) -> None:
    _reset_cache()
    _install_fake_flashinfer(monkeypatch)
    ubatch_id = 0
    monkeypatch.setattr(flashinfer_b12x_moe, "dbo_current_ubatch_id", lambda: ubatch_id)

    experts = _make_experts(w4a16=True)

    wrapper_0 = experts._ensure_wrapper(torch.device("cuda:0"))
    ubatch_id = 1
    wrapper_1 = experts._ensure_wrapper(torch.device("cuda:0"))

    assert wrapper_0 is not wrapper_1
    assert len(_FakeB12xMoEWrapper.instances) == 2


def test_fp4_b12x_wrapper_is_shared_for_matching_shape(monkeypatch) -> None:
    _reset_cache()
    _install_fake_flashinfer(monkeypatch)

    experts_0 = _make_experts(w4a16=False)
    experts_1 = _make_experts(w4a16=False)

    wrapper_0 = experts_0._ensure_wrapper(torch.device("cuda:0"))
    wrapper_1 = experts_1._ensure_wrapper(torch.device("cuda:0"))

    assert wrapper_0 is wrapper_1
    assert len(_FakeB12xMoEWrapper.instances) == 1
    assert wrapper_0.kwargs["activation_precision"] == "fp4"


def test_b12x_nvfp4_oracle_uses_w4a16_quant_config() -> None:
    ones = torch.ones(2, dtype=torch.float32)

    quant_config = make_nvfp4_moe_quant_config(
        NvFp4MoeBackend.FLASHINFER_B12X,
        w13_scale=ones,
        w2_scale=ones,
        w13_scale_2=ones,
        w2_scale_2=ones,
        a13_scale=ones,
        a2_scale=ones,
        source_format="compressed_tensors",
    )

    assert quant_config.use_nvfp4_w4a16
    assert quant_config.quant_dtype is None
    assert quant_config.weight_quant_dtype == "nvfp4"
    assert quant_config.a1_gscale is None
    assert quant_config.a2_gscale is None
    assert quant_config.source_format == "compressed_tensors"

    experts = _make_experts(w4a16=True)
    assert experts.activation_precision == "bf16"


def test_w4a16_process_weights_prepares_and_releases_original_layout(
    monkeypatch,
) -> None:
    prepared, calls = _install_fake_w4a16_prepare(monkeypatch)
    layer = _make_layer()
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=layer.w13_weight_scale_2,
        g2_alphas=layer.w2_weight_scale_2,
        a1_gscale=None,
        a2_gscale=None,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        source_format="compressed_tensors",
    )
    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        num_local_experts=2,
        num_logical_experts=2,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=64,
    )
    experts = FlashInferB12xExperts(moe_config=moe_config, quant_config=quant_config)

    experts.process_weights_after_loading(layer)

    assert experts._w4a16_prepared_weights is prepared
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].shape == (2, 512, 64)
    assert args[3].shape == (2, 128, 128)
    assert kwargs["activation"] == "silu"
    assert kwargs["params_dtype"] == torch.bfloat16
    assert kwargs["source_format"] == "compressed_tensors"

    assert layer.w13_weight.shape == (0, 512, 64)
    assert layer.w2_weight.shape == (0, 128, 128)
    assert layer.w13_weight_scale.numel() == 0
    assert layer.w2_weight_scale.numel() == 0
    assert experts.quant_config.w1_scale is None
    assert experts.quant_config.w2_scale is None
    assert experts._w4a16_empty_float32 is not None
    assert experts._w4a16_empty_float32.numel() == 0


def test_w4a16_route_pack_chunk_size_keeps_qwen_shape_under_triton_limit() -> None:
    experts = _make_experts(
        w4a16=True,
        num_experts=256,
        experts_per_token=8,
        max_num_tokens=32768,
    )

    assert experts._w4a16_route_pack_tensor_numel(16384) == 1 << 20
    assert experts._w4a16_route_pack_tensor_numel(32768) == 1 << 21

    chunk_size = experts._max_w4a16_chunk_tokens()

    assert chunk_size < 16384
    assert experts._w4a16_route_pack_tensor_numel(chunk_size) <= 1 << 19
    assert experts._w4a16_route_pack_tensor_numel(chunk_size + 1) > 1 << 19


def test_w4a16_apply_chunks_large_batches(monkeypatch) -> None:
    experts = _make_experts(w4a16=True)
    wrapper = object()
    calls: list[int] = []

    experts._w4a16_prepared_weights = object()
    monkeypatch.setattr(experts, "_ensure_wrapper", lambda device: wrapper)
    monkeypatch.setattr(experts, "_max_w4a16_chunk_tokens", lambda: 3)

    def fake_run_w4a16_prepared(
        wrapper_arg,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    ):
        assert wrapper_arg is wrapper
        assert topk_weights.size(0) == hidden_states.size(0)
        assert topk_ids.size(0) == hidden_states.size(0)
        calls.append(hidden_states.size(0))
        return torch.full_like(hidden_states, len(calls))

    monkeypatch.setattr(
        experts,
        "_run_w4a16_prepared",
        fake_run_w4a16_prepared,
    )

    hidden_states = torch.zeros(5, experts.hidden_dim, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    topk_weights = torch.ones(5, experts.topk, dtype=torch.float32)
    topk_ids = torch.zeros(5, experts.topk, dtype=torch.int32)
    w1 = torch.empty(0)
    w2 = torch.empty(0)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=experts.global_num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=None,
    )

    assert calls == [3, 2]
    assert torch.all(output[:3] == 1)
    assert torch.all(output[3:] == 2)
