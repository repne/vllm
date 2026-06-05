# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from weakref import WeakValueDictionary

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
    kNvfp4StaticGroupScale,
    kStaticTensorScale,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


@dataclass(frozen=True)
class _B12xWrapperKey:
    num_experts: int
    topk: int
    hidden_dim: int
    intermediate_size: int
    max_num_tokens: int
    num_local_experts: int
    activation: str
    activation_precision: str
    source_format: str
    device: str
    ubatch_id: int


_B12X_WRAPPERS: WeakValueDictionary[_B12xWrapperKey, object] = WeakValueDictionary()
_TRITON_MAX_TENSOR_NUMEL = 1 << 20
_W4A16_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)
_W4A16_ROUTE_PACK_TARGET_FILL = 0.9


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    Weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``
    and cached as ``w1_sf_mma`` / ``w2_sf_mma``.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.weight_quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]

        self.activation_precision = (
            "fp4" if quant_config.a1_gscale is not None else "bf16"
        )

        # source_format selects how the FlashInfer kernel interprets the
        # FP4 byte payload. Set by the parent quant method on
        # FusedMoEQuantConfig; fall back to "modelopt" if upstream hasn't
        # been wired through yet.
        self.source_format = quant_config.source_format or "modelopt"

        # Lazily created on first apply() call. The wrapper owns large
        # shape-scoped scratch/output buffers, so it is shared across layers
        # with matching geometry.
        self._wrappers: dict[_B12xWrapperKey, object] = {}
        # W4A16 B12x uses a FlashInfer-specific packed layout. Once populated,
        # apply() uses this representation and the original checkpoint-layout
        # parameters can be released.
        self._w4a16_prepared_weights: object | None = None
        self._w4a16_empty_float32: torch.Tensor | None = None
        # Populated in process_weights_after_loading.
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:
        #   block_scale = max_abs * w_gs / fp4_max,  g1_alphas = 1/w_gs
        # The SM12x kernel treats w1_alpha (= g1_alphas) as a per-expert weight
        # dequant multiplier separate from input_gs (activation scale).  We bake
        # w_gs into the block scales so that w1_alpha = 1.0 and the kernel sees
        # the simpler form:
        #   block_scale = max_abs / fp4_max,  w1_alpha = 1.0
        # The FP4-packed values and dequantised results are identical in both
        # representations.  We set scale_2 = 1.0 to signal that the bake-in is
        # already done.
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w13_weight_scale.dtype)
        layer.w13_weight_scale_2.data.fill_(1.0)

        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w2_weight_scale.dtype)
        layer.w2_weight_scale_2.data.fill_(1.0)

        # The SM12x kernel uses dynamic per-block quantization for FC2 input
        # activations (the SwiGLU output before the down projection).  The
        # calibrated a2_gscale from the modelopt checkpoint (~tens to hundreds)
        # is intended for static-quantisation backends (TRTLLM/CUTLASS) and
        # causes every intermediate activation to saturate at max FP4 when
        # multiplied by values that large.  Force to 1.0 so the kernel uses
        # its own per-block dynamic scale.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)
            self._fc2_input_scale = self.a2_gscale
        else:
            # W4A16 NVFP4 checkpoints have no calibrated a2_gscale; b12x
            # performs dynamic per-block FC2-input quantization, so a uniform
            # 1.0 scale per expert is equivalent to the bake-in above for
            # static-quant checkpoints. Allocate once here so apply() stays
            # alloc-free.
            self._fc2_input_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight_scale.device,
                dtype=torch.float32,
            )

        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.g1_alphas is not None and self.g2_alphas is not None
        if self.activation_precision == "bf16":
            if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
                self._prepare_w4a16_weights(
                    layer.w13_weight,
                    layer.w2_weight,
                )
                if isinstance(layer, torch.nn.Module):
                    self._release_w4a16_original_layout(layer)
        else:
            self._prepare_nvfp4_scale_views()

    def _prepare_nvfp4_scale_views(self) -> None:
        # Precompute MMA-layout views of the (now-rewritten) weight scale
        # factors once here rather than recomputing on every forward pass.
        # Converts swizzled 3D scale factors [E, M, K_sf] to the 6D MMA
        # layout expected by the SM12x kernel's _get_weight_views().
        assert self.w1_scale is not None and self.w2_scale is not None
        sf_vec_size = 16
        e_w1, m_w1, k_sf_w1 = self.w1_scale.shape
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(e_w1 * m_w1, k_sf_w1),
            m=m_w1,
            k=k_sf_w1 * sf_vec_size,
            num_groups=e_w1,
            sf_vec_size=sf_vec_size,
        )
        e_w2, m_w2, k_sf_w2 = self.w2_scale.shape
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(e_w2 * m_w2, k_sf_w2),
            m=m_w2,
            k=k_sf_w2 * sf_vec_size,
            num_groups=e_w2,
            sf_vec_size=sf_vec_size,
        )

    def _prepare_w4a16_weights(
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> None:
        if self._w4a16_prepared_weights is not None:
            return

        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
            prepare_w4a16_packed_weights,
        )

        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.g1_alphas is not None and self.g2_alphas is not None
        self._w4a16_prepared_weights = prepare_w4a16_packed_weights(
            w13_weight,
            self.w1_scale,
            self.g1_alphas,
            w2_weight,
            self.w2_scale,
            self.g2_alphas,
            activation=self._activation_str,
            params_dtype=self.out_dtype,
            source_format=self.source_format,
        )
        self._w4a16_empty_float32 = torch.empty(
            0,
            dtype=torch.float32,
            device=w13_weight.device,
        )

    def _release_w4a16_original_layout(self, layer: torch.nn.Module) -> None:
        # The prepared layout owns the runtime weight tensors. Drop the original
        # checkpoint-layout payloads so FlashInfer B12x W4A16 does not keep a
        # second copy of every MoE layer's weights and block scales.
        w13_rows = (2 if self._activation_str == "silu" else 1) * (
            self.intermediate_size_per_partition
        )
        replacements = {
            "w13_weight": (0, w13_rows, self.hidden_dim // 2),
            "w2_weight": (
                0,
                self.hidden_dim,
                self.intermediate_size_per_partition // 2,
            ),
            "w13_weight_scale": (0,),
            "w2_weight_scale": (0,),
        }
        for name, shape in replacements.items():
            old_param = getattr(layer, name)
            replace_parameter(
                layer,
                name,
                torch.empty(shape, dtype=old_param.dtype, device=old_param.device),
            )
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Original W4A4 NVFP4 (modelopt format).
        if (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic):
            return True

        # W4A16 NVFP4 compressed-tensors `nvfp4-pack-quantized`
        return (
            weight_key is not None
            and weight_key.dtype == torch.uint8
            and weight_key.scale == kNvfp4StaticGroupScale
            and weight_key.scale2 == kStaticTensorScale
            and weight_key.symmetric
            and activation_key is None
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    def _make_wrapper(self, device: torch.device | str) -> object:
        from flashinfer.fused_moe import B12xMoEWrapper

        return B12xMoEWrapper(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
            activation_precision=self.activation_precision,
            source_format=self.source_format,
            device=str(device),
        )

    def _wrapper_key(self, device: torch.device | str) -> _B12xWrapperKey:
        return _B12xWrapperKey(
            num_experts=self.global_num_experts,
            topk=self.topk,
            hidden_dim=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
            activation_precision=self.activation_precision,
            source_format=self.source_format,
            device=str(device),
            ubatch_id=dbo_current_ubatch_id(),
        )

    def _ensure_wrapper(self, device: torch.device | str) -> object:
        """Lazily create B12xMoEWrapper on first use."""
        key = self._wrapper_key(device)
        wrapper = self._wrappers.get(key)
        if wrapper is not None:
            return wrapper

        wrapper = _B12X_WRAPPERS.get(key)
        if wrapper is None:
            wrapper = self._make_wrapper(device)
            _B12X_WRAPPERS[key] = wrapper
        self._wrappers[key] = wrapper
        return wrapper

    def _run_w4a16_prepared(
        self,
        wrapper: object,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        assert self._w4a16_prepared_weights is not None
        assert self._w4a16_empty_float32 is not None
        num_tokens = topk_ids.size(0)
        moe_output = wrapper._moe_output[:num_tokens]  # type: ignore[attr-defined]
        return launch_sm120_moe(
            a=hidden_states,
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights,
            w1_weight=w1,
            w1_weight_sf=w1,
            w1_alpha=self._w4a16_empty_float32,
            fc2_input_scale=None,
            w2_weight=w2,
            w2_weight_sf=w2,
            w2_alpha=self._w4a16_empty_float32,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.num_local_experts,
            scatter_output=moe_output,
            activation=self._activation_str,
            quant_mode="w4a16",
            source_format=self.source_format,
            _workspace=wrapper._static_workspace,  # type: ignore[attr-defined]
            _prepared_weights=self._w4a16_prepared_weights,
        )

    @staticmethod
    def _next_power_of_2(x: int) -> int:
        return 1 << (max(int(x), 1) - 1).bit_length()

    @staticmethod
    def _select_w4a16_route_block_size_m(
        num_tokens: int,
        topk: int,
        num_experts: int,
    ) -> int:
        avg_routes_per_expert = (int(num_tokens) * int(topk)) / int(num_experts)
        for routed_size in _W4A16_ALLOWED_ROUTED_SIZES:
            if avg_routes_per_expert < _W4A16_ROUTE_PACK_TARGET_FILL * routed_size:
                return routed_size
        return _W4A16_ALLOWED_ROUTED_SIZES[-1]

    @staticmethod
    def _max_w4a16_packed_route_slots(
        numel: int,
        block_size: int,
        num_experts: int,
    ) -> int:
        max_packed_routes = int(numel) + int(num_experts) * (int(block_size) - 1)
        if int(numel) < int(num_experts):
            max_packed_routes = min(
                int(numel) * int(block_size),
                max_packed_routes,
            )
        return max(max_packed_routes, 1)

    def _w4a16_route_pack_tensor_numel(self, num_tokens: int) -> int:
        # FlashInfer's W4A16 route-packing prefix kernel materializes
        # BLOCK_ROUTE_INIT and [BLOCK_E, BLOCK_M] Triton tensors. Triton
        # rejects tensors above 1,048,576 elements, so keep large prefill
        # batches below those shapes.
        route_num_experts = (
            self.global_num_experts
            if self.num_local_experts != self.global_num_experts
            else self.num_local_experts
        )
        block_size = self._select_w4a16_route_block_size_m(
            num_tokens,
            self.topk,
            route_num_experts,
        )
        max_packed_routes = self._max_w4a16_packed_route_slots(
            int(num_tokens) * self.topk,
            block_size,
            route_num_experts,
        )
        max_route_blocks = (max_packed_routes + block_size - 1) // block_size
        block_route_init = self._next_power_of_2(max_packed_routes)
        block_e = self._next_power_of_2(route_num_experts)
        block_m = self._next_power_of_2(max_route_blocks)
        return max(block_route_init, block_e * block_m)

    def _max_w4a16_chunk_tokens(self) -> int:
        override = os.environ.get("VLLM_FLASHINFER_B12X_W4A16_CHUNK_SIZE")
        if override:
            return max(1, int(override))

        limit = _TRITON_MAX_TENSOR_NUMEL // 2
        if self._w4a16_route_pack_tensor_numel(self.max_num_tokens) <= limit:
            return self.max_num_tokens

        low = 1
        high = self.max_num_tokens
        while low < high:
            mid = (low + high + 1) // 2
            if self._w4a16_route_pack_tensor_numel(mid) <= limit:
                low = mid
            else:
                high = mid - 1
        return low

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        wrapper = self._ensure_wrapper(hidden_states.device)
        if self.activation_precision == "bf16":
            self._prepare_w4a16_weights(w1, w2)
            chunk_size = self._max_w4a16_chunk_tokens()
            for start in range(0, hidden_states.size(0), chunk_size):
                end = min(start + chunk_size, hidden_states.size(0))
                result = self._run_w4a16_prepared(
                    wrapper,
                    hidden_states[start:end],
                    w1,
                    w2,
                    topk_weights[start:end],
                    topk_ids[start:end],
                )
                output[start:end].copy_(result)
            return

        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for FlashInferB12xExperts"
        )
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        result = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
        )
        output.copy_(result)
