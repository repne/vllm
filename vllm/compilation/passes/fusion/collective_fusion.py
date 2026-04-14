# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from dataclasses import dataclass

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.fx.experimental.symbolic_shapes import statically_known_true

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .sequence_parallelism import (
    get_effective_sp_min_token_num,
    is_sp_applicable_for_range,
)

FP8_DTYPE = current_platform.fp8_dtype()
FLASHINFER_BMM_FP8_MIN_M = 64

logger = init_logger(__name__)


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(
                input,
                mat2=mat2,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            return torch.ops.aten._scaled_mm.default(
                all_gather,
                mat2=weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )
            return cutlass_scaled_mm[1]

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


FLASHINFER_BMM_FP8_DTYPE = torch.float8_e4m3fn
VIEW_LIKE_OPS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
)
LAYOUT_PRESERVING_OPS = (
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
)


def _get_node_arg(node: fx.Node, name: str, index: int) -> object:
    return node.kwargs.get(name, node.args[index] if len(node.args) > index else None)


def _is_view_like(node: fx.Node) -> bool:
    return any(is_func(node, op) for op in VIEW_LIKE_OPS)


def _is_passthrough(node: fx.Node) -> bool:
    return _is_view_like(node) or any(is_func(node, op) for op in LAYOUT_PRESERVING_OPS)


def _strip_view_like(node: fx.Node) -> fx.Node:
    while _is_view_like(node):
        parent = node.args[0]
        if not isinstance(parent, fx.Node):
            break
        node = parent
    return node


@dataclass(frozen=True)
class _UserPathState:
    node: fx.Node
    saw_slice_scatter: bool = False


def _walk_user_paths(
    start_nodes: list[fx.Node],
    *,
    track_slice_scatter: bool = False,
) -> list[_UserPathState]:
    worklist = [_UserPathState(node=user) for user in start_nodes]
    visited: set[_UserPathState] = set()
    reachable: list[_UserPathState] = []

    while worklist:
        state = worklist.pop()
        if state in visited:
            continue
        visited.add(state)
        reachable.append(state)

        if _is_passthrough(state.node):
            worklist.extend(
                _UserPathState(
                    node=child,
                    saw_slice_scatter=state.saw_slice_scatter,
                )
                for child in state.node.users
            )
            continue

        if track_slice_scatter and is_func(
            state.node, torch.ops.aten.slice_scatter.default
        ):
            worklist.extend(
                _UserPathState(node=child, saw_slice_scatter=True)
                for child in state.node.users
            )

    return reachable


def _node_shape(node: fx.Node) -> list[object] | None:
    val = node.meta.get("val")
    if hasattr(val, "shape"):
        return list(val.shape)
    return None


def _node_first_dim(node: fx.Node) -> object | None:
    shape = _node_shape(node)
    if shape:
        return shape[0]
    return None


def _node_ndim(node: fx.Node) -> int | None:
    shape = _node_shape(node)
    if shape is None:
        return None
    return len(shape)


def _dim_is_statically_lt(dim: int | torch.SymInt, threshold: int) -> bool:
    if isinstance(dim, int):
        return dim < threshold
    try:
        return bool(statically_known_true(dim < threshold))
    except Exception:
        return False


def _passes_min_m(node: fx.Node) -> bool:
    gemm_m = _node_first_dim(node)
    if gemm_m is None or not isinstance(gemm_m, int | torch.SymInt):
        return True
    return not _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)


def _passes_min_m_after_reduce_scatter(node: fx.Node, world_size: int) -> bool:
    gemm_m = _node_first_dim(node)
    if gemm_m is None or not isinstance(gemm_m, int | torch.SymInt):
        return True
    return not _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M * world_size)


def _copy_replacement_meta(src: fx.Node, dst: fx.Node) -> None:
    dst.meta = {
        key: value for key, value in src.meta.items() if key != "eager_input_vals"
    }


def _arg_numel_is_one(arg: object) -> bool:
    if isinstance(arg, torch.Tensor):
        return arg.numel() == 1
    if isinstance(arg, fx.Node):
        val = arg.meta.get("val")
        if hasattr(val, "numel"):
            return val.numel() == 1
    return False


def _unwrap_bmm_fp8_arg_to_2d(arg: object) -> fx.Node | None:
    if not isinstance(arg, fx.Node):
        return None

    node = _strip_view_like(arg)
    if is_func(node, torch.ops.aten.unsqueeze.default):
        dim = _get_node_arg(node, "dim", 1)
        if dim != 0:
            return None
        src = _get_node_arg(node, "self", 0)
        if not isinstance(src, fx.Node):
            return None
        src = _strip_view_like(src)
        ndim = _node_ndim(src)
        if ndim is not None and ndim != 2:
            return None
        return src

    ndim = _node_ndim(node)
    if ndim is not None and ndim != 2:
        return None
    return node


@dataclass
class _CollectiveOp:
    input_node: fx.Node
    dim: object
    world_size: object
    group_name: object


@dataclass
class _BmmFp8Op:
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: object
    backend: object


def _parse_collective_op(
    node: fx.Node,
    op,
) -> _CollectiveOp | None:
    if not is_func(node, op):
        return None

    input_node = _get_node_arg(node, "tensor", 0)
    dim = _get_node_arg(node, "dim", 1)
    world_size = _get_node_arg(node, "world_size", 2)
    group_name = _get_node_arg(node, "group_name", 3)
    if not isinstance(input_node, fx.Node):
        return None
    return _CollectiveOp(
        input_node=input_node,
        dim=dim,
        world_size=world_size,
        group_name=group_name,
    )


def _parse_all_gather(
    node: fx.Node,
) -> _CollectiveOp | None:
    return _parse_collective_op(node, torch.ops.vllm.all_gather.default)


def _parse_collective_group_name(collective: _CollectiveOp) -> str | None:
    if (
        collective.dim != 0
        or not isinstance(collective.world_size, int)
        or not isinstance(collective.group_name, str)
    ):
        return None
    return collective.group_name


def _parse_bmm_fp8(
    node: fx.Node,
) -> _BmmFp8Op | None:
    if not is_func(node, torch.ops.vllm.bmm_fp8.default):
        return None

    a = _get_node_arg(node, "A", 0)
    b = _get_node_arg(node, "B", 1)
    a_scale = _get_node_arg(node, "A_scale", 2)
    b_scale = _get_node_arg(node, "B_scale", 3)
    out_dtype = _get_node_arg(node, "dtype", 4)
    backend = _get_node_arg(node, "backend", 5)

    a_2d = _unwrap_bmm_fp8_arg_to_2d(a)
    b_2d = _unwrap_bmm_fp8_arg_to_2d(b)
    if a_2d is None or b_2d is None:
        return None
    return _BmmFp8Op(
        a_2d=a_2d,
        b_2d=b_2d,
        a_scale=a_scale,
        b_scale=b_scale,
        out_dtype=out_dtype,
        backend=backend,
    )


def _is_supported_flashinfer_bmm_fp8(parsed: _BmmFp8Op | None) -> bool:
    return (
        parsed is not None
        and parsed.backend == "auto"
        and _arg_numel_is_one(parsed.a_scale)
        and _arg_numel_is_one(parsed.b_scale)
    )


def _flashinfer_bmm_fp8_extra_check(match: pm.Match) -> bool:
    return _is_supported_flashinfer_bmm_fp8(_get_match_bmm_fp8(match))


def _find_match_node(match: pm.Match, op) -> fx.Node | None:
    for node in match.nodes:
        if is_func(node, op):
            return node
    return None


def _get_match_bmm_fp8(match: pm.Match) -> _BmmFp8Op | None:
    node = _find_match_node(match, torch.ops.vllm.bmm_fp8.default)
    if node is None:
        return None
    return _parse_bmm_fp8(node)


def _is_qkv_split(node: fx.Node) -> bool:
    if not is_func(node, torch.ops.aten.split_with_sizes.default):
        return False

    split_sizes = _get_node_arg(node, "split_sizes", 1)
    dim = _get_node_arg(node, "dim", 2)
    return (
        isinstance(split_sizes, (list, tuple))
        and len(split_sizes) == 3
        and dim in (-1, 1)
    )


def _classify_qkv_branch(node: fx.Node) -> str | None:
    if any(_is_qkv_split(state.node) for state in _walk_user_paths(list(node.users))):
        return "direct"
    if any(
        state.saw_slice_scatter and _is_qkv_split(state.node)
        for state in _walk_user_paths(list(node.users), track_slice_scatter=True)
    ):
        return "rotary"
    return None


class _FlashInferCollectivePatternBase(BasePattern):
    def empty_fp8(self, *shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=FLASHINFER_BMM_FP8_DTYPE, device=self.device)

    def empty_fp32_scalar(self) -> torch.Tensor:
        return torch.empty([], dtype=torch.float32, device=self.device)

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_fp8(128, 16),
            self.empty_fp8(16, 16),
            self.empty_fp32_scalar(),
            self.empty_fp32_scalar(),
        ]


class FlashInferBMMFP8ReduceScatterPattern(_FlashInferCollectivePatternBase):
    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            a = torch.ops.aten.unsqueeze.default(a_2d, 0)
            b = torch.ops.aten.unsqueeze.default(b_2d, 0)
            bmm = torch.ops.vllm.bmm_fp8.default(
                a, b, a_scale, b_scale, self.dtype, "auto"
            )
            output = torch.ops.aten.view.default(bmm, [a_2d.shape[0], b_2d.shape[1]])
            return torch.ops.vllm.reduce_scatter.default(
                output, 0, self.tp_size, self.tp.unique_name
            )

        def replacement(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            output_shape = [a_2d.shape[0], b_2d.shape[1]]
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default(
                a_2d,
                b_2d,
                a_scale,
                b_scale,
                "sum",
                0,
                0,
                self.tp.unique_name,
                output_shape,
                self.dtype,
            )

        def extra_check(match: pm.Match) -> bool:
            parsed = _get_match_bmm_fp8(match)
            if parsed is None:
                return False
            return _flashinfer_bmm_fp8_extra_check(match) and (
                _passes_min_m_after_reduce_scatter(parsed.a_2d, self.tp_size)
            )

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=extra_check,
        )


class FlashInferAllGatherBMMFP8Pattern(_FlashInferCollectivePatternBase):
    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gather.default(
                a_shard_2d, 0, self.tp_size, self.tp.unique_name
            )
            a = torch.ops.aten.unsqueeze.default(gathered, 0)
            b = torch.ops.aten.unsqueeze.default(b_2d, 0)
            bmm = torch.ops.vllm.bmm_fp8.default(
                a, b, a_scale, b_scale, self.dtype, "auto"
            )
            return torch.ops.aten.view.default(bmm, [gathered.shape[0], b_2d.shape[1]])

        def replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale,
                b_scale,
                0,
                self.tp.unique_name,
                self.dtype,
            )

        def extra_check(match: pm.Match) -> bool:
            parsed = _get_match_bmm_fp8(match)
            if parsed is None:
                return False
            return _flashinfer_bmm_fp8_extra_check(match) and _passes_min_m(parsed.a_2d)

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=extra_check,
        )


class FlashInferAllGatherBMMFP8QKVPattern(_FlashInferCollectivePatternBase):
    def register(self, pm_pass: PatternMatcherPass) -> None:
        pattern = pm.CallFunction(
            torch.ops.aten.split_with_sizes.default,
            pm.CallFunction(
                torch.ops.aten.view.default,
                pm.CallFunction(
                    torch.ops.vllm.bmm_fp8.default,
                    pm.CallFunction(
                        torch.ops.aten.unsqueeze.default,
                        pm.CallFunction(
                            torch.ops.vllm.all_gather.default,
                            pm.KeywordArg("a_shard_2d"),
                            0,
                            self.tp_size,
                            self.tp.unique_name,
                        ),
                        0,
                    ),
                    pm.CallFunction(
                        torch.ops.aten.unsqueeze.default,
                        pm.KeywordArg("b_2d"),
                        0,
                    ),
                    pm.KeywordArg("a_scale"),
                    pm.KeywordArg("b_scale"),
                    self.dtype,
                    "auto",
                    _users=pm.MULTIPLE,
                ),
                pm.Ignored(),
                _users=pm.MULTIPLE,
            ),
            pm.Ignored(),
            pm.Ignored(),
            _users=pm.MULTIPLE,
        )

        def extra_check(match: pm.Match) -> bool:
            if not _flashinfer_bmm_fp8_extra_check(match):
                return False

            split_node = _find_match_node(
                match, torch.ops.aten.split_with_sizes.default
            )
            if split_node is None:
                return False

            ag_match = _match_ag_bmm_from_split(split_node)
            return ag_match is not None and _passes_min_m(ag_match.replace_nodes[0])

        @pm.register_graph_pattern(
            pattern,
            pass_dict=pm_pass,
            extra_check=extra_check,
        )
        def handler(
            match: pm.Match,
            a_shard_2d: fx.Node,
            b_2d: fx.Node,
            a_scale: object,
            b_scale: object,
        ) -> None:
            del a_shard_2d, b_2d, a_scale, b_scale
            split_node = _find_match_node(
                match, torch.ops.aten.split_with_sizes.default
            )
            if split_node is None:
                return

            ag_match = _match_ag_bmm_from_split(split_node)
            if ag_match is None or not _passes_min_m(ag_match.replace_nodes[0]):
                return

            _lower_ag_bmm(match.graph, ag_match)
            match.erase_nodes()


def register_flashinfer_bmm_fp8_collective_patterns(
    pm_pass: PatternMatcherPass,
    dtype: torch.dtype,
    device: str | None,
) -> None:
    FlashInferBMMFP8ReduceScatterPattern(dtype, device).register(pm_pass)
    FlashInferAllGatherBMMFP8Pattern(dtype, device).register(pm_pass)
    FlashInferAllGatherBMMFP8QKVPattern(dtype, device).register(pm_pass)


@dataclass
class _FP8CollectiveGemmMatch:
    replace_nodes: list[fx.Node]
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: object
    group_name: str


def _match_ag_bmm_from_split(
    split_node: fx.Node,
) -> _FP8CollectiveGemmMatch | None:
    if not _is_qkv_split(split_node):
        return None

    split_input = _get_node_arg(split_node, "self", 0)
    if (
        not isinstance(split_input, fx.Node)
        or not _is_passthrough(split_input)
        or _node_ndim(split_input) != 2
    ):
        return None

    bmm_node = _strip_view_like(split_input)
    parsed_bmm = _parse_bmm_fp8(bmm_node)
    if parsed_bmm is None or not _is_supported_flashinfer_bmm_fp8(parsed_bmm):
        return None

    ag_node = _strip_view_like(parsed_bmm.a_2d)
    parsed_ag = _parse_all_gather(ag_node)
    if parsed_ag is None:
        return None

    parsed_group_name = _parse_collective_group_name(parsed_ag)
    if parsed_group_name is None:
        return None

    replace_nodes = [split_input]
    sibling_targets = [
        user
        for user in bmm_node.users
        if user is not split_input
        and _is_passthrough(user)
        and _node_ndim(user) == 2
        and _node_shape(user) == _node_shape(split_input)
    ]
    if sibling_targets:
        if len(sibling_targets) != 1:
            return None

        branch_kinds = {
            _classify_qkv_branch(split_input),
            _classify_qkv_branch(sibling_targets[0]),
        }
        if branch_kinds != {"direct", "rotary"}:
            return None
        replace_nodes.append(sibling_targets[0])

    return _FP8CollectiveGemmMatch(
        replace_nodes=replace_nodes,
        a_2d=parsed_ag.input_node,
        b_2d=parsed_bmm.b_2d,
        a_scale=parsed_bmm.a_scale,
        b_scale=parsed_bmm.b_scale,
        out_dtype=parsed_bmm.out_dtype,
        group_name=parsed_group_name,
    )


def _first_node_in_graph(graph: fx.Graph, nodes: list[fx.Node]) -> fx.Node:
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    return min(nodes, key=node_order.__getitem__)


def _lower_ag_bmm(graph: fx.Graph, match: _FP8CollectiveGemmMatch) -> None:
    replace_node = _first_node_in_graph(graph, match.replace_nodes)
    with graph.inserting_before(replace_node):
        replacement = graph.call_function(
            torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default,
            args=(
                match.a_2d,
                match.b_2d,
                match.a_scale,
                match.b_scale,
                0,
                match.group_name,
                match.out_dtype,
            ),
        )

    _copy_replacement_meta(replace_node, replacement)
    for node in match.replace_nodes:
        node.replace_all_uses_with(replacement)
    for node in match.replace_nodes:
        graph.erase_node(node)


class AsyncTPPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.min_token_num = get_effective_sp_min_token_num(config)

        # Enable symmetric memory for the TP process group
        tp_device_group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(tp_device_group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="async_tp_pass"
        )
        GEMMReduceScatterPattern(self.model_dtype, self.device).register(self.patterns)

        AllGatherGEMMPattern(self.model_dtype, self.device).register(self.patterns)

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherScaledMMPattern(self.model_dtype, self.device).register(
                self.patterns
            )

            CutlassScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherCutlassScaledMMPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            with suppress(ImportError):
                import vllm.utils.flashinfer  # noqa: F401
            if hasattr(torch.ops.vllm, "bmm_fp8"):
                register_flashinfer_bmm_fp8_collective_patterns(
                    self.patterns,
                    self.model_dtype,
                    self.device,
                )

        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # AsyncTP is a follow-up optimization on top of sequence parallelism,
        # so reuse the same compile-range gate.
        return is_sp_applicable_for_range(
            self.compilation_config,
            self.min_token_num,
            compile_range,
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
