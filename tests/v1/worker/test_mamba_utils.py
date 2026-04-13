# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import (
    MambaCopyBuffers,
    MambaGPUContext,
    postprocess_mamba,
    preprocess_mamba,
)


def _make_scheduler_output(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None,
    resumed_req_ids: set[str],
) -> SchedulerOutput:
    cached = CachedRequestData.make_empty()
    cached.resumed_req_ids = resumed_req_ids
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted_req_ids,
    )


def test_resumed_req_ids_cleared_from_mamba_state_idx():
    """When a request is force-preempted (e.g. reset_prefix_cache),
    it appears in resumed_req_ids but NOT in preempted_req_ids.
    preprocess_mamba must still reset its mamba_state_idx_cpu entry to -1,
    otherwise stale indices can point beyond the new block allocation.

    Note: finished and preempted requests are now handled by
    input_batch.remove_request() which sets mamba_state_idx_cpu[req_index] = -1.
    preprocess_mamba only handles resumed_req_ids that weren't removed.
    """
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)

    # Set up input_batch with mamba_state_idx_cpu as a numpy array.
    # The "resumed" request is at index 0 with a stale mamba_state_idx of 3.
    input_batch = MagicMock(req_ids=[])
    input_batch.mamba_state_idx_cpu = np.array([3, -1, -1, -1], dtype=np.int32)
    input_batch.req_id_to_index = {"resumed": 0}

    copy_bufs = MagicMock(mamba_group_ids=[0], mamba_spec=spec)

    sched = _make_scheduler_output(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        resumed_req_ids={"resumed"},
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            sched,
            MagicMock(),  # kv_cache_config
            cache_config,
            input_batch,
            {},  # requests
            {},  # forward_context
            (),  # mamba_state_copy_funcs
            copy_bufs,
        )

    # The resumed request's mamba_state_idx should be reset to -1
    assert input_batch.mamba_state_idx_cpu[0] == -1


# -----------------------------------------------------------------------------
# Golden tests for postprocess_mamba_fused_kernel
# -----------------------------------------------------------------------------


@dataclass
class _TestConfig:
    """Common test configuration for fused kernel tests."""

    block_size: int = 16
    num_blocks: int = 32
    num_layers: int = 2
    num_reqs: int = 4
    max_num_reqs: int = 8
    # Conv state shape: [num_blocks, conv_width, inner_dim]
    conv_width: int = 4
    conv_inner_dim: int = 64
    # Temporal state shape: [num_blocks, state_dim]
    temporal_state_dim: int = 128
    dtype: torch.dtype = torch.float16


class _MockCpuGpuBuffer:
    """Mock CpuGpuBuffer for testing without pinned memory."""

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.gpu = torch.zeros(size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


def _make_postprocess_scheduler_output(
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    scheduled_spec_decode_tokens: dict[str, list] | None = None,
) -> SchedulerOutput:
    """Create a minimal SchedulerOutput for postprocess testing."""
    cached = CachedRequestData.make_empty()
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
    )


def _make_mock_attention(
    conv_state: torch.Tensor, temporal_state: torch.Tensor
) -> MagicMock:
    """Create a mock attention object with kv_cache."""
    attention = MagicMock()
    attention.kv_cache = [conv_state, temporal_state]
    return attention


def _make_kv_cache_config(cfg: _TestConfig, layer_names: list[str]) -> KVCacheConfig:
    """Create a KVCacheConfig with mamba groups."""
    mamba_spec = MambaSpec(
        block_size=cfg.block_size,
        shapes=(
            (cfg.conv_width, cfg.conv_inner_dim),
            (cfg.temporal_state_dim,),
        ),
        dtypes=(cfg.dtype, cfg.dtype),
        mamba_cache_mode="all",
    )
    group = KVCacheGroupSpec(
        layer_names=layer_names,
        kv_cache_spec=mamba_spec,
    )
    return KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
    )


def _make_input_batch(
    req_ids: list[str],
    num_accepted_tokens: list[int],
    mamba_state_idx: list[int],
) -> MagicMock:
    """Create a mock GPUInputBatch."""
    batch = MagicMock()
    batch.req_ids = req_ids
    batch.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    # Use numpy arrays so modifications persist
    batch.num_accepted_tokens_cpu = np.array(num_accepted_tokens, dtype=np.int32)
    batch.mamba_state_idx_cpu = np.array(mamba_state_idx, dtype=np.int32)
    return batch


def _make_requests(
    req_ids: list[str],
    num_computed_tokens: list[int],
    block_ids_per_req: list[list[int]],
) -> dict[str, MagicMock]:
    """Create mock CachedRequestState objects."""
    requests = {}
    for i, req_id in enumerate(req_ids):
        req = MagicMock()
        req.num_computed_tokens = num_computed_tokens[i]
        req.block_ids = {0: block_ids_per_req[i]}  # group_id=0
        requests[req_id] = req
    return requests


def _make_copy_bufs(
    cfg: _TestConfig, kv_cache_config: KVCacheConfig, device: torch.device
) -> MambaCopyBuffers:
    """Create MambaCopyBuffers for the Python path."""

    def make_buffer(n, dtype):
        return _MockCpuGpuBuffer(n, dtype, device)

    return MambaCopyBuffers.create(
        max_num_reqs=cfg.max_num_reqs,
        kv_cache_config=kv_cache_config,
        copy_funcs=(get_conv_copy_spec, get_temporal_copy_spec),
        make_buffer=make_buffer,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPostprocessMambaFusedKernel:
    """Tests for postprocess_mamba_fused_kernel comparing GPU vs CPU paths."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @pytest.fixture
    def test_config(self):
        return _TestConfig()

    def test_matches_python_postprocess_mamba(self, device, test_config):
        """
        Golden test: GPU kernel produces identical results to Python impl.

        This test:
        1. Sets up identical initial state for both paths
        2. Runs Python postprocess_mamba (modifies states via batch_memcpy)
        3. Runs GPU fused kernel (modifies states directly)
        4. Compares resulting state tensors and num_accepted_tokens
        """
        cfg = test_config
        torch.manual_seed(42)

        # Test scenario: 4 requests with different copy conditions
        # Copy needed when: aligned_new_computed >= num_tokens_running_state
        # where: num_tokens_running_state = num_computed + num_scheduled - num_draft
        #        new_num_computed = num_tokens_running_state + num_accepted - 1
        #        aligned_new_computed = (new_num_computed // block_size) * block_size
        req_ids = ["req_0", "req_1", "req_2", "req_3"]

        # Configure requests so some need copies, some don't
        # block_size = 16
        # req_0: running=60+5-2=63, new=63+3-1=65, aligned=64 >= 63 -> COPY
        # req_1: running=30+3-0=33, new=33+2-1=34, aligned=32 < 33 -> NO COPY
        # req_2: running=45+8-3=50, new=50+4-1=53, aligned=48 < 50 -> NO COPY
        # req_3: running=10+6-0=16, new=16+2-1=17, aligned=16 >= 16 -> COPY
        num_computed_tokens = [60, 30, 45, 10]
        num_scheduled_tokens = {"req_0": 5, "req_1": 3, "req_2": 8, "req_3": 6}
        num_draft_tokens = {"req_0": 2, "req_1": 0, "req_2": 3, "req_3": 0}
        num_accepted_tokens = [3, 2, 4, 2]
        mamba_state_idx = [3, 1, 2, 0]  # source block indices

        # Block IDs for each request (simulate block table)
        block_ids_per_req = [
            list(range(8)),  # req_0: blocks 0-7
            list(range(8, 16)),  # req_1: blocks 8-15
            list(range(16, 24)),  # req_2: blocks 16-23
            list(range(24, 32)),  # req_3: blocks 24-31
        ]

        layer_names = [f"layer_{i}" for i in range(cfg.num_layers)]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        # Create state tensors - one set for Python, one for GPU
        conv_states_py = [
            torch.randn(
                cfg.num_blocks,
                cfg.conv_width,
                cfg.conv_inner_dim,
                dtype=cfg.dtype,
                device=device,
            )
            for _ in range(cfg.num_layers)
        ]
        temporal_states_py = [
            torch.randn(
                cfg.num_blocks,
                cfg.temporal_state_dim,
                dtype=cfg.dtype,
                device=device,
            )
            for _ in range(cfg.num_layers)
        ]

        # Clone for GPU path (deep copy before any modifications)
        conv_states_gpu = [s.clone() for s in conv_states_py]
        temporal_states_gpu = [s.clone() for s in temporal_states_py]

        # Create forward_context for both paths
        forward_context_py = {
            name: _make_mock_attention(conv_states_py[i], temporal_states_py[i])
            for i, name in enumerate(layer_names)
        }
        forward_context_gpu = {
            name: _make_mock_attention(conv_states_gpu[i], temporal_states_gpu[i])
            for i, name in enumerate(layer_names)
        }

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)
        copy_funcs = (get_conv_copy_spec, get_temporal_copy_spec)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            copy_funcs,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = MambaGPUContext.create(
            max_num_reqs=cfg.max_num_reqs,
            kv_cache_config=kv_cache_config,
            num_state_types=2,  # conv + temporal
            device=device,
        )
        gpu_ctx.initialize_from_forward_context(
            kv_cache_config, forward_context_gpu, copy_funcs
        )

        # Build GPU input tensors
        num_reqs = len(req_ids)
        num_accepted_tokens_gpu = torch.tensor(
            num_accepted_tokens, dtype=torch.int32, device=device
        )
        mamba_state_idx_gpu = torch.tensor(
            mamba_state_idx, dtype=torch.int32, device=device
        )
        num_scheduled_tokens_gpu = torch.tensor(
            [num_scheduled_tokens[r] for r in req_ids], dtype=torch.int32, device=device
        )
        num_computed_tokens_gpu = torch.tensor(
            num_computed_tokens, dtype=torch.int32, device=device
        )
        num_draft_tokens_gpu = torch.tensor(
            [num_draft_tokens.get(r, 0) for r in req_ids],
            dtype=torch.int32,
            device=device,
        )

        # Build block table: [num_reqs, max_blocks]
        max_blocks = max(len(b) for b in block_ids_per_req)
        block_table_gpu = torch.zeros(
            num_reqs, max_blocks, dtype=torch.int32, device=device
        )
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_gpu[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        gpu_ctx.run_fused_postprocess(
            num_reqs=num_reqs,
            num_accepted_tokens_gpu=num_accepted_tokens_gpu,
            mamba_state_idx_gpu=mamba_state_idx_gpu,
            num_scheduled_tokens_gpu=num_scheduled_tokens_gpu,
            num_computed_tokens_gpu=num_computed_tokens_gpu,
            num_draft_tokens_gpu=num_draft_tokens_gpu,
            block_table_gpu=block_table_gpu,
        )
        torch.accelerator.synchronize()

        # --- Compare results ---
        # 1. Compare state tensors
        for i in range(cfg.num_layers):
            torch.testing.assert_close(
                conv_states_gpu[i],
                conv_states_py[i],
                msg=f"Conv state mismatch at layer {i}",
            )
            torch.testing.assert_close(
                temporal_states_gpu[i],
                temporal_states_py[i],
                msg=f"Temporal state mismatch at layer {i}",
            )

        # 2. Compare num_accepted_tokens updates
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="num_accepted_tokens mismatch",
        )

    def test_no_copy_when_not_needed(self, device, test_config):
        """Kernel should not modify state when no copy is needed."""
        cfg = test_config
        torch.manual_seed(123)

        # Single request where no copy is needed:
        # running = 30 + 3 = 33, new = 33 + 1 - 1 = 33, aligned = 32 < 33
        req_ids = ["req_0"]
        num_computed_tokens = [30]
        num_scheduled_tokens = {"req_0": 3}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [1]
        mamba_state_idx = [1]
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        # Create state tensor
        conv_state = torch.randn(
            cfg.num_blocks,
            cfg.conv_width,
            cfg.conv_inner_dim,
            dtype=cfg.dtype,
            device=device,
        )
        temporal_state = torch.randn(
            cfg.num_blocks, cfg.temporal_state_dim, dtype=cfg.dtype, device=device
        )

        # Clone to verify no modification
        conv_state_orig = conv_state.clone()
        temporal_state_orig = temporal_state.clone()

        forward_context = {"layer_0": _make_mock_attention(conv_state, temporal_state)}
        copy_funcs = (get_conv_copy_spec, get_temporal_copy_spec)

        gpu_ctx = MambaGPUContext.create(
            max_num_reqs=cfg.max_num_reqs,
            kv_cache_config=kv_cache_config,
            num_state_types=2,
            device=device,
        )
        gpu_ctx.initialize_from_forward_context(
            kv_cache_config, forward_context, copy_funcs
        )

        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        gpu_ctx.run_fused_postprocess(
            num_reqs=num_reqs,
            num_accepted_tokens_gpu=torch.tensor(
                num_accepted_tokens, dtype=torch.int32, device=device
            ),
            mamba_state_idx_gpu=torch.tensor(
                mamba_state_idx, dtype=torch.int32, device=device
            ),
            num_scheduled_tokens_gpu=torch.tensor(
                [num_scheduled_tokens[r] for r in req_ids],
                dtype=torch.int32,
                device=device,
            ),
            num_computed_tokens_gpu=torch.tensor(
                num_computed_tokens, dtype=torch.int32, device=device
            ),
            num_draft_tokens_gpu=torch.tensor(
                [num_draft_tokens.get(r, 0) for r in req_ids],
                dtype=torch.int32,
                device=device,
            ),
            block_table_gpu=block_table_gpu,
        )
        torch.accelerator.synchronize()

        # State should be unchanged
        torch.testing.assert_close(conv_state, conv_state_orig)
        torch.testing.assert_close(temporal_state, temporal_state_orig)

    @pytest.mark.parametrize("num_reqs", [1, 2, 8, 16])
    def test_various_batch_sizes(self, device, test_config, num_reqs):
        """Verify kernel works correctly with different batch sizes."""
        cfg = _TestConfig(max_num_reqs=max(16, num_reqs))
        torch.manual_seed(456)

        req_ids = [f"req_{i}" for i in range(num_reqs)]
        # All requests will trigger a copy
        num_computed_tokens = [60] * num_reqs
        num_scheduled_tokens = {r: 5 for r in req_ids}
        num_draft_tokens = {r: 0 for r in req_ids}
        num_accepted_tokens = [3] * num_reqs
        mamba_state_idx = [3] * num_reqs
        # Each request gets unique blocks
        block_ids_per_req = [list(range(i * 8, (i + 1) * 8)) for i in range(num_reqs)]

        # Ensure we have enough blocks
        total_blocks = num_reqs * 8
        cfg = _TestConfig(num_blocks=total_blocks, max_num_reqs=max(16, num_reqs))

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)
        copy_funcs = (get_conv_copy_spec, get_temporal_copy_spec)

        # Create states for Python path
        conv_state_py = torch.randn(
            total_blocks,
            cfg.conv_width,
            cfg.conv_inner_dim,
            dtype=cfg.dtype,
            device=device,
        )
        temporal_state_py = torch.randn(
            total_blocks, cfg.temporal_state_dim, dtype=cfg.dtype, device=device
        )

        # Clone for GPU path
        conv_state_gpu = conv_state_py.clone()
        temporal_state_gpu = temporal_state_py.clone()

        forward_context_py = {
            "layer_0": _make_mock_attention(conv_state_py, temporal_state_py)
        }
        forward_context_gpu = {
            "layer_0": _make_mock_attention(conv_state_gpu, temporal_state_gpu)
        }

        # Run Python path
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            copy_funcs,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # Run GPU path
        gpu_ctx = MambaGPUContext.create(
            max_num_reqs=cfg.max_num_reqs,
            kv_cache_config=kv_cache_config,
            num_state_types=2,
            device=device,
        )
        gpu_ctx.initialize_from_forward_context(
            kv_cache_config, forward_context_gpu, copy_funcs
        )

        max_blocks_per_req = 8
        block_table_gpu = torch.zeros(
            num_reqs, max_blocks_per_req, dtype=torch.int32, device=device
        )
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_gpu[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        gpu_ctx.run_fused_postprocess(
            num_reqs=num_reqs,
            num_accepted_tokens_gpu=torch.tensor(
                num_accepted_tokens, dtype=torch.int32, device=device
            ),
            mamba_state_idx_gpu=torch.tensor(
                mamba_state_idx, dtype=torch.int32, device=device
            ),
            num_scheduled_tokens_gpu=torch.tensor(
                [num_scheduled_tokens[r] for r in req_ids],
                dtype=torch.int32,
                device=device,
            ),
            num_computed_tokens_gpu=torch.tensor(
                num_computed_tokens, dtype=torch.int32, device=device
            ),
            num_draft_tokens_gpu=torch.tensor(
                [num_draft_tokens.get(r, 0) for r in req_ids],
                dtype=torch.int32,
                device=device,
            ),
            block_table_gpu=block_table_gpu,
        )
        torch.accelerator.synchronize()

        # Compare results
        torch.testing.assert_close(
            conv_state_gpu, conv_state_py, msg="Conv state mismatch"
        )
        torch.testing.assert_close(
            temporal_state_gpu, temporal_state_py, msg="Temporal state mismatch"
        )

    def test_block_table_with_realistic_stride(self, device, test_config):
        """
        Test kernel with realistic block table strides.

        In real usage, the block table is pre-allocated with shape
        [max_num_reqs, max_num_blocks_per_req] and then sliced to
        [:num_reqs]. This means stride(0) = max_num_blocks_per_req,
        which is typically much larger than the actual blocks used.

        This test verifies the kernel handles non-tight strides correctly,
        catching bugs where stride is incorrectly treated as bytes vs elements.
        """
        cfg = test_config
        torch.manual_seed(789)

        # Use multiple requests to exercise stride-based indexing
        num_reqs = 4
        req_ids = [f"req_{i}" for i in range(num_reqs)]

        # All requests trigger copies (same setup as test_various_batch_sizes)
        num_computed_tokens = [60] * num_reqs
        num_scheduled_tokens = {r: 5 for r in req_ids}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [3] * num_reqs
        mamba_state_idx = [3] * num_reqs

        # Each request uses only 8 blocks, but we allocate much more
        blocks_used_per_req = 8
        block_ids_per_req = [
            list(range(i * blocks_used_per_req, (i + 1) * blocks_used_per_req))
            for i in range(num_reqs)
        ]

        total_blocks = num_reqs * blocks_used_per_req
        cfg = _TestConfig(num_blocks=total_blocks, max_num_reqs=max(16, num_reqs))

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)
        copy_funcs = (get_conv_copy_spec, get_temporal_copy_spec)

        # Create states for Python path
        conv_state_py = torch.randn(
            total_blocks,
            cfg.conv_width,
            cfg.conv_inner_dim,
            dtype=cfg.dtype,
            device=device,
        )
        temporal_state_py = torch.randn(
            total_blocks, cfg.temporal_state_dim, dtype=cfg.dtype, device=device
        )

        # Clone for GPU path
        conv_state_gpu = conv_state_py.clone()
        temporal_state_gpu = temporal_state_py.clone()

        forward_context_py = {
            "layer_0": _make_mock_attention(conv_state_py, temporal_state_py)
        }
        forward_context_gpu = {
            "layer_0": _make_mock_attention(conv_state_gpu, temporal_state_gpu)
        }

        # Run Python path
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            copy_funcs,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # Run GPU path with REALISTIC block table stride
        gpu_ctx = MambaGPUContext.create(
            max_num_reqs=cfg.max_num_reqs,
            kv_cache_config=kv_cache_config,
            num_state_types=2,
            device=device,
        )
        gpu_ctx.initialize_from_forward_context(
            kv_cache_config, forward_context_gpu, copy_funcs
        )

        # KEY DIFFERENCE: Create a large block table like real code does
        # Real system has max_num_blocks_per_req >> blocks actually used
        max_num_reqs_full = 16
        max_blocks_per_req_full = 512  # Much larger than blocks_used_per_req=8

        # Allocate full-size table (simulates pre-allocated CpuGpuBuffer)
        block_table_full = torch.zeros(
            max_num_reqs_full, max_blocks_per_req_full, dtype=torch.int32, device=device
        )

        # Fill in actual block IDs (only first few columns used)
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_full[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        # Slice like real code: block_table.gpu[:num_reqs]
        # This preserves stride(0) = 512, not 8!
        block_table_gpu = block_table_full[:num_reqs]

        # Verify stride is large (the key property we're testing)
        assert block_table_gpu.stride(0) == max_blocks_per_req_full, (
            f"Expected stride {max_blocks_per_req_full}, "
            f"got {block_table_gpu.stride(0)}"
        )

        gpu_ctx.run_fused_postprocess(
            num_reqs=num_reqs,
            num_accepted_tokens_gpu=torch.tensor(
                num_accepted_tokens, dtype=torch.int32, device=device
            ),
            mamba_state_idx_gpu=torch.tensor(
                mamba_state_idx, dtype=torch.int32, device=device
            ),
            num_scheduled_tokens_gpu=torch.tensor(
                [num_scheduled_tokens[r] for r in req_ids],
                dtype=torch.int32,
                device=device,
            ),
            num_computed_tokens_gpu=torch.tensor(
                num_computed_tokens, dtype=torch.int32, device=device
            ),
            num_draft_tokens_gpu=torch.tensor(
                [num_draft_tokens.get(r, 0) for r in req_ids],
                dtype=torch.int32,
                device=device,
            ),
            block_table_gpu=block_table_gpu,
        )
        torch.accelerator.synchronize()

        # Compare results - this will fail if stride handling is incorrect
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="Conv state mismatch - possible stride bug in kernel",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="Temporal state mismatch - possible stride bug in kernel",
        )
