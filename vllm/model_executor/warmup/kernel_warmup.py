# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def _resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / _flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    # NOTE: Autotuning may have already been run early (before KV cache
    # allocation) in gpu_worker.determine_available_memory() to avoid OOM.
    # When early autotuning ran, we skip the second call here because the
    # autotuner workspace buffers would compete with the now-allocated KV
    # cache for GPU memory, causing OOM on shapes not covered by the early
    # run's cache.
    _did_early_autotune = getattr(worker, "_did_flashinfer_autotune_early", False)
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif _did_early_autotune:
        logger.info(
            "Skipping FlashInfer autotune in kernel_warmup — "
            "already completed early (before KV cache allocation)."
        )
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


# TODO: remove once FlashInfer upstream fixes the persistent file cache
# to resolve collisions like `use_8x4_sf_layout=True/False`, which causes
# invalid tactics to be chosen
_FLASHINFER_USE_PERSISTENT_CACHE = False

def _spec_decode_uses_eagle(runner: "GPUModelRunner") -> bool:
    spec_config = getattr(runner, "speculative_config", None)
    if spec_config is None:
        return False
    use_eagle = getattr(spec_config, "use_eagle", None)
    if not callable(use_eagle):
        return False
    return bool(use_eagle())


def _spec_decode_query_len(runner: "GPUModelRunner") -> int:
    query_len = int(getattr(runner, "uniform_decode_query_len", 0) or 0)
    if query_len > 0:
        return query_len

    spec_config = getattr(runner, "speculative_config", None)
    if spec_config is None:
        return 1

    num_spec_tokens = int(getattr(spec_config, "num_speculative_tokens", 0) or 0)
    return 1 + num_spec_tokens


def _runner_max_num_seqs(runner: "GPUModelRunner") -> int:
    scheduler_config = runner.scheduler_config
    return int(getattr(scheduler_config, "max_num_seqs", 1) or 1)


def _runner_max_model_len(runner: "GPUModelRunner") -> int:
    max_model_len = int(getattr(runner, "max_model_len", 0) or 0)
    if max_model_len > 0:
        return max_model_len

    model_config = getattr(runner, "model_config", None)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    if max_model_len > 0:
        return max_model_len

    vllm_config = getattr(runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    return int(getattr(model_config, "max_model_len", 0) or 0)


def _runner_cudagraph_capture_sizes(runner: "GPUModelRunner") -> tuple[int, ...]:
    compilation_config = getattr(runner, "compilation_config", None)
    if compilation_config is None:
        vllm_config = getattr(runner, "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return ()

    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if not capture_sizes:
        return ()
    return tuple(sorted(set(int(size) for size in capture_sizes if int(size) > 0)))


def _add_packed_residual_counts(
    token_counts: list[int],
    *,
    residual_tokens: int,
    max_num_tokens: int,
    max_num_seqs: int,
) -> None:
    if residual_tokens <= 0:
        return

    max_pack = min(max_num_seqs, max_num_tokens // residual_tokens)
    if max_pack <= 0:
        return

    # Cover a single tail, the observed four-way concurrency tail, and the
    # largest tail pack the scheduler can place in one batch. Warming every
    # possible tail size is too expensive because each exact count replays the
    # whole model under FlashInfer autotune.
    token_counts.append(residual_tokens)
    token_counts.append(min(4, max_pack) * residual_tokens)
    token_counts.append(max_pack * residual_tokens)


def _add_flashinfer_cudagraph_counts(
    token_counts: list[int],
    runner: "GPUModelRunner",
    max_num_tokens: int,
) -> None:
    capture_sizes = _runner_cudagraph_capture_sizes(runner)
    if not capture_sizes:
        return

    max_capture_size = min(capture_sizes[-1], max_num_tokens)
    if max_capture_size <= 0:
        return

    spec_decode_offset = 0
    query_len = _spec_decode_query_len(runner)
    if query_len > 1:
        # EAGLE/MTP graph warmups can run the model FP8 GEMMs at a smaller
        # effective M than the captured graph size. For num_speculative_tokens=3,
        # a 64-token graph has shown FP8 GEMM shapes at M=56.
        spec_decode_offset = 2 * query_len

    important_sizes = [max_capture_size]
    if spec_decode_offset > 0 and max_capture_size > spec_decode_offset:
        important_sizes.append(max_capture_size - spec_decode_offset)

    if max_capture_size > 64 and 64 in capture_sizes:
        important_sizes.append(64)
        if spec_decode_offset > 0 and spec_decode_offset < 64:
            important_sizes.append(64 - spec_decode_offset)

    token_counts.extend(important_sizes)


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    """Exact token counts to execute during FlashInfer autotune.

    Some FlashInfer runners include raw input shapes in their cache-key extras.
    For those runners, executing only a 16,384-token dummy batch may not cover
    the exact 15,008-token shape produced by vLLM's Mamba-aligned chunked
    prefill. Run the scheduler-aligned prefill, packed tail, and graph capture
    shapes during autotune while keeping FlashInfer's default bucket mapping
    active, so runtime cache lookups use the same tuning-config identity.
    """
    max_num_tokens = int(runner.scheduler_config.max_num_batched_tokens)
    token_counts = [max_num_tokens]
    max_num_seqs = _runner_max_num_seqs(runner)
    max_model_len = _runner_max_model_len(runner)

    cache_config = getattr(runner, "cache_config", None)
    if cache_config is None:
        cache_config = runner.vllm_config.cache_config
    block_size = int(getattr(cache_config, "block_size", 0) or 0)
    chunk_tokens = max_num_tokens
    if block_size > 0:
        aligned_tokens = max_num_tokens // block_size * block_size
        if aligned_tokens > 0:
            token_counts.append(aligned_tokens)
            chunk_tokens = aligned_tokens
        if _spec_decode_uses_eagle(runner) and aligned_tokens > block_size:
            chunk_tokens = aligned_tokens - block_size
            token_counts.append(chunk_tokens)

    _add_packed_residual_counts(
        token_counts,
        residual_tokens=max_num_tokens - chunk_tokens,
        max_num_tokens=max_num_tokens,
        max_num_seqs=max_num_seqs,
    )
    if max_model_len > 0:
        _add_packed_residual_counts(
            token_counts,
            residual_tokens=max_model_len % chunk_tokens,
            max_num_tokens=max_num_tokens,
            max_num_seqs=max_num_seqs,
        )

    _add_flashinfer_cudagraph_counts(token_counts, runner, max_num_tokens)

    return tuple(dict.fromkeys(t for t in token_counts if 0 < t <= max_num_tokens))


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    warmup_token_counts = _flashinfer_autotune_token_counts(runner)

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            for num_tokens in warmup_token_counts:
                runner._dummy_run(
                    num_tokens=num_tokens,
                    skip_eplb=True,
                    is_profile=True,
                )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = _resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, FlashInfer autotunes operations
    # for its generated buckets up to m. Some runners also include raw tensor
    # shapes in cache-key extras, so run the exact scheduler-aligned shapes too.
    dummy_run_kwargs = dict(
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(tune_mode=True, cache=str(cache_path)):
                for num_tokens in warmup_token_counts:
                    runner._dummy_run(**dummy_run_kwargs, num_tokens=num_tokens)
        else:
            for num_tokens in warmup_token_counts:
                runner._dummy_run(**dummy_run_kwargs, num_tokens=num_tokens)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        if not is_leader and world.local_rank == 0:
            with open(cache_path, "wb") as f:
                f.write(tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
