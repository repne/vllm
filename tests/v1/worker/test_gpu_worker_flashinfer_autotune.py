# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker


def _make_worker(*, enable_flashinfer_autotune: bool = True):
    worker = object.__new__(Worker)
    worker.device = "cuda:0"
    worker.model_runner = object()
    worker.vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(
            enable_flashinfer_autotune=enable_flashinfer_autotune
        )
    )
    return worker


def test_flashinfer_early_autotune_returns_retained_memory(monkeypatch) -> None:
    worker = _make_worker()
    autotune_calls = []

    class FakeMemorySnapshot:
        free_memory_values = [10_000, 7_500]

        def __init__(self, device):
            self.device = device
            self.free_memory = self.free_memory_values.pop(0)

    monkeypatch.setattr(gpu_worker, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        gpu_worker.current_platform, "has_device_capability", lambda capability: True
    )
    monkeypatch.setattr(gpu_worker, "MemorySnapshot", FakeMemorySnapshot)
    monkeypatch.setattr(gpu_worker.gc, "collect", lambda: None)
    monkeypatch.setattr(gpu_worker.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        gpu_worker, "flashinfer_autotune", lambda runner: autotune_calls.append(runner)
    )

    retained_memory = Worker._maybe_flashinfer_autotune_early(worker)

    assert retained_memory == 2_500
    assert worker.flashinfer_autotune_retained_memory == 2_500
    assert worker._did_flashinfer_autotune_early is True
    assert autotune_calls == [worker.model_runner]

    assert Worker._maybe_flashinfer_autotune_early(worker) == 0
    assert autotune_calls == [worker.model_runner]


def test_flashinfer_early_autotune_disabled_returns_zero(monkeypatch) -> None:
    worker = _make_worker(enable_flashinfer_autotune=False)

    monkeypatch.setattr(
        gpu_worker,
        "has_flashinfer",
        lambda: (_ for _ in ()).throw(AssertionError("should not check FlashInfer")),
    )
    monkeypatch.setattr(
        gpu_worker,
        "flashinfer_autotune",
        lambda runner: (_ for _ in ()).throw(AssertionError("should not autotune")),
    )

    assert Worker._maybe_flashinfer_autotune_early(worker) == 0


def test_determine_available_memory_accounts_for_retained_autotune_memory(
    monkeypatch,
) -> None:
    worker = object.__new__(Worker)
    worker.device = "cuda:0"
    worker.init_snapshot = SimpleNamespace(free_memory=10_000, total_memory=10_000)
    worker.requested_memory = 6_000
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None,
        gpu_memory_utilization=0.9,
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=None)
    )
    profile_calls = []
    worker.model_runner = SimpleNamespace(
        model_memory_usage=1_000,
        profile_run=lambda: profile_calls.append("profile"),
        profile_cudagraph_memory=lambda: 0,
    )

    @contextmanager
    def fake_memory_profiling(init_snapshot, weights_memory):
        yield SimpleNamespace(
            before_profile=SimpleNamespace(torch_peak=100),
            after_profile=SimpleNamespace(free_memory=8_000),
            non_torch_increase=200,
            torch_peak_increase=0,
            weights_memory=weights_memory,
            non_kv_cache_memory=0,
        )

    monkeypatch.setattr(gpu_worker, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "memory_stats",
        lambda device: {"allocated_bytes.all.peak": 400},
    )
    monkeypatch.setattr(Worker, "_maybe_flashinfer_autotune_early", lambda self: 700)

    available_memory = Worker.determine_available_memory(worker)

    assert available_memory == 3_800
    assert worker.available_kv_cache_memory_bytes == 3_800
    assert worker.non_torch_memory == 200
    assert worker.peak_activation_memory == 300
    assert profile_calls == ["profile"]
