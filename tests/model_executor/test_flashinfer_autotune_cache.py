# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

import vllm.utils.flashinfer as fi_utils
from vllm.distributed import parallel_state
from vllm.model_executor.warmup import kernel_warmup


def test_resolve_flashinfer_autotune_file_default_layout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_jit = SimpleNamespace(
        env=SimpleNamespace(
            FLASHINFER_WORKSPACE_DIR=Path("/flashinfer-cache/0.6.11.post2/103a")
        )
    )
    fake_flashinfer = SimpleNamespace(jit=fake_jit)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fake_jit)
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None)

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == (
        tmp_path
        / "flashinfer_autotune_cache"
        / "0.6.11.post2"
        / "103a"
        / cache_hash
        / "autotune_configs.json"
    )
    assert path.parent.is_dir()


def test_resolve_flashinfer_autotune_file_uses_override_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == tmp_path / cache_hash / "autotune_configs.json"


def test_flashinfer_autotune_counts_include_mamba_aligned_mtp_chunks() -> None:
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        cache_config=SimpleNamespace(block_size=1072),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
    )

    token_counts = kernel_warmup._flashinfer_autotune_token_counts(runner)

    assert token_counts == (16384, 16080, 15008, 1376)


def test_flashinfer_autotune_counts_include_concurrent_prefill_tails() -> None:
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384,
            max_num_seqs=16,
        ),
        cache_config=SimpleNamespace(block_size=1072),
        speculative_config=SimpleNamespace(
            use_eagle=lambda: True,
            num_speculative_tokens=3,
        ),
        uniform_decode_query_len=4,
        max_model_len=262144,
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[
                4,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
                168,
                176,
                184,
                192,
                200,
                208,
                216,
                224,
                232,
                240,
                248,
                256,
            ],
        ),
    )

    token_counts = kernel_warmup._flashinfer_autotune_token_counts(runner)

    assert token_counts == (
        16384,
        16080,
        15008,
        1376,
        5504,
        15136,
        7008,
        14016,
        256,
        248,
        64,
        56,
    )


def test_flashinfer_autotune_counts_cover_250k_2048_config() -> None:
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=8,
        ),
        cache_config=SimpleNamespace(block_size=1072),
        speculative_config=SimpleNamespace(
            use_eagle=lambda: True,
            num_speculative_tokens=3,
        ),
        uniform_decode_query_len=4,
        max_model_len=250000,
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[4, 8, 16, 24, 32, 40, 48, 56, 64],
        ),
    )

    token_counts = kernel_warmup._flashinfer_autotune_token_counts(runner)

    assert token_counts == (
        2048,
        1072,
        976,
        1952,
        224,
        896,
        1792,
        64,
        56,
    )


def test_flashinfer_autotune_persists_expanded_buckets(monkeypatch, tmp_path) -> None:
    autotune_kwargs = []
    dummy_runs = []
    barriers = []
    broadcasts = []
    loaded_configs = []
    cache_path = tmp_path / "autotune_configs.json"

    @contextmanager
    def fake_autotune(**kwargs):
        autotune_kwargs.append(kwargs)
        yield
        Path(kwargs["cache"]).write_bytes(b'{"configs": []}')

    class FakeAutoTuner:
        @classmethod
        def get(cls):
            return cls()

        def load_configs(self, path):
            loaded_configs.append(path)

    class FakeWorld:
        rank_in_group = 0
        local_rank = 0

        def broadcast_object(self, obj, src):
            broadcasts.append((obj, src))
            return obj

        def barrier(self):
            barriers.append("barrier")

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        cache_config=SimpleNamespace(block_size=1072),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
        _dummy_run=lambda **kwargs: dummy_runs.append(kwargs),
    )

    fake_autotuner = ModuleType("flashinfer.autotuner")
    fake_autotuner.AutoTuner = FakeAutoTuner
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setattr(fi_utils, "autotune", fake_autotune)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: FakeWorld())
    monkeypatch.setattr(
        kernel_warmup, "_resolve_flashinfer_autotune_file", lambda _: cache_path
    )

    kernel_warmup.flashinfer_autotune(runner)

    assert autotune_kwargs == [
        {"tune_mode": True, "cache": str(cache_path)},
    ]
    assert dummy_runs == [
        {
            "num_tokens": 16384,
            "skip_eplb": True,
            "is_profile": True,
        },
        {
            "num_tokens": 16080,
            "skip_eplb": True,
            "is_profile": True,
        },
        {
            "num_tokens": 15008,
            "skip_eplb": True,
            "is_profile": True,
        },
        {
            "num_tokens": 1376,
            "skip_eplb": True,
            "is_profile": True,
        },
    ]
    assert broadcasts == [(b'{"configs": []}', 0)]
    assert barriers == ["barrier"]
    assert loaded_configs == [str(cache_path)]
