# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GPU-resident int4 n-gram table (VLLM_PLE_GPU_QUANT).

The AMD and NVIDIA PLE twins register custom ops under the same names
(``qwen4_exp_ple_short_conv``), so they cannot coexist in one process. This
file skips itself when the NVIDIA module was imported first, e.g. when the
whole ``tests/models/qwen4_exp`` directory is collected together with
``test_ple.py``; run this file alone to exercise the AMD implementation.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "qwen4_exp_ple_short_conv"):
    pytest.skip(
        "the NVIDIA PLE module already registered the twin custom ops",
        allow_module_level=True,
    )

_REAL_SIDECAR_DIR = "/models/qwen38-flash-next-ple/ples_int4"
_REAL_FP8_SIDECAR_DIR = "/models/qwen38-flash-next-ple/ples_fp8"


def _write_mini_sidecar(
    tmp_path,
    *,
    shards: int,
    rows_per_shard: int,
    width: int,
    group: int,
) -> str:
    """Write a synthetic int4 group-quantized sidecar directory."""
    from safetensors.torch import save_file

    rng = torch.Generator().manual_seed(20260907)
    rows = shards * rows_per_shard
    for shard in range(shards):
        packed = torch.randint(
            0, 256, (rows_per_shard, width // 2), generator=rng, dtype=torch.uint8
        )
        scales = (
            torch.rand(
                rows_per_shard, width // group, generator=rng, dtype=torch.float32
            )
            * 0.05
            + 0.01
        )
        save_file(
            {
                "weight_i4": packed,
                "weight_scale": scales.to(torch.float16),
            },
            str(tmp_path / f"shard_{shard}.safetensors"),
        )
    (tmp_path / "META.json").write_text(
        json.dumps(
            {
                "layout": f"group{group}_int4_fp16scale_lownibblefirst",
                "shards": shards,
                "rows": rows,
                "width": width,
            }
        )
    )
    return str(tmp_path)


def _reference_rows(table, ids: torch.Tensor, rows_per_shard: int) -> torch.Tensor:
    """Reference dequant through the deployed ``_PleQuantTable``, one row at a time."""
    rows = []
    for row_id in ids.tolist():
        shard = row_id // rows_per_shard
        local = row_id - shard * rows_per_shard
        rows.append(table._dequant(shard, torch.tensor([local]))[0])
    return torch.stack(rows)


def _make_reference_table(quant_dir: str, rows: int, width: int, rows_per_shard: int):
    from vllm.v1.ple_offload.worker import _PleQuantTable

    original = _PleQuantTable.ROWS_PER_SHARD
    _PleQuantTable.ROWS_PER_SHARD = rows_per_shard
    try:
        return _PleQuantTable(quant_dir, rows, width)
    finally:
        _PleQuantTable.ROWS_PER_SHARD = original


def test_gpu_quant_lookup_matches_cpu_dequant(tmp_path) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import (
        _gpu_quant_lookup,
        _load_gpu_quant_table,
    )

    quant_dir = _write_mini_sidecar(
        tmp_path, shards=4, rows_per_shard=8, width=8, group=2
    )
    rows, width = 32, 8
    table = _make_reference_table(quant_dir, rows, width, rows_per_shard=8)

    q, s, _ = _load_gpu_quant_table(
        quant_dir, rows, width, tp_start=0, tp_end=rows, device=torch.device("cpu")
    )
    embedding = SimpleNamespace(
        _ple_quant_q=q,
        _ple_quant_s=s,
        _ple_quant_kind="int4",
        _ple_quant_lut=None,
        tp_size=1,
        embedding_dim=width,
    )
    ids = torch.arange(rows, dtype=torch.long).reshape(4, rows // 4)
    output = torch.empty(4, (rows // 4) * width, dtype=torch.float16)
    _gpu_quant_lookup(embedding, ids, output)

    reference = _reference_rows(table, torch.arange(rows), 8).to(torch.float16)
    torch.testing.assert_close(output, reference.view(4, -1), rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_gpu_quant_row_slicing_across_tp(tmp_path, tp_size: int) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _load_gpu_quant_table

    shards, rows_per_shard, width, group = 4, 8, 8, 2
    rows = shards * rows_per_shard
    quant_dir = _write_mini_sidecar(
        tmp_path,
        shards=shards,
        rows_per_shard=rows_per_shard,
        width=width,
        group=group,
    )
    table = _make_reference_table(quant_dir, rows, width, rows_per_shard)

    ids = torch.tensor([0, 1, 7, 8, 15, 16, 23, 24, 31], dtype=torch.long)
    reference = _reference_rows(table, ids, rows_per_shard)
    rows_per_rank = rows // tp_size
    gathered = torch.zeros(len(ids), width, dtype=torch.float16)
    for rank in range(tp_size):
        start, end = rank * rows_per_rank, (rank + 1) * rows_per_rank
        q, s, _ = _load_gpu_quant_table(
            quant_dir,
            rows,
            width,
            tp_start=start,
            tp_end=end,
            device=torch.device("cpu"),
        )
        # same row math as the vocab-parallel lookup: local index, zero elsewhere
        mask = (ids >= start) & (ids < end)
        local = (ids - start).masked_fill(~mask, 0)
        packed = q[local]
        scales = s[local].to(torch.float32)
        low = (packed & 0xF).to(torch.float32)
        high = (packed >> 4).to(torch.float32)
        nibbles = torch.stack((low, high), dim=-1).view(len(ids), width)
        ranked = (nibbles - 8.0) * scales.repeat_interleave(group, dim=1)
        gathered += ranked.to(torch.float16) * mask.unsqueeze(-1)

    torch.testing.assert_close(gathered, reference.to(torch.float16), rtol=0, atol=0)


def test_gpu_quant_stub_replaces_table_parameter() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _stub_gpu_quant_table

    embedding = nn.Module()
    embedding.embedding_dim = 8
    embedding.weight = nn.Parameter(torch.zeros(4, 8))
    _stub_gpu_quant_table(embedding)
    assert embedding.weight.numel() == 0
    assert embedding.weight.shape == (0, 8)

    meta_embedding = nn.Module()
    meta_embedding.embedding_dim = 8
    meta_embedding.weight = nn.Parameter(torch.empty(4, 8, device="meta"))
    _stub_gpu_quant_table(meta_embedding)
    assert meta_embedding.weight.numel() == 0


def test_gpu_quant_rejects_unknown_layout(tmp_path) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _load_gpu_quant_table

    quant_dir = _write_mini_sidecar(
        tmp_path, shards=1, rows_per_shard=8, width=8, group=2
    )
    meta_path = tmp_path / "META.json"
    meta = json.loads(meta_path.read_text())
    meta["layout"] = "group2_int4_fp16scale_highnibblefirst"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="low-nibble-first"):
        _load_gpu_quant_table(
            quant_dir, 8, 8, tp_start=0, tp_end=8, device=torch.device("cpu")
        )


@pytest.mark.skipif(
    not os.path.exists(_REAL_SIDECAR_DIR + "/META.json"),
    reason="real int4 sidecar not present on this host",
)
def test_gpu_quant_real_sidecar_parity() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _load_gpu_quant_table

    rows, width = 320001536, 160
    rows_per_shard = 2500012
    table = _make_reference_table(_REAL_SIDECAR_DIR, rows, width, rows_per_shard)
    for shard in (0, 127):
        start = shard * rows_per_shard
        q, s, _ = _load_gpu_quant_table(
            _REAL_SIDECAR_DIR,
            rows,
            width,
            tp_start=start,
            tp_end=start + rows_per_shard,
            device=torch.device("cpu"),
        )
        local_ids = torch.tensor([0, 1, 4095, rows_per_shard - 1], dtype=torch.long)
        packed = q[local_ids]
        scales = s[local_ids].to(torch.float32)
        low = (packed & 0xF).to(torch.float32)
        high = (packed >> 4).to(torch.float32)
        nibbles = torch.stack((low, high), dim=-1).view(len(local_ids), width)
        ranked = (nibbles - 8.0) * scales.repeat_interleave(16, dim=1)
        reference = table._dequant(shard, local_ids)
        torch.testing.assert_close(
            ranked.to(torch.float16), reference.to(torch.float16), rtol=0, atol=0
        )


def test_gpu_quant_lazy_attach_on_first_lookup(tmp_path) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _gpu_quant_lookup

    quant_dir = _write_mini_sidecar(
        tmp_path, shards=4, rows_per_shard=8, width=8, group=2
    )
    rows, width = 32, 8
    table = _make_reference_table(quant_dir, rows, width, rows_per_shard=8)
    embedding = SimpleNamespace(
        tp_size=1,
        embedding_dim=width,
        _ple_quant_pending=(quant_dir, rows, width, 0, rows),
    )
    ids = torch.arange(rows, dtype=torch.long).reshape(4, rows // 4)
    output = torch.empty(4, (rows // 4) * width, dtype=torch.float16)
    _gpu_quant_lookup(embedding, ids, output)

    assert embedding._ple_quant_pending is None
    assert embedding._ple_quant_q is not None
    reference = _reference_rows(table, torch.arange(rows), 8).to(torch.float16)
    torch.testing.assert_close(output, reference.view(4, -1), rtol=0, atol=0)


def _write_mini_fp8_sidecar(
    tmp_path, *, shards: int, rows_per_shard: int, width: int
) -> str:
    """Write a synthetic fp8 per-row sidecar (e4m3 bytes + fp32 row scales)."""
    from safetensors.torch import save_file

    rng = torch.Generator().manual_seed(20260915)
    for shard in range(shards):
        raw = torch.randint(
            0, 256, (rows_per_shard, width), generator=rng, dtype=torch.uint8
        )
        raw[((raw >> 3) & 0xF) == 15] = 0  # avoid inf/NaN encodings
        scales = (
            torch.rand(rows_per_shard, generator=rng, dtype=torch.float32) * 0.1
            + 0.01
        )
        save_file(
            {"weight_fp8": raw, "weight_scale": scales},
            str(tmp_path / f"shard_{shard}.safetensors"),
        )
    (tmp_path / "META.json").write_text(
        json.dumps(
            {
                "layout": "per_row_e4m3",
                "shards": shards,
                "rows": shards * rows_per_shard,
                "width": width,
            }
        )
    )
    return str(tmp_path)


def _fp8_reference(quant_dir: str, shards: int, width: int) -> torch.Tensor:
    from safetensors.torch import load_file

    from vllm.models.qwen4_exp.amd.ple_layer import _e4m3_lut

    lut = _e4m3_lut()
    rows = []
    for shard in range(shards):
        data = load_file(quant_dir + f"/shard_{shard}.safetensors")
        rows.append(
            lut[data["weight_fp8"].long()] * data["weight_scale"].unsqueeze(-1)
        )
    return torch.cat(rows)


def test_gpu_quant_fp8_lookup_matches_reference(tmp_path) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _gpu_quant_lookup

    shards, rows_per_shard, width = 4, 8, 8
    rows = shards * rows_per_shard
    quant_dir = _write_mini_fp8_sidecar(
        tmp_path, shards=shards, rows_per_shard=rows_per_shard, width=width
    )
    embedding = SimpleNamespace(
        tp_size=1,
        embedding_dim=width,
        _ple_quant_pending=(quant_dir, rows, width, 0, rows),
    )
    ids = torch.arange(rows, dtype=torch.long).reshape(4, rows // 4)
    output = torch.empty(4, (rows // 4) * width, dtype=torch.float16)
    _gpu_quant_lookup(embedding, ids, output)

    reference = _fp8_reference(quant_dir, shards, width).to(torch.float16)
    torch.testing.assert_close(output, reference.view(4, -1), rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_gpu_quant_fp8_row_slicing_across_tp(tmp_path, tp_size: int) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _e4m3_lut, _load_gpu_quant_table

    shards, rows_per_shard, width = 4, 8, 8
    rows = shards * rows_per_shard
    quant_dir = _write_mini_fp8_sidecar(
        tmp_path, shards=shards, rows_per_shard=rows_per_shard, width=width
    )
    lut = _e4m3_lut()
    reference = _fp8_reference(quant_dir, shards, width)
    ids = torch.tensor([0, 1, 7, 8, 15, 16, 23, 24, 31], dtype=torch.long)
    rows_per_rank = rows // tp_size
    gathered = torch.zeros(len(ids), width, dtype=torch.float16)
    for rank in range(tp_size):
        start, end = rank * rows_per_rank, (rank + 1) * rows_per_rank
        q, s, layout = _load_gpu_quant_table(
            quant_dir,
            rows,
            width,
            tp_start=start,
            tp_end=end,
            device=torch.device("cpu"),
        )
        assert layout == "per_row_e4m3"
        mask = (ids >= start) & (ids < end)
        local = (ids - start).masked_fill(~mask, 0)
        ranked = lut[q[local].long()] * s[local].unsqueeze(-1)
        gathered += ranked.to(torch.float16) * mask.unsqueeze(-1)

    torch.testing.assert_close(
        gathered, reference[ids].to(torch.float16), rtol=0, atol=0
    )


@pytest.mark.skipif(
    not os.path.exists(_REAL_FP8_SIDECAR_DIR + "/shard_127.safetensors"),
    reason="real fp8 sidecar not present on this host",
)
def test_gpu_quant_real_fp8_sidecar_parity() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import _e4m3_lut, _load_gpu_quant_table

    rows, width = 320001536, 160
    rows_per_shard = 2500012
    table = _make_reference_table(_REAL_FP8_SIDECAR_DIR, rows, width, rows_per_shard)
    lut = _e4m3_lut()
    for shard in (0, 127):
        start = shard * rows_per_shard
        q, s, layout = _load_gpu_quant_table(
            _REAL_FP8_SIDECAR_DIR,
            rows,
            width,
            tp_start=start,
            tp_end=start + rows_per_shard,
            device=torch.device("cpu"),
        )
        assert layout == "per_row_e4m3"
        local_ids = torch.tensor([0, 1, 4095, rows_per_shard - 1], dtype=torch.long)
        ranked = lut[q[local_ids].long()] * s[local_ids].unsqueeze(-1)
        reference = table._dequant(shard, local_ids)
        torch.testing.assert_close(
            ranked.to(torch.float16), reference.to(torch.float16), rtol=0, atol=0
        )
