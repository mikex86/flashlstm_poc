from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import math
import torch

RECOMPUTE = 4


def _find_library(root: Path) -> Path:
    candidates = (
        root / "build" / "libflashlstm.so",
        root / "cmake-build-debug" / "libflashlstm.so",
        root / "cmake-build-release" / "libflashlstm.so",
        root / "cmake-build-debug" / "flashlstm" / "libflashlstm.so",
        root / "cmake-build-release" / "flashlstm" / "libflashlstm.so",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate libflashlstm.so. Build the project with CMake before running this script."
    )


def _as_void_p(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


class GateCacheHost(ctypes.Structure):
    _fields_ = [
        ("h_ptr", ctypes.c_void_p),
        ("c_ptr", ctypes.c_void_p),
    ]


class StreamingLstmOptions(ctypes.Structure):
    _fields_ = [
        ("h_dtype", ctypes.c_int),
        ("c_dtype", ctypes.c_int),
    ]


def _gate_dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    raise ValueError(f"Unsupported gate cache dtype: {dtype}")


@dataclass(frozen=True)
class LstmConfig:
    time_steps: int
    batch_size: int
    input_size: int
    hidden_size: int
    weight_sets: int = 1
    time_oversample: bool = False

    def describe(self) -> str:
        return (
            f"T={self.time_steps}, B={self.batch_size}, "
            f"I={self.input_size}, H={self.hidden_size}, S={self.weight_sets}, O={int(self.time_oversample)}"
        )


def _prepare_function(lib: ctypes.CDLL) -> None:
    lib.flstm_StreamingLstmForward.restype = None
    lib.flstm_StreamingLstmForward.argtypes = [
        ctypes.c_size_t,  # time_steps
        ctypes.c_size_t,  # batch
        ctypes.c_size_t,  # input
        ctypes.c_size_t,  # hidden
        ctypes.c_size_t,  # recompute_interval
        ctypes.c_size_t,  # weight_set_count
        ctypes.c_bool,    # time_oversample
        ctypes.c_void_p,  # x host
        ctypes.c_void_p,  # h0 device
        ctypes.c_void_p,  # c0 device
        ctypes.c_void_p,  # weight_ih
        ctypes.c_void_p,  # weight_hh
        ctypes.c_void_p,  # bias_ih
        ctypes.c_void_p,  # bias_hh
        ctypes.c_void_p,  # y host
        GateCacheHost,    # gate cache host
        ctypes.POINTER(StreamingLstmOptions),  # gate cache options
        ctypes.c_void_p,  # hy device
        ctypes.c_void_p,  # cy device
        ctypes.c_void_p,  # compute stream
        ctypes.c_void_p,  # h2d stream
        ctypes.c_void_p,  # d2h stream
    ]


def _run_case(lib: ctypes.CDLL, cfg: LstmConfig):
    torch.manual_seed(0)
    device = torch.device("cuda")

    weight_set_count = cfg.weight_sets
    oversample_factor = weight_set_count if (cfg.time_oversample and weight_set_count > 1) else 1
    effective_time_steps = cfg.time_steps * oversample_factor

    x_fp32 = torch.randn(cfg.time_steps, cfg.batch_size, cfg.input_size, dtype=torch.float32).contiguous()
    x_host_logical = x_fp32.to(dtype=torch.float16).contiguous().pin_memory()
    x_torch = x_fp32.to(device)
    x_host = x_host_logical
    if oversample_factor > 1:
        x_host = x_host_logical.repeat_interleave(oversample_factor, dim=0).contiguous()

    h0_init = torch.randn(1, cfg.batch_size, cfg.hidden_size, dtype=torch.float32)
    c0_init = torch.randn(1, cfg.batch_size, cfg.hidden_size, dtype=torch.float32)

    h0_torch = h0_init.to(device)
    c0_torch = c0_init.to(device)

    h0_device = h0_init.squeeze(0).to(device=device, dtype=torch.float16).contiguous()
    c0_device = c0_init.squeeze(0).to(device=device, dtype=torch.float16).contiguous()

    gate_dim = 4 * cfg.hidden_size
    std = 1.0 / math.sqrt(cfg.hidden_size)
    weight_ih = torch.empty(
        weight_set_count, gate_dim, cfg.input_size, device=device, dtype=torch.float32
    ).uniform_(-std, std)
    weight_hh = torch.empty(
        weight_set_count, gate_dim, cfg.hidden_size, device=device, dtype=torch.float32
    ).uniform_(-std, std)
    bias_ih = torch.zeros(weight_set_count, gate_dim, device=device, dtype=torch.float32)
    bias_hh = torch.zeros_like(bias_ih)

    lstm_cells = [torch.nn.LSTMCell(cfg.input_size, cfg.hidden_size).to(device) for _ in range(weight_set_count)]
    with torch.no_grad():
        for idx, cell in enumerate(lstm_cells):
            cell.eval()
            cell.weight_ih.copy_(weight_ih[idx])
            cell.weight_hh.copy_(weight_hh[idx])
            cell.bias_ih.copy_(bias_ih[idx])
            cell.bias_hh.copy_(bias_hh[idx])

    with torch.no_grad():
        h_cell = h0_torch.squeeze(0).clone()
        c_cell = c0_torch.squeeze(0).clone()
        h_states_ref = []
        for t in range(cfg.time_steps):
            for set_idx in range(weight_set_count if cfg.time_oversample else 1):
                effective_idx = set_idx if cfg.time_oversample else (t % weight_set_count)
                h_cell, c_cell = lstm_cells[effective_idx](x_torch[t], (h_cell, c_cell))
            h_states_ref.append(h_cell.unsqueeze(0))
        h_states_ref = torch.cat(h_states_ref, dim=0)
        y_ref = h_states_ref
        h_n_ref = h_cell.unsqueeze(0)
        c_n_ref = c_cell.unsqueeze(0)

    y_host = torch.empty(effective_time_steps, cfg.batch_size, cfg.hidden_size, dtype=torch.float16).contiguous().pin_memory()

    checkpoint_steps = (effective_time_steps + RECOMPUTE - 1) // RECOMPUTE
    gate_cache_h = torch.empty(
        checkpoint_steps,
        cfg.batch_size,
        cfg.hidden_size,
        dtype=torch.float16,
    ).contiguous().pin_memory()
    gate_cache_c = torch.empty(
        checkpoint_steps,
        cfg.batch_size,
        cfg.hidden_size,
        dtype=torch.float32,
    ).contiguous().pin_memory()
    gate_cache_struct = GateCacheHost(_as_void_p(gate_cache_h), _as_void_p(gate_cache_c))
    gate_cache_options = StreamingLstmOptions(
        _gate_dtype_enum(gate_cache_h.dtype),
        _gate_dtype_enum(gate_cache_c.dtype),
    )

    hy_device = torch.empty(cfg.batch_size, cfg.hidden_size, dtype=torch.float16, device=device).contiguous()
    cy_device = torch.empty(cfg.batch_size, cfg.hidden_size, dtype=torch.float16, device=device).contiguous()

    compute_stream = torch.cuda.Stream()
    h2d_stream = torch.cuda.Stream()
    d2h_stream = torch.cuda.Stream()

    stream_handles = {
        compute_stream.cuda_stream,
        h2d_stream.cuda_stream,
        d2h_stream.cuda_stream,
    }

    if len(stream_handles) != 3:
        raise RuntimeError("Streaming LSTM forward requires three distinct CUDA streams")

    lib.flstm_StreamingLstmForward(
        ctypes.c_size_t(effective_time_steps),
        ctypes.c_size_t(cfg.batch_size),
        ctypes.c_size_t(cfg.input_size),
        ctypes.c_size_t(cfg.hidden_size),
        ctypes.c_size_t(RECOMPUTE),
        ctypes.c_size_t(weight_set_count),
        ctypes.c_bool(False),  # oversample handled in test harness
        _as_void_p(x_host),
        _as_void_p(h0_device),
        _as_void_p(c0_device),
        _as_void_p(weight_ih),
        _as_void_p(weight_hh),
        _as_void_p(bias_ih),
        _as_void_p(bias_hh),
        _as_void_p(y_host),
        gate_cache_struct,
        ctypes.byref(gate_cache_options),
        _as_void_p(hy_device),
        _as_void_p(cy_device),
        ctypes.c_void_p(compute_stream.cuda_stream),
        ctypes.c_void_p(h2d_stream.cuda_stream),
        ctypes.c_void_p(d2h_stream.cuda_stream),
    )

    torch.cuda.synchronize()

    y_ref_cpu = y_ref.cpu()
    y_custom_eff = y_host
    if oversample_factor > 1:
        y_custom_eff = y_custom_eff.view(cfg.time_steps, oversample_factor, cfg.batch_size, cfg.hidden_size)[:, -1, ...].contiguous()
    y_custom = y_custom_eff.to(dtype=torch.float32)
    h_states_custom = y_custom
    h_states_ref_cpu = h_states_ref.cpu()
    h_custom = hy_device.to(dtype=torch.float32).cpu()
    c_custom = cy_device.to(dtype=torch.float32).cpu()
    h_ref = h_n_ref.squeeze(0).cpu()
    c_ref = c_n_ref.squeeze(0).cpu()

    tol_atol = 5e-2
    tol_rtol = 5e-2
    torch.testing.assert_close(y_custom, y_ref_cpu, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(h_states_custom, h_states_ref_cpu, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(h_custom, h_ref, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(c_custom, c_ref, atol=tol_atol, rtol=tol_rtol)

    y_delta = (y_custom - y_ref_cpu).abs().max().item()
    h_state_delta = (h_states_custom - h_states_ref_cpu).abs().max().item()
    h_delta = (h_custom - h_ref).abs().max().item()
    c_delta = (c_custom - c_ref).abs().max().item()

    return y_delta, h_delta, c_delta, h_state_delta


def _gather_cases() -> Iterable[LstmConfig]:
    return (
        LstmConfig(4, 2, 3, 5, 3, True),
        LstmConfig(16, 8, 64, 32),
        LstmConfig(32, 4, 128, 16),
        LstmConfig(64, 32, 256, 256),
        LstmConfig(2048, 32, 1024, 1024),
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available – cannot run validation")

    root = Path(__file__).resolve().parents[1]
    lib_path = _find_library(root)
    lib = ctypes.CDLL(str(lib_path))
    _prepare_function(lib)

    print(f"Loaded {lib_path}")

    for cfg in _gather_cases():
        y_diff, h_diff, c_diff, hs_diff = _run_case(lib, cfg)
        print(
            f"[PASS] {cfg.describe()} :: "
            f"max|Δy|={y_diff:.3e}, max|Δh|={h_diff:.3e}, max|Δc|={c_diff:.3e}, "
            f"max|Δh_t|={hs_diff:.3e}"
        )

    print("All configurations validated successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
