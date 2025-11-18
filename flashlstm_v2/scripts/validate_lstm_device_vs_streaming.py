from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

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
        ("time_oversample", ctypes.c_int),
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
            f"I={self.input_size}, H={self.hidden_size}, S={self.weight_sets}, "
            f"oversample={self.time_oversample}"
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

    lib.flstm_LstmForward.restype = None
    lib.flstm_LstmForward.argtypes = [
        ctypes.c_size_t,  # time_steps
        ctypes.c_size_t,  # batch
        ctypes.c_size_t,  # input
        ctypes.c_size_t,  # hidden
        ctypes.c_size_t,  # recompute_interval
        ctypes.c_size_t,  # weight_set_count
        ctypes.c_void_p,  # x device
        ctypes.c_void_p,  # h0 device
        ctypes.c_void_p,  # c0 device
        ctypes.c_void_p,  # weight_ih
        ctypes.c_void_p,  # weight_hh
        ctypes.c_void_p,  # bias_ih
        ctypes.c_void_p,  # bias_hh
        ctypes.c_void_p,  # y device
        GateCacheHost,    # gate cache device
        ctypes.POINTER(StreamingLstmOptions),  # gate cache options
        ctypes.c_void_p,  # hy device
        ctypes.c_void_p,  # cy device
        ctypes.c_void_p,  # compute stream
        ctypes.c_void_p,  # h2d stream
        ctypes.c_void_p,  # d2h stream
    ]

    lib.flstm_StreamingLstmBackward.restype = None
    lib.flstm_StreamingLstmBackward.argtypes = [
        ctypes.c_size_t,  # time_steps
        ctypes.c_size_t,  # batch
        ctypes.c_size_t,  # input
        ctypes.c_size_t,  # hidden
        ctypes.c_size_t,  # recompute_interval
        ctypes.c_size_t,  # weight_set_count
        ctypes.c_void_p,  # x host
        ctypes.c_void_p,  # y host
        GateCacheHost,    # gate cache host
        ctypes.c_void_p,  # dY host
        ctypes.c_void_p,  # d_hn device
        ctypes.c_void_p,  # d_cn device
        ctypes.c_void_p,  # h0 device
        ctypes.c_void_p,  # c0 device
        ctypes.c_void_p,  # weight_ih
        ctypes.c_void_p,  # weight_hh
        ctypes.c_void_p,  # bias_ih
        ctypes.c_void_p,  # bias_hh
        ctypes.c_void_p,  # dx host
        ctypes.c_void_p,  # dW_ih
        ctypes.c_void_p,  # dW_hh
        ctypes.c_void_p,  # db_ih
        ctypes.c_void_p,  # db_hh
        ctypes.c_void_p,  # dh0
        ctypes.c_void_p,  # dc0
        ctypes.c_void_p,  # compute stream
        ctypes.c_void_p,  # h2d stream
        ctypes.c_void_p,  # d2h stream
        ctypes.POINTER(StreamingLstmOptions),  # gate cache options
    ]

    lib.flstm_LstmBackward.restype = None
    lib.flstm_LstmBackward.argtypes = [
        ctypes.c_size_t,  # time_steps
        ctypes.c_size_t,  # batch
        ctypes.c_size_t,  # input
        ctypes.c_size_t,  # hidden
        ctypes.c_size_t,  # recompute_interval
        ctypes.c_size_t,  # weight_set_count
        ctypes.c_void_p,  # x device
        ctypes.c_void_p,  # y device
        GateCacheHost,    # gate cache device
        ctypes.c_void_p,  # dY device
        ctypes.c_void_p,  # d_hn device
        ctypes.c_void_p,  # d_cn device
        ctypes.c_void_p,  # h0 device
        ctypes.c_void_p,  # c0 device
        ctypes.c_void_p,  # weight_ih
        ctypes.c_void_p,  # weight_hh
        ctypes.c_void_p,  # bias_ih
        ctypes.c_void_p,  # bias_hh
        ctypes.c_void_p,  # dx device
        ctypes.c_void_p,  # dW_ih
        ctypes.c_void_p,  # dW_hh
        ctypes.c_void_p,  # db_ih
        ctypes.c_void_p,  # db_hh
        ctypes.c_void_p,  # dh0
        ctypes.c_void_p,  # dc0
        ctypes.c_void_p,  # compute stream
        ctypes.c_void_p,  # h2d stream
        ctypes.c_void_p,  # d2h stream
        ctypes.POINTER(StreamingLstmOptions),  # gate cache options
    ]


def _alloc_params(cfg: LstmConfig, device: torch.device):
    gate_dim = 4 * cfg.hidden_size
    std = 1.0 / math.sqrt(cfg.hidden_size)
    weight_ih = torch.empty(
        cfg.weight_sets, gate_dim, cfg.input_size, device=device, dtype=torch.float32
    ).uniform_(-std, std)
    weight_hh = torch.empty(
        cfg.weight_sets, gate_dim, cfg.hidden_size, device=device, dtype=torch.float32
    ).uniform_(-std, std)
    bias_ih = torch.zeros(cfg.weight_sets, gate_dim, device=device, dtype=torch.float32)
    bias_hh = torch.zeros_like(bias_ih)
    return weight_ih, weight_hh, bias_ih, bias_hh


def _run_case(lib: ctypes.CDLL, cfg: LstmConfig):
    torch.manual_seed(0)
    device = torch.device("cuda")
    weight_set_count = cfg.weight_sets
    time_oversample = cfg.time_oversample and weight_set_count > 1

    # Inputs
    x_fp32 = torch.randn(cfg.time_steps, cfg.batch_size, cfg.input_size, dtype=torch.float32).contiguous()
    x_host = x_fp32.to(dtype=torch.float16).contiguous().pin_memory()
    x_device = x_host.to(device=device)
    h0 = torch.randn(cfg.batch_size, cfg.hidden_size, device=device, dtype=torch.float16)
    c0 = torch.randn_like(h0)

    weight_ih, weight_hh, bias_ih, bias_hh = _alloc_params(cfg, device)

    # Streams
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_c = torch.cuda.Stream()

    # Gate cache + output buffers (streaming)
    checkpoint_steps = (cfg.time_steps + RECOMPUTE - 1) // RECOMPUTE
    gate_cache_h_host = torch.empty(
        checkpoint_steps,
        cfg.batch_size,
        cfg.hidden_size,
        dtype=torch.float16,
    ).contiguous().pin_memory()
    gate_cache_c_host = torch.empty(
        checkpoint_steps,
        cfg.batch_size,
        cfg.hidden_size,
        dtype=torch.float32,
    ).contiguous().pin_memory()
    gate_cache_host_struct = GateCacheHost(_as_void_p(gate_cache_h_host), _as_void_p(gate_cache_c_host))
    gate_cache_opts = StreamingLstmOptions(
        _gate_dtype_enum(gate_cache_h_host.dtype),
        _gate_dtype_enum(gate_cache_c_host.dtype),
        int(time_oversample),
    )
    y_host = torch.empty(cfg.time_steps, cfg.batch_size, cfg.hidden_size, dtype=torch.float16).contiguous().pin_memory()
    hy_stream = torch.empty(cfg.batch_size, cfg.hidden_size, dtype=torch.float16, device=device)
    cy_stream = torch.empty_like(hy_stream)

    lib.flstm_StreamingLstmForward(
        ctypes.c_size_t(cfg.time_steps),
        ctypes.c_size_t(cfg.batch_size),
        ctypes.c_size_t(cfg.input_size),
        ctypes.c_size_t(cfg.hidden_size),
        ctypes.c_size_t(RECOMPUTE),
        ctypes.c_size_t(weight_set_count),
        _as_void_p(x_host),
        _as_void_p(h0),
        _as_void_p(c0),
        _as_void_p(weight_ih),
        _as_void_p(weight_hh),
        _as_void_p(bias_ih),
        _as_void_p(bias_hh),
        _as_void_p(y_host),
        gate_cache_host_struct,
        ctypes.byref(gate_cache_opts),
        _as_void_p(hy_stream),
        _as_void_p(cy_stream),
        ctypes.c_void_p(stream_a.cuda_stream),
        ctypes.c_void_p(stream_b.cuda_stream),
        ctypes.c_void_p(stream_c.cuda_stream),
    )

    # Device forward
    gate_cache_h_dev = torch.empty_like(gate_cache_h_host, device=device)
    gate_cache_c_dev = torch.empty_like(gate_cache_c_host, device=device)
    gate_cache_dev_struct = GateCacheHost(_as_void_p(gate_cache_h_dev), _as_void_p(gate_cache_c_dev))
    y_device = torch.empty(cfg.time_steps, cfg.batch_size, cfg.hidden_size, device=device, dtype=torch.float16)
    hy_device = torch.empty_like(hy_stream)
    cy_device = torch.empty_like(hy_stream)

    lib.flstm_LstmForward(
        ctypes.c_size_t(cfg.time_steps),
        ctypes.c_size_t(cfg.batch_size),
        ctypes.c_size_t(cfg.input_size),
        ctypes.c_size_t(cfg.hidden_size),
        ctypes.c_size_t(RECOMPUTE),
        ctypes.c_size_t(weight_set_count),
        _as_void_p(x_device),
        _as_void_p(h0),
        _as_void_p(c0),
        _as_void_p(weight_ih),
        _as_void_p(weight_hh),
        _as_void_p(bias_ih),
        _as_void_p(bias_hh),
        _as_void_p(y_device),
        gate_cache_dev_struct,
        ctypes.byref(gate_cache_opts),
        _as_void_p(hy_device),
        _as_void_p(cy_device),
        ctypes.c_void_p(stream_a.cuda_stream),
        ctypes.c_void_p(stream_b.cuda_stream),
        ctypes.c_void_p(stream_c.cuda_stream),
    )

    torch.cuda.synchronize()

    # Forward comparisons
    y_stream = y_host.to(device=device, dtype=torch.float32)
    h_stream = hy_stream.to(dtype=torch.float32)
    c_stream = cy_stream.to(dtype=torch.float32)
    y_dev = y_device.to(dtype=torch.float32)
    h_dev = hy_device.to(dtype=torch.float32)
    c_dev = cy_device.to(dtype=torch.float32)

    torch.testing.assert_close(y_stream, y_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(h_stream, h_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(c_stream, c_dev, atol=5e-2, rtol=5e-2)

    # Upstream grads
    torch.manual_seed(1)
    dY_host = torch.randn_like(y_host)
    dY_dev = dY_host.to(device=device)
    d_hn = torch.randn_like(hy_stream)
    d_cn = torch.randn_like(cy_stream)

    dx_host = torch.empty_like(x_host)
    dW_ih_host = torch.zeros_like(weight_ih)
    dW_hh_host = torch.zeros_like(weight_hh)
    db_ih_host = torch.zeros_like(bias_ih)
    db_hh_host = torch.zeros_like(bias_hh)
    dh0_host = torch.empty_like(h0, dtype=torch.float32)
    dc0_host = torch.empty_like(c0, dtype=torch.float32)

    lib.flstm_StreamingLstmBackward(
        ctypes.c_size_t(cfg.time_steps),
        ctypes.c_size_t(cfg.batch_size),
        ctypes.c_size_t(cfg.input_size),
        ctypes.c_size_t(cfg.hidden_size),
        ctypes.c_size_t(RECOMPUTE),
        ctypes.c_size_t(weight_set_count),
        _as_void_p(x_host),
        _as_void_p(y_host),
        gate_cache_host_struct,
        _as_void_p(dY_host),
        _as_void_p(d_hn),
        _as_void_p(d_cn),
        _as_void_p(h0),
        _as_void_p(c0),
        _as_void_p(weight_ih),
        _as_void_p(weight_hh),
        _as_void_p(bias_ih),
        _as_void_p(bias_hh),
        _as_void_p(dx_host),
        _as_void_p(dW_ih_host),
        _as_void_p(dW_hh_host),
        _as_void_p(db_ih_host),
        _as_void_p(db_hh_host),
        _as_void_p(dh0_host),
        _as_void_p(dc0_host),
        ctypes.c_void_p(stream_a.cuda_stream),
        ctypes.c_void_p(stream_b.cuda_stream),
        ctypes.c_void_p(stream_c.cuda_stream),
        ctypes.byref(gate_cache_opts),
    )

    dx_dev = torch.empty_like(x_device)
    dW_ih_dev = torch.zeros_like(weight_ih)
    dW_hh_dev = torch.zeros_like(weight_hh)
    db_ih_dev = torch.zeros_like(bias_ih)
    db_hh_dev = torch.zeros_like(bias_hh)
    dh0_dev = torch.empty_like(h0, dtype=torch.float32)
    dc0_dev = torch.empty_like(c0, dtype=torch.float32)

    lib.flstm_LstmBackward(
        ctypes.c_size_t(cfg.time_steps),
        ctypes.c_size_t(cfg.batch_size),
        ctypes.c_size_t(cfg.input_size),
        ctypes.c_size_t(cfg.hidden_size),
        ctypes.c_size_t(RECOMPUTE),
        ctypes.c_size_t(weight_set_count),
        _as_void_p(x_device),
        _as_void_p(y_device),
        gate_cache_dev_struct,
        _as_void_p(dY_dev),
        _as_void_p(d_hn),
        _as_void_p(d_cn),
        _as_void_p(h0),
        _as_void_p(c0),
        _as_void_p(weight_ih),
        _as_void_p(weight_hh),
        _as_void_p(bias_ih),
        _as_void_p(bias_hh),
        _as_void_p(dx_dev),
        _as_void_p(dW_ih_dev),
        _as_void_p(dW_hh_dev),
        _as_void_p(db_ih_dev),
        _as_void_p(db_hh_dev),
        _as_void_p(dh0_dev),
        _as_void_p(dc0_dev),
        ctypes.c_void_p(stream_a.cuda_stream),
        ctypes.c_void_p(stream_b.cuda_stream),
        ctypes.c_void_p(stream_c.cuda_stream),
        ctypes.byref(gate_cache_opts),
    )

    torch.cuda.synchronize()

    # Backward comparisons
    torch.testing.assert_close(dx_host.to(device=device), dx_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dW_ih_host, dW_ih_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dW_hh_host, dW_hh_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(db_ih_host, db_ih_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(db_hh_host, db_hh_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dh0_host, dh0_dev, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dc0_host, dc0_dev, atol=5e-2, rtol=5e-2)

    y_delta = (y_stream - y_dev).abs().max().item()
    dx_delta = (dx_host.to(device=device) - dx_dev).abs().max().item()
    grad_delta = max(
        (dW_ih_host - dW_ih_dev).abs().max().item(),
        (dW_hh_host - dW_hh_dev).abs().max().item(),
        (db_ih_host - db_ih_dev).abs().max().item(),
        (db_hh_host - db_hh_dev).abs().max().item(),
    )

    return y_delta, dx_delta, grad_delta


def _gather_cases() -> Iterable[LstmConfig]:
    return (
        LstmConfig(4, 2, 3, 5, 2),
        LstmConfig(16, 8, 64, 32),
        LstmConfig(64, 16, 128, 64),
        LstmConfig(12, 4, 16, 8, 2, True),
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
        y_delta, dx_delta, grad_delta = _run_case(lib, cfg)
        print(
            f"[PASS] {cfg.describe()} :: max|Δy|={y_delta:.3e}, max|Δdx|={dx_delta:.3e}, max|Δgrad|={grad_delta:.3e}"
        )

    print("Streaming and device LSTM paths match.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
