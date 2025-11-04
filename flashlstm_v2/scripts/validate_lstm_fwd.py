from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


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


@dataclass(frozen=True)
class LstmConfig:
    time_steps: int
    batch_size: int
    input_size: int
    hidden_size: int

    def describe(self) -> str:
        return (
            f"T={self.time_steps}, B={self.batch_size}, "
            f"I={self.input_size}, H={self.hidden_size}"
        )


def _prepare_function(lib: ctypes.CDLL) -> None:
    lib.flstm_StreamingLstmForward.restype = None
    lib.flstm_StreamingLstmForward.argtypes = [
        ctypes.c_size_t,  # time_steps
        ctypes.c_size_t,  # batch
        ctypes.c_size_t,  # input
        ctypes.c_size_t,  # hidden
        ctypes.c_void_p,  # x host
        ctypes.c_void_p,  # h0 device
        ctypes.c_void_p,  # c0 device
        ctypes.c_void_p,  # weight_ih
        ctypes.c_void_p,  # weight_hh
        ctypes.c_void_p,  # bias_ih
        ctypes.c_void_p,  # bias_hh
        ctypes.c_void_p,  # y host
        ctypes.c_void_p,  # z cache device
        ctypes.c_void_p,  # h cache device
        ctypes.c_void_p,  # c cache device
        ctypes.c_void_p,  # gate cache device
        ctypes.c_void_p,  # compute stream
        ctypes.c_void_p,  # h2d stream
        ctypes.c_void_p,  # d2h stream
    ]


def _run_case(lib: ctypes.CDLL, cfg: LstmConfig):
    torch.manual_seed(0)
    device = torch.device("cuda")

    lstm = torch.nn.LSTM(cfg.input_size, cfg.hidden_size, batch_first=False).to(device)
    lstm.eval()

    x_fp32 = torch.randn(cfg.time_steps, cfg.batch_size, cfg.input_size, dtype=torch.float32).contiguous()
    x_host = x_fp32.to(dtype=torch.float16).contiguous().pin_memory()
    x_torch = x_fp32.to(device)

    h0_init = torch.randn(1, cfg.batch_size, cfg.hidden_size, dtype=torch.float32)
    c0_init = torch.randn(1, cfg.batch_size, cfg.hidden_size, dtype=torch.float32)

    h0_torch = h0_init.to(device)
    c0_torch = c0_init.to(device)

    h0_device = h0_init.squeeze(0).to(device=device, dtype=torch.float16).contiguous()
    c0_device = c0_init.squeeze(0).to(device=device, dtype=torch.float16).contiguous()

    with torch.no_grad():
        y_ref, (h_n_ref, c_n_ref) = lstm(x_torch, (h0_torch, c0_torch))

    lstm_cell = torch.nn.LSTMCell(cfg.input_size, cfg.hidden_size).to(device)
    lstm_cell.eval()
    with torch.no_grad():
        lstm_cell.weight_ih.copy_(lstm.weight_ih_l0)
        lstm_cell.weight_hh.copy_(lstm.weight_hh_l0)
        lstm_cell.bias_ih.copy_(lstm.bias_ih_l0)
        lstm_cell.bias_hh.copy_(lstm.bias_hh_l0)

    with torch.no_grad():
        h_cell = h0_torch.squeeze(0).clone()
        c_cell = c0_torch.squeeze(0).clone()
        h_states_ref = []
        c_states_ref = []
        gate_states_ref = []
        for t in range(cfg.time_steps):
            x_step = x_torch[t]
            linear_input = F.linear(x_step, lstm.weight_ih_l0)
            linear_hidden = F.linear(h_cell, lstm.weight_hh_l0)
            gates_linear = linear_input + linear_hidden + lstm.bias_ih_l0 + lstm.bias_hh_l0
            gi, gf, gg, go = gates_linear.chunk(4, dim=1)
            i_act = torch.sigmoid(gi)
            f_act = torch.sigmoid(gf)
            g_act = torch.tanh(gg)
            o_act = torch.sigmoid(go)
            gate_states_ref.append(torch.cat((i_act, f_act, g_act, o_act), dim=1).unsqueeze(0))
            h_cell, c_cell = lstm_cell(x_step, (h_cell, c_cell))
            h_states_ref.append(h_cell.unsqueeze(0))
            c_states_ref.append(c_cell.unsqueeze(0))
        h_states_ref = torch.cat(h_states_ref, dim=0)
        c_states_ref = torch.cat(c_states_ref, dim=0)
        gate_states_ref = torch.cat(gate_states_ref, dim=0)

    weight_ih = lstm.weight_ih_l0.detach().clone().contiguous().to(device)
    weight_hh = lstm.weight_hh_l0.detach().clone().contiguous().to(device)
    bias_ih = lstm.bias_ih_l0.detach().clone().contiguous().to(device)
    bias_hh = lstm.bias_hh_l0.detach().clone().contiguous().to(device)

    y_host = torch.empty(cfg.time_steps, cfg.batch_size, cfg.hidden_size, dtype=torch.float16).contiguous().pin_memory()

    z_cache = torch.empty(cfg.input_size + cfg.hidden_size, cfg.time_steps * cfg.batch_size,
                          device=device, dtype=torch.float16)
    h_cache = torch.empty(cfg.time_steps + 1, cfg.batch_size, cfg.hidden_size,
                          device=device, dtype=torch.float16)
    c_cache = torch.empty_like(h_cache)
    gate_cache = torch.empty(cfg.time_steps, cfg.batch_size, 4 * cfg.hidden_size,
                             device=device, dtype=torch.float16)

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
        ctypes.c_size_t(cfg.time_steps),
        ctypes.c_size_t(cfg.batch_size),
        ctypes.c_size_t(cfg.input_size),
        ctypes.c_size_t(cfg.hidden_size),
        _as_void_p(x_host),
        _as_void_p(h0_device),
        _as_void_p(c0_device),
        _as_void_p(weight_ih),
        _as_void_p(weight_hh),
        _as_void_p(bias_ih),
        _as_void_p(bias_hh),
        _as_void_p(y_host),
        _as_void_p(z_cache),
        _as_void_p(h_cache),
        _as_void_p(c_cache),
        _as_void_p(gate_cache),
        ctypes.c_void_p(0),
        ctypes.c_void_p(h2d_stream.cuda_stream),
        ctypes.c_void_p(d2h_stream.cuda_stream),
    )

    torch.cuda.synchronize()

    y_ref_cpu = y_ref.cpu()
    y_custom = y_host.to(dtype=torch.float32)
    h_states_custom = h_cache[1:].to(dtype=torch.float32).cpu()
    c_states_custom = c_cache[1:].to(dtype=torch.float32).cpu()
    h_states_ref_cpu = h_states_ref.cpu()
    c_states_ref_cpu = c_states_ref.cpu()
    h_custom = h_states_custom[-1].cpu()
    c_custom = c_states_custom[-1].cpu()
    h_ref = h_n_ref.squeeze(0).cpu()
    c_ref = c_n_ref.squeeze(0).cpu()

    tol_atol = 5e-2
    tol_rtol = 5e-2
    torch.testing.assert_close(y_custom, y_ref_cpu, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(h_states_custom, h_states_ref_cpu, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(c_states_custom, c_states_ref_cpu, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(h_custom, h_ref, atol=tol_atol, rtol=tol_rtol)
    torch.testing.assert_close(c_custom, c_ref, atol=tol_atol, rtol=tol_rtol)

    y_delta = (y_custom - y_ref_cpu).abs().max().item()
    h_state_delta = (h_states_custom - h_states_ref_cpu).abs().max().item()
    c_state_delta = (c_states_custom - c_states_ref_cpu).abs().max().item()
    h_delta = (h_custom - h_ref).abs().max().item()
    c_delta = (c_custom - c_ref).abs().max().item()

    return y_delta, h_delta, c_delta, h_state_delta, c_state_delta


def _gather_cases() -> Iterable[LstmConfig]:
    return (
        LstmConfig(4, 2, 3, 5),
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
        y_diff, h_diff, c_diff, hs_diff, cs_diff = _run_case(lib, cfg)
        print(
            f"[PASS] {cfg.describe()} :: "
            f"max|Δy|={y_diff:.3e}, max|Δh|={h_diff:.3e}, max|Δc|={c_diff:.3e}, "
            f"max|Δh_t|={hs_diff:.3e}, max|Δc_t|={cs_diff:.3e}"
        )

    print("All configurations validated successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
