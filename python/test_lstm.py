#!/usr/bin/env python3
import argparse
import ctypes
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

LSTM_COMPUTE_PRECISION_FP16_ACC32 = 0
LSTM_COMPUTE_PRECISION_FP16_ACC16 = 1


class LstmBuffers(ctypes.Structure):
    _fields_ = [("impl", ctypes.c_void_p)]


def resolve_library_path(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Specified library path does not exist: {path}")
        return path

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "build" / "libflashlstm.so",
        root / "build" / "libflashlstm.dylib",
        root / "build" / "flashlstm.dll",
        root / "cmake-build-debug" / "libflashlstm.so",
        root / "cmake-build-release" / "libflashlstm.so",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate flashlstm shared library. "
        "Build the project or provide --lib-path explicitly."
    )


def load_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.lstm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(LstmBuffers),
        ctypes.c_int,
    ]
    lib.lstm_forward.restype = ctypes.c_int
    lib.lstm_create_buffers.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(LstmBuffers),
    ]
    lib.lstm_create_buffers.restype = ctypes.c_int
    lib.lstm_destroy_buffers.argtypes = [ctypes.POINTER(LstmBuffers)]
    lib.lstm_destroy_buffers.restype = ctypes.c_int
    lib.lstm_pack_weights.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(LstmBuffers),
    ]
    lib.lstm_pack_weights.restype = ctypes.c_int
    return lib


def run_single_case(
        lib: ctypes.CDLL,
        seq_len: int,
        batch: int,
        input_size: int,
        hidden_size: int,
        seed: int,
        atol: float,
        rtol: float,
) -> None:
    torch.manual_seed(seed)

    lstm_cpu = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size).eval()

    with torch.no_grad():
        x = torch.randn(seq_len, batch, input_size, dtype=torch.float32)
        h0 = torch.randn(1, batch, hidden_size, dtype=torch.float32)
        c0 = torch.randn(1, batch, hidden_size, dtype=torch.float32)
        torch_out, (torch_hn, torch_cn) = lstm_cpu(x, (h0, c0))

    lstm_cuda = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size).cuda().eval()
    lstm_cuda.load_state_dict(lstm_cpu.state_dict())

    x_cuda = x.cuda().contiguous()
    h0_cuda = h0.cuda().contiguous()
    c0_cuda = c0.cuda().contiguous()

    x_dev = x_cuda.view(-1)
    h0_dev = h0_cuda.view(-1)
    c0_dev = c0_cuda.view(-1)

    weight_ih_dev = lstm_cuda.weight_ih_l0.detach().contiguous()
    weight_hh_dev = lstm_cuda.weight_hh_l0.detach().contiguous()
    b_ih_dev = lstm_cuda.bias_ih_l0.detach().contiguous().view(-1)
    b_hh_dev = lstm_cuda.bias_hh_l0.detach().contiguous().view(-1)

    output_dev = torch.empty(seq_len * batch * hidden_size, dtype=torch.float32, device="cuda")
    hn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
    cn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")

    input_proj_chsz = 64

    buffers = LstmBuffers()
    buffers_ptr = ctypes.byref(buffers)
    status = lib.lstm_create_buffers(
        LSTM_COMPUTE_PRECISION_FP16_ACC16,
        seq_len,
        batch,
        input_size,
        hidden_size,
        input_proj_chsz,
        buffers_ptr,
    )
    if status != 0:
        raise RuntimeError(f"lstm_create_buffers failed with error code {status}")

    try:
        status = lib.lstm_pack_weights(
            LSTM_COMPUTE_PRECISION_FP16_ACC16,
            ctypes.c_void_p(weight_ih_dev.data_ptr()),
            ctypes.c_void_p(weight_hh_dev.data_ptr()),
            input_size,
            hidden_size,
            buffers_ptr,
        )
        if status != 0:
            raise RuntimeError(f"lstm_pack_weights failed with error code {status}")

        status = lib.lstm_forward(
            ctypes.c_void_p(x_dev.data_ptr()),
            ctypes.c_void_p(b_ih_dev.data_ptr()),
            ctypes.c_void_p(b_hh_dev.data_ptr()),
            ctypes.c_void_p(h0_dev.data_ptr()),
            ctypes.c_void_p(c0_dev.data_ptr()),
            ctypes.c_void_p(output_dev.data_ptr()),
            ctypes.c_void_p(hn_dev.data_ptr()),
            ctypes.c_void_p(cn_dev.data_ptr()),
            seq_len,
            batch,
            input_size,
            hidden_size,
            buffers_ptr,
            LSTM_COMPUTE_PRECISION_FP16_ACC16,
        )
        if status != 0:
            raise RuntimeError(f"lstm_forward failed with error code {status}")

        torch.cuda.synchronize()

        output_np = output_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
        hn_np = hn_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
        cn_np = cn_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)

        output_ref = torch_out.detach().cpu().numpy().reshape(-1).astype(np.float32)
        hn_ref = torch_hn[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
        cn_ref = torch_cn[0].detach().cpu().numpy().reshape(-1).astype(np.float32)

        np.testing.assert_allclose(output_np, output_ref, rtol=rtol, atol=atol)

        if seq_len > 1:
            seq_len_small = max(1, seq_len // 2)
            with torch.no_grad():
                x_small = torch.randn(seq_len_small, batch, input_size, dtype=torch.float32)
                torch_out_small, (torch_hn_small, torch_cn_small) = lstm_cpu(x_small, (h0, c0))
            x_small_dev = x_small.cuda().contiguous()
            x_small_flat = x_small_dev.view(-1)
            output_small_dev = torch.empty(seq_len_small * batch * hidden_size, dtype=torch.float32, device="cuda")
            hn_small_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
            cn_small_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")

            status = lib.lstm_forward(
                ctypes.c_void_p(x_small_flat.data_ptr()),
                ctypes.c_void_p(b_ih_dev.data_ptr()),
                ctypes.c_void_p(b_hh_dev.data_ptr()),
                ctypes.c_void_p(h0_dev.data_ptr()),
                ctypes.c_void_p(c0_dev.data_ptr()),
                ctypes.c_void_p(output_small_dev.data_ptr()),
                ctypes.c_void_p(hn_small_dev.data_ptr()),
                ctypes.c_void_p(cn_small_dev.data_ptr()),
                seq_len_small,
                batch,
                input_size,
                hidden_size,
                buffers_ptr,
                LSTM_COMPUTE_PRECISION_FP16_ACC16,
            )
            if status != 0:
                raise RuntimeError(f"lstm_forward (small) failed with error code {status}")

            torch.cuda.synchronize()

            output_small_np = output_small_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
            hn_small_np = hn_small_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
            cn_small_np = cn_small_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)

            output_small_ref = torch_out_small.detach().cpu().numpy().reshape(-1).astype(np.float32)
            hn_small_ref = torch_hn_small[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
            cn_small_ref = torch_cn_small[0].detach().cpu().numpy().reshape(-1).astype(np.float32)

            np.testing.assert_allclose(output_small_np, output_small_ref, rtol=rtol, atol=atol)
            np.testing.assert_allclose(hn_small_np, hn_small_ref, rtol=rtol, atol=atol)
            np.testing.assert_allclose(cn_small_np, cn_small_ref, rtol=rtol, atol=atol)

        seq_len_large = seq_len + 1
        with torch.no_grad():
            x_large = torch.randn(seq_len_large, batch, input_size, dtype=torch.float32)
            torch_out_large, (torch_hn_large, torch_cn_large) = lstm_cpu(x_large, (h0, c0))
        x_large_dev = x_large.cuda().contiguous()
        x_large_flat = x_large_dev.view(-1)
        output_large_dev = torch.empty(seq_len_large * batch * hidden_size, dtype=torch.float32, device="cuda")
        hn_large_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
        cn_large_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")

        status = lib.lstm_forward(
            ctypes.c_void_p(x_large_flat.data_ptr()),
            ctypes.c_void_p(b_ih_dev.data_ptr()),
            ctypes.c_void_p(b_hh_dev.data_ptr()),
            ctypes.c_void_p(h0_dev.data_ptr()),
            ctypes.c_void_p(c0_dev.data_ptr()),
            ctypes.c_void_p(output_large_dev.data_ptr()),
            ctypes.c_void_p(hn_large_dev.data_ptr()),
            ctypes.c_void_p(cn_large_dev.data_ptr()),
            seq_len_large,
            batch,
            input_size,
            hidden_size,
            buffers_ptr,
            LSTM_COMPUTE_PRECISION_FP16_ACC16,
        )
        if status != 0:
            raise RuntimeError(f"lstm_forward (large) failed with error code {status}")

        torch.cuda.synchronize()

        output_large_np = output_large_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
        hn_large_np = hn_large_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
        cn_large_np = cn_large_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)

        output_large_ref = torch_out_large.detach().cpu().numpy().reshape(-1).astype(np.float32)
        hn_large_ref = torch_hn_large[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
        cn_large_ref = torch_cn_large[0].detach().cpu().numpy().reshape(-1).astype(np.float32)

        np.testing.assert_allclose(output_large_np, output_large_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(hn_large_np, hn_large_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(cn_large_np, cn_large_ref, rtol=rtol, atol=atol)
    finally:
        lib.lstm_destroy_buffers(buffers_ptr)


def build_test_plan(
        mode: str, random_cases: int, max_dims: Tuple[int, int, int, int], seed: int
) -> List[Tuple[int, int, int, int, int]]:
    rng = np.random.default_rng(seed)
    if mode == "quick":
        deterministic: Sequence[Tuple[int, int, int, int]] = [
            (1, 1, 4, 4),
            (2, 2, 8, 8),
            (4, 3, 16, 16),
        ]
    elif mode == "standard":
        deterministic = [
            (1, 1, 4, 4),
            (3, 2, 8, 8),
            (5, 4, 16, 16),
            (7, 2, 32, 24),
            (10, 3, 24, 32),
        ]
    elif mode == "stress":
        deterministic = [
            (1, 1, 4, 4),
            (4, 4, 32, 32),
            (8, 8, 48, 64),
            (12, 6, 64, 32),
            (16, 4, 96, 128),
            (32, 2, 128, 96),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    plan: List[Tuple[int, int, int, int, int]] = []
    for idx, dims in enumerate(deterministic):
        plan.append((*dims, seed + idx))

    max_seq, max_batch, max_input, max_hidden = max_dims
    for i in range(random_cases):
        seq = int(rng.integers(1, max_seq + 1))
        batch = int(rng.integers(1, max_batch + 1))
        input_size = int(rng.integers(1, max_input + 1))
        hidden_size = int(rng.integers(1, max_hidden + 1))
        plan.append((seq, batch, input_size, hidden_size, seed + len(plan) + i))
    return plan


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CUDA LSTM against PyTorch.")
    parser.add_argument(
        "--lib-path",
        type=str,
        default=None,
        help="Path to flashlstm shared library (defaults to build/libflashlstm.*).",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "stress"],
        default="standard",
        help="Select preset for deterministic test coverage.",
    )
    parser.add_argument(
        "--random-cases",
        type=int,
        default=4,
        help="Number of additional random configurations to test.",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=24,
        help="Maximum sequence length for random cases.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Maximum batch size for random cases.",
    )
    parser.add_argument(
        "--max-input",
        type=int,
        default=64,
        help="Maximum input size for random cases.",
    )
    parser.add_argument(
        "--max-hidden",
        type=int,
        default=64,
        help="Maximum hidden size for random cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed used for deterministic and random tests.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for parity checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-2,
        help="Relative tolerance for parity checks.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run tests.")

    lib_path = resolve_library_path(args.lib_path)
    lib = load_library(lib_path)

    plan = build_test_plan(
        args.mode,
        args.random_cases,
        (args.max_seq, args.max_batch, args.max_input, args.max_hidden),
        args.seed,
    )

    for idx, (seq_len, batch, input_size, hidden_size, seed) in enumerate(plan, start=1):
        print(
            f"[{idx}/{len(plan)}] Testing seq={seq_len}, batch={batch}, "
            f"input={input_size}, hidden={hidden_size}, seed={seed}"
        )
        run_single_case(
            lib,
            seq_len,
            batch,
            input_size,
            hidden_size,
            seed,
            args.atol,
            args.rtol,
        )

    print(f"All {len(plan)} test cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
