#!/usr/bin/env python3
"""
Benchmark the custom CUDA LSTM implementation against PyTorch's nn.LSTM.
"""

import argparse
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

LSTM_COMPUTE_PRECISION_FP16_ACC32 = 0
LSTM_COMPUTE_PRECISION_FP16_ACC16 = 1

NUM_ATTENTION_HEADS = 8
ATTENTION_HEAD_DIM = 64
CHUNK_SIZE_CANDIDATES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
AUTOTUNE_WARMUP_ITERS = 5
AUTOTUNE_TIMED_ITERS = 25


class LstmBuffers(ctypes.Structure):
    _fields_ = [("impl", ctypes.c_void_p)]


@dataclass
class BenchmarkResult:
    implementation: str
    avg_ms: float
    std_ms: float

    @property
    def iters_per_second(self) -> float:
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else float("inf")


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
    raise FileNotFoundError("Unable to locate flashlstm shared library. "
                            "Build the project or provide --lib-path explicitly.")


def load_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.lstm_forward.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # b_ih
        ctypes.c_void_p,  # b_hh
        ctypes.c_void_p,  # h0
        ctypes.c_void_p,  # c0
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # hn
        ctypes.c_void_p,  # cn
        ctypes.c_size_t,  # seq_len
        ctypes.c_size_t,  # batch
        ctypes.c_size_t,  # input_size
        ctypes.c_size_t,  # hidden_size
        ctypes.POINTER(LstmBuffers),  # buffers
        ctypes.c_int,  # precision
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


def benchmark_flashlstm(lib: ctypes.CDLL,
                        x_dev: torch.Tensor,
                        weight_ih_dev: torch.Tensor,
                        weight_hh_dev: torch.Tensor,
                        b_ih_dev: torch.Tensor,
                        b_hh_dev: torch.Tensor,
                        h0_dev: torch.Tensor,
                        c0_dev: torch.Tensor,
                        seq_len: int,
                        batch: int,
                        input_size: int,
                        hidden_size: int,
                        warmup: int,
                        repeats: int,
                        chunk_size: int) -> BenchmarkResult:
    output_dev = torch.empty(seq_len * batch * hidden_size, dtype=torch.float32, device="cuda")
    hn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
    cn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")

    buffers = LstmBuffers()
    buffers_ptr = ctypes.byref(buffers)
    status = lib.lstm_create_buffers(
        LSTM_COMPUTE_PRECISION_FP16_ACC16,
        seq_len,
        batch,
        input_size,
        hidden_size,
        chunk_size,
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

        args = (
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

        for _ in range(max(warmup, 0)):
            ret = lib.lstm_forward(*args)
            if ret != 0:
                raise RuntimeError(f"FlashLSTM warmup failed with error code {ret}")
        torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            ret = lib.lstm_forward(*args)
            if ret != 0:
                raise RuntimeError(f"FlashLSTM forward failed with error code {ret}")
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)
    finally:
        lib.lstm_destroy_buffers(buffers_ptr)

    avg_ms = float(np.mean(timings))
    std_ms = float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0
    return BenchmarkResult("FlashLSTM CUDA", avg_ms, std_ms)


def autotune_chunk_size(lib: ctypes.CDLL,
                        x_dev: torch.Tensor,
                        weight_ih_dev: torch.Tensor,
                        weight_hh_dev: torch.Tensor,
                        b_ih_dev: torch.Tensor,
                        b_hh_dev: torch.Tensor,
                        h0_dev: torch.Tensor,
                        c0_dev: torch.Tensor,
                        seq_len: int,
                        batch: int,
                        input_size: int,
                        hidden_size: int,
                        candidates: Sequence[int] = CHUNK_SIZE_CANDIDATES,
                        warmup: int = AUTOTUNE_WARMUP_ITERS,
                        repeats: int = AUTOTUNE_TIMED_ITERS) -> Tuple[int, float]:
    if seq_len <= 0:
        raise ValueError("Sequence length must be positive for auto-tuning.")
    timed_iters = max(repeats, 1)
    output_dev = torch.empty(seq_len * batch * hidden_size, dtype=torch.float32, device="cuda")
    hn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
    cn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    best_chunk = None
    best_time_ms = float("inf")

    torch.cuda.synchronize()
    for chunk_size in candidates:
        buffers = LstmBuffers()
        buffers_ptr = ctypes.byref(buffers)
        status = lib.lstm_create_buffers(
            LSTM_COMPUTE_PRECISION_FP16_ACC16,
            seq_len,
            batch,
            input_size,
            hidden_size,
            int(chunk_size),
            buffers_ptr,
        )
        if status != 0:
            raise RuntimeError(f"lstm_create_buffers failed during auto-tuning with error code {status}")

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
                raise RuntimeError(f"lstm_pack_weights failed during auto-tuning with error code {status}")

            args = (
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

            for _ in range(max(warmup, 0)):
                ret = lib.lstm_forward(*args)
                if ret != 0:
                    raise RuntimeError(f"Auto-tuning warmup failed with error code {ret}")
            torch.cuda.synchronize()

            total_ms = 0.0
            for _ in range(timed_iters):
                start_event.record()
                ret = lib.lstm_forward(*args)
                if ret != 0:
                    raise RuntimeError(f"Auto-tuning iteration failed with error code {ret}")
                stop_event.record()
                stop_event.synchronize()
                total_ms += start_event.elapsed_time(stop_event)
            avg_ms = total_ms / float(timed_iters)
        finally:
            lib.lstm_destroy_buffers(buffers_ptr)

        torch.cuda.synchronize()
        if avg_ms < best_time_ms:
            best_time_ms = avg_ms
            best_chunk = int(chunk_size)

    if best_chunk is None:
        raise RuntimeError("Auto-tuning failed to evaluate any chunk sizes.")
    return best_chunk, best_time_ms


def benchmark_pytorch(lstm: torch.nn.LSTM,
                      x_cuda: torch.Tensor,
                      h0_cuda: torch.Tensor,
                      c0_cuda: torch.Tensor,
                      warmup: int,
                      repeats: int) -> BenchmarkResult:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            lstm(x_cuda, (h0_cuda, c0_cuda))

        timings: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            lstm(x_cuda, (h0_cuda, c0_cuda))
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    avg_ms = float(torch.tensor(timings).mean().item())
    std_ms = float(torch.tensor(timings).std(unbiased=True).item()) if len(timings) > 1 else 0.0
    return BenchmarkResult("PyTorch nn.LSTM (cuDNN)", avg_ms, std_ms)


def benchmark_flash_attention(seq_len: int,
                              batch: int,
                              num_heads: int,
                              head_dim: int,
                              warmup: int,
                              repeats: int) -> BenchmarkResult:
    dtype = torch.float16
    device = torch.device("cuda")
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    def run_attention() -> torch.Tensor:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    for _ in range(max(warmup, 0)):
        run_attention()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_attention()
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    avg_ms = float(np.mean(timings))
    std_ms = float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0
    return BenchmarkResult(f"FlashAttention (heads={num_heads})", avg_ms, std_ms)


def parity_check(lib: ctypes.CDLL,
                 torch_out: torch.Tensor,
                 torch_hn: torch.Tensor,
                 torch_cn: torch.Tensor,
                 x_dev: torch.Tensor,
                 weight_ih_dev: torch.Tensor,
                 weight_hh_dev: torch.Tensor,
                 b_ih_dev: torch.Tensor,
                 b_hh_dev: torch.Tensor,
                 h0_dev: torch.Tensor,
                 c0_dev: torch.Tensor,
                 seq_len: int,
                 batch: int,
                 input_size: int,
                 hidden_size: int,
                 atol: float,
                 rtol: float,
                 chunk_size: int) -> None:
    output_dev = torch.empty(seq_len * batch * hidden_size, dtype=torch.float32, device="cuda")
    hn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")
    cn_dev = torch.empty(batch * hidden_size, dtype=torch.float32, device="cuda")

    buffers = LstmBuffers()
    buffers_ptr = ctypes.byref(buffers)
    status = lib.lstm_create_buffers(
        LSTM_COMPUTE_PRECISION_FP16_ACC16,
        seq_len,
        batch,
        input_size,
        hidden_size,
        chunk_size,
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

        ret = lib.lstm_forward(
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
        if ret != 0:
            raise RuntimeError(f"FlashLSTM parity check failed with error code {ret}")
    finally:
        lib.lstm_destroy_buffers(buffers_ptr)

    torch.cuda.synchronize()

    output_np = output_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
    hn_np = hn_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)
    cn_np = cn_dev.detach().cpu().numpy().reshape(-1).astype(np.float32)

    output_ref = torch_out.detach().cpu().numpy().reshape(-1).astype(np.float32)
    hn_ref = torch_hn[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
    cn_ref = torch_cn[0].detach().cpu().numpy().reshape(-1).astype(np.float32)

    np.testing.assert_allclose(output_np, output_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(hn_np, hn_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cn_np, cn_ref, rtol=rtol, atol=atol)


def format_result(result: BenchmarkResult) -> str:
    return (f"{result.implementation:>24}: "
            f"{result.avg_ms:8.3f} ms ± {result.std_ms:6.3f} ms "
            f"({result.iters_per_second:6.2f} it/s)")


def determine_sequence_lengths(seq_lens_arg: Optional[Sequence[int]],
                               seq_len_single: Optional[int],
                               seq_len_min: int,
                               seq_len_max: int,
                               seq_len_step: int) -> List[int]:
    if seq_lens_arg:
        return sorted(set(int(v) for v in seq_lens_arg if int(v) > 0))
    if seq_len_single:
        return [seq_len_single]
    if seq_len_step <= 0:
        raise ValueError("--seq-len-step must be positive.")
    if seq_len_min <= 0 or seq_len_max <= 0:
        raise ValueError("Sequence lengths must be positive.")
    if seq_len_min > seq_len_max:
        raise ValueError("--seq-len-min must be <= --seq-len-max.")
    seq_values = list(range(seq_len_min, seq_len_max + 1, seq_len_step))
    if not seq_values:
        raise ValueError("No sequence lengths generated; adjust range parameters.")
    return seq_values


def plot_performance(seq_lens: Sequence[int],
                     flash_results: Sequence[BenchmarkResult],
                     torch_results: Sequence[BenchmarkResult],
                     attention_results: Sequence[BenchmarkResult],
                     output_path: Path) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        )
    flash_ms = [res.avg_ms for res in flash_results]
    torch_ms = [res.avg_ms for res in torch_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq_lens, flash_ms, label="FlashLSTM CUDA", linewidth=2)
    ax.plot(seq_lens, torch_ms, label="PyTorch nn.LSTM (CUDA)", linewidth=2)
    if attention_results:
        attn_ms = [res.avg_ms for res in attention_results]
        ax.plot(seq_lens,
                attn_ms,
                label=f"FlashAttention (heads={NUM_ATTENTION_HEADS})",
                linewidth=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Average time (ms)")
    ax.set_title("Sequence Length Performance")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FlashLSTM against PyTorch LSTM.")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Benchmark a single sequence length (overrides range options).")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None,
                        help="Explicit list of sequence lengths to benchmark.")
    parser.add_argument("--seq-len-min", type=int, default=64,
                        help="Minimum sequence length for range sweep (inclusive).")
    parser.add_argument("--seq-len-max", type=int, default=8192,
                        help="Maximum sequence length for range sweep (inclusive).")
    parser.add_argument("--seq-len-step", type=int, default=128,
                        help="Step between sequence lengths in the sweep.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--input-size", type=int, default=1024, help="Input feature size.")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden feature size.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument("--repeats", type=int, default=20, help="Number of timed iterations.")
    parser.add_argument("--lib-path", type=str, default=None, help="Path to flashlstm shared library.")
    parser.add_argument("--seed", type=int, default=2025, help="RNG seed.")
    parser.add_argument("--check", action="store_true", help="Run parity check before benchmarking.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for parity check.")
    parser.add_argument("--rtol", type=float, default=5e-2, help="Relative tolerance for parity check.")
    parser.add_argument("--plot-path", type=str, default="benchmark_seq_length.png",
                        help="Path to save the performance plot (set to empty to disable saving).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating the performance plot.")
    return parser


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for benchmarking.")

    parser = build_argument_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    seq_lens = determine_sequence_lengths(args.seq_lens,
                                          args.seq_len,
                                          args.seq_len_min,
                                          args.seq_len_max,
                                          args.seq_len_step)

    lib = load_library(resolve_library_path(args.lib_path))

    lstm_cpu = torch.nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size).eval()
    lstm_cuda = torch.nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size).cuda().eval()
    lstm_cuda.load_state_dict(lstm_cpu.state_dict())

    h0 = torch.randn(1, args.batch, args.hidden_size, dtype=torch.float32)
    c0 = torch.randn(1, args.batch, args.hidden_size, dtype=torch.float32)
    h0_cuda = h0.cuda().contiguous()
    c0_cuda = c0.cuda().contiguous()
    h0_dev = h0_cuda.view(-1)
    c0_dev = c0_cuda.view(-1)

    weight_ih_dev = lstm_cuda.weight_ih_l0.detach().contiguous()
    weight_hh_dev = lstm_cuda.weight_hh_l0.detach().contiguous()
    b_ih_dev = lstm_cuda.bias_ih_l0.detach().contiguous().view(-1)
    b_hh_dev = lstm_cuda.bias_hh_l0.detach().contiguous().view(-1)

    flash_results: List[BenchmarkResult] = []
    torch_results: List[BenchmarkResult] = []
    attention_results: List[BenchmarkResult] = []

    print("Benchmark configuration:")
    print(f"  batch={args.batch}, input_size={args.input_size}, hidden_size={args.hidden_size}")
    print(f"  warmup={args.warmup}, repeats={args.repeats}")
    print(f"  attention heads={NUM_ATTENTION_HEADS}, head_dim={ATTENTION_HEAD_DIM}")
    print(f"  attention embed_dim={NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM}")
    print(f"  sequence lengths: {seq_lens}")

    for idx, seq_len in enumerate(seq_lens, start=1):
        torch.manual_seed(args.seed + seq_len)
        x = torch.randn(seq_len, args.batch, args.input_size, dtype=torch.float32)
        x_cuda = x.cuda().contiguous()
        x_dev = x_cuda.view(-1)

        best_chunk_size, best_chunk_time = autotune_chunk_size(
            lib,
            x_dev,
            weight_ih_dev,
            weight_hh_dev,
            b_ih_dev,
            b_hh_dev,
            h0_dev,
            c0_dev,
            seq_len,
            args.batch,
            args.input_size,
            args.hidden_size,
        )
        print(f"[{idx}/{len(seq_lens)}] seq_len={seq_len}: auto-tuned chunk size {best_chunk_size} "
              f"({best_chunk_time:.3f} ms average over {AUTOTUNE_TIMED_ITERS} iters)")

        if args.check:
            with torch.no_grad():
                torch_out_cpu, (torch_hn_cpu, torch_cn_cpu) = lstm_cpu(x, (h0, c0))
            parity_check(lib,
                         torch_out_cpu,
                         torch_hn_cpu,
                         torch_cn_cpu,
                         x_dev,
                         weight_ih_dev,
                         weight_hh_dev,
                         b_ih_dev,
                         b_hh_dev,
                         h0_dev,
                         c0_dev,
                         seq_len,
                         args.batch,
                         args.input_size,
                         args.hidden_size,
                         args.atol,
                         args.rtol,
                         best_chunk_size)
            print(f"[{idx}/{len(seq_lens)}] seq_len={seq_len}: parity check passed.")

        result_flash = benchmark_flashlstm(
            lib,
            x_dev,
            weight_ih_dev,
            weight_hh_dev,
            b_ih_dev,
            b_hh_dev,
            h0_dev,
            c0_dev,
            seq_len,
            args.batch,
            args.input_size,
            args.hidden_size,
            args.warmup,
            args.repeats,
            best_chunk_size,
        )

        result_torch = benchmark_pytorch(
            lstm_cuda,
            x_cuda,
            h0_cuda,
            c0_cuda,
            args.warmup,
            args.repeats,
        )

        torch.manual_seed(args.seed + seq_len * 17)
        result_attention = benchmark_flash_attention(
            seq_len,
            args.batch,
            NUM_ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            args.warmup,
            args.repeats,
        )

        flash_results.append(result_flash)
        torch_results.append(result_torch)
        attention_results.append(result_attention)

        speedup = (result_torch.avg_ms / result_flash.avg_ms
                   if result_flash.avg_ms > 0 else float("inf"))
        print(f"\n[{idx}/{len(seq_lens)}] seq_len={seq_len}")
        print(f"  {format_result(result_flash)}")
        print(f"  {format_result(result_torch)}")
        print(f"  {format_result(result_attention)}")
        print(f"  Speedup (PyTorch / FlashLSTM): {speedup:6.3f}x")

    if not args.no_plot and args.plot_path:
        output_path = Path(args.plot_path)
        plot_performance(seq_lens, flash_results, torch_results, attention_results, output_path)
        print(f"\nSaved performance plot to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
