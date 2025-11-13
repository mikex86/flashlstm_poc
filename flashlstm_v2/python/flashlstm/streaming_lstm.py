from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function

from . import streaming_lstm_backward as _streaming_lstm_backward
from . import streaming_lstm_forward as _streaming_lstm_forward

_GATE_CACHE_DTYPE_TO_ENUM = {
    torch.float32: 0,
    torch.float64: 1,
}
_GATE_CACHE_ALLOWED_DTYPES: Tuple[torch.dtype, ...] = tuple(_GATE_CACHE_DTYPE_TO_ENUM.keys())


def _check_pinned_half(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must reside on the CPU (pinned host memory).")
    if tensor.dtype != torch.float16:
        raise ValueError(f"{name} must use dtype torch.float16, got {tensor.dtype}.")
    if not tensor.is_pinned():
        raise ValueError(f"{name} must be allocated in pinned memory (tensor.pin_memory()).")


def _check_pinned_float(
    tensor: torch.Tensor, name: str, allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32,)
) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must reside on the CPU (pinned host memory).")
    if tensor.dtype not in allowed_dtypes:
        allowed = ", ".join(str(d) for d in allowed_dtypes)
        raise ValueError(f"{name} must use dtype in {{{allowed}}}, got {tensor.dtype}.")
    if not tensor.is_pinned():
        raise ValueError(f"{name} must be allocated in pinned memory (tensor.pin_memory()).")


def _ensure_half_cuda(
    tensor: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    name: str,
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(shape, device="cuda", dtype=torch.float16)
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must reside on CUDA, got device {tensor.device}.")
    if tensor.dtype != torch.float16:
        raise ValueError(f"{name} must use dtype torch.float16, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


class _StreamingLSTMFunction(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_host: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
        recompute_interval: int,
        gate_cache_dtype: torch.dtype,
    ):
        _check_pinned_half(x_host, "x_host")
        if not x_host.is_contiguous():
            x_host = x_host.contiguous()

        if recompute_interval <= 0:
            raise ValueError(f"recompute_interval must be >= 1, got {recompute_interval}")

        if gate_cache_dtype not in _GATE_CACHE_DTYPE_TO_ENUM:
            allowed = ", ".join(str(d) for d in _GATE_CACHE_ALLOWED_DTYPES)
            raise ValueError(
                f"gate_cache_dtype must be one of {{{allowed}}}, got {gate_cache_dtype}."
            )
        gate_cache_type_id = _GATE_CACHE_DTYPE_TO_ENUM[gate_cache_dtype]

        for name, param in (
            ("weight_ih", weight_ih),
            ("weight_hh", weight_hh),
            ("bias_ih", bias_ih),
            ("bias_hh", bias_hh),
        ):
            if param.device.type != "cuda":
                raise ValueError(f"{name} must reside on CUDA.")
            if param.dtype != torch.float32:
                raise ValueError(f"{name} must use dtype torch.float32.")

        h0 = _ensure_half_cuda(h0, (x_host.size(1), weight_hh.size(1)), "h0")
        c0 = _ensure_half_cuda(c0, (x_host.size(1), weight_hh.size(1)), "c0")

        time_steps, batch_size, input_size = x_host.shape
        hidden_size = weight_hh.size(1)
        gate_dim = 4 * hidden_size
        checkpoint_steps = (time_steps + recompute_interval - 1) // recompute_interval

        if weight_ih.shape != (gate_dim, input_size):
            raise ValueError(
                f"weight_ih must have shape {(gate_dim, input_size)}, "
                f"got {tuple(weight_ih.shape)}"
            )
        if weight_hh.shape != (gate_dim, hidden_size):
            raise ValueError(
                f"weight_hh must have shape {(gate_dim, hidden_size)}, "
                f"got {tuple(weight_hh.shape)}"
            )
        if bias_ih.numel() != gate_dim or bias_hh.numel() != gate_dim:
            raise ValueError("Bias tensors must have length 4 * hidden_size.")

        y_host = torch.empty(
            (time_steps, batch_size, hidden_size),
            dtype=torch.float16,
            pin_memory=True,
        )
        gate_cache_host = torch.empty(
            (checkpoint_steps, 2, batch_size, hidden_size),
            dtype=gate_cache_dtype,
            pin_memory=True,
        )
        hy_device = torch.empty(
            (batch_size, hidden_size),
            device="cuda",
            dtype=torch.float16,
        )
        cy_device = torch.empty_like(hy_device)

        compute_stream = torch.cuda.Stream()
        h2d_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()

        _streaming_lstm_forward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            x_host.data_ptr(),
            h0.data_ptr(),
            c0.data_ptr(),
            weight_ih.data_ptr(),
            weight_hh.data_ptr(),
            bias_ih.data_ptr(),
            bias_hh.data_ptr(),
            y_host.data_ptr(),
            gate_cache_host.data_ptr(),
            gate_cache_type_id,
            hy_device.data_ptr(),
            cy_device.data_ptr(),
            compute_stream.cuda_stream,
            h2d_stream.cuda_stream,
            d2h_stream.cuda_stream,
        )

        compute_stream.synchronize()
        h2d_stream.synchronize()
        d2h_stream.synchronize()
        torch.cuda.current_stream().wait_stream(compute_stream)

        ctx.save_for_backward(
            x_host,
            h0,
            c0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            y_host,
            gate_cache_host,
        )
        ctx.meta = (time_steps, batch_size, input_size, hidden_size, recompute_interval, gate_cache_type_id)
        ctx.mark_non_differentiable(gate_cache_host)

        return y_host, gate_cache_host, hy_device, cy_device

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y_host: Optional[torch.Tensor],
        _grad_gate_cache: Optional[torch.Tensor],
        grad_hy: Optional[torch.Tensor],
        grad_cy: Optional[torch.Tensor],
    ):
        (
            x_host,
            h0,
            c0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            y_host,
            gate_cache_host,
        ) = ctx.saved_tensors
        (
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            gate_cache_type_id,
        ) = ctx.meta
        _check_pinned_float(gate_cache_host, "gate_cache_host", _GATE_CACHE_ALLOWED_DTYPES)

        if grad_y_host is None:
            grad_y_host = torch.zeros_like(y_host)
        else:
            if grad_y_host.device.type != "cpu":
                raise ValueError("grad_y_host must reside on the CPU.")
            if grad_y_host.dtype != torch.float16:
                grad_y_host = grad_y_host.to(dtype=torch.float16)
            if not grad_y_host.is_contiguous():
                grad_y_host = grad_y_host.contiguous()
            if not grad_y_host.is_pinned():
                grad_y_host = grad_y_host.pin_memory()

        _check_pinned_half(grad_y_host, "grad_y_host")

        grad_hy_half: Optional[torch.Tensor] = None
        grad_cy_half: Optional[torch.Tensor] = None
        grad_hy_ptr = 0
        grad_cy_ptr = 0

        if grad_hy is not None:
            if grad_hy.device.type != "cuda":
                raise ValueError("grad_hy must reside on CUDA.")
            if grad_hy.dtype != torch.float16:
                grad_hy = grad_hy.to(dtype=torch.float16)
            if not grad_hy.is_contiguous():
                grad_hy = grad_hy.contiguous()
            grad_hy_half = grad_hy
            grad_hy_ptr = grad_hy_half.data_ptr()

        if grad_cy is not None:
            if grad_cy.device.type != "cuda":
                raise ValueError("grad_cy must reside on CUDA.")
            if grad_cy.dtype != torch.float16:
                grad_cy = grad_cy.to(dtype=torch.float16)
            if not grad_cy.is_contiguous():
                grad_cy = grad_cy.contiguous()
            grad_cy_half = grad_cy
            grad_cy_ptr = grad_cy_half.data_ptr()

        dx_host = torch.empty_like(x_host)
        dW_ih = torch.zeros_like(weight_ih)
        dW_hh = torch.zeros_like(weight_hh)
        db_ih = torch.zeros_like(bias_ih)
        db_hh = torch.zeros_like(bias_hh)
        dh0_float = torch.empty((batch_size, hidden_size), device="cuda", dtype=torch.float32)
        dc0_float = torch.empty_like(dh0_float)

        compute_stream = torch.cuda.Stream()
        h2d_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()

        _streaming_lstm_backward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            x_host.data_ptr(),
            y_host.data_ptr(),
            gate_cache_host.data_ptr(),
            gate_cache_type_id,
            grad_y_host.data_ptr(),
            grad_hy_ptr,
            grad_cy_ptr,
            h0.data_ptr(),
            c0.data_ptr(),
            weight_ih.data_ptr(),
            weight_hh.data_ptr(),
            bias_ih.data_ptr(),
            bias_hh.data_ptr(),
            dx_host.data_ptr(),
            dW_ih.data_ptr(),
            dW_hh.data_ptr(),
            db_ih.data_ptr(),
            db_hh.data_ptr(),
            dh0_float.data_ptr(),
            dc0_float.data_ptr(),
            compute_stream.cuda_stream,
            h2d_stream.cuda_stream,
            d2h_stream.cuda_stream,
        )

        compute_stream.synchronize()
        h2d_stream.synchronize()
        d2h_stream.synchronize()
        torch.cuda.current_stream().wait_stream(compute_stream)

        grad_h0 = dh0_float.to(dtype=h0.dtype)
        grad_c0 = dc0_float.to(dtype=c0.dtype)

        return (
            dx_host,
            grad_h0,
            grad_c0,
            dW_ih,
            dW_hh,
            db_ih,
            db_hh,
            None,
            None
        )


def streaming_lstm(
    x_host: torch.Tensor,
    h0: Optional[torch.Tensor],
    c0: Optional[torch.Tensor],
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    *,
    recompute_interval: int = 1,
    gate_cache_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional wrapper for the streaming LSTM kernels.
    Returns pinned host outputs, gate cache, and the final CUDA states (hy, cy).
    """
    return _StreamingLSTMFunction.apply(
        x_host,
        h0,
        c0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        recompute_interval,
        gate_cache_dtype,
    )


class StreamingLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        gate_dim = 4 * hidden_size
        self.weight_ih = nn.Parameter(
            torch.empty(gate_dim, input_size, device="cuda", dtype=torch.float32)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(gate_dim, hidden_size, device="cuda", dtype=torch.float32)
        )
        self.bias_ih = nn.Parameter(
            torch.zeros(gate_dim, device="cuda", dtype=torch.float32)
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(gate_dim, device="cuda", dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)

    def forward(
        self,
        x_host: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
        *,
        recompute_interval: int = 1,
        gate_cache_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _check_pinned_half(x_host, "x_host")
        batch_size = x_host.size(1)
        h0 = _ensure_half_cuda(h0, (batch_size, self.hidden_size), "h0")
        c0 = _ensure_half_cuda(c0, (batch_size, self.hidden_size), "c0")

        y_host, gate_cache_host, hy, cy = streaming_lstm(
            x_host,
            h0,
            c0,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            recompute_interval=recompute_interval,
            gate_cache_dtype=gate_cache_dtype,
        )
        return y_host, gate_cache_host, (hy, cy)
