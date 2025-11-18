from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function

from . import streaming_lstm_backward as _streaming_lstm_backward
from . import streaming_lstm_forward as _streaming_lstm_forward


class GateCache(NamedTuple):
    h: torch.Tensor
    c: torch.Tensor


def _gate_cache_dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    raise ValueError(f"gate cache dtype must be float16 or float32, got {dtype}")


def _check_pinned_half(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must reside on the CPU (pinned host memory).")
    if tensor.dtype != torch.float16:
        raise ValueError(f"{name} must use dtype torch.float16, got {tensor.dtype}.")
    if not tensor.is_pinned():
        raise ValueError(f"{name} must be allocated in pinned memory (tensor.pin_memory()).")


def _check_pinned_float(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must reside on the CPU (pinned host memory).")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use dtype torch.float32, got {tensor.dtype}.")
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


def _normalize_weight_sets(
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    gate_dim: int,
    input_size: int,
    hidden_size: int,
    requested_sets: Optional[int],
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if requested_sets is not None and requested_sets <= 0:
        raise ValueError(f"weight_set_count must be >= 1, got {requested_sets}")
    def _validate_and_maybe_expand(
        param: torch.Tensor,
        expected_last_shape: Tuple[int, ...],
        label: str,
        allow_rank1: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        if param.dim() == 2:
            # Bias tensors may come as (S, 4H) when allow_rank1 is True.
            if allow_rank1 and param.shape[1:] == expected_last_shape:
                if param.shape[0] <= 0:
                    raise ValueError(f"{label} must have at least one weight set, got shape {tuple(param.shape)}")
                return param.shape[0], param
            set_count = 1
            if param.shape != expected_last_shape:
                raise ValueError(f"{label} must have shape {expected_last_shape}, got {tuple(param.shape)}")
            return set_count, param
        if allow_rank1 and param.dim() == 1:
            if param.numel() != expected_last_shape[0]:
                raise ValueError(f"{label} must have length {expected_last_shape[0]}, got {param.numel()}")
            return 1, param
        if param.dim() == 3:
            if param.shape[1:] != expected_last_shape:
                raise ValueError(f"{label} must have shape (S, {', '.join(map(str, expected_last_shape))}), got {tuple(param.shape)}")
            return param.shape[0], param
        raise ValueError(f"{label} must be 2D or 3D, got rank {param.dim()}")

    set_ih, weight_ih = _validate_and_maybe_expand(weight_ih, (gate_dim, input_size), "weight_ih")
    set_hh, weight_hh = _validate_and_maybe_expand(weight_hh, (gate_dim, hidden_size), "weight_hh")
    set_bih, bias_ih = _validate_and_maybe_expand(bias_ih, (gate_dim,), "bias_ih", allow_rank1=True)
    set_bhh, bias_hh = _validate_and_maybe_expand(bias_hh, (gate_dim,), "bias_hh", allow_rank1=True)

    inferred_sets = set_ih
    for name, count in (("weight_hh", set_hh), ("bias_ih", set_bih), ("bias_hh", set_bhh)):
        if count != inferred_sets:
            raise ValueError(f"{name} must have the same number of weight sets as weight_ih ({inferred_sets}), "
                             f"got {count}")

    weight_set_count = inferred_sets
    if requested_sets is not None and requested_sets != weight_set_count:
        raise ValueError(
            f"weight_set_count mismatch: expected {requested_sets} sets but tensors provide {weight_set_count}"
        )

    return weight_set_count, weight_ih.contiguous(), weight_hh.contiguous(), bias_ih.contiguous(), bias_hh.contiguous()


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
        gate_cache_dtypes: Tuple[torch.dtype, torch.dtype],
        weight_set_count: Optional[int],
        time_oversample: bool,
    ):
        _check_pinned_half(x_host, "x_host")
        if not x_host.is_contiguous():
            x_host = x_host.contiguous()

        if recompute_interval <= 0:
            raise ValueError(f"recompute_interval must be >= 1, got {recompute_interval}")

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

        time_steps, batch_size, input_size = x_host.shape
        hidden_size = weight_hh.shape[-1]
        gate_dim = 4 * hidden_size
        weight_set_count, weight_ih, weight_hh, bias_ih, bias_hh = _normalize_weight_sets(
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            gate_dim,
            input_size,
            hidden_size,
            weight_set_count,
        )
        checkpoint_steps = (time_steps + recompute_interval - 1) // recompute_interval

        h0 = _ensure_half_cuda(h0, (x_host.size(1), hidden_size), "h0")
        c0 = _ensure_half_cuda(c0, (x_host.size(1), hidden_size), "c0")

        gate_cache_h_dtype, gate_cache_c_dtype = gate_cache_dtypes
        gate_cache_h_enum = _gate_cache_dtype_enum(gate_cache_h_dtype)
        gate_cache_c_enum = _gate_cache_dtype_enum(gate_cache_c_dtype)

        y_host = torch.empty(
            (time_steps, batch_size, hidden_size),
            dtype=torch.float16,
            pin_memory=True,
        )
        gate_cache_h = torch.empty(
            (checkpoint_steps, batch_size, hidden_size),
            dtype=gate_cache_h_dtype,
            pin_memory=True,
        )
        gate_cache_c = torch.empty(
            (checkpoint_steps, batch_size, hidden_size),
            dtype=gate_cache_c_dtype,
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
        current_stream = torch.cuda.current_stream()
        compute_stream.wait_stream(current_stream)
        h2d_stream.wait_stream(current_stream)
        d2h_stream.wait_stream(current_stream)

        _streaming_lstm_forward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            weight_set_count,
            x_host.data_ptr(),
            h0.data_ptr(),
            c0.data_ptr(),
            weight_ih.data_ptr(),
            weight_hh.data_ptr(),
            bias_ih.data_ptr(),
            bias_hh.data_ptr(),
            y_host.data_ptr(),
            gate_cache_h.data_ptr(),
            gate_cache_c.data_ptr(),
            gate_cache_h_enum,
            gate_cache_c_enum,
            int(time_oversample),
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
            gate_cache_h,
            gate_cache_c,
        )
        ctx.meta = (
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            weight_set_count,
            gate_cache_h_enum,
            gate_cache_c_enum,
            time_oversample,
        )
        ctx.mark_non_differentiable(gate_cache_h, gate_cache_c)

        return y_host, gate_cache_h, gate_cache_c, hy_device, cy_device

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y_host: Optional[torch.Tensor],
        _grad_gate_cache_h: Optional[torch.Tensor],
        _grad_gate_cache_c: Optional[torch.Tensor],
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
            gate_cache_h,
            gate_cache_c,
        ) = ctx.saved_tensors
        (
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            weight_set_count,
            gate_cache_h_enum,
            gate_cache_c_enum,
            time_oversample,
        ) = ctx.meta
        if gate_cache_h.dtype == torch.float32:
            _check_pinned_float(gate_cache_h, "gate_cache_h")
        elif gate_cache_h.dtype == torch.float16:
            _check_pinned_half(gate_cache_h, "gate_cache_h")
        else:
            raise ValueError(f"Unsupported dtype for gate_cache_h: {gate_cache_h.dtype}")
        if gate_cache_c.dtype == torch.float32:
            _check_pinned_float(gate_cache_c, "gate_cache_c")
        elif gate_cache_c.dtype == torch.float16:
            _check_pinned_half(gate_cache_c, "gate_cache_c")
        else:
            raise ValueError(f"Unsupported dtype for gate_cache_c: {gate_cache_c.dtype}")

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
        current_stream = torch.cuda.current_stream()
        compute_stream.wait_stream(current_stream)
        h2d_stream.wait_stream(current_stream)
        d2h_stream.wait_stream(current_stream)

        _streaming_lstm_backward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            weight_set_count,
            x_host.data_ptr(),
            y_host.data_ptr(),
            gate_cache_h.data_ptr(),
            gate_cache_c.data_ptr(),
            gate_cache_h_enum,
            gate_cache_c_enum,
            int(time_oversample),
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
            None,
            None,
            None,
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
    gate_cache_dtypes: Tuple[torch.dtype, torch.dtype] = (torch.float32, torch.float32),
    weight_set_count: Optional[int] = None,
    time_oversample: bool = False,
) -> Tuple[torch.Tensor, GateCache, torch.Tensor, torch.Tensor]:
    """
    Functional wrapper for the streaming LSTM kernels.
    Returns pinned host outputs, gate cache, and the final CUDA states (hy, cy).
    """
    outputs = _StreamingLSTMFunction.apply(
        x_host,
        h0,
        c0,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        recompute_interval,
        gate_cache_dtypes,
        weight_set_count,
        time_oversample,
    )
    y_host, gate_cache_h, gate_cache_c, hy, cy = outputs
    gate_cache = GateCache(gate_cache_h, gate_cache_c)
    return y_host, gate_cache, hy, cy


class StreamingLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_set_count: int = 1,
        time_oversample: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_set_count = int(weight_set_count)
        self.time_oversample = bool(time_oversample)

        gate_dim = 4 * hidden_size
        weight_ih_shape = (gate_dim, input_size) if self.weight_set_count == 1 else (
            self.weight_set_count,
            gate_dim,
            input_size,
        )
        weight_hh_shape = (gate_dim, hidden_size) if self.weight_set_count == 1 else (
            self.weight_set_count,
            gate_dim,
            hidden_size,
        )
        bias_shape = (gate_dim,) if self.weight_set_count == 1 else (self.weight_set_count, gate_dim)
        self.weight_ih = nn.Parameter(
            torch.empty(weight_ih_shape, device="cuda", dtype=torch.float32)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(weight_hh_shape, device="cuda", dtype=torch.float32)
        )
        self.bias_ih = nn.Parameter(
            torch.zeros(bias_shape, device="cuda", dtype=torch.float32)
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(bias_shape, device="cuda", dtype=torch.float32)
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
        gate_cache_dtypes: Tuple[torch.dtype, torch.dtype] = (torch.float32, torch.float32),
        time_oversample: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, GateCache, Tuple[torch.Tensor, torch.Tensor]]:
        _check_pinned_half(x_host, "x_host")
        batch_size = x_host.size(1)
        h0 = _ensure_half_cuda(h0, (batch_size, self.hidden_size), "h0")
        c0 = _ensure_half_cuda(c0, (batch_size, self.hidden_size), "c0")
        use_time_oversample = self.time_oversample if time_oversample is None else bool(time_oversample)

        y_host, gate_cache_host, hy, cy = streaming_lstm(
            x_host,
            h0,
            c0,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            recompute_interval=recompute_interval,
            gate_cache_dtypes=gate_cache_dtypes,
            weight_set_count=self.weight_set_count,
            time_oversample=use_time_oversample,
        )
        return y_host, gate_cache_host, (hy, cy)
