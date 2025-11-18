from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function

from . import lstm_backward as _lstm_backward
from . import lstm_forward as _lstm_forward
from .streaming_lstm import (
    GateCache,
    _ensure_half_cuda,
    _gate_cache_dtype_enum,
    _normalize_weight_sets,
)


class _LstmFunction(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
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
        if x.device.type != "cuda":
            raise ValueError("x must reside on CUDA.")
        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

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

        time_steps, batch_size, input_size = x.shape
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

        h0 = _ensure_half_cuda(h0, (batch_size, hidden_size), "h0")
        c0 = _ensure_half_cuda(c0, (batch_size, hidden_size), "c0")

        gate_cache_h_dtype, gate_cache_c_dtype = gate_cache_dtypes
        gate_cache_h_enum = _gate_cache_dtype_enum(gate_cache_h_dtype)
        gate_cache_c_enum = _gate_cache_dtype_enum(gate_cache_c_dtype)

        checkpoint_steps = (time_steps + recompute_interval - 1) // recompute_interval
        y_device = torch.empty(
            (time_steps, batch_size, hidden_size),
            device="cuda",
            dtype=torch.float16,
        )
        gate_cache_h = torch.empty(
            (checkpoint_steps, batch_size, hidden_size),
            device="cuda",
            dtype=gate_cache_h_dtype,
        )
        gate_cache_c = torch.empty_like(gate_cache_h, dtype=gate_cache_c_dtype)
        hy_device = torch.empty((batch_size, hidden_size), device="cuda", dtype=torch.float16)
        cy_device = torch.empty_like(hy_device)

        compute_stream = torch.cuda.Stream()
        h2d_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()

        _lstm_forward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            weight_set_count,
            x.data_ptr(),
            h0.data_ptr(),
            c0.data_ptr(),
            weight_ih.data_ptr(),
            weight_hh.data_ptr(),
            bias_ih.data_ptr(),
            bias_hh.data_ptr(),
            y_device.data_ptr(),
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
            x,
            h0,
            c0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            y_device,
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
        return y_device, GateCache(gate_cache_h, gate_cache_c), hy_device, cy_device

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_y: Optional[torch.Tensor],
        _grad_gate_cache_h: Optional[torch.Tensor],
        _grad_gate_cache_c: Optional[torch.Tensor],
        grad_hy: Optional[torch.Tensor],
        grad_cy: Optional[torch.Tensor],
    ):
        (
            x,
            h0,
            c0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            y,
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

        if grad_y is None:
            grad_y = torch.zeros_like(y)
        if grad_y.device.type != "cuda":
            raise ValueError("grad_y must reside on CUDA.")
        if grad_y.dtype != torch.float16:
            grad_y = grad_y.to(dtype=torch.float16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()

        grad_hy_ptr = 0
        grad_cy_ptr = 0
        if grad_hy is not None:
            if grad_hy.device.type != "cuda":
                raise ValueError("grad_hy must reside on CUDA.")
            if grad_hy.dtype != torch.float16:
                grad_hy = grad_hy.to(dtype=torch.float16)
            if not grad_hy.is_contiguous():
                grad_hy = grad_hy.contiguous()
            grad_hy_ptr = grad_hy.data_ptr()

        if grad_cy is not None:
            if grad_cy.device.type != "cuda":
                raise ValueError("grad_cy must reside on CUDA.")
            if grad_cy.dtype != torch.float16:
                grad_cy = grad_cy.to(dtype=torch.float16)
            if not grad_cy.is_contiguous():
                grad_cy = grad_cy.contiguous()
            grad_cy_ptr = grad_cy.data_ptr()

        dx = torch.empty_like(x)
        dW_ih = torch.zeros_like(weight_ih)
        dW_hh = torch.zeros_like(weight_hh)
        db_ih = torch.zeros_like(bias_ih)
        db_hh = torch.zeros_like(bias_hh)
        dh0_float = torch.empty((batch_size, hidden_size), device="cuda", dtype=torch.float32)
        dc0_float = torch.empty_like(dh0_float)

        compute_stream = torch.cuda.Stream()
        h2d_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()

        _lstm_backward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            recompute_interval,
            weight_set_count,
            x.data_ptr(),
            y.data_ptr(),
            gate_cache_h.data_ptr(),
            gate_cache_c.data_ptr(),
            gate_cache_h_enum,
            gate_cache_c_enum,
            int(time_oversample),
            grad_y.data_ptr(),
            grad_hy_ptr,
            grad_cy_ptr,
            h0.data_ptr(),
            c0.data_ptr(),
            weight_ih.data_ptr(),
            weight_hh.data_ptr(),
            bias_ih.data_ptr(),
            bias_hh.data_ptr(),
            dx.data_ptr(),
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
            dx,
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


def flashlstm(
    x: torch.Tensor,
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
    outputs = _LstmFunction.apply(
        x,
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
    y, gate_cache, hy, cy = outputs
    return y, gate_cache, hy, cy


class FlashLstm(nn.Module):
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
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
        *,
        recompute_interval: int = 1,
        gate_cache_dtypes: Tuple[torch.dtype, torch.dtype] = (torch.float32, torch.float32),
        time_oversample: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, GateCache, Tuple[torch.Tensor, torch.Tensor]]:
        if x.device.type != "cuda":
            raise ValueError("FlashLstm expects CUDA inputs.")
        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()
        batch_size = x.size(1)
        h0 = _ensure_half_cuda(h0, (batch_size, self.hidden_size), "h0")
        c0 = _ensure_half_cuda(c0, (batch_size, self.hidden_size), "c0")
        use_time_oversample = self.time_oversample if time_oversample is None else bool(time_oversample)

        y_device, gate_cache, hy, cy = flashlstm(
            x,
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
        return y_device, gate_cache, (hy, cy)
