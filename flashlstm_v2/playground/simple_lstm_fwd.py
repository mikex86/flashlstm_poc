"""
Minimal, self-contained LSTM forward pass using ONE GEMM + ONE pointwise op
per time step. Educational and correctness-checked against torch.nn.LSTM.

Key idea (per step)
-------------------
z_t = [x_t, h_{t-1}]        ∈ R^{B×(I+H)}           # concat inputs & hidden
W   = [W_ih^T ; W_hh^T]     ∈ R^{(I+H)×(4H)}        # concat weights (transposed)
G_t = z_t @ W + b           ∈ R^{B×(4H)}            # ONE GEMM (all 4 gates)
(h_t, c_t) = pointwise(G_t, c_{t-1})                # ONE pointwise op

Gate order matches PyTorch: [i, f, g, o].

API (matches the tests in the prompt)
-------------------------------------
- create_lstm_handle(weight_ih, weight_hh, bias_ih, bias_hh) -> handle
- destroy_lstm_handle(handle)
- lstm_forward(handle, x, h0, c0) -> (y, (hn, cn))
  * x can be (B, T, I) or (T, B, I)
  * h0/c0 can be (1, B, H) or (B, H)

No Triton or custom kernels — just PyTorch ops.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

# Disable TF32 so our reference comparisons are nice and tight (float32 only).
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = False


@dataclass
class LSTMHandle:
    """Holds pre-fused parameters for a 1-GEMM LSTM forward."""
    weight_concat: torch.Tensor  # (I+H, 4H), contiguous
    bias: torch.Tensor  # (4H,),     contiguous
    input_size: int
    hidden_size: int


def create_lstm_handle(
        weight_ih: torch.Tensor,  # (4H, I)
        weight_hh: torch.Tensor,  # (4H, H)
        bias_ih: torch.Tensor,  # (4H,)
        bias_hh: torch.Tensor,  # (4H,)
) -> LSTMHandle:
    """Prepare concatenated weights and fused bias for the one-GEMM formulation."""
    # Transpose weights so we can do z_t @ W where z_t is (B, I+H) and W is (I+H, 4H)
    W_ih_T = weight_ih.detach().transpose(0, 1).contiguous()  # (I, 4H)
    W_hh_T = weight_hh.detach().transpose(0, 1).contiguous()  # (H, 4H)
    weight_concat = torch.cat([W_ih_T, W_hh_T], dim=0).contiguous()  # (I+H, 4H)
    bias = (bias_ih.detach() + bias_hh.detach()).contiguous()  # (4H,)
    return LSTMHandle(
        weight_concat=weight_concat,
        bias=bias,
        input_size=W_ih_T.shape[0],
        hidden_size=W_hh_T.shape[0],
    )


def destroy_lstm_handle(handle: LSTMHandle) -> None:
    """Zero out tensors/metadata in the handle (simple lifecycle helper)."""
    handle.weight_concat = None  # type: ignore
    handle.bias = None  # type: ignore
    handle.input_size = 0
    handle.hidden_size = 0


def _lstm_pointwise(gates: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise LSTM update.

    gates: (B, 4H) pre-activations with order [i, f, g, o]
    c_prev: (B, H)

    returns (h_next, c_next) both (B, H)
    """
    B, fourH = gates.shape
    H = fourH // 4
    gi = gates[:, 0:H]
    gf = gates[:, H:2 * H]
    gg = gates[:, 2 * H:3 * H]
    go = gates[:, 3 * H:4 * H]

    i = torch.sigmoid(gi)
    f = torch.sigmoid(gf)
    g = torch.tanh(gg)
    c_next = f * c_prev + i * g
    o = torch.sigmoid(go)
    h_next = o * torch.tanh(c_next)
    return h_next, c_next


def _normalize_states(h0: torch.Tensor, c0: torch.Tensor, device, dtype) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Ensure states are (B, H); return states and a flag whether caller gave (1, B, H)."""
    gave_batched = (h0.dim() == 3)
    h = (h0[0] if gave_batched else h0).to(device=device, dtype=dtype).contiguous()
    c = (c0[0] if gave_batched else c0).to(device=device, dtype=dtype).contiguous()
    return h, c, gave_batched


def lstm_forward(
        handle: LSTMHandle,
        x: torch.Tensor,  # (B, T, I) or (T, B, I)
        h_prev: torch.Tensor,  # (1, B, H) or (B, H)
        c_prev: torch.Tensor,  # (1, B, H) or (B, H)
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Single-layer LSTM forward pass using one GEMM + one pointwise per step.

    Returns:
      y:  (B, T, H) if input was (B, T, I)  OR  (T, B, H) if input was (T, B, I)
      (hn, cn): both shaped like PyTorch's API for 1 layer: (1, B, H)
    """
    if handle.weight_concat is None or handle.bias is None:
        raise RuntimeError("LSTM handle was destroyed or not initialised")

    device = x.device
    dtype_in = x.dtype
    compute_dtype = torch.float32  # stick to fp32 for stable comparisons

    # Convert states to (B, H)
    h_t, c_t, states_batched = _normalize_states(h_prev, c_prev, device, compute_dtype)

    # Decide input layout. We treat x as (B, T, I) iff its first dim equals batch B.
    # Otherwise we assume (T, B, I). This matches the test which passes (B, T, I).
    B = h_t.shape[0]
    if x.dim() != 3:
        raise ValueError("x must be 3D (B,T,I) or (T,B,I).")
    if x.shape[0] == B:
        # (B, T, I) -> transpose to (T, B, I) for the loop
        inputs = x.transpose(0, 1).contiguous()
        will_transpose_back = True
    elif x.shape[1] == B:
        # already (T, B, I)
        inputs = x.contiguous()
        will_transpose_back = False
    else:
        raise ValueError(f"Cannot infer layout from x.shape={tuple(x.shape)} and batch={B}")

    T, B_in, I = inputs.shape
    if B_in != B:
        raise ValueError(f"Batch mismatch: B_from_states={B}, B_from_x={B_in}")
    if h_t.shape[1] != handle.hidden_size:
        raise ValueError(f"Hidden size mismatch: handle.H={handle.hidden_size}, h0.H={h_t.shape[1]}")

    W = handle.weight_concat.to(device=device, dtype=compute_dtype)  # (I+H, 4H)
    b = handle.bias.to(device=device, dtype=compute_dtype)  # (4H,)

    y = torch.empty((T, B, handle.hidden_size), device=device, dtype=compute_dtype)

    for t in range(T):
        x_t = inputs[t].to(dtype=compute_dtype)  # (B, I)
        z_t = torch.cat([x_t, h_t], dim=1)  # (B, I+H)
        gates = z_t @ W  # ONE GEMM -> (B, 4H)
        gates = gates + b  # fused bias add
        h_t, c_t = _lstm_pointwise(gates, c_t)  # ONE pointwise op
        y[t] = h_t

    # Return to caller's layout & dtype
    y_out = y.transpose(0, 1) if will_transpose_back else y
    if y_out.dtype != dtype_in:
        y_out = y_out.to(dtype=dtype_in)

    hn = h_t.unsqueeze(0)  # (1, B, H)
    cn = c_t.unsqueeze(0)
    if hn.dtype != dtype_in:
        hn = hn.to(dtype=dtype_in)
    if cn.dtype != dtype_in:
        cn = cn.to(dtype=dtype_in)
    return y_out, (hn, cn)


# ---------------------------
# Quick self-check (CPU or CUDA)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Try both layouts
    for layout in ["BTH", "TBI"]:
        B, T, I, H = 8, 16, 5, 7
        if layout == "BTH":
            x = torch.randn(B, T, I, device=device, dtype=dtype)
        else:
            x = torch.randn(T, B, I, device=device, dtype=dtype)

        h0 = torch.randn(1, B, H, device=device, dtype=dtype)
        c0 = torch.randn(1, B, H, device=device, dtype=dtype)

        lstm_ref = torch.nn.LSTM(I, H, batch_first=(layout == "BTH")).to(device=device, dtype=dtype)
        with torch.no_grad():
            handle = create_lstm_handle(
                lstm_ref.weight_ih_l0, lstm_ref.weight_hh_l0,
                lstm_ref.bias_ih_l0, lstm_ref.bias_hh_l0
            )

        out_ref, (hn_ref, cn_ref) = lstm_ref(x, (h0, c0))
        out_man, (hn_man, cn_man) = lstm_forward(handle, x, h0, c0)

        atol, rtol = 1e-5, 1e-6
        assert torch.allclose(out_ref, out_man, atol=atol, rtol=rtol), f"Output mismatch ({layout})"
        assert torch.allclose(hn_ref, hn_man, atol=atol, rtol=rtol), f"Hidden mismatch ({layout})"
        assert torch.allclose(cn_ref, cn_man, atol=atol, rtol=rtol), f"Cell mismatch ({layout})"
        print(f"✅ Parity confirmed for layout {layout}")

    print("All checks passed.")
