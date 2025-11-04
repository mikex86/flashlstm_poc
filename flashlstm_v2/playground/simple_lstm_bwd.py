"""
LSTM (single-layer, unidirectional) laid out to mimic cuDNN’s backward rhythm:

Per time step (reverse):
  • ONE GEMM: dZ_t = dG_t @ W^T                  # backprop to inputs/hidden
  • ONE fused elementwise "bp1" kernel:
      consumes (dh_t, dc_t, i,f,g,o,c_t,c_{t-1}) and
      produces (dG_t, dc_{t-1}) AND contributes to db in-place

After the loop (once):
  • ONE GEMM total: dWcat = Z_all^T @ D_all      # param gradients for the whole seq

Forward per time step is ONE GEMM + ONE fused pointwise as before.

Gate order: [i, f, g, o]
Shapes follow PyTorch default (batch_first=False).

Requirements:
  pip install torch
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


@torch.no_grad()
def fused_lstm_pointwise_forward_kernel(
        gates: torch.Tensor,  # (B, 4H) pre-activations from GEMM
        c_prev: torch.Tensor,  # (B, H)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused forward pointwise for one time step.
    Returns: h_next, c_next, i, f, g, o   (all (B, H))
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
    return h_next, c_next, i, f, g, o


@torch.no_grad()
def fused_lstm_pointwise_backward_kernel(
        dh_t: torch.Tensor,  # (B, H) upstream wrt h_t (includes dY_t + gate-path from future)
        dc_t: torch.Tensor,  # (B, H) upstream wrt c_t (from future)
        i: torch.Tensor,  # (B, H) saved gate activations
        f: torch.Tensor,  # (B, H)
        g: torch.Tensor,  # (B, H)
        o: torch.Tensor,  # (B, H)
        c_t: torch.Tensor,  # (B, H) saved cell at time t (after update)
        c_prev: torch.Tensor,  # (B, H) saved cell at time t-1
        # Accumulators updated IN-PLACE by the kernel (so no outer .sum() kernels):
        db_accum: torch.Tensor,  # (4H,) running bias grad, updated in-place
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused backward pointwise: computes dG_t and dc_{t-1}, and accumulates db in-place.
    Returns:
      dG_t    (B, 4H)  # gate pre-activation grads
      dc_prev (B, H)
    """
    tanh_c = torch.tanh(c_t)
    do = dh_t * tanh_c
    dc = dc_t + dh_t * o * (1.0 - tanh_c * tanh_c)

    di = dc * g
    df = dc * c_prev
    dg = dc * i
    dc_prev = dc * f

    dai = di * i * (1.0 - i)  # sigmoid'
    daf = df * f * (1.0 - f)  # sigmoid'
    dag = dg * (1.0 - g * g)  # tanh'
    dao = do * o * (1.0 - o)  # sigmoid'

    dG = torch.cat([dai, daf, dag, dao], dim=1)  # (B, 4H)

    # Accumulate bias gradient inside the fused kernel interface
    db_accum.add_(dG.sum(dim=0))  # in a real CUDA kernel this reduction is fused

    return dG, dc_prev


# ============================================================
# Concatenated-weight handle (forward uses 1-GEMM per step)
# ============================================================
@dataclass
class LSTMHandle:
    weight_concat: torch.Tensor  # (I+H, 4H), float32, contiguous
    bias_fused: torch.Tensor  # (4H,),     float32, contiguous
    input_size: int
    hidden_size: int


@torch.no_grad()
def create_lstm_handle(
        weight_ih: torch.Tensor,  # (4H, I)
        weight_hh: torch.Tensor,  # (4H, H)
        bias_ih: torch.Tensor,  # (4H,)
        bias_hh: torch.Tensor,  # (4H,)
) -> LSTMHandle:
    W_ih_T = weight_ih.detach().float().t().contiguous()  # (I, 4H)
    W_hh_T = weight_hh.detach().float().t().contiguous()  # (H, 4H)
    Wcat = torch.cat([W_ih_T, W_hh_T], dim=0).contiguous()  # (I+H, 4H)
    b = (bias_ih.detach().float() + bias_hh.detach().float()).contiguous()
    return LSTMHandle(Wcat, b, W_ih_T.shape[0], W_hh_T.shape[0])


# ============================================================
# Forward: ONE GEMM + ONE fused pointwise per step
# ============================================================
@torch.no_grad()
def lstm_forward(
        handle: LSTMHandle,
        x: torch.Tensor,  # (T, B, I)
        h0: torch.Tensor,  # (1, B, H) or (B, H)
        c0: torch.Tensor,  # (1, B, H) or (B, H)
        return_cache: bool = True,
):
    T, B, I = x.shape
    H = handle.hidden_size
    assert I == handle.input_size

    W = handle.weight_concat
    b = handle.bias_fused

    h_t = (h0[0] if h0.dim() == 3 else h0).contiguous().float()
    c_t = (c0[0] if c0.dim() == 3 else c0).contiguous().float()
    x = x.contiguous().float()
    y = torch.empty((T, B, H), dtype=torch.float32, device=x.device)

    if return_cache:
        h_s = torch.empty((T + 1, B, H), dtype=torch.float32, device=x.device);
        h_s[0] = h_t
        c_s = torch.empty((T + 1, B, H), dtype=torch.float32, device=x.device);
        c_s[0] = c_t
        i_s = torch.empty((T, B, H), dtype=torch.float32, device=x.device)
        f_s = torch.empty((T, B, H), dtype=torch.float32, device=x.device)
        g_s = torch.empty((T, B, H), dtype=torch.float32, device=x.device)
        o_s = torch.empty((T, B, H), dtype=torch.float32, device=x.device)

    for t in range(T):
        x_t = x[t]  # (B, I)
        z_t = torch.cat([x_t, h_t], dim=1)  # (B, I+H)
        gates = z_t @ W  # ONE GEMM
        gates = gates + b
        h_t, c_t, i, f, g, o = fused_lstm_pointwise_forward_kernel(gates, c_t)  # fused pointwise
        y[t] = h_t
        if return_cache:
            h_s[t + 1] = h_t
            c_s[t + 1] = c_t
            i_s[t] = i
            f_s[t] = f
            g_s[t] = g
            o_s[t] = o

    hn = h_t.unsqueeze(0)
    cn = c_t.unsqueeze(0)

    if not return_cache:
        return y, (hn, cn)

    cache = {
        "x": x, "Wcat": W, "bias": b,
        "h_s": h_s, "c_s": c_s, "i_s": i_s, "f_s": f_s, "g_s": g_s, "o_s": o_s,
    }
    return y, (hn, cn), cache


# ============================================================
# Backward: per-step ONE GEMM + ONE fused pointwise
#   • We compute dZ_t = dG_t @ W^T each step (ONE GEMM per t).
#   • We buffer z_t and dG_t across time, and after the loop do ONE final GEMM:
#         dWcat = Z_all^T @ D_all
#   • Bias db is accumulated inside the fused pointwise kernel (no outer .sum()).
# ============================================================
@torch.no_grad()
def lstm_backward_cudnn_like(
        dY: torch.Tensor,  # (T, B, H)
        cache: dict,
        d_hn: torch.Tensor,  # (1, B, H)
        d_cn: torch.Tensor,  # (1, B, H)
):
    x = cache["x"]  # (T, B, I)
    W = cache["Wcat"]  # (I+H, 4H)
    h_s = cache["h_s"]  # (T+1, B, H)
    c_s = cache["c_s"]  # (T+1, B, H)
    i_s = cache["i_s"]
    f_s = cache["f_s"]
    g_s = cache["g_s"]
    o_s = cache["o_s"]

    T, B, I = x.shape
    H = h_s.shape[2]

    dx = torch.zeros_like(x)
    dh_next = torch.zeros((B, H), dtype=torch.float32, device=x.device)
    dc_next = torch.zeros((B, H), dtype=torch.float32, device=x.device)
    if d_hn is not None: dh_next += d_hn[0].to(dh_next.dtype)
    if d_cn is not None: dc_next += d_cn[0].to(dc_next.dtype)

    # Buffers to batch one final parameter GEMM after the loop
    Zbuf = torch.empty((T * B, I + H), dtype=torch.float32, device=x.device)
    Dbuf = torch.empty((T * B, 4 * H), dtype=torch.float32, device=x.device)

    # Bias grad accumulator (updated inside fused kernel)
    db_fused = torch.zeros((4 * H,), dtype=torch.float32, device=x.device)

    for step, t in enumerate(reversed(range(T))):
        # Upstream at time t
        dh_t = dY[t].to(torch.float32) + dh_next  # (B, H)
        c_t = c_s[t + 1]
        c_prev = c_s[t]
        h_prev = h_s[t]  # (B,H)
        i = i_s[t]
        f = f_s[t]
        g = g_s[t]
        o = o_s[t]

        # Fused pointwise: dG_t and dc_{t-1}, and in-place db accumulation
        dG_t, dc_prev = fused_lstm_pointwise_backward_kernel(
            dh_t, dc_next, i, f, g, o, c_t, c_prev, db_accum=db_fused
        )  # (B,4H), (B,H)

        # ONE GEMM per step: backprop to inputs/hidden via W^T
        dZ_t = dG_t @ W.t()  # GEMM   (B, I+H)
        dx[t] = dZ_t[:, :I]
        dh_next = dZ_t[:, I:]  # to previous time step
        dc_next = dc_prev

        # Save z_t and dG_t for single param-GEMM after the loop
        z_t = torch.cat([x[t], h_prev], dim=1)  # (B, I+H)
        start = step * B
        Zbuf[start:start + B] = z_t
        Dbuf[start:start + B] = dG_t

    # Final ONE GEMM for all parameter grads at once
    # dWcat: (I+H, 4H) = (I+H, TB) @ (TB, 4H)
    dWcat = Zbuf.t() @ Dbuf

    # Split to PyTorch parameter layout
    dW_ih = dWcat[:I, :].t().contiguous()  # (4H, I)
    dW_hh = dWcat[I:, :].t().contiguous()  # (4H, H)

    # Fused bias was b = b_ih + b_hh
    db_ih = db_fused.clone()
    db_hh = db_fused.clone()

    dh0 = dh_next
    dc0 = dc_next
    return dx, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0


# ============================================================
# Parity test vs torch.nn.LSTM
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float32

    # Dimensions
    T, B, I, H = 5, 3, 4, 6

    # Reference LSTM
    lstm = torch.nn.LSTM(I, H, num_layers=1, bias=True, batch_first=False).to(device, dtype)

    # Pull weights
    with torch.no_grad():
        W_ih_t = lstm.weight_ih_l0.detach().clone()
        W_hh_t = lstm.weight_hh_l0.detach().clone()
        b_ih_t = lstm.bias_ih_l0.detach().clone()
        b_hh_t = lstm.bias_hh_l0.detach().clone()

    # Inputs & states
    x = torch.randn(T, B, I, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.randn(1, B, H, device=device, dtype=dtype, requires_grad=True)
    c0 = torch.randn(1, B, H, device=device, dtype=dtype, requires_grad=True)

    # Forward (manual)
    handle = create_lstm_handle(W_ih_t, W_hh_t, b_ih_t, b_hh_t)
    y_man, (hn_man, cn_man), cache = lstm_forward(handle, x, h0, c0, return_cache=True)

    # Forward (torch)
    y_ref, (hn_ref, cn_ref) = lstm(x, (h0, c0))

    # Forward checks
    atol, rtol = 1e-5, 1e-6
    assert torch.allclose(y_man, y_ref, atol=atol, rtol=rtol)
    assert torch.allclose(hn_man, hn_ref, atol=atol, rtol=rtol)
    assert torch.allclose(cn_man, cn_ref, atol=atol, rtol=rtol)

    # Build a scalar loss touching y, hn, cn
    y_tgt = torch.randn_like(y_ref)
    hn_tgt = torch.randn_like(hn_ref)
    cn_tgt = torch.randn_like(cn_ref)

    # Upstream grads for L = 0.5||· - target||^2
    dY = (y_man - y_tgt).detach()
    dHn = (hn_man - hn_tgt).detach()
    dCn = (cn_man - cn_tgt).detach()

    # Manual backward (cuDNN-like structure)
    dx_man, dWih_man, dWhh_man, dbih_man, dbhh_man, dh0_man, dc0_man = lstm_backward_cudnn_like(
        dY, cache, d_hn=dHn, d_cn=dCn
    )

    # Torch autograd on same loss
    lstm.zero_grad(set_to_none=True)
    loss = 0.5 * (y_ref - y_tgt).pow(2).sum()
    loss = loss + 0.5 * (hn_ref - hn_tgt).pow(2).sum()
    loss = loss + 0.5 * (cn_ref - cn_tgt).pow(2).sum()
    loss.backward()

    # Collect refs
    dx_ref = x.grad.detach()
    dh0_ref = h0.grad.detach()[0]
    dc0_ref = c0.grad.detach()[0]
    dW_ih_ref = lstm.weight_ih_l0.grad.detach()
    dW_hh_ref = lstm.weight_hh_l0.grad.detach()
    db_ih_ref = lstm.bias_ih_l0.grad.detach()
    db_hh_ref = lstm.bias_hh_l0.grad.detach()

    # Comparisons
    assert torch.allclose(dx_man, dx_ref, atol=atol, rtol=rtol), "dx mismatch"
    assert torch.allclose(dh0_man, dh0_ref, atol=atol, rtol=rtol), "dh0 mismatch"
    assert torch.allclose(dc0_man, dc0_ref, atol=atol, rtol=rtol), "dc0 mismatch"
    assert torch.allclose(dWih_man, dW_ih_ref, atol=atol, rtol=rtol), "dW_ih mismatch"
    assert torch.allclose(dWhh_man, dW_hh_ref, atol=atol, rtol=rtol), "dW_hh mismatch"
    assert torch.allclose(dbih_man, db_ih_ref, atol=atol, rtol=rtol), "db_ih mismatch"
    assert torch.allclose(dbhh_man, db_hh_ref, atol=atol, rtol=rtol), "db_hh mismatch"

    print("✅ Parity OK. Backward uses 1 GEMM/step + 1 fused pointwise, and a single post-loop GEMM for params.")
