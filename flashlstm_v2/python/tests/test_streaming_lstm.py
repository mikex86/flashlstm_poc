import pytest
import torch
import torch.nn.functional as F

from flashlstm.streaming_lstm import StreamingLSTM


def _random_pinned_half(shape):
    tensor = torch.empty(shape, dtype=torch.float16, pin_memory=True)
    tensor.copy_(torch.randn_like(tensor, dtype=torch.float16))
    return tensor


def _alternating_reference(
    x: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    time_oversample: bool = False,
):
    weight_set_count = weight_ih.shape[0] if weight_ih.dim() == 3 else 1
    h = h0
    c = c0
    outputs = []
    for t in range(x.size(0)):
        repeats = weight_set_count if time_oversample and weight_set_count > 1 else 1
        for set_idx in range(repeats):
            idx = set_idx if time_oversample else (t % weight_set_count)
            W_ih = weight_ih[idx] if weight_ih.dim() == 3 else weight_ih
            W_hh = weight_hh[idx] if weight_hh.dim() == 3 else weight_hh
            b_ih = bias_ih[idx] if bias_ih.dim() == 2 else bias_ih
            b_hh = bias_hh[idx] if bias_hh.dim() == 2 else bias_hh
            gates = F.linear(x[t], W_ih, b_ih) + F.linear(h, W_hh, b_hh)
            gi, gf, gg, go = gates.chunk(4, dim=1)
            i_act = torch.sigmoid(gi)
            f_act = torch.sigmoid(gf)
            g_act = torch.tanh(gg)
            o_act = torch.sigmoid(go)
            c = f_act * c + i_act * g_act
            h = o_act * torch.tanh(c)
        outputs.append(h.unsqueeze(0))
    y = torch.cat(outputs, dim=0)
    return y, h, c


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_streaming_lstm_matches_torch_lstm():
    torch.manual_seed(1234)

    time_steps = 4
    batch_size = 2
    input_size = 3
    hidden_size = 5

    module = StreamingLSTM(input_size, hidden_size)

    torch.manual_seed(1234)

    reference = torch.nn.LSTM(input_size, hidden_size, batch_first=False, device='cuda')

    x_host = _random_pinned_half((time_steps, batch_size, input_size)).contiguous()
    x_ref = x_host.to(device="cuda", dtype=torch.float32)

    h0 = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    c0 = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    h0_ref = h0.to(dtype=torch.float32).unsqueeze(0)
    c0_ref = c0.to(dtype=torch.float32).unsqueeze(0)

    module.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)

    y_host, _, (hy, cy) = module(x_host, h0, c0)
    y = y_host.to(device="cuda", dtype=torch.float32)
    hy_f = hy.to(dtype=torch.float32)
    cy_f = cy.to(dtype=torch.float32)

    y_ref, (hy_ref, cy_ref) = reference(x_ref, (h0_ref, c0_ref))
    hy_ref = hy_ref.squeeze(0)
    cy_ref = cy_ref.squeeze(0)

    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(hy_f, hy_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(cy_f, cy_ref, rtol=1e-3, atol=2e-3)

    loss = (
        y.pow(2).mean()
        + hy_f.pow(2).mean()
        + cy_f.pow(2).mean()
    )
    loss.backward()

    loss_ref = (
        y_ref.pow(2).mean()
        + hy_ref.pow(2).mean()
        + cy_ref.pow(2).mean()
    )
    loss_ref.backward()

    torch.testing.assert_close(
        module.weight_ih.grad,
        reference.weight_ih_l0.grad,
        rtol=1e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        module.weight_hh.grad,
        reference.weight_hh_l0.grad,
        rtol=1e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        module.bias_ih.grad,
        reference.bias_ih_l0.grad,
        rtol=1e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        module.bias_hh.grad,
        reference.bias_hh_l0.grad,
        rtol=1e-3,
        atol=5e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_streaming_lstm_alternating_weights():
    torch.manual_seed(7)

    time_steps = 6
    batch_size = 3
    input_size = 4
    hidden_size = 5
    weight_sets = 3

    module = StreamingLSTM(input_size, hidden_size, weight_set_count=weight_sets)

    x_host = _random_pinned_half((time_steps, batch_size, input_size)).contiguous()
    x_ref = x_host.to(device="cuda", dtype=torch.float32)

    h0 = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    c0 = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    h0_ref = h0.to(dtype=torch.float32)
    c0_ref = c0.to(dtype=torch.float32)

    module.zero_grad(set_to_none=True)

    y_host, _, (hy, cy) = module(x_host, h0, c0)
    y = y_host.to(device="cuda", dtype=torch.float32)
    hy_f = hy.to(dtype=torch.float32)
    cy_f = cy.to(dtype=torch.float32)

    with torch.no_grad():
        y_ref, hy_ref, cy_ref = _alternating_reference(
            x_ref,
            h0_ref,
            c0_ref,
            module.weight_ih,
            module.weight_hh,
            module.bias_ih,
            module.bias_hh,
        )

    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(hy_f, hy_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(cy_f, cy_ref, rtol=1e-3, atol=2e-3)

    loss = (
        y.pow(2).mean()
        + hy_f.pow(2).mean()
        + cy_f.pow(2).mean()
    )
    loss.backward()

    ref_weight_ih = module.weight_ih.detach().clone().requires_grad_(True)
    ref_weight_hh = module.weight_hh.detach().clone().requires_grad_(True)
    ref_bias_ih = module.bias_ih.detach().clone().requires_grad_(True)
    ref_bias_hh = module.bias_hh.detach().clone().requires_grad_(True)
    y_ref_fwd, hy_ref_fwd, cy_ref_fwd = _alternating_reference(
        x_ref,
        h0_ref,
        c0_ref,
        ref_weight_ih,
        ref_weight_hh,
        ref_bias_ih,
        ref_bias_hh,
    )
    loss_ref = (
        y_ref_fwd.pow(2).mean()
        + hy_ref_fwd.pow(2).mean()
        + cy_ref_fwd.pow(2).mean()
    )
    grads = torch.autograd.grad(
        loss_ref,
        (ref_weight_ih, ref_weight_hh, ref_bias_ih, ref_bias_hh),
    )

    torch.testing.assert_close(
        module.weight_ih.grad,
        grads[0],
        rtol=1e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        module.weight_hh.grad,
        grads[1],
        rtol=1e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        module.bias_ih.grad,
        grads[2],
        rtol=1e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        module.bias_hh.grad,
        grads[3],
        rtol=1e-3,
        atol=5e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_streaming_lstm_time_oversample():
    torch.manual_seed(9)

    time_steps = 3
    batch_size = 2
    input_size = 4
    hidden_size = 6
    weight_sets = 2

    module = StreamingLSTM(input_size, hidden_size, weight_set_count=weight_sets)

    x_host = _random_pinned_half((time_steps, batch_size, input_size)).contiguous()
    x_ref = x_host.to(device="cuda", dtype=torch.float32)

    h0 = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    c0 = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    h0_ref = h0.to(dtype=torch.float32)
    c0_ref = c0.to(dtype=torch.float32)

    module.zero_grad(set_to_none=True)

    y_host, _, (hy, cy) = module(x_host, h0, c0, time_oversample=True)
    y = y_host.to(device="cuda", dtype=torch.float32)
    hy_f = hy.to(dtype=torch.float32)
    cy_f = cy.to(dtype=torch.float32)

    y_ref, hy_ref, cy_ref = _alternating_reference(
        x_ref,
        h0_ref,
        c0_ref,
        module.weight_ih,
        module.weight_hh,
        module.bias_ih,
        module.bias_hh,
        time_oversample=True,
    )

    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(hy_f, hy_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(cy_f, cy_ref, rtol=1e-3, atol=2e-3)

    loss = (y.pow(2).mean() + hy_f.pow(2).mean() + cy_f.pow(2).mean())
    loss.backward()

    ref_wi = module.weight_ih.detach().clone().requires_grad_(True)
    ref_wh = module.weight_hh.detach().clone().requires_grad_(True)
    ref_bi = module.bias_ih.detach().clone().requires_grad_(True)
    ref_bh = module.bias_hh.detach().clone().requires_grad_(True)
    y_ref_fwd, hy_ref_fwd, cy_ref_fwd = _alternating_reference(
        x_ref,
        h0_ref,
        c0_ref,
        ref_wi,
        ref_wh,
        ref_bi,
        ref_bh,
        time_oversample=True,
    )
    loss_ref = (y_ref_fwd.pow(2).mean() + hy_ref_fwd.pow(2).mean() + cy_ref_fwd.pow(2).mean())
    grads = torch.autograd.grad(loss_ref, (ref_wi, ref_wh, ref_bi, ref_bh))

    torch.testing.assert_close(module.weight_ih.grad, grads[0], rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(module.weight_hh.grad, grads[1], rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(module.bias_ih.grad, grads[2], rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(module.bias_hh.grad, grads[3], rtol=1e-3, atol=5e-3)

if __name__ == '__main__':
    pytest.main([__file__])
