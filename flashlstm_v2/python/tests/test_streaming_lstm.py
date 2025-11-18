import pytest
import torch
import torch.nn.functional as F

from flashlstm import FlashLstm
from flashlstm.streaming_lstm import StreamingLSTM


def _random_pinned_half(shape):
    tensor = torch.empty(shape, dtype=torch.float16, pin_memory=True)
    tensor.copy_(torch.randn_like(tensor, dtype=torch.float16))
    return tensor


def _random_device_half(shape):
    return torch.randn(shape, device="cuda", dtype=torch.float16)


def _alternating_reference(
    x: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    *,
    time_oversample: bool = False,
):
    weight_set_count = weight_ih.shape[0] if weight_ih.dim() == 3 else 1
    h = h0
    c = c0
    outputs = []
    for t in range(x.size(0)):
        if time_oversample and weight_set_count > 1:
            for idx in range(weight_set_count):
                W_ih = weight_ih[idx]
                W_hh = weight_hh[idx]
                b_ih = bias_ih[idx]
                b_hh = bias_hh[idx]
                gates = F.linear(x[t], W_ih, b_ih) + F.linear(h, W_hh, b_hh)
                gi, gf, gg, go = gates.chunk(4, dim=1)
                i_act = torch.sigmoid(gi)
                f_act = torch.sigmoid(gf)
                g_act = torch.tanh(gg)
                o_act = torch.sigmoid(go)
                c = f_act * c + i_act * g_act
                h = o_act * torch.tanh(c)
            outputs.append(h.unsqueeze(0))
        else:
            idx = t % weight_set_count
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
def test_flashlstm_matches_streaming():
    torch.manual_seed(123)

    time_steps = 5
    batch_size = 3
    input_size = 4
    hidden_size = 6

    streaming = StreamingLSTM(input_size, hidden_size)
    flash = FlashLstm(input_size, hidden_size)
    flash.load_state_dict(streaming.state_dict())

    x_host = _random_pinned_half((time_steps, batch_size, input_size)).contiguous()
    x_device = x_host.to(device="cuda")

    h0 = _random_device_half((batch_size, hidden_size))
    c0 = _random_device_half((batch_size, hidden_size))

    streaming.zero_grad(set_to_none=True)
    flash.zero_grad(set_to_none=True)

    y_host, _, (hy_stream, cy_stream) = streaming(x_host, h0, c0)
    y_stream = y_host.to(device="cuda", dtype=torch.float16)

    y_flash, _, (hy_flash, cy_flash) = flash(x_device, h0, c0)

    torch.testing.assert_close(y_flash.float(), y_stream.float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(hy_flash.float(), hy_stream.float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(cy_flash.float(), cy_stream.float(), rtol=1e-3, atol=2e-3)

    loss_stream = (y_stream.float().pow(2).mean() + hy_stream.float().pow(2).mean() + cy_stream.float().pow(2).mean())
    loss_flash = (y_flash.float().pow(2).mean() + hy_flash.float().pow(2).mean() + cy_flash.float().pow(2).mean())

    loss_stream.backward()
    loss_flash.backward()

    torch.testing.assert_close(streaming.weight_ih.grad, flash.weight_ih.grad, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(streaming.weight_hh.grad, flash.weight_hh.grad, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(streaming.bias_ih.grad, flash.bias_ih.grad, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(streaming.bias_hh.grad, flash.bias_hh.grad, rtol=1e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flashlstm_matches_torch_lstm():
    torch.manual_seed(42)

    time_steps = 4
    batch_size = 2
    input_size = 3
    hidden_size = 5

    flash = FlashLstm(input_size, hidden_size)
    reference = torch.nn.LSTM(input_size, hidden_size, batch_first=False, device="cuda")

    x = _random_device_half((time_steps, batch_size, input_size)).contiguous()
    h0 = _random_device_half((batch_size, hidden_size))
    c0 = _random_device_half((batch_size, hidden_size))
    h0_ref = h0.to(dtype=torch.float32).unsqueeze(0)
    c0_ref = c0.to(dtype=torch.float32).unsqueeze(0)

    flash.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)

    y_flash, _, (hy_flash, cy_flash) = flash(x, h0, c0)
    y_ref, (hy_ref, cy_ref) = reference(x.to(dtype=torch.float32), (h0_ref, c0_ref))
    hy_ref = hy_ref.squeeze(0)
    cy_ref = cy_ref.squeeze(0)

    torch.testing.assert_close(y_flash.float(), y_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(hy_flash.float(), hy_ref, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(cy_flash.float(), cy_ref, rtol=1e-3, atol=2e-3)

    loss_flash = y_flash.float().pow(2).mean() + hy_flash.float().pow(2).mean() + cy_flash.float().pow(2).mean()
    loss_ref = y_ref.pow(2).mean() + hy_ref.pow(2).mean() + cy_ref.pow(2).mean()

    loss_flash.backward()
    loss_ref.backward()

    torch.testing.assert_close(flash.weight_ih.grad, reference.weight_ih_l0.grad, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(flash.weight_hh.grad, reference.weight_hh_l0.grad, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(flash.bias_ih.grad, reference.bias_ih_l0.grad, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(flash.bias_hh.grad, reference.bias_hh_l0.grad, rtol=1e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_streaming_lstm_time_oversample():
    torch.manual_seed(17)

    time_steps = 5
    batch_size = 2
    input_size = 3
    hidden_size = 4
    weight_sets = 2

    module = StreamingLSTM(
        input_size,
        hidden_size,
        weight_set_count=weight_sets,
        time_oversample=True,
    )

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
            time_oversample=True,
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
        time_oversample=True,
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

if __name__ == '__main__':
    pytest.main([__file__])
