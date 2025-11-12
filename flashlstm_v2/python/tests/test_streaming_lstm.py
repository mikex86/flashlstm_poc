import pytest
import torch

from flashlstm.streaming_lstm import StreamingLSTM


def _random_pinned_half(shape):
    tensor = torch.empty(shape, dtype=torch.float16, pin_memory=True)
    tensor.copy_(torch.randn_like(tensor, dtype=torch.float16))
    return tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_streaming_lstm_matches_torch_lstm():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    time_steps = 4
    batch_size = 2
    input_size = 3
    hidden_size = 5

    module = StreamingLSTM(input_size, hidden_size).cuda()
    reference = torch.nn.LSTM(input_size, hidden_size, batch_first=False).cuda()

    with torch.no_grad():
        reference.weight_ih_l0.copy_(module.weight_ih)
        reference.weight_hh_l0.copy_(module.weight_hh)
        reference.bias_ih_l0.copy_(module.bias_ih)
        reference.bias_hh_l0.copy_(module.bias_hh)

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
