# lstm_backward_demo.py
# Minimal self-contained PyTorch LSTM example:
# bs=32, seqlen=2048, nembed=1024, nhidden=1024 + backward pass.

import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    # Config
    bs = 32
    seqlen = 2048
    nembed = 1024   # input_size
    nhid = 1024     # hidden_size
    lr = 1e-3

    # Device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    # Model
    lstm = nn.LSTM(
        input_size=nembed,
        hidden_size=nhid,
        num_layers=1,
        batch_first=True,
    ).to(device=device, dtype=dtype)

    # Simple head to produce a scalar from the final hidden state
    head = nn.Linear(nhid, 1).to(device=device, dtype=dtype)

    # Optimizer
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=lr)

    # Fake data: (batch, seq_len, nembed)
    x = torch.randn(bs, seqlen, nembed, device=device, dtype=dtype)

    # Target for a regression-style loss (one scalar per sequence)
    y = torch.randn(bs, 1, device=device, dtype=dtype)

    # Forward
    opt.zero_grad(set_to_none=True)
    # LSTM returns (output, (h_n, c_n))
    # output: (bs, seqlen, nhid), h_n: (num_layers, bs, nhid)
    output, (h_n, c_n) = lstm(x)

    # Take the final layer's hidden state for each sequence: (bs, nhid)
    last_h = h_n[-1]  # shape: (bs, nhid)

    # Predict and compute loss
    pred = head(last_h)           # (bs, 1)
    loss = F.mse_loss(pred, y)    # scalar

    # Backward
    loss.backward()

    # (Optional) inspect grad norm to prove grads exist
    total_norm = 0.0
    for p in lstm.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.float().norm(2)
            total_norm += param_norm.item() ** 2
    total_norm **= 0.5

    # Optimizer step
    opt.step()

    print(f"loss={loss.item():.6f}  grad_norm~={total_norm:.3f}  device={device}  dtype={dtype}")

if __name__ == "__main__":
    main()