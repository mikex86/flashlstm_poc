"\"\"\"Thin Python bindings for the flash LSTM streaming kernels.\"\"\""

from ._flashlstm import streaming_lstm_backward, streaming_lstm_forward

__all__ = ["streaming_lstm_forward", "streaming_lstm_backward"]
