from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn, Tensor


class DistributedEMA(nn.Module):
    """Exponential moving average of a scalar, averaged across distributed ranks on ``compute``."""

    _ema: Tensor

    def __init__(self, decay: float, initial_value: float = 0.0, device: str | int | torch.device = "cpu"):
        super().__init__()
        self.decay = decay
        # Defaults to CPU; pass the training device when ``update`` will be fed CUDA tensors, else the
        # `_ema * decay + batch_mean` below hits a device mismatch.
        self.register_buffer("_ema", torch.tensor(initial_value, dtype=torch.double, device=device))

    def update(self, values: Tensor) -> None:
        """Update EMA with the mean of ``values`` (local to this rank)."""
        batch_mean = values.detach().to(dtype=torch.double).mean()
        self._ema = self.decay * self._ema + (1 - self.decay) * batch_mean

    def all_reduce(self) -> None:
        """Average the EMA in-place across ranks."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._ema, op=dist.ReduceOp.AVG)

    def compute(self) -> float:
        """Average the EMA across ranks and return the value."""
        self.all_reduce()
        return self._ema.item()

    @property
    def value(self) -> float:
        return self._ema.item()