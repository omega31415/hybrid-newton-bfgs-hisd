"""Network definition and parameter order for the embedding experiment."""

import torch
from torch import nn

SMALL_WIDTHS = (2, 2, 2, 2, 3, 1)
WIDE_WIDTHS = (2, 10, 2, 7, 3, 1)


class FiveLayerNet(nn.Module):
    """Four tanh hidden layers followed by a bias-free linear output."""

    def __init__(self, widths=SMALL_WIDTHS):
        super().__init__()
        self.widths = tuple(widths)
        self.layers = nn.ModuleList(
            [
                nn.Linear(a, b, bias=i < len(widths) - 2)
                for i, (a, b) in enumerate(zip(widths, widths[1:]))
            ]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def layout(widths):
    offset, result = (0, [])
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        ws = slice(offset, offset + a * b)
        offset += a * b
        bs = slice(offset, offset + b) if i < len(widths) - 2 else None
        offset += b if bs is not None else 0
        result.append((a, b, ws, bs))
    return (result, offset)
