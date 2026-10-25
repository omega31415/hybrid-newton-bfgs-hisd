"""HiSD: standard high-index saddle dynamics with fixed step size."""

from torch.optim.optimizer import required

from accelerated_hisd import AcceleratedHiSD


class HiSD(AcceleratedHiSD):
    """Use the same reflected-gradient update with zero momentum."""

    def __init__(self, params, lr=0.1, v=required):
        super().__init__(params, lr=lr, gamma=0.0, v=v, v_lr=0.0)
