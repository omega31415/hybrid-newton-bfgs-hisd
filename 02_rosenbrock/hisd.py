"""HiSD: standard high-index saddle dynamics with fixed step size."""

from accelerated_hisd import AcceleratedHiSD


class HiSD(AcceleratedHiSD):
    """Use the same reflected-gradient update with zero momentum."""

    def __init__(self, beta=1e-3):
        super().__init__(beta=beta, gamma=0.0)
