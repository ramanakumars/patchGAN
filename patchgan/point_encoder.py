from torch import nn


class PointEncoder(nn.Sequential):
    def __init__(self, filt):
        super().__init__(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2, True),
            nn.LayerNorm(16),
            nn.Linear(16, filt),
            nn.LeakyReLU(0.2, True),
        )
