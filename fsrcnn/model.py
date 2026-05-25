import math
import torch
from torch import nn


class FSRCNN(nn.Module):
    """
    FSRCNN: Accelerating the Super-Resolution Convolutional Neural Network
    https://arxiv.org/abs/1608.00367

    Args:
        scale: upscaling factor (2, 3, or 4)
        num_channels: number of image channels (default: 3 for RGB)
        d: feature dimension in extraction/expansion layers
        s: shrunken feature dimension in mapping layers
        m: number of mapping layers
    """

    def __init__(self, scale: int, num_channels: int = 3, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        if not (2 <= scale <= 4):
            raise ValueError("scale must be 2, 3, or 4")
        self.scale = scale

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )

        self.mapping = nn.Sequential(
            *[layer for _ in range(m) for layer in (
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s),
            )]
        )

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )

        self.deconv = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9,
            stride=scale,
            padding=4,
            output_padding=scale - 1,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight[0][0].numel())))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        return self.deconv(x)


class FSRCNN_s(FSRCNN):
    """
    Lightweight variant of FSRCNN with reduced d, s, m for faster inference.
    Same architecture, smaller capacity: d=32, s=5, m=1.
    """

    def __init__(self, scale: int, num_channels: int = 3):
        super().__init__(scale=scale, num_channels=num_channels, d=32, s=5, m=1)
