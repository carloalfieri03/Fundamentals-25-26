import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, **_):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DeconvBlock1D(nn.Module):
    """
    Deconvolution block for 1D signals:
    ConvTranspose1d → (BatchNorm1d optional) → ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        use_batchnorm=True,
    ):
        super().__init__()

        layers = [
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
