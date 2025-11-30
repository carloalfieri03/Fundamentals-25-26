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

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, **_):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=0 
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.relu(x)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x