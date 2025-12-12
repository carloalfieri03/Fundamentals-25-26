import torch.nn as nn

class ConvBlock(nn.Module):
    ### Same padding così piu facile
     
    def __init__(self, in_channels, out_channels, kernel_size=3,pool_type='avg', pool_size=2, stride=2,pool_stride=2, **_):
        super(ConvBlock, self).__init__()
        pad = (kernel_size-1) //2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad)
        self.relu = nn.ReLU()
        if pool_type == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)
        elif pool_type == 'max':  
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")

    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class DeconvBlock1D(nn.Module):
    """
    Deconvolution block for 1D signals:
    ConvTranspose1d -> (BatchNorm1d optional) -> (ReLU optional)
    """
    def __init__( self, in_channels, out_channels, kernel_size=4, stride=4, padding=None, output_padding=0):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.net=nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )

    def forward(self, x):
        return self.net(x)
