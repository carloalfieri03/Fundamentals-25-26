import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, stride=2, **_):
        super(ConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
       # x = self.pool(x)
        return x
    
### CAA classe nuova
class EncoderWrapper(nn.Module):
    def __init__(self, in_channels, layer_widths, kernel_size,stride=2):
        super().__init__()
        
        layers = []
        current_in = in_channels
        
       # multiple layers
        for width in layer_widths:
            layers.append(
                ConvBlock(
                    in_channels=current_in, 
                    out_channels=width, 
                    kernel_size=kernel_size,
                    stride=stride 
                )
            )
            current_in = width # Update for next layer (e.g., 9 -> 64, then 64 -> 64)
            
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)

class DeconvBlock1D(nn.Module):
    """
    Deconvolution block for 1D signals:
    ConvTranspose1d -> (BatchNorm1d optional) -> (ReLU optional)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=None,
        output_padding=1,
        use_batchnorm=True,
        apply_activation=True, 
    ):
        super().__init__()
        if padding is None:
            pads = (kernel_size - 1) // 2
        layers = [
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                
                padding=pads,
                output_padding=output_padding,
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))

        if apply_activation:
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- The Decoder Wrapper ---
class DecoderWrapper(nn.Module):
    def __init__(self, in_channels, layer_widths, kernel_size, stride=2, output_padding=1, use_batchnorm=True):
        """
        Args:
            in_channels (int): Channels coming from the latent representation.
            layer_widths (list): List of output channel widths for each layer. 
                                 Usually the reverse of the Encoder's layer_widths.
            kernel_size (int): Kernel size for deconvolution.
            stride (int): Stride (upscaling factor).
            output_padding (int): Additional size added to one side of the output shape. 
                                  Usually needed to exactly match Encoder input dimensions.
        """
        super().__init__()
        
        layers = []
        current_in = in_channels
        
        # Loop through widths with index to check for the final layer
        for i, width in enumerate(layer_widths):
            is_last_layer = (i == len(layer_widths) - 1)
            
            # For the final layer, we typically disable BatchNorm and Activation 
            # to allow the output to take any value (real numbers).
            block_batchnorm = use_batchnorm and not is_last_layer
            block_activation = not is_last_layer

            layers.append(
                DeconvBlock1D(
                    in_channels=current_in,
                    out_channels=width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=None, 
                    output_padding=output_padding,
                    use_batchnorm=block_batchnorm,
                    apply_activation=block_activation
                )
            )
            current_in = width 

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

