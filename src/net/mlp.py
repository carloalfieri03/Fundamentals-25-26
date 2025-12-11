import torch.nn as nn
class ConfigurableMLP(nn.Module):
    def __init__(self,hidden_size,output_size,dropout,activation_type):
        super().__init__()
        layers=[]
        activation=getattr(nn,activation_type)
        num_layers=len(hidden_size)
        
        if isinstance(dropout, (list, tuple)):
             dropout_rates = dropout
        else:
             dropout_rates = [dropout] * num_layers 
             ## so if you want to have same dropout for alllayers you just write it once
       
        # First Layer
        layers.extend([
            nn.LazyLinear(hidden_size[0]),
            activation(),  # Instantiate the activation
            nn.Dropout(dropout_rates[0])
        ])
        
        # Middle Layers
        for i in range(len(hidden_size) - 1):
            layers.extend([
                nn.Linear(hidden_size[i], hidden_size[i+1]),
                activation(), # Instantiate the activation
                nn.Dropout(dropout_rates[i+1])
            ])
            
        # Output Layer
        layers.append(nn.Linear(hidden_size[-1], output_size))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)