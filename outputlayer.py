import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

class OutputLayer(nn.Module):
    """A module that defines multi-layer fully connected neural networks."""

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 initialization, use_bias=True, logit=False):
        """Build multi-layer FC."""
        super(OutputLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.initialization = initialization
        self.use_bias = use_bias
        self.logit = logit

        if num_layers > 0:
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size
                fc = nn.Linear(layer_input_size, hidden_size, bias=use_bias)
                setattr(self, 'fc_{}'.format(layer), fc)
            self.out = nn.Linear(hidden_size, output_size, bias=use_bias)
        else:
            self.out = nn.Linear(input_size, output_size, bias=use_bias)
        self.reset_parameters()

    def get_fc(self, layer):
        """Get FC layer by layer number."""
        return getattr(self, 'fc_{}'.format(layer))

    def reset_parameters(self):
        """Initialise parameters for all layers."""
        init_method = getattr(init, self.initialization)
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            init_method(fc.weight.data)
            if self.use_bias:
                init.constant(fc.bias.data, val=0)
        init_method(self.out.weight.data)
        init.constant(self.out.bias.data, val=0)

    def forward(self, x):
        """Complete multi-layer DNN network."""
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            x = F.relu(fc(x))
        output = self.out(x)
        if self.logit:
            return output
        else:
            return F.sigmoid(output)
