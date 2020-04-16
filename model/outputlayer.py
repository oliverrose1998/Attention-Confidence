import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

class OutputLayer(nn.Module):
    """A module that defines multi-layer fully connected neural networks."""

    def __init__(self, opt):
        nn.Module.__init__(self)

        """Build multi-layer FC."""
        #super(OutputLayer, self).__init__()
        num_directions = 2 if opt.bidirectional else 1
        self.input_size = num_directions * opt.hiddenSize
        self.hidden_size = opt.linearSize
        self.output_size = 1
        self.num_layers = opt.nFCLayers
        self.initialization = opt.init_word
        self.use_bias = True
        self.logit = True

        if self.num_layers > 0:
            for layer in range(self.num_layers):
                layer_input_size = self.input_size if layer == 0 else self.hidden_size
                fc = nn.Linear(layer_input_size, self.hidden_size, bias=self.use_bias)
                setattr(self, 'fc_{}'.format(layer), fc)
            self.out = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)
        else:
            self.out = nn.Linear(self.input_size, self.output_size, bias=self.use_bias)
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
