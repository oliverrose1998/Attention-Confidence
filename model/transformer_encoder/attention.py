import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class Attention(torch.nn.Module):
    """ Luong attention layer as defined in: https://arxiv.org/pdf/1508.04025.pdf
        Specifically defined with grapheme combination in mind.
    """
    def __init__(self, attn_type, attn_heads, num_features, initialisation):
        """ Initialise the Attention layer

            Arguments:
                attn_type {string}: The type of attention similarity function to apply
                num_features {int}: The number of input feature dimensions per grapheme
                initialisation {string}: The type of weight initialisation to use
        """
        super(Attention, self).__init__()
        self.num_features = num_features
        self.attn_type = attn_type
        self.initialisation = initialisation
        self.use_bias = True

        self.inlinear = nn.ModuleList([nn.Linear(self.num_features, self.num_features, self.use_bias) for _ in range(3)])

        if self.attn_type not in ['dot', 'mult', 'concat', 'scaled-dot', 'concat-enc-key']:
            raise ValueError(self.attn_type, "is not an appropriate attention type.")

        if self.attn_type == 'mult':
            self.attn = torch.nn.Linear(self.num_features, self.num_features, self.use_bias)
            self.initialise_parameters()
        elif self.attn_type == 'concat':
            self.attn = nn.Linear(self.num_features, 1, self.use_bias)
            self.initialise_parameters()

    def encode_inputs(self, query, key, value):
        """ Feed forward layer to encode inputs  """
        query, key, value = [F.relu(l(x)) for l, x in zip(self.inlinear, (query, key, value))]
        return query, key, value

    def dot_score(self, key, query):
        """ Dot product similarity function """
        return torch.sum(key * query, dim=1).view(1, -1)

    def mult_score(self, key, query):
        """ Multiplicative similarity function (also called general) """
        output = self.attn(query)
        return torch.sum(key * output, dim=1).view(1, -1)

    def concat_score(self, key, query):
        """ Concatinative similarity function (also called additive) """
        # Concat context with hidden representation
        #output = torch.cat((query, key), dim=1)
        output = self.out(query).view(1, -1)
        return F.tanh(output)

    def forward(self, key, query, value):
        """ Compute and return the attention weights and the result of the weighted sum.
            key, query, val are of the tensor form: (Arcs, Graphemes, Features)
        """
       
        # Feed forward layer to encodw inputs
        key, query, value = self.encode_inputs(query, key, value)

        # Calculate the attention weights (alpha) based on the given attention type
        if self.attn_type == 'mult':
            attn_energies = self.mult_score(key, query)
        elif self.attn_type == 'concat':
            attn_energies = self.concat_score(key, query)
        elif self.attn_type == 'dot':
            attn_energies = self.dot_score(key, query)
        elif self.attn_type == 'scaled-dot':
            attn_energies = self.dot_score(key, query) / self.num_features
        elif self.attn_type == 'concat-enc-key':
            attn_energies = self.concat_score(key, query)

        # Alpha is the softmax normalized probability scores
        alpha = F.softmax(attn_energies, dim=1)

        # The context is the result of the weighted summation
        context = torch.mm(alpha, value)

        return context

    def initialise_parameters(self):
        """Initialise parameters for all layers."""
        init_method = getattr(init, self.initialisation)
        for l in self.modules():
            if isinstance(l, nn.Linear):
                init_method(l.weight.data)
                if self.use_bias:
                    init.constant(l.bias.data, val=0)

