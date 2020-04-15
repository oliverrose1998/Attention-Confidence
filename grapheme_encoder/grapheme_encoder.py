""" `grapheme_encoder.py` defines:
    * Optional BiDirectional RNN, GRU, or LSTM encoder
"""


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

class GraphemeEncoder(nn.Module):
    """ Bi-directional recurrent neural network designed 
        to encode a grapheme feature sequence.
    """
    def __init__(self, opt):
        nn.Module.__init__(self)

        # Defining some parameters
        self.hidden_size = opt.grapheme_hidden_size
        self.num_layers = opt.grapheme_num_layers
        self.initialisation = opt.init_grapheme
        self.use_bias = True

        if opt.grapheme_encoder == 'RNN':
            self.encoder = nn.RNN(
                input_size=opt.grapheme_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=opt.grapheme_bidirectional,
                batch_first=True,
                dropout=opt.encoding_dropout,
                bias=True
            )
        elif opt.grapheme_encoder == 'LSTM':
            self.encoder = nn.LSTM(
                input_size=opt.grapheme_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=opt.grapheme_bidirectional,
                batch_first=True,
                dropout=opt.encoding_dropout,
                bias=True
            )
        elif opt.grapheme_encoder == 'GRU':
            self.encoder = nn.GRU(
                input_size=opt.grapheme_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=opt.grapheme_bidirectional,
                batch_first=True,
                dropout=opt.encoding_dropout,
                bias=True
            )
        else:
            raise ValueError('Unexpected encoder type: Got {} but expected RNN, GRU, or LSTM'.format(opt.encoder_type))


        self.initialise_parameters()

    def forward(self, x):
        """ Passing in the input into the model and obtaining outputs and the updated hidden state """
        out, hidden_state = self.encoder(x)
        return out, hidden_state

    def init_hidden_state(self, batch_size):
        """ Generate the first hidden state of zeros """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def initialise_parameters(self):
        """ Initialise parameters for all layers. """
        init_method = getattr(init, self.initialisation)
        init_method(self.encoder.weight_ih_l0.data)
        init_method(self.encoder.weight_hh_l0.data)
        if self.use_bias:
            init.constant(self.encoder.bias_ih_l0.data, val=0)
            init.constant(self.encoder.bias_hh_l0.data, val=0)
