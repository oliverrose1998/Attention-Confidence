""" `lstmcell.py` defines:
    * basic LSTM cell,
"""


import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init


DURATION_IDX = 50


class LSTMCell(nn.LSTMCell):
    """ Overriding initialization and naming methods of LSTMCell. """

    def reset_parameters(self):
        """ Orthogonal Initialization """
        init.orthogonal(self.weight_ih.data)
        self.weight_hh.data.set_(torch.eye(self.hidden_size).repeat(4, 1))
        # The bias is just set to zero vectors.
        if self.bias:
            init.constant(self.bias_ih.data, val=0)
            init.constant(self.bias_hh.data, val=0)

    def __repr__(self):
        """ Rename """
        string = '{name}({input_size}, {hidden_size})'
        if 'bias' in self.__dict__ and self.bias is False:
            string += ', bias={bias}'
        return string.format(name=self.__class__.__name__, **self.__dict__)
