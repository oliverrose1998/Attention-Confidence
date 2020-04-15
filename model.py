""" `model.py` defines:
    * DNN layers including the output layer,
    * LatticeRNN model that connects LSTM layers and DNN layers.
    * If requested, the grapheme encoder is added to the model.
"""

from outputlayer import OutputLayer
from recurrent_encoder import LSTM
from transformer_encoder import Transformer

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class Model(nn.Module):
    """Bidirectional LSTM model on lattices."""

    def __init__(self, opt):
        """Basic model building blocks."""
        nn.Module.__init__(self)
        self.opt = opt

        if self.opt.grapheme_combination != 'None':
            self.is_graphemic = True

            if self.opt.grapheme_encoding:
                self.grapheme_encoder = GraphemeEncoder(self.opt)
                self.grapheme_attention = LuongAttention(
                    attn_type=self.opt.grapheme_combination,
                    num_features=self.opt.grapheme_hidden_size * 2,
                    initialisation=self.opt.init_grapheme
                )
                self.has_grapheme_encoding = True
            else:
                self.grapheme_attention = LuongAttention(
                    attn_type=self.opt.grapheme_combination,
                    num_features=self.opt.grapheme_features,
                    initialisation=self.opt.init_grapheme
                )
                self.has_grapheme_encoding = False
        else:
            self.is_graphemic = False

        if self.opt.encoder == 'TRANSFORMER':
            self.model_encoder = Transformer(self.opt)

        elif self.opt.encoder == 'RECURRENT':
            self.model_encoder = LSTM(self.opt)

        self.model_output = OutputLayer(self.opt)

    def forward(self, lattice):
        """Forward pass through the model."""
        # Apply attention over the grapheme information
        if self.is_graphemic:

            if self.has_grapheme_encoding:
                grapheme_encoding, _ = self.grapheme_encoder.forward(lattice.grapheme_data)
                reduced_grapheme_info, _ = self.grapheme_attention.forward(
                    key=self.create_key(lattice, grapheme_encoding),
                    query=grapheme_encoding,
                    val=grapheme_encoding
                )
            else:
                reduced_grapheme_info, _ = self.grapheme_attention.forward(
                    key=self.create_key(lattice, None),
                    query=lattice.grapheme_data,
                    val=lattice.grapheme_data
                )
            reduced_grapheme_info = reduced_grapheme_info.squeeze(1)
            lattice.edges = torch.cat((lattice.edges, reduced_grapheme_info), dim=1)

        output = self.model_encoder.forward(lattice, self.opt.arc_combine_method)
        output = self.model_output.forward(output)
        return output

    def create_key(self, lattice, grapheme_encoding):
        """ Concat features to create a key for grapheme attention"""
        if self.grapheme_attention.attn_type == 'concat-enc-key':
            padded_grapheme_dim = lattice.grapheme_data.shape[1]
            word_durations = torch.unsqueeze(torch.unsqueeze(lattice.edges[:, DURATION_IDX], 1).repeat(1, padded_grapheme_dim), 2)
            mask = torch.unsqueeze((torch.sum(lattice.grapheme_data, dim=2) != 0), 2)

            masked_word_durations = word_durations * mask.type(torch.FloatTensor)

            if self.has_grapheme_encoding:
                if grapheme_encoding is None:
                    raise Exception('No grapheme encoding to use for a key')
                key = torch.cat((grapheme_encoding, masked_word_durations), dim=2)
            else:
                key = torch.cat((lattice.grapheme_data, masked_word_durations), dim=2)
        else:
            # For all self-attention schemes
            if self.has_grapheme_encoding:
                if grapheme_encoding is None:
                    raise Exception('No grapheme encoding to use for a key')
                key = grapheme_encoding
            else:
                key = lattice.grapheme_data
        return key

def create_model(opt):
    """New Model object."""
    model = Model(opt)
    model.share_memory()
    return model
