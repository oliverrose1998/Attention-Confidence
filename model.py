""" `model.py` defines:
    * DNN layers including the output layer,
    * LatticeRNN model that connects LSTM layers and DNN layers.
    * If requested, the grapheme encoder is added to the model.
"""

from attention import AddAttention, MultiHeadedAttention
from encoder import Encoder
from grapheme_encoder import LuongAttention, GraphemeEncoder
from lstm import LSTM, LSTMCell

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class DNN_output(nn.Module):
    """A module that defines multi-layer fully connected neural networks."""

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 initialization, use_bias=True, logit=False):
        """Build multi-layer FC."""
        super(DNN_output, self).__init__()
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


class Model(nn.Module):
    """Bidirectional LSTM model on lattices."""

    def __init__(self, opt):
        """Basic model building blocks."""
        nn.Module.__init__(self)
        self.opt = opt

        if self.opt.arc_combine_method == 'attention':
            #self.attention = MultiHeadedAttention(h=self.opt.attn_heads, d_model=hidden)
            #self.attention = MultiHeadedAttention(h=1, d_model=self.opt.inputSize + self.opt.keySize)
            print(self.opt.inputSize)
            print(self.opt.keySize)
            print('================')
            self.attention = AddAttention(self.opt.inputSize + self.opt.keySize,
                                       self.opt.attentionSize,
                                       self.opt.attentionLayers, self.opt.init_word,
                                       use_bias=True)
        else:
            self.attention = None

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

        num_directions = 2 if self.opt.bidirectional else 1

        if self.opt.encoder_type == 'ATTENTION':
            self.model_intermediate = Encoder(self.opt.inputSize, self.opt.hiddenSize, self.opt.hiddenSize,
                                              self.opt.init_word, self.opt.nLSTMLayers, use_bias=True,
                                              birdirectional=self.opt.bidirectional, attention=self.attention, 
                                              attention_order=self.opt.attention_order, attention_key=self.opt.attention_key, 
                                              dropout=self.opt.intermediate_dropout)
        else:
            self.model_intermediate = LSTM(LSTMCell, self.opt.inputSize, self.opt.hiddenSize, 
                                         self.opt.nLSTMLayers, use_bias=True,
                                         bidirectional=self.opt.bidirectional,
                                         attention=self.attention)

        self.model_output = DNN_output(num_directions * self.opt.hiddenSize,
                                      self.opt.linearSize, 1, self.opt.nFCLayers,
                                      self.opt.init_word, use_bias=True, logit=True)

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

        # BiLSTM -> FC(relu) -> LayerOut (sigmoid if not logit)
        output = self.model_intermediate.forward(lattice, self.opt.arc_combine_method)
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
