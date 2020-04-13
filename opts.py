"""Argument parser for genral and model / data specific options."""

import os
import argparse
import torch
import utils

class Opts():
    """All configurations and hyperparameters."""

    def __init__(self):
        """Process calling arguments."""
        self.parse()

        self.args.data = os.path.join(self.args.rootDir, self.args.dataset)
        self.args.model = os.path.join(self.args.rootDir, 'exp')

        # Set torch default tensor type and random seed
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.manual_seed(self.args.manualSeed)

        # Indicate the model type. This indicates what is expected from
        # the lattice objects
        if self.args.lattice_type.lower() == 'grapheme':
            lattice_type_tag = 'G'
        elif self.args.lattice_type.lower() == 'word':
            self.args.grapheme_features = 0
            self.args.grapheme_hidden_size = 0
            self.args.grapheme_combination  = 'None'
            lattice_type_tag = 'W'
        else:
            raise Exception('Not a valid lattice type')

        # Customized parameters for the network
        arch = self.args.arch.split('-')
        assert len(arch) == 4, 'bad architecture input argument'
        self.args.nLSTMLayers = int(arch[0])
        self.args.hiddenSize = int(arch[1])
        self.args.nFCLayers = int(arch[2])
        self.args.linearSize = int(arch[3])
        self.args.bidirectional = True

        # Grapheme mergering architecture
        grapheme_arch = self.args.grapheme_arch.split('-')
        assert len(grapheme_arch) == 2, 'bad grapheme model architecture input argument'
        self.args.grapheme_num_layers = int(grapheme_arch[0])
        self.args.grapheme_hidden_size = int(grapheme_arch[1])
        self.args.grapheme_bidirectional = True

        if self.args.grapheme_combination == 'None':
            # Won't read in the grapheme information
            self.args.grapheme_features = 0
            self.args.grapheme_hidden_size = 0
        
        # Attention mechanism architecture
        attention_arch = self.args.grapheme_arch.split('-')
        assert len(grapheme_arch) == 2, 'bad grapheme model architecture input argument'
        self.args.attentionLayers = int(attention_arch[0])
        self.args.attentionSize = int(attention_arch[1])

        # Customized parameters for dataset
        if 'onebest' in self.args.dataset:
            # TODO: Test this input size
            print('Warning: Untested code')
            self.args.inputSize = 52
            self.args.onebest = True
        elif self.args.dataset.startswith('lattice') or self.args.dataset.endswith('-lat'):
            # Is a lattice dataset
            self.args.inputSize = 54
            if self.args.grapheme_encoding:
                self.args.inputSize += self.args.grapheme_hidden_size * 2
            else:
                self.args.inputSize += self.args.grapheme_features
        elif self.args.dataset.startswith('confnet') or self.args.dataset.endswith('-cn'):
            # Is a confusion network dataset
            self.args.inputSize = 52
            if 'lm' in self.args.dataset and 'am' in self.args.dataset:
                # Include LM and AM
                self.args.inputSize += 2
            if self.args.grapheme_encoding:
                self.args.inputSize += self.args.grapheme_hidden_size * 2
            else:
                self.args.inputSize += self.args.grapheme_features
        else:
            # TODO: Make cleaner
            self.args.inputSize = 54 + self.args.grapheme_features
            # raise ValueError('Expecting the dataset name to indicate if 1-best, lattice, or confusion network')

        if self.args.forceInputSize != -1:
            # Override implicit input size code above
            self.args.inputSize = self.args.forceInputSize

        # Attention mechanism keys and input size
        if self.args.attention_key == 'self':
            self.args.keySize = 0
        elif self.args.attention_key == 'dist':
            self.args.keySize = 1
        else:
            self.args.keySize = 4

        # Settings for debug mode
        if self.args.debug:
            self.args.nEpochs = 2
            self.args.nThreads = 1

        # Build a useful hash key and model directory
        if self.args.grapheme_encoding:
            assert self.args.encoding_dropout <= 1 and self.args.encoding_dropout >= 0, \
                'The dropout CLI argument must be a valid ratio'
            grapheme_encoding_tag = 'E=' + str(self.args.grapheme_arch) + '-' + str(self.args.encoding_dropout) + '-' + str(self.args.encoder_type)
        else:
            self.args.encoding_dropout = 0
            grapheme_encoding_tag = 'E=None'

        # Setup model directory
        self.args.hashKey = self.args.dataset \
                            + '_' + self.args.arch \
                            + '_' + self.args.arc_combine_method \
                            + '_' + 'L='+str(self.args.LR) \
                            + '_' + 'M='+str(self.args.momentum) \
                            + '_' + 'S='+str(self.args.batchSize) \
                            + '_' + 'O='+str(self.args.optimizer) \
                            + '_' + 'D='+self.args.LRDecay \
                            + '-' + str(self.args.LRDParam) \
                            + '_' + str(lattice_type_tag) \
                            + '_' + self.args.attention_arch \
                            + '_' + 'o='+self.args.attention_order \
                            + '_' + 'd='+str(self.args.intermediate_dropout) \
                            + '_' + 'h='+str(self.args.attention_heads) \
                            + '_' + self.args.suffix  
  
        if self.args.debug: 
            self.args.hashKey += '_debug'
        self.args.resume = os.path.join(self.args.model, self.args.hashKey)
        utils.mkdir(self.args.resume)

        # Display all options
        utils.print_options(self.args)

    def parse(self):
        """Parsing calling arguments."""
        parser = argparse.ArgumentParser(description='parser for latticeRNN')
        # General options
        parser.add_argument('--debug', default=False, action="store_true",
                            help='Debug mode, only run 2 epochs and 1 thread')
        parser.add_argument('--manualSeed', default=1, type=int,
                            help='Manual seed')
        # Path options
        parser.add_argument('--rootDir', type=str, required=True,
                            help='path to experiment root directory')
        # Data options
        parser.add_argument('--dataset', default='lattice_mapped_0.1_prec', type=str,
                            help='Name of dataset')
        parser.add_argument('--target', default='target', type=str,
                            help='Name of target directory within the data directory')
        parser.add_argument('--nThreads', default=10, type=int,
                            help='Number of data loading threads')
        parser.add_argument('--trainPctg', default=1.0, type=float,
                            help='Percentage of taining data to use')
        parser.add_argument('--shuffle', default=False, action="store_true",
                            help='Flag to shuffle the dataset before training')
        parser.add_argument('--subtrain', default=False, action='store_true',
                            help='Run training on a subset of the dataset, but cross validation and test on the full sets')
        parser.add_argument('--forceInputSize', default=-1, type=int,
                            help='Explicitly dictate the number of features to use')
        # Grapheme data options
        parser.add_argument('--lattice-type', default='word', choices=['grapheme', 'word'],
                            help='Indicate whether the grapheme information should be read from the lattice or not.')
        parser.add_argument('--grapheme-features', default=5, type=int,
                            help='The number of grapheme features to consider, if any exists in the data.')
        # Training/testing options
        parser.add_argument('--nEpochs', default=15, type=int,
                            help='Number of total epochs to run')
        parser.add_argument('--epochNum', default=0, type=int,
                            help='0=retrain|-1=latest|-2=best',
                            choices=[0, -1, -2])
        parser.add_argument('--batchSize', default=1, type=int,
                            help='Mini-batch size')
        parser.add_argument('--saveOne', default=False, action="store_true",
                            help='Only preserve one saved model')
        parser.add_argument('--valOnly', default=False, action="store_true",
                            help='Run on validation set only')
        parser.add_argument('--testOnly', default=False, action="store_true",
                            help='Run the test to see the performance')
        parser.add_argument('--onebest', default=False, action="store_true",
                            help='Train on one-best path only')
        parser.add_argument('--test_epochs', default=True, action="store_true",
                            help='Test model after every training epoch')
        parser.add_argument('--attention_stats', default=False, action="store_true",
                            help='Retrieve attention weight statistics when test')
        parser.add_argument('--seq_length_stats', default=False, action="store_true",
                            help='Display a NCE/AUC values for binned sequence lengths')
        # Optimization options
        parser.add_argument('--LR', default=0.05, type=float,
                            help='Initial learning rate')
        parser.add_argument('--LRDecay', default='none', type=str,
                            help='Learning rate decay method',
                            choices=['anneal', 'stepwise', 'newbob', 'none'])
        parser.add_argument('--LRDParam', default=0.5, type=float,
                            help='Param for learning rate decay')
        parser.add_argument('--momentum', default=0.5, type=float,
                            help='Momentum')
        parser.add_argument('--weightDecay', default=1e-3, type=float,
                            help='Weight decay')
        parser.add_argument('--clip', default=1.0, type=float,
                            help='Gradient clipping')
        parser.add_argument('--optimizer', default='SGD', type=str,
                            help='Optimizer type',
                            choices=['SGD', 'Adam'])
        # Word level model options
        parser.add_argument('--init-word', default='kaiming_normal', type=str,
                            help='Initialisation method for linear layers',
                            choices=['uniform', 'normal',
                                     'xavier_uniform', 'xavier_normal',
                                     'kaiming_uniform', 'kaiming_normal'])
        parser.add_argument('--arch', default='3-64-1-64', type=str,
                            help='Model architecture: '\
                                 'nLSTMLayer-LSTMSize-nFCLayer-nFCSize')
        parser.add_argument('--arc_combine-method', default='mean', type=str,
                            help='method for combining edges',
                            choices=['mean', 'max', 'posterior', 'attention'])
        # Grapheme level model options
        parser.add_argument('--init-grapheme', default='kaiming_normal', type=str,
                            help='Initialisation method for linear layers',
                            choices=['uniform', 'normal',
                                     'xavier_uniform', 'xavier_normal',
                                     'kaiming_uniform', 'kaiming_normal'])
        parser.add_argument('--grapheme-combination', default='None', type=str,
                            help='The method to use for grapheme combination',
                            choices=['None', 'dot', 'mult', 'concat', 'scaled-dot', 'concat-enc-key'])
        parser.add_argument('--grapheme-encoding', default=False, action="store_true",
                            help='Use a bidirectional recurrent structure to encode the grapheme information')
        parser.add_argument('--encoder-type', default='RNN', type=str,
                            help='The type of bidirectional recurrent encoder to use for grapheme combination',
                            choices=['RNN', 'GRU', 'LSTM', 'ATTENTION'])
        parser.add_argument('--encoding-dropout', default=0, type=float,
                            help='The amount of dropout to apply in the bidirectional grapheme encoding')
        parser.add_argument('--grapheme-arch', default='1-10', type=str,
                            help='Grapheme model architecture: num_layers-layer_size')
        # Attention Mechanism options
        parser.add_argument('--attention_order', default='one', type=str,
                            choices=['zero','one','two','inf','all'],
                            help='The order of the neighbours (distance of historical states) taken as input to the attention mechanism')
        parser.add_argument('--attention-arch', default='1-64', type=str,
                            help='Attention model architecture: num_layers-layer_size')
        parser.add_argument('--attention_key', default='dist', type=str,
                            choices=['self','dist','global','local'],
                            help='The key used for the attention mechanism')
        parser.add_argument('--intermediate_dropout', default=0, type=float,
                            help='The amount of dropout to apply in the intermediate DNN stage')
        parser.add_argument('--attention_heads', default=1, type=int,
                            help='The number of attention heads used')
        # Naming options
        parser.add_argument('--suffix', default='LatticeRNN', type=str,
                            help='Suffix for saving the model')
        self.args = parser.parse_args()
