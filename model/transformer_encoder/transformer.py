""" `transformer.py` defines:
    * Transformer contains feed forward layers and attention mechanism.
"""

from .attention import Attention

import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F

DURATION_IDX = 50



class Transformer(nn.Module):
    """
    Contains the feed forwards layers and attention mechanism.
  
    Input:   Single lattice
    Output:  Encoded hidden states for each arc
    """

    def __init__(self, opt):
        nn.Module.__init__(self)
        
        # Intermediate architecture parameters
        self.input_size = opt.inputSize
        self.key_size = opt.keySize
        self.hidden_size = opt.hiddenSize
        self.output_size = opt.hiddenSize
        self.num_layers = opt.nEncoderLayers
        self.initialization = opt.init_word
        self.use_bias = True
        self.bidirectional = opt.bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Attention mechanism
        self.transformer_order = opt.transformer_order
        self.attn_key = opt.attn_key
        self.attn_dmetric = opt.attn_dmetric
        self.attn = Attention(attn_type=opt.attn_type, 
                              attn_heads=opt.attn_heads,
                              num_features=opt.inputSize+opt.keySize, 
                              initialisation=opt.init_grapheme)

        # Intermediate architecure
        if self.num_layers > 0:
            for layer in range(self.num_layers):
                layer_input_size = 2*self.input_size + self.key_size if layer == 0 else self.hidden_size
                fc = nn.Linear(layer_input_size, self.hidden_size, bias=self.use_bias)
                setattr(self, 'attention_{}'.format(layer), fc)
            self.out = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)
        else:
            self.out = nn.Linear(2*self.input_size, self.output_size, bias=self.use_bias)
        self.reset_parameters()

        # Attention mechanism input edges
        if self.transformer_order == 'zero':
            self.transformer_limits = None
        elif self.transformer_order == 'one':
            self.transformer_limits = (-2,0)
        elif self.transformer_order == 'two':
            self.transformer_limits = (-3,0) 
        elif self.transformer_order == 'inf':
            self.transformer_limits = (np.NINF,0)
        elif self.transformer_order == 'all':  
            self.transformer_limits = (np.NINF, np.inf)
        else:
            raise ValueError

    def get_fc(self, layer):
        #""Get FC layer by layer number.""
        return getattr(self, 'attention_{}'.format(layer))

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

    #def enrich_edge(self, edges):

    def forward_edge(self, x):
        """Complete multi-layer DNN network."""
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            x = F.relu(fc(x))
        return self.out(x)


    def forward_lattice(self, lattice, input_, attention_method):
        """ Forward through one layer of LSTM. """

        index = -1
        node_outputs = []
      
        # For each node in the lattice
        for node_index, each_node in enumerate(lattice.nodes):
 
            distances_t = []
            distances_n = []
            weighted_in_edges = Variable(torch.zeros([1,self.input_size+self.key_size]), requires_grad=False)

            if self.transformer_limits is not None:
    
                # Define empty list for incoming edges, the distance to the edges
                in_edges = []
                distances = []

                # If the node has parents and therefore incoming edges
                if each_node in lattice.child_dict:

                    # node_distances is the distances between each_node and all other nodes.                 
                    node_distances = lattice.matrix[each_node].tolist()

                    # parent_nodes is a dictionary (key: node, value: distance) for all nodes up to given historical distance
                    parent_nodes = {i:x for i, x in enumerate(node_distances) if x > self.transformer_limits[0] and x < self.transformer_limits[1]}   

                    # Determine the mid time of the outgoing arcs   
                    if lattice.has_times: 
                        if each_node in lattice.parent_dict.keys():
                            for child, edge_id in lattice.parent_dict[each_node].items():
                                if isinstance(edge_id, float):
                                    edge_id = [edge_id]
                                out_mid_t = 0.5*lattice.end_times[edge_id[0]]+0.5*lattice.start_times[edge_id[0]]

                    # for all nodes in parent_nodes, append the edge_id and distance of all incoming arcs
                    for parent, distance in parent_nodes.items():
                        if parent in lattice.parent_dict.keys():
                            for child, edge_id in lattice.parent_dict[parent].items():
                                if child == each_node or child in parent_nodes.keys():
                                    if isinstance(edge_id, float):
                                        edge_id = [edge_id]
                                    for e_id in edge_id:
                                        in_edges.append(e_id)
                                        distances_n.append(distance)

                                        if lattice.has_times:
                                            edge_mid_t = 0.5*lattice.end_times[e_id]+0.5*lattice.start_times[e_id]
                                            distances_t.append(round(edge_mid_t-out_mid_t,2))

                    # Assert in_edges is a list of items
                    if all(isinstance(item, list) for item in in_edges):
                        in_edges = [item for sublist in in_edges
                                    for item in sublist]
                    else:
                        assert all(isinstance(item, int) for item in in_edges)
           
                    # If edges ids in in_edgess, convert distances to type Variable and set in_edges to a matrix with column vectors 
                    # equal to the edge features for each edge_id.        
                    if in_edges:
                        dists_n = Variable(torch.Tensor([[distances_n[ind]] for ind,i in enumerate(in_edges)]), requires_grad=False)
                        if lattice.has_times:
                            dists_t = Variable(torch.Tensor([[distances_t[ind]] for ind,i in enumerate(in_edges)]), requires_grad=False)
                        posterior = torch.cat([lattice.edges[i, index] for i in in_edges]).view(-1, 1)
                        posterior = posterior * lattice.word_std[0, index] + lattice.word_mean[0, index]
                        in_edges = torch.cat([lattice.edges[i].view(1,-1) for i in in_edges], 0)
                         
                        key = torch.cat((posterior, torch.ones_like(posterior) * torch.mean(posterior),
                                             torch.ones_like(posterior)*torch.std(posterior)), dim=1)

                        if self.attn_dmetric == 'nodes':
                            key = torch.cat((key, dists_n), dim=1)
                        elif self.attn_dmetric == 'time':
                            key = torch.cat((key, dists_t), dim=1)
                        elif self.attn_dmetric == 'both':
                            key = torch.cat((key, dists_n, dists_t), dim=1)                                      
                        
                    else:
                        key = Variable(torch.zeros(1, self.opt.keySize))
                        in_edges = Variable(torch.zeros([1, len(lattice.edges[0])]), requires_grad=False)

                    in_edges = torch.cat((in_edges, key), dim=1)
         
                    #print(f'INTO ATTENTION {in_edges.shape}')
                    # Caculate attention weights and multiply with matrix of edge features  
                    weighted_in_edges  = self.attn.forward(query=in_edges, key=in_edges, value=in_edges)           
                    #print(f'OUT ATTENTION {weighted_in_edges.shape}')

            out_edges = []

            # If node has children and therefore outgoing edges
            if each_node in lattice.parent_dict:

                # out_edges is a list of outgoing edges
                edge_id = lattice.parent_dict[each_node].values()
                out_edges.extend(edge_id)

                # Assert out_edges is a list of items
                if all(isinstance(item, list) for item in out_edges):
                    out_edges = [item for sublist in out_edges
                                 for item in sublist]
                else:
                    assert all(isinstance(item, int) for item in out_edges)

                # If edges ids in out_edges, set out_edges to a matrix with column vectors equal to the edge features for each edge_id.
                if out_edges:
                    out_edges = torch.cat([lattice.edges[i].view(1,-1) for i in out_edges], 0) 
                else:
                    out_edges = torch.zeros([1, len(lattice.edges[0])])

                # Ensure out_edges is of type Variable
                if type(out_edges) != torch.autograd.variable.Variable:
                    out_edges = Variable(out_edges, requires_grad=False)

                # For each outgoing edge calculate the hidden state
                for each_edge in out_edges:
                    node_input = torch.cat((each_edge.view(1,-1), weighted_in_edges.view(1,-1)), 1)[0]
                    hidden_state = self.forward_edge(node_input.view(1,-1))
                    node_outputs.append(hidden_state)

        return torch.cat(node_outputs,0) 

    def forward(self, lattice, combine_method):
        #"" Complete multi-layer LSTM network. ""
        # Set initial states to zero
        output = lattice.edges
        cur_output = []

        for direction in range(self.num_directions):
            
            node_output = Transformer.forward_lattice(
                self, lattice=lattice, input_=output,
                attention_method=combine_method)

            if self.bidirectional:
                lattice.reverse()

            cur_output.append(node_output)       
        output = torch.cat(cur_output, 1)
        return output
