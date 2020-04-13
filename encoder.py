""" `LSTM.py` defines:
    * basic LSTM cell,
    * LSTM layers for lattices building from LSTM cell,
"""


import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F

DURATION_IDX = 50



class Encoder(nn.Module):
    """A module that defines multi-layer fully connected neural networks."""

    def __init__(self, input_size, hidden_size, output_size, initialization,
                num_layers, use_bias=True, bidirectional=True, attention=None, 
                attention_order=None, attention_key=None, dropout=0, **kwargs):

        """Build multi-layer FC."""
        super(Encoder, self).__init__()
        
        # Intermediate architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.initialization = initialization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Attention mechanism
        self.attention = attention
        self.attention_key = attention_key

        # Attention weight statistics
        self.sum_weights = np.zeros(40)
        self.squ_weights = np.zeros(40)
        self.num_weights = 0

        # Intermediate architecure
        if num_layers > 0:
            for layer in range(num_layers):
                layer_input_size = 2*input_size if layer == 0 else hidden_size
                fc = nn.Linear(layer_input_size, hidden_size, bias=use_bias)
                setattr(self, 'attention_{}'.format(layer), fc)
            self.out = nn.Linear(hidden_size, output_size, bias=use_bias)
        else:
            self.out = nn.Linear(2*input_size, output_size, bias=use_bias)
        self.drop_layer = nn.Dropout(p=dropout)
        self.reset_parameters()

        # Attention mechanism input edges
        if attention_order == 'zero':
            self.attention_limits = None
        elif attention_order == 'one':
            self.attention_limits = (-2,0)
        elif attention_order == 'two':
            self.attention_limits = (-3,0) 
        elif attention_order == 'inf':
            self.attention_limits = (np.NINF,0)
        elif attention_order == 'all':  
            self.attention_limits = (np.NINF, np.inf)
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

    def forward_edge(self, x):
        """Complete multi-layer DNN network."""
        for layer in range(self.num_layers):
            fc = self.get_fc(layer)
            x = self.drop_layer(F.relu(fc(x)))
        return self.out(x)


    def forward_lattice(self, lattice, input_, attention_method):
        """ Forward through one layer of LSTM. """

        index = -1
        matrix = lattice.matrix
        node_outputs = []
        sum_weights = np.zeros(40)
        squ_weights = np.zeros(40)
        num_weights = 0

        # For each node in the lattice
        for node_index, each_node in enumerate(lattice.nodes):
 
            weighted_in_edges = Variable(torch.zeros([1,self.input_size]), requires_grad=False)

            if self.attention_limits is not None:
    
                # Define empty list for incoming edges, the distance to the edges
                in_edges = []
                distances = []

                # If the node has parents and therefore incoming edges
                if each_node in lattice.child_dict:

                    # node_distances is the distances between each_node and all other nodes.                 
                    node_distances = matrix[each_node].tolist()

                    # parent_nodes is a dictionary (key: node, value: distance) for all nodes up to given historical distance
                    parent_nodes = {i:x for i, x in enumerate(node_distances) if x > self.attention_limits[0] and x < self.attention_limits[1]}   
    
                    # for all nodes in parent_nodes, append the edge_id and distance of all incoming arcs
                    for parent, distance in parent_nodes.items():
                        if parent in lattice.parent_dict.keys():
                            for child, edge_id in lattice.parent_dict[parent].items():
                                if child == each_node or child in parent_nodes.keys():
                                    if isinstance(edge_id, float):
                                        edge_id = [edge_id]
                                    for e_id in edge_id:
                                        in_edges.append(e_id)
                                        distances.append(distance)

                    # Assert in_edges is a list of items
                    if all(isinstance(item, list) for item in in_edges):
                        in_edges = [item for sublist in in_edges
                                    for item in sublist]
                    else:
                        assert all(isinstance(item, int) for item in in_edges)
           
                    # If edges ids in in_edgess, convert distances to type Variable and set in_edges to a matrix with column vectors 
                    # equal to the edge features for each edge_id.        

                    if in_edges: 
                        distances = Variable(torch.Tensor([[distances[ind]] for ind,i in enumerate(in_edges)]), requires_grad=False)
                        posterior = torch.cat([lattice.edges[i, index] for i in in_edges]).view(-1, 1)
                        posterior = posterior * lattice.word_std[0, index] + lattice.word_mean[0, index]
                        in_edges = torch.cat([lattice.edges[i].view(1,-1) for i in in_edges], 0)
                         
                        if self.attention_key == 'dist':
                            key = distances
                        elif self.attention_key == 'global':
                            key = torch.cat((posterior, torch.ones_like(posterior) * torch.mean(posterior),
                                             torch.ones_like(posterior)*torch.std(posterior),distances), dim=1)                        
                        
                    else:
                        #key = Variable(torch.zeros(1, self.opt.keySize))
                        in_edges = Variable(torch.zeros([1, len(lattice.edges[0])]), requires_grad=False)

                    # Caculate attention weights and multiply with matrix of edge features  
                    weighted_in_edges = self.attention.forward(query=in_edges, key=in_edges, value=in_edges)           
   
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
                    #print(f'1 {node_input}')
                    #node_input = torch.cat((node_input, node_input),0)
                    #print(f'2 {node_input}')
                    hidden_state = self.forward_edge(node_input.view(1,-1))
                    node_outputs.append(hidden_state)

        return torch.cat(node_outputs,0) 

    def forward(self, lattice, combine_method):
        #"" Complete multi-layer LSTM network. ""
        # Set initial states to zero
        output = lattice.edges
        cur_output = []

        for direction in range(self.num_directions):
            
            node_output = Encoder.forward_lattice(
                self, lattice=lattice, input_=output,
                attention_method=combine_method)

            if self.bidirectional:
                lattice.reverse()

            cur_output.append(node_output)       
        output = torch.cat(cur_output, 1)
        return output
