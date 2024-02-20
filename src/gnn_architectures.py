#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains the GNN architecture, as the GNN class.
One of the fundamental steps in the GNN's update rule is the
use of an appropriate convolution. We define 2 convolutions,
one for coloured edges, and one for colourless edges.

@author: ----
"""
from typing import List

import torch
import torch_geometric.nn.conv

from torch_geometric.nn import MessagePassing

import torch.nn.functional as F
from torch.nn import Parameter


class EC_GCNConv(MessagePassing):
    # in_channels (int) - Size of each input sample
    # out_channels (int) - Size of each output sample
    def __init__(self, in_channels, out_channels, edge_colours, aggregation):
        if aggregation == 'sum':
            super(EC_GCNConv, self).__init__(aggr='add')
        if aggregation == 'max':
            super(EC_GCNConv, self).__init__(aggr='max')
        self.weights = Parameter(torch.Tensor(edge_colours, out_channels, in_channels))
        self.weights.data.normal_(0, 0.001)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_colours = edge_colours
        
    def forward(self, x, edge_index, edge_colour):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.edge_colours):
            edge_mask = edge_colour == i
            temp_edges = edge_index[:, edge_mask]
            out += F.linear(self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0))), self.weights[i], bias=None)
        return out
    
    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, num_layers, feature_dimension, num_edge_colours, aggregation):
        super(GNN, self).__init__()

        self.num_colours = num_edge_colours
        self.num_layers = num_layers
        assert self.num_layers >= 1

        # Dimensions of the layers, 0 to L corresponds to left to right.
        self.dimensions = [feature_dimension] + ([2 * feature_dimension] * (self.num_layers - 1)) + [feature_dimension]

        self.conv_layers = []
        self.linear_layers = []

        for i in range(self.num_layers):
            self.conv_layers.append(EC_GCNConv(self.dimensions[i], self.dimensions[i + 1],
                                               num_edge_colours, aggregation))
            self.linear_layers.append(torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))

        self.conv_layers: List[EC_GCNConv] = torch.nn.ModuleList(self.conv_layers)  # type: ignore
        self.linear_layers: List[EC_GCNConv] = torch.nn.ModuleList(self.linear_layers)  # type: ignore

        self.activation_fn = torch.relu
        self.output_fn = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type

        for i in range(self.num_layers):
            x = self.linear_layers[i](x) + self.conv_layers[i](x, edge_index, edge_colour)
            if i != self.num_layers - 1:  # ReLU on all except last layer
                x = self.activation_fn(x)
        
        # Note: this translation is irrelevant since the bias vectors are not
        # constrained to the positive reals, therefore it isn't mentioned in
        # the report. However, I've left it here for completeness since the
        # models were trained with it.
        return self.output_fn(x - 10)

    def all_labels(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type

        # Layer 0
        return_list = [x]

        for i in range(self.num_layers):
            # Layer i + 1
            x = self.linear_layers[i](x) + self.conv_layers[i](x, edge_index, edge_colour)
            if i != self.num_layers - 1:  # ReLU on all except last layer
                x = self.activation_fn(x)
                return_list.append(x)

        return_list.append(self.output_fn(x - 10))
        return return_list

    def layer_dimension(self, layer):
        return self.dimensions[layer]

    def matrix_A(self, layer):
        return self.linear_layers[layer - 1].weight.detach()

    def matrix_B(self, layer, colour):
        return self.conv_layers[layer - 1].weights[colour].detach()

    def bias(self, layer):
        if layer == self.num_layers:
            return self.linear_layers[layer - 1].bias.detach() - 10
        return self.linear_layers[layer - 1].bias.detach()

    def activation(self, layer):
        if layer == self.num_layers:
            return self.output_fn
        return self.activation_fn


# Do not use the below
# Left in for reference
# TODO: remove later
class RGCN(torch.nn.Module):
    def __init__(self, feature_dimension, hidden_dimension, num_layers, num_edge_colours):
        super(RGCN, self).__init__()

        self.num_edge_colours = num_edge_colours
        self.in_channels = feature_dimension
        self.out_channels = feature_dimension
        self.hidden_channels = hidden_dimension
        self.num_layers = num_layers

        self.activation = torch.relu
        self.final_activation = torch.sigmoid

        self.dimensions = []
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.hidden_channels
            if i == self.num_layers - 1:
                out_channels = self.out_channels
            else:
                out_channels = self.hidden_channels

            self.dimensions.append(in_channels)
            self.layers.append(torch_geometric.nn.conv.RGCNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                num_relations=self.num_edge_colours,
                aggr='add',
                root_weight=True,
                bias=True,
            ))

        self.layers: List[torch_geometric.nn.conv.RGCNConv] = torch.nn.ModuleList(self.layers)  # type: ignore

        self.dimensions.append(self.out_channels)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        for layer in self.layers[:-1]:
            x = self.activation(layer(x, edge_index, edge_type))

        x = self.final_activation(self.layers[-1](x, edge_index, edge_type))
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # Helper methods to be used when extracting sound rules:

    def layer_dimension(self, layer):  # layer 0 is input dimension
        return self.dimensions[layer]

    def matrix_A(self, layer):  # layer 1 is first layer of R-GCN
        return self.layers[layer - 1].root.detach()

    def matrix_B(self, layer, colour):  # colour is from 0 to num_colours (exclusive)
        return self.layers[layer - 1].weight[colour].detach()

    def bias(self, layer):
        return self.layers[layer - 1].bias.detach()

    def activation(self, layer):
        if 0 < layer < self.num_layers:
            return self.activation
        elif layer == self.num_layers:
            return self.final_activation
        else:
            raise Exception('No layer exists with ID', layer)

#
# class EC_GCNConv_FW(MessagePassing):
#     # in_channels (int) - Size of each input sample
#     # out_channels (int) - Size of each output sample
#     def __init__(self, matrix_b1, matrix_b2, matrix_b3, matrix_b4, in_channels, out_channels, num_edge_types=1):
#         super(EC_GCNConv_FW, self).__init__(aggr='max')  # "Max" aggregation
#         self.weights = Parameter(torch.Tensor(num_edge_types, out_channels, in_channels))
#         self.weights[0] = matrix_b1
#         self.weights[1] = matrix_b2
#         self.weights[2] = matrix_b3
#         self.weights[3] = matrix_b4
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_edge_types = num_edge_types
#
#     def forward(self, x, edge_index, edge_type):
#         out = torch.zeros(x.size(0), self.out_channels, device=x.device)
#         for i in range(self.num_edge_types):
#             edge_mask = edge_type == i
#             temp_edges = edge_index[:, edge_mask]
#             out += F.linear(self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0))), self.weights[i], bias=None)
#         return out
#
#     def message(self, x_j):
#         return x_j
#
#     def update(self, aggr_out):
#         return aggr_out
#
#
# class GNN_layer1(torch.nn.Module):
#     def __init__(self, feature_dimension, matrix_a, matrix_b1, matrix_b2, matrix_b3, matrix_b4, bias, num_edge_types=1):
#         super(GNN_layer1, self).__init__()
#         self.conv1 = EC_GCNConv_FW(feature_dimension, 2 * feature_dimension, num_edge_types)
#         self.lin_self_1 = torch.nn.Linear(feature_dimension, 2 * feature_dimension)
#         self.lin_self_1.weight = matrix_a
#         self.lin_self_1.bias = bias
#
#     def forward(self, data):
#         x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
#         x = self.lin_self_1(x) + self.conv1(x, edge_index, edge_type)
#
#         return torch.relu(x)

