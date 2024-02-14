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
    def __init__(self, feature_dimension, num_edge_colours, aggregation):
        super(GNN, self).__init__()

        self.num_colours = num_edge_colours

        self.num_layers = 2
        # Dimensions of the layers, 0 to L corresponds to left to right.
        self.dimensions = [feature_dimension, 2*feature_dimension, feature_dimension]

        agg_1 = 'max'
        agg_2 = 'max'
        if aggregation == 'sum-max':
            agg_1 = 'sum'
        elif aggregation == 'max-sum':
            agg_2 = 'sum'
        if aggregation == 'sum-sum':
            agg_1 = 'sum'
            agg_2 = 'sum'

        self.conv1 = EC_GCNConv(self.dimensions[0], self.dimensions[1], num_edge_colours, agg_1)
        self.conv2 = EC_GCNConv(self.dimensions[1], self.dimensions[2], num_edge_colours, agg_2)

        self.lin_self_1 = torch.nn.Linear(self.dimensions[0], self.dimensions[1])
        self.lin_self_2 = torch.nn.Linear(self.dimensions[1], self.dimensions[2])
        
        self.output = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type
        
        x = self.lin_self_1(x) + self.conv1(x, edge_index, edge_colour)
        x = torch.relu(x)
        x = self.lin_self_2(x) + self.conv2(x, edge_index, edge_colour)
        
        # Note: this translation is irrelevant since the bias vectors are not
        # constrained to the positive reals, therefore it isn't mentioned in
        # the report. However, I've left it here for completeness since the
        # models were trained with it.
        return self.output(x - 10)

    def all_labels(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type

        # Layer 0
        return_list = [x]
        # Layer 1
        x = self.lin_self_1(x) + self.conv1(x, edge_index, edge_colour)
        x = torch.relu(x)
        return_list.append(x)
        # Layer 2
        x = self.lin_self_2(x) + self.conv2(x, edge_index, edge_colour)
        return_list.append(self.output(x - 10))

        return return_list

    def layer_dimension(self, layer):
        return self.dimensions[layer]

    def matrix_A(self, layer):
        if layer == 1:
            return self.lin_self_1.weight.detach()
        elif layer == 2:
            return self.lin_self_2.weight.detach()
        else:
            return None

    def matrix_B(self, layer, colour):
        if layer == 1:
            return self.conv1.weights[colour].detach()
        elif layer == 2:
            return self.conv2.weights[colour].detach()
        else:
            return None

    def bias(self, layer):
        if layer == 1:
            return self.lin_self_1.bias.detach()
        elif layer == 2:
            return self.lin_self_2.bias.detach() - 10
        else:
            return None

    def activation(self, layer):
        if layer == 1:
            return torch.relu
        elif layer == 2:
            m = torch.nn.Sigmoid()
            return m
        else:
            return None


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

