import torch

import gnn_architectures


# if negative_only==True, will only threshold negative values
def threshold_matrix_values(matrix: torch.tensor, threshold: float, negative_only=True):
    below_threshold_mask = matrix <= -threshold
    above_threshold_mask = matrix >= threshold
    if negative_only:
        outside_threshold_mask = above_threshold_mask
    else:
        outside_threshold_mask = torch.logical_or(below_threshold_mask, above_threshold_mask)
    inside_threshold_mask = torch.logical_not(outside_threshold_mask)
    matrix[inside_threshold_mask] = 0


def weight_cutoff_model(model: gnn_architectures.GNN, weight_cutoff: float):
    for layer in range(1, model.num_layers + 1):
        matrix_a = model.matrix_A(layer)
        threshold_matrix_values(matrix_a, weight_cutoff)
        for colour in range(model.num_colours):
            matrix_b = model.matrix_B(layer, colour)
            threshold_matrix_values(matrix_b, weight_cutoff)


def max_abs_value_in_matrix(matrix: torch.tensor):
    return torch.max(torch.abs(matrix))


# Get the weight with max absolute value
def max_weight_size_in_model(model: gnn_architectures.GNN):
    max_weight = 0
    for layer in range(1, model.num_layers + 1):
        matrix_a = model.matrix_A(layer)
        matrix_max = max_abs_value_in_matrix(matrix_a)
        if matrix_max > max_weight:
            max_weight = matrix_max
        for colour in range(model.num_colours):
            matrix_b = model.matrix_B(layer, colour)
            matrix_max = max_abs_value_in_matrix(matrix_b)
            if matrix_max > max_weight:
                max_weight = matrix_max
    return max_weight
