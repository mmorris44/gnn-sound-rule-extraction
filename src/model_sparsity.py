import torch

import gnn_architectures


def threshold_matrix_values(matrix: torch.tensor, threshold: float):
    below_threshold_mask = matrix <= -threshold
    above_threshold_mask = matrix >= threshold
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
