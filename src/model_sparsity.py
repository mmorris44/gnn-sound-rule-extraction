import torch

import gnn_architectures


# if negative_only==True, will only threshold negative values
def threshold_matrix_values(matrix: torch.tensor, threshold: float, negative_only=True):
    below_threshold_mask = matrix <= -threshold
    above_threshold_mask = matrix >= threshold
    if negative_only:
        outside_threshold_mask = torch.logical_or(below_threshold_mask, matrix >= 0)
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


# not in use for now
def uniform_matrix_rows_by_count(matrix: torch.tensor, max_number_positions_to_trim: int):
    if max_number_positions_to_trim != 0:
        print(max_number_positions_to_trim)
        print(matrix)
    for row_index, row in enumerate(matrix):
        positive_entries = row > 0
        negative_entries = row < 0
        positive_count_tensor = torch.zeros_like(positive_entries)
        negative_count_tensor = torch.zeros_like(negative_entries)
        positive_count_tensor[positive_entries] = 1
        negative_count_tensor[negative_entries] = 1
        positive_row_count = torch.sum(positive_count_tensor).item()
        negative_row_count = torch.sum(negative_count_tensor).item()
        if positive_row_count >= negative_row_count and negative_row_count <= max_number_positions_to_trim:
            matrix[row_index][negative_entries] = 0
        elif positive_row_count <= max_number_positions_to_trim:
            matrix[row_index][positive_entries] = 0
    if max_number_positions_to_trim != 0:
        print(matrix)
        print()


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
