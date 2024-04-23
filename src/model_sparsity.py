import torch

import gnn_architectures


# if negative_only==True, will only threshold negative values
def threshold_matrix_values(matrix: torch.tensor, threshold: float, negative_only=False):
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


# Find which channels are closest to being NABN, as well as which weights need to be zeroed
# TODO: come back to this, computation blows up exponentially
def closest_to_nabn_channels(model: gnn_architectures.GNN):
    s0 = [[{}] for _ in range(model.layer_dimension(0))]

    layer = 1
    s1 = [[] for _ in range(model.layer_dimension(layer))]
    for i in range(model.layer_dimension(layer)):  # row index
        changes = {}

        # All matrices in layer
        matrices = [model.matrix_A(layer)] + [model.matrix_B(layer, colour) for colour in range(model.num_colours)]
        for matrix_id, matrix in enumerate(matrices):
            row_indices_to_trim, col_indices_to_trim = [], []  # Track which indices to zero out
            for j in range(model.layer_dimension(layer - 1)):  # column index
                if matrix[i][j] < 0:
                    row_indices_to_trim.append(i)
                    col_indices_to_trim.append(j)
            changes[(layer, matrix_id)] = (row_indices_to_trim, col_indices_to_trim)
        s1[i].append(changes)

    layer = 2
    s1 = [[] for _ in range(model.layer_dimension(layer))]
    for i in range(model.layer_dimension(layer)):  # row index
        changes = {}

        # All matrices in layer
        matrices = [model.matrix_A(layer)] + [model.matrix_B(layer, colour) for colour in range(model.num_colours)]
        for matrix_id, matrix in enumerate(matrices):
            row_indices_to_trim, col_indices_to_trim = [], []  # Track which indices to zero out
            for j in range(model.layer_dimension(layer - 1)):  # column index
                if matrix[i][j] < 0:
                    row_indices_to_trim.append(i)
                    col_indices_to_trim.append(j)
                elif matrix[i][j] > 0:
                    changes[(layer, matrix_id)] = s1[j]
                    # TODO: need to check all possible ways of zeroing positive values in layer 2 (too large)
            changes[(layer, matrix_id)] = (row_indices_to_trim, col_indices_to_trim)
        s1[i].append(changes)


# UpDown algorithm, but also track which weights would need to be zeroed to make the channel UP, DOWN, or ZERO
# TODO: revisit later, blows up massively computationally
def up_down_with_trim(model):
    # Outer list contains an entry for each channel
    # Inner list contains pairs
    # Pair is (State, Dict[(Int, Int), tensor]), where State is UP, DOWN, ZERO (only UP for now)
    # Each key in dictionary is (layer, matrix_id)
    # Each matrix ID is: A = 0, B_0 = 1, B_1 = 2, ...
    # Each tensor is a mask of the minimal entries that need to become zero to obtain the given state
    #
    # If a key isn't given for a matrix, it is assumed that no changes are needed to that matrix
    # There can be multiple entries in the list for a single state, giving different options for how to obtain it
    s0 = [[(1, {})] for _ in model.layer_dimension(0)]
    states = [s0]

    for layer in range(1, model.num_layers + 1):
        sl = [[] for _ in model.layer_dimension(layer)]

        for i in range(model.layer_dimension(layer)):  # row index
            # Try to make the channel UP
            changes = {}

            # All matrices in layer
            matrices = [model.matrix_A(layer)] + [model.matrix_B(layer, colour) for colour in range(model.num_colours)]
            for matrix_id, matrix in enumerate(matrices):
                row_indices_to_trim, col_indices_to_trim = [], []  # Track which indices to zero out
                for j in range(model.layer_dimension(layer - 1)):  # column index
                    if matrix[i][j] < 0:  # Assuming all previous states were UP
                        row_indices_to_trim.append(i)
                        col_indices_to_trim.append(j)
                changes[(layer, matrix_id)] = (row_indices_to_trim, col_indices_to_trim)

            sl[i].append((1, changes))

        states.append(sl)
    return states[-1]
