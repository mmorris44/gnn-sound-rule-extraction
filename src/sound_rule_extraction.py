import argparse
import math
from enum import Enum
from typing import List

import torch

import gnn_architectures

parser = argparse.ArgumentParser(description="Extract sound rules")
parser.add_argument('--model-path', help='Path to model file')
parser.add_argument('--weight-cutoff', help='Threshold size below which weights are clamped to 0', default=0, type=float)
parser.add_argument('--extraction-algorithm', help='Algorithm to use for extraction', choices=['stats', 'nabn', 'up-down', 'neg-inf-search'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: gnn_architectures.GNN = torch.load(args.model_path).to(device)


def threshold_matrix_values(matrix: torch.tensor, threshold: float):
    below_threshold_mask = matrix <= -threshold
    above_threshold_mask = matrix >= threshold
    outside_threshold_mask = torch.logical_or(below_threshold_mask, above_threshold_mask)
    inside_threshold_mask = torch.logical_not(outside_threshold_mask)
    matrix[inside_threshold_mask] = 0


if args.weight_cutoff != 0:
    for layer in range(1, model.num_layers + 1):
        matrix_a = model.matrix_A(layer)
        threshold_matrix_values(matrix_a, args.weight_cutoff)
        for colour in range(model.num_colours):
            matrix_b = model.matrix_B(layer, colour)
            threshold_matrix_values(matrix_b, args.weight_cutoff)


def value_breakdown(matrix: torch.tensor, ratio=True):
    positive_mask = matrix > 0
    n_pos = torch.sum(positive_mask).item()
    negative_mask = matrix < 0
    n_neg = torch.sum(negative_mask).item()
    n_zero = matrix.numel() - matrix.nonzero().size(0)

    if ratio:
        n_pos = n_pos / matrix.numel()
        n_neg = n_neg / matrix.numel()
        n_zero = n_zero / matrix.numel()

    return n_pos, n_neg, n_zero


if args.extraction_algorithm == 'stats':
    tot_positive, tot_negative, tot_zeroes = 0, 0, 0
    print('Layer || Matrix || Positive || Negative || Zero')
    for layer in range(1, model.num_layers + 1):
        matrix_a = model.matrix_A(layer)
        positive, negative, zeroes = value_breakdown(matrix_a)
        tot_positive += positive
        tot_negative += negative
        tot_zeroes += zeroes
        print(str(layer).ljust(5), 'A'.ljust(6),
              "{:.2f}".format(positive), "{:.2f}".format(negative), "{:.2f}".format(zeroes), sep=' || ')

        for colour in range(model.num_colours):
            matrix_b = model.matrix_B(layer, colour)
            positive, negative, zeroes = value_breakdown(matrix_b)
            tot_positive += positive
            tot_negative += negative
            tot_zeroes += zeroes
            print(str(layer).ljust(5), ('B_' + str(colour)).ljust(6),
                  "{:.2f}".format(positive), "{:.2f}".format(negative), "{:.2f}".format(zeroes), sep=' || ')

    print('\nTotals:')
    print('Positive || Negative || Zero')
    total = tot_positive + tot_negative + tot_zeroes
    tot_positive = tot_positive / total
    tot_negative = tot_negative / total
    tot_zeroes = tot_zeroes / total
    print("{:.2f}".format(tot_positive), "{:.2f}".format(tot_negative), "{:.2f}".format(tot_zeroes), sep=' || ')

if args.extraction_algorithm == 'nabn':
    s0 = [0] * model.layer_dimension(0)
    states = [s0]
    for layer in range(1, model.num_layers + 1):
        sl = [0] * model.layer_dimension(layer)

        for i in range(model.layer_dimension(layer)):
            for j in range(model.layer_dimension(layer - 1)):
                # Negative weights create ABN
                if model.matrix_A(layer)[i][j] < 0:
                    sl[i] = 1
                    break
                for colour in range(model.num_colours):
                    if model.matrix_B(layer, colour)[i][j] < 0:
                        sl[i] = 1
                        break
                if sl[i] == 1:
                    break

                # ABN propagation by non-zero weights
                if states[layer - 1][j] == 1:
                    if not math.isclose(model.matrix_A(layer)[i][j].item(), 0):
                        sl[i] = 1
                        break
                    for colour in range(model.num_colours):
                        if not math.isclose(model.matrix_B(layer, colour)[i][j].item(), 0):
                            sl[i] = 1
                            break
                    if sl[i] == 1:
                        break

        states.append(sl)
    print(states[-1])
    print('Different rule heads that can be checked:', states[-1].count(0))


class UpDownStates(Enum):
    UP = 0
    DOWN = 1
    ZERO = 2
    UNKNOWN = 3


if args.extraction_algorithm == 'up-down':
    # 0 = up, 1 = down, 2 = 0, 3 = ?
    s0 = [UpDownStates.UP] * model.layer_dimension(0)
    states = [s0]
    for layer in range(1, model.num_layers + 1):
        sl = [UpDownStates.UNKNOWN] * model.layer_dimension(layer)

        for i in range(model.layer_dimension(layer)):
            up_exists_pair, up_excluded_pair = False, False
            down_exists_pair, down_excluded_pair = False, False
            zero_excluded_pair = False

            for j in range(model.layer_dimension(layer - 1)):
                pos, neg = False, False
                if model.matrix_A(layer)[i][j] > 0:
                    pos = True
                for colour in range(model.num_colours):
                    if model.matrix_B(layer, colour)[i][j] > 0:
                        pos = True
                        break
                if model.matrix_A(layer)[i][j] < 0:
                    neg = True
                for colour in range(model.num_colours):
                    if model.matrix_B(layer, colour)[i][j] < 0:
                        neg = True
                        break

                zero = not (pos or neg)
                sj = states[layer - 1][j]

                # Monotonically increasing
                if (pos and sj == UpDownStates.UP) or (neg and sj == UpDownStates.DOWN):
                    up_exists_pair = True
                if (neg and sj == UpDownStates.UP) or (pos and sj == UpDownStates.DOWN) or (neg and sj == UpDownStates.UNKNOWN) or (pos and sj == UpDownStates.UNKNOWN):
                    up_excluded_pair = True

                # Monotonically decreasing
                if (neg and sj == UpDownStates.UP) or (pos and sj == UpDownStates.DOWN):
                    down_exists_pair = True
                if (pos and sj == UpDownStates.UP) or (neg and sj == UpDownStates.DOWN) or (pos and sj == UpDownStates.UNKNOWN) or (neg and sj == UpDownStates.UNKNOWN):
                    down_excluded_pair = True

                # Zero
                if not ((pos and sj == UpDownStates.ZERO) or (neg and sj == UpDownStates.ZERO) or zero):
                    zero_excluded_pair = True

            if up_exists_pair and not up_excluded_pair:
                sl[i] = UpDownStates.UP
            elif down_exists_pair and not down_excluded_pair:
                sl[i] = UpDownStates.DOWN
            elif not zero_excluded_pair:
                sl[i] = UpDownStates.ZERO

        states.append(sl)
    print(states[-1])
    print('Monotonically increasing:', states[-1].count(UpDownStates.UP))
    print('Monotonically decreasing:', states[-1].count(UpDownStates.DOWN))
    print('Zero:', states[-1].count(UpDownStates.ZERO))
    print('Unknown:', states[-1].count(UpDownStates.UNKNOWN))


class MinMaxStates(Enum):
    VALUE = 0
    INF = 1
    NEG_INF = 2
    UNKNOWN = 3


# This is not working, but there may be something useful here for later
def apply_gnn_layer_to_arbitrary_node(initial_value: torch.tensor, edges_per_colour: List[int], gnn: gnn_architectures.GNN, layer_num: int):
    a_matrix = gnn.matrix_A(layer_num)
    b_matrices = [gnn.matrix_B(layer_num, colour_id) for colour_id in range(gnn.num_colours)]
    base_tensor = torch.matmul(a_matrix, initial_value)
    channel_states = [(MinMaxStates.VALUE, MinMaxStates.VALUE) for _ in range(base_tensor.size()[0])]
    agg_tensor_summed = torch.zeros_like(base_tensor)
    for colour, b_matrix in enumerate(b_matrices):
        agg_tensor = torch.matmul(b_matrix, initial_value)
        agg_tensor_summed += agg_tensor * edges_per_colour[colour]
    for i, entry in enumerate(agg_tensor_summed):
        if entry < 0:
            channel_states[i] = (MinMaxStates.NEG_INF, MinMaxStates.VALUE)
        elif entry > 0:
            channel_states[i] = (MinMaxStates.VALUE, MinMaxStates.INF)
    return model.activation(layer_num)(base_tensor + agg_tensor_summed + gnn.bias(layer_num)), channel_states


# Node has no incoming edges
# Final layer of GNN is not applied
def apply_gnn_to_arbitrary_node(initial_value: torch.tensor, model: gnn_architectures.GNN):
    node_value = initial_value
    for layer in range(1, model.num_layers):
        # \sigma(Av + b)
        node_value = model.activation(layer)(torch.matmul(model.matrix_A(layer), node_value) + model.bias(layer))
    return node_value


if args.extraction_algorithm == 'neg-inf-search':
    rand_mask = torch.rand(model.layer_dimension(0)) > 0.5
    initial_value = torch.zeros(model.layer_dimension(0))
    initial_value[rand_mask] = 1

    arb_node_value = apply_gnn_to_arbitrary_node(initial_value, model)
    is_neg_inf = torch.zeros(model.layer_dimension(model.num_layers), dtype=torch.bool)
    for colour in range(model.num_colours):
        matrix_b = model.matrix_B(model.num_layers, colour)
        passed_value = torch.matmul(matrix_b, arb_node_value)
        passed_value_neg_mask = passed_value < 0
        is_neg_inf[passed_value_neg_mask] = 1

    print(is_neg_inf)
    print(torch.count_nonzero(is_neg_inf).item(), '/', model.layer_dimension(model.num_layers), 'channels found which can be negative infinity before the final activation function')


