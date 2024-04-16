import argparse
import copy
import math
import random
from enum import Enum
from typing import List

import torch

import gnn_architectures
from model_sparsity import weight_cutoff_model, max_weight_size_in_model


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


def print_model(model):
    print('Model A matrices:')
    for layer in range(1, model.num_layers + 1):
        print(f'Layer {layer}')
        matrix_a = model.matrix_A(layer)
        for row in matrix_a:
            print([f'{el:.3f}' for el in row.tolist()])


def model_stats(model):
    tot_positive, tot_negative, tot_zeroes = 0, 0, 0
    print('Layer || Matrix || Positive || Negative || Zero')
    for layer in range(1, model.num_layers + 1):
        matrix_a = model.matrix_A(layer)
        positive, negative, zeroes = value_breakdown(matrix_a)
        tot_positive += positive
        tot_negative += negative
        tot_zeroes += zeroes
        print(str(layer).ljust(5), 'A'.ljust(6),
              "{:.10f}".format(positive), "{:.10f}".format(negative), "{:.10f}".format(zeroes), sep=' || ')

        for colour in range(model.num_colours):
            matrix_b = model.matrix_B(layer, colour)
            positive, negative, zeroes = value_breakdown(matrix_b)
            tot_positive += positive
            tot_negative += negative
            tot_zeroes += zeroes
            print(str(layer).ljust(5), ('B_' + str(colour)).ljust(6),
                  "{:.10f}".format(positive), "{:.10f}".format(negative), "{:.10f}".format(zeroes), sep=' || ')

    print('\nTotals:')
    print('Positive || Negative || Zero')
    total = tot_positive + tot_negative + tot_zeroes
    tot_positive = tot_positive / total
    tot_negative = tot_negative / total
    tot_zeroes = tot_zeroes / total
    print("{:.10f}".format(tot_positive), "{:.10f}".format(tot_negative), "{:.10f}".format(tot_zeroes), sep=' || ')
    return tot_positive, tot_negative, tot_zeroes


def nabn(model):
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
    return states[-1]


class UpDownStates(Enum):
    UP = 0
    DOWN = 1
    ZERO = 2
    UNKNOWN = 3


def up_down(model):
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
    return states[-1]


# Node has no incoming edges
# Final layer of GNN is not applied
def apply_gnn_to_arbitrary_node(initial_value: torch.tensor, model: gnn_architectures.GNN):
    node_value = initial_value
    for layer in range(1, model.num_layers):
        # \sigma(Av + b)
        node_value = model.activation(layer)(torch.matmul(model.matrix_A(layer), node_value) + model.bias(layer))
    return node_value


def random_binary_tensor(size: int, probability_of_one: float):
    rand_mask = torch.rand(size) > probability_of_one
    rand_tensor = torch.zeros(size)
    rand_tensor[rand_mask] = 1  # some 0s, some 1s
    return rand_tensor


# Algorithm only works for two-layer GNNs
# Since v_0=0 cancels out B_0 matrices, and model is only two layers
def neg_inf_fan(model):
    assert model.num_layers == 2
    initial_value = torch.zeros(model.layer_dimension(0))  # Init to zeroes to pass nothing in first layer

    # Get value for node after L - 1 GNN layers, where node has no incoming edges
    arb_node_value = apply_gnn_to_arbitrary_node(initial_value, model)
    # By default, nothing is negative infinity
    is_neg_inf = torch.zeros(model.layer_dimension(model.num_layers), dtype=torch.bool)
    for colour in range(model.num_colours):
        matrix_b = model.matrix_B(model.num_layers, colour)
        passed_value = torch.matmul(matrix_b, arb_node_value)  # Possible product from a neighbouring node
        # Different channels do not need to be negative in conjunction with each other
        # All that matters is that there is one such graph in which a channel can be made negative infinity
        passed_value_neg_mask = passed_value < 0
        is_neg_inf[passed_value_neg_mask] = 1

    return is_neg_inf


# Only works with ReLU
# Line of nodes to the root node, fan of size d at the end away from the root node
def neg_inf_line(model, algorithm_iterations=1000):
    # Channels by default cannot have negative infinity passed to them
    is_neg_inf = torch.zeros(model.layer_dimension(model.num_layers), dtype=torch.bool)

    for _ in range(algorithm_iterations):
        # All nodes given the same initial value
        initial_value = random_binary_tensor(model.layer_dimension(0), 0.5)
        # Sequence of colours to use for the edges. First colour is for the fan. Last connects to root node
        colour_sequence = random.sample(range(model.num_colours), model.num_layers)

        # track current value of latest reached node
        # x, y in (x + dy)
        neighbour_x_value = torch.zeros(model.layer_dimension(0))  # set initial x-component to 0
        # TODO: do we actually need to track the x_value? Seems like not
        neighbour_y_value = torch.clone(initial_value)  # set y-component to initial value
        current_unreached_value = torch.clone(initial_value)  # tracks current value of an unreached node in the line

        # In first layer:
        # neighbour_x_value, neighbour_y_value refer to the fan nodes
        # d(Bv) = B(dv), so y-component is set to initial_value v
        # This tracks the sum of all the neighbours of the node connected to the fan
        # Equivalent to having one neighbour with feature equal to that sum
        # current_unreached_value refers to the node connected to the fan

        # Stop one short of final layer, since final layer is what will pass the infinities to the root node
        for layer in range(1, model.num_layers):
            # Compute default x and y values (as if y > 0)
            # first part of x_value comes from current unreached value
            x_value = model.bias(layer) + torch.matmul(model.matrix_A(layer), current_unreached_value)
            # add x_value from neighbour as well
            x_value += torch.matmul(model.matrix_B(layer, colour_sequence[layer - 1]), neighbour_x_value)
            # y_value comes from neighbour, which are "current" values
            y_value = torch.matmul(model.matrix_B(layer, colour_sequence[layer - 1]), neighbour_y_value)

            # Set new x and y values
            # Set to non-zero where y > 0
            # y > 0 means that as d tends to infinity, (x + dy) tends to infinity, so the ReLU drops away
            # y < 0 means that as d tends to infinity, (x + dy) tends to negative infinity, so the output of ReLU is 0
            y_mask = y_value > 0
            neighbour_x_value = torch.where(y_mask > 0, x_value, 0)
            neighbour_y_value = torch.where(y_mask > 0, y_value, 0)

            # Update unreached value
            # \sigma(Av + b + Bu), where u, v are both the current unreached value
            current_unreached_value = model.activation(layer)(
                torch.matmul(model.matrix_A(layer), current_unreached_value)
                + model.bias(layer)
                + torch.matmul(model.matrix_B(layer, colour_sequence[layer - 1]), current_unreached_value)
            )  # TODO: using wrong colour values here

        # Compute final values that will be passed
        y_value = torch.matmul(model.matrix_B(model.num_layers, colour_sequence[model.num_layers - 1]), neighbour_y_value)

        # If y value is negative, then d can be made arbitrary large to pass negative infinity
        y_value_neg_mask = y_value < 0
        is_neg_inf[y_value_neg_mask] = 1

        # For logging this iteration's result
        neg_inf_channels_for_iter = torch.zeros_like(is_neg_inf)
        neg_inf_channels_for_iter[y_value_neg_mask] = 1

        # Uncomment the below to see number of found channels for each iteration of the algorithm
        # print(torch.count_nonzero(neg_inf_channels_for_iter).item(), '/', model.layer_dimension(model.num_layers))

    return is_neg_inf


# min_threshold_increment determines how finely the model should adjust the threshold
# min_ratio_learnable is a strict lower bound on the number of channels which are 0 or UP
def find_weight_cutoff_for_ratio_rule_channels(model: gnn_architectures.GNN,
                                               min_ratio_learnable=0.5,
                                               min_cutoff_increment=0.0001):
    min_cutoff = 0
    max_cutoff = max_weight_size_in_model(model)

    final_ratio_up = 0
    final_ratio_zero = 0

    while max_cutoff - min_cutoff > min_cutoff_increment:
        cutoff_model = copy.deepcopy(model)
        middle_cutoff = (max_cutoff + min_cutoff) / 2  # binary search
        weight_cutoff_model(cutoff_model, middle_cutoff)
        alg_final_state = up_down(cutoff_model)

        up_count = alg_final_state.count(UpDownStates.UP)
        zero_count = alg_final_state.count(UpDownStates.ZERO)
        ratio_learnable = (up_count + zero_count) / len(alg_final_state)

        if ratio_learnable > min_ratio_learnable:
            max_cutoff = middle_cutoff
            final_ratio_up = up_count / len(alg_final_state)
            final_ratio_zero = zero_count / len(alg_final_state)
        else:
            min_cutoff = middle_cutoff
    return max_cutoff.item(), final_ratio_up, final_ratio_zero


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract sound rules")
    parser.add_argument('--model-path', help='Path to model file')
    parser.add_argument('--weight-cutoff', help='Threshold size below which weights are clamped to 0', default=0,
                        type=float)
    parser.add_argument('--extraction-algorithm', help='Algorithm to use for extraction',
                        choices=['stats', 'nabn', 'up-down', 'neg-inf-fan', 'neg-inf-line', 'find-cutoff'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model: gnn_architectures.GNN = torch.load(args.model_path).to(device)

    if args.weight_cutoff != 0:
        weight_cutoff_model(loaded_model, args.weight_cutoff)

    if args.extraction_algorithm == 'stats':
        model_stats(loaded_model)

    if args.extraction_algorithm == 'nabn':
        final_state = nabn(loaded_model)
        print(final_state)
        print('Different rule heads that can be checked:', final_state.count(0))

    if args.extraction_algorithm == 'up-down':
        final_state = up_down(loaded_model)
        print(final_state)
        print('Monotonically increasing:', final_state.count(UpDownStates.UP))
        print('Monotonically decreasing:', final_state.count(UpDownStates.DOWN))
        print('Zero:', final_state.count(UpDownStates.ZERO))
        print('Unknown:', final_state.count(UpDownStates.UNKNOWN))

    if args.extraction_algorithm == 'neg-inf-fan':
        neg_inf_channels = neg_inf_fan(loaded_model)
        print(neg_inf_channels)
        print(torch.count_nonzero(neg_inf_channels).item(), '/', loaded_model.layer_dimension(loaded_model.num_layers),
              'channels found which can be negative infinity before the final activation function')

    if args.extraction_algorithm == 'neg-inf-line':
        neg_inf_channels = neg_inf_line(loaded_model)
        print(neg_inf_channels)
        print(torch.count_nonzero(neg_inf_channels).item(), '/', loaded_model.layer_dimension(loaded_model.num_layers),
              'channels found which can be negative infinity before the final activation function')

    if args.extraction_algorithm == 'find-cutoff':
        cutoff, ratio_up, ratio_zero = find_weight_cutoff_for_ratio_rule_channels(loaded_model)
        print('\n\n----\n\n')
        print(f'cutoff: {cutoff}, ratio_up: {ratio_up}, ratio_zero: {ratio_zero}')
