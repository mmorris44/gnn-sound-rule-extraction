import argparse
import copy
import math
import random
from enum import Enum
from typing import List

import torch
from torch_geometric.data import Data

import gnn_architectures
from model_sparsity import weight_cutoff_model, max_weight_size_in_model
from encoding_schemes import ICLREncoderDecoder, CanonicalEncoderDecoder


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


def old_up_down(model):
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
                if (neg and sj == UpDownStates.UP) or (pos and sj == UpDownStates.DOWN) or (
                        neg and sj == UpDownStates.UNKNOWN) or (pos and sj == UpDownStates.UNKNOWN):
                    up_excluded_pair = True

                # Monotonically decreasing
                if (neg and sj == UpDownStates.UP) or (pos and sj == UpDownStates.DOWN):
                    down_exists_pair = True
                if (pos and sj == UpDownStates.UP) or (neg and sj == UpDownStates.DOWN) or (
                        pos and sj == UpDownStates.UNKNOWN) or (neg and sj == UpDownStates.UNKNOWN):
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


def up_down(model):
    s0 = [UpDownStates.UP] * model.layer_dimension(0)
    states = [s0]
    for layer in range(1, model.num_layers + 1):
        sl = [UpDownStates.UNKNOWN] * model.layer_dimension(layer)

        for i in range(model.layer_dimension(layer)):
            up_excluded_pair, down_excluded_pair, zero_excluded_pair = False, False, False

            for j in range(model.layer_dimension(layer - 1)):
                pos_a, neg_a, pos_b, neg_b = False, False, False, False
                if model.matrix_A(layer)[i][j] > 0:
                    pos_a = True
                for colour in range(model.num_colours):
                    if model.matrix_B(layer, colour)[i][j] > 0:
                        pos_b = True
                        break
                if model.matrix_A(layer)[i][j] < 0:
                    neg_a = True
                for colour in range(model.num_colours):
                    if model.matrix_B(layer, colour)[i][j] < 0:
                        neg_b = True
                        break

                zero_a = not (pos_a or neg_a)
                zero_b = not (pos_b or neg_b)
                sj = states[layer - 1][j]  # state of channel j at layer l-1

                # Stable
                if not ((sj == UpDownStates.ZERO and zero_b) or (zero_a and zero_b)):
                    zero_excluded_pair = True

                # Increasing
                if (neg_a and sj in {UpDownStates.UP, UpDownStates.UNKNOWN}) or \
                        (pos_a and sj in {UpDownStates.DOWN, UpDownStates.UNKNOWN}) or \
                        (not zero_b and sj in {UpDownStates.DOWN, UpDownStates.UNKNOWN}) or \
                        neg_b:
                    up_excluded_pair = True

                # Decreasing
                if (pos_a and sj in {UpDownStates.UP, UpDownStates.UNKNOWN}) or \
                        (neg_a and sj in {UpDownStates.DOWN, UpDownStates.UNKNOWN}) or \
                        (not zero_b and sj in {UpDownStates.DOWN, UpDownStates.UNKNOWN}) or \
                        pos_b:
                    down_excluded_pair = True

            if not zero_excluded_pair:
                sl[i] = UpDownStates.ZERO
            elif not up_excluded_pair:
                sl[i] = UpDownStates.UP
            elif not down_excluded_pair:
                sl[i] = UpDownStates.DOWN

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
        neighbour_y_value = torch.clone(initial_value)  # set y-component to initial value

        # In first layer:
        # d(Bv) = B(dv), so y-component is set to initial_value v
        # This tracks the sum of all the neighbours of the node connected to the fan
        # Equivalent to having one neighbour with feature equal to that sum

        # Stop one short of final layer, since final layer is what will pass the infinities to the root node
        for layer in range(1, model.num_layers):
            # Compute default x and y values (as if y > 0)
            # y_value comes from neighbour, which are "current" values
            y_value = torch.matmul(model.matrix_B(layer, colour_sequence[layer - 1]), neighbour_y_value)

            # Set new x and y values
            # Set to non-zero where y > 0
            # y > 0 means that as d tends to infinity, (x + dy) tends to infinity, so the ReLU drops away
            # y < 0 means that as d tends to infinity, (x + dy) tends to negative infinity, so the output of ReLU is 0
            y_mask = y_value > 0
            neighbour_y_value = torch.where(y_mask > 0, y_value, 0)  # ReLU

        # Compute final values that will be passed
        y_value = torch.matmul(model.matrix_B(model.num_layers, colour_sequence[model.num_layers - 1]),
                               neighbour_y_value)

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


# compute all partitions of a set
def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def all_variable_groundings(var_set: set):
    var_list = list(var_set)
    partitions = list(partition(var_list))
    groundings = []
    for p in partitions:
        grounding = {}
        for i, group in enumerate(p):
            grounding.update({var: var_list[i] for var in group})
        groundings.append(grounding)
    return groundings


def is_monotonic_rule_captured(
        model: gnn_architectures.GNN,
        threshold: float,
        iclr_encoder_decoder: ICLREncoderDecoder,
        can_encoder_decoder: CanonicalEncoderDecoder,
        rule: str,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parts = rule.split(' implies ')
    assert len(parts) == 2, 'Should only be body and head'
    body_str, head_str = parts
    assert ' not ' not in body_str, 'Only monotonic rules can be extracted'
    body_atoms = body_str.split(' and ')
    rule_body = []
    var_set = set()

    # get rule body
    for atom in body_atoms:
        # get variables
        atom_comma_parts = atom.split(',')
        assert len(atom_comma_parts) == 2, 'Only binary predicates supported for now'
        var1 = atom_comma_parts[0][-1]
        var2 = atom_comma_parts[1][0]
        var_set.add(var1)
        var_set.add(var2)
        # get predicate
        atom_bracket_parts = atom.split('(')
        assert len(atom_bracket_parts) == 2, 'Should only be a single opening bracket in rule'
        predicate = atom_bracket_parts[0]
        rule_body.append((var1, predicate, var2))

    # get rule head
    atom = head_str
    # get variables
    atom_comma_parts = atom.split(',')
    assert len(atom_comma_parts) == 2, 'Only binary predicates supported for now'
    var1 = atom_comma_parts[0][-1]
    var2 = atom_comma_parts[1][0]
    # get predicate
    atom_bracket_parts = atom.split('(')
    assert len(atom_bracket_parts) == 2, 'Should only be a single opening bracket in rule'
    predicate = atom_bracket_parts[0]
    rule_head = (var1, predicate, var2)

    # consider all possible ways of grounding the body
    variable_groundings = all_variable_groundings(var_set)
    entailed_in_all_groundings = True

    for grounding in variable_groundings:
        ground_body = [(grounding[var1], predicate, grounding[var2]) for (var1, predicate, var2) in rule_body]
        ground_head = (grounding[rule_head[0]], rule_head[1], grounding[rule_head[2]])
        cd_dataset = iclr_encoder_decoder.encode_dataset(ground_body)
        (gr_features, gr_nodes, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)
        gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
        gnn_output_gr = model(gr_dataset)

        cd_output_dataset_scores_dict = can_encoder_decoder.decode_graph(gr_nodes, gnn_output_gr, threshold)
        facts_scores_dict = {}
        for (s, p, o) in cd_output_dataset_scores_dict:
            ss, pp, oo = iclr_encoder_decoder.decode_fact(s, p, o)
            facts_scores_dict[(ss, pp, oo)] = cd_output_dataset_scores_dict[(s, p, o)]

        if ground_head not in facts_scores_dict:
            entailed_in_all_groundings = False
            break

    return entailed_in_all_groundings


# check if all rules with given number of body atoms are captured
def check_all_rules(
    model: gnn_architectures.GNN,
    canonical_encoder_file: str,
    iclr22_encoder_file: str,
    model_threshold: float,
    num_body_atoms: int,
):
    if num_body_atoms == 1:
        return check_all_rules_one_body_atom(model, canonical_encoder_file, iclr22_encoder_file, model_threshold)
    elif num_body_atoms == 2:
        return check_all_rules_two_body_atoms(model, canonical_encoder_file, iclr22_encoder_file, model_threshold)
    raise Exception('Number of body atoms not supported for checking yet')


# check if all rules with one body atom are captured
def check_all_rules_one_body_atom(
    model: gnn_architectures.GNN,
    canonical_encoder_file: str,
    iclr22_encoder_file: str,
    model_threshold: float,
):
    rules = []
    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=canonical_encoder_file)
    iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=iclr22_encoder_file)

    # get alg outputs
    final_up_down_state = up_down(model)

    # list of predicates
    predicate_list = list(iclr_encoder_decoder.input_predicate_to_unary_canonical_dict.keys())

    body_predicate_options = predicate_list[:]
    body_variable_options = [('x', 'y'), ('x', 'x')]
    bodies = []
    for body_predicate in body_predicate_options:
        for body_variables in body_variable_options:
            bodies.append((body_predicate, body_variables[0], body_variables[1]))

    head_predicate_options = predicate_list[:]
    head_variable_options = [('x', 'y'), ('x', 'x'), ('y', 'x'), ('y', 'y')]
    heads = []
    for index, head_predicate in enumerate(head_predicate_options):
        if final_up_down_state[index] not in {UpDownStates.UP, UpDownStates.ZERO}:
            continue
        for head_variables in head_variable_options:
            heads.append((head_predicate, head_variables[0], head_variables[1]))

    for body in bodies:
        for head in heads:
            b1, b2, b3 = body
            h1, h2, h3 = head

            # check rule safety - only check soundness of safe rules
            rule_safe = True
            for variable in {h2, h3}:
                if variable not in {b2, b3}:
                    rule_safe = False
                    break
            if not rule_safe:
                continue

            rules.append(f'{b1}({b2},{b3}) implies {h1}({h2},{h3})')

    print(f'Checking a total of {len(rules)} rules')
    random.shuffle(rules)
    count = 0
    first_rule_captured = None
    for rule in rules:
        if is_monotonic_rule_captured(
                model,
                model_threshold,
                iclr_encoder_decoder,
                can_encoder_decoder,
                rule,
        ):
            count += 1
            if first_rule_captured is None:
                first_rule_captured = rule
    return count, first_rule_captured


# check if all rules with two body atoms are captured
def check_all_rules_two_body_atoms(
    model: gnn_architectures.GNN,
    canonical_encoder_file: str,
    iclr22_encoder_file: str,
    model_threshold: float,
):
    rules = []
    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=canonical_encoder_file)
    iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=iclr22_encoder_file)

    # get alg outputs
    final_up_down_state = up_down(model)

    # list of predicates
    predicate_list = list(iclr_encoder_decoder.input_predicate_to_unary_canonical_dict.keys())

    body_predicate_options = []
    for p1 in predicate_list:
        for p2 in predicate_list:
            body_predicate_options.append((p1, p2))
    body_variable_options = [  # y is connecting variable in tree-like rule
        ('x', 'y', 'y', 'z'),
        ('y', 'x', 'y', 'z'),
        ('x', 'y', 'z', 'y'),
        ('y', 'x', 'z', 'y'),

        ('x', 'y', 'y', 'y'),
        ('y', 'x', 'y', 'y'),

        ('y', 'y', 'y', 'z'),
        ('y', 'y', 'z', 'y'),

        ('y', 'y', 'y', 'y'),
    ]
    bodies = []
    for (pred1, pred2) in body_predicate_options:
        for body_variables in body_variable_options:
            bodies.append((pred1, body_variables[0], body_variables[1],
                           pred2, body_variables[2], body_variables[3]))

    head_predicate_options = predicate_list[:]
    head_variable_options = [
        ('x', 'y'),
        ('y', 'x'),
        ('x', 'z'),
        ('z', 'x'),
        ('z', 'y'),
        ('y', 'z'),

        ('x', 'x'),
        ('y', 'y'),
        ('z', 'z'),
    ]
    heads = []
    for index, head_predicate in enumerate(head_predicate_options):
        if final_up_down_state[index] not in {UpDownStates.UP, UpDownStates.ZERO}:
            continue
        for head_variables in head_variable_options:
            heads.append((head_predicate, head_variables[0], head_variables[1]))

    for body in bodies:
        for head in heads:
            b1, b2, b3, b4, b5, b6 = body
            h1, h2, h3 = head

            # check rule safety - only check soundness of safe rules
            rule_safe = True
            for variable in {h2, h3}:
                if variable not in {b2, b3, b5, b6}:
                    rule_safe = False
                    break
            if not rule_safe:
                continue

            rules.append(f'{b1}({b2},{b3}) and {b4}({b5},{b6}) implies {h1}({h2},{h3})')

    print(f'Checking a total of {len(rules)} rules')
    random.shuffle(rules)
    count = 0
    first_rule_captured = None
    for rule in rules:
        if is_monotonic_rule_captured(
            model,
            model_threshold,
            iclr_encoder_decoder,
            can_encoder_decoder,
            rule,
        ):
            count += 1
            if first_rule_captured is None:
                first_rule_captured = rule
    return count, first_rule_captured


class RuleCaptureStates(Enum):
    Yes = 0
    NoNegInf = 1
    NoBodyNotEntail = 2
    CannotCheck = 3


# check if rules given in file are captured
def check_given_rules(
        model: gnn_architectures.GNN,
        rules_file: str,
        canonical_encoder_file: str,
        iclr22_encoder_file: str,
        model_threshold: float,
):
    rules = []
    with open(rules_file) as file_in:
        for line in file_in:
            rules.append(line)

    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=canonical_encoder_file)
    iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=iclr22_encoder_file)

    # get alg outputs
    final_up_down_state = up_down(model)
    neg_inf_channels = neg_inf_line(model)

    # values are either 0 = "yes", 1 = "no due to neg-inf-line", 2 = "no due to body not entailing"
    # 3 = "cannot be checked", for each monotonic rule in the rules file
    # entry in dict for each monotonic rule in the file
    captured = {}

    # key is each unique head predicate
    # value is True iff predicate is UP for UpDown algorithm
    rule_head_predicates_checkable = {}

    for rule in rules:
        if ' not ' in rule:
            continue  # cannot check non-monotonic rules

        # get head predicate index
        head_predicate = rule.split(' implies ')[1].split('(')[0]
        cd_dataset = iclr_encoder_decoder.encode_dataset([('x', head_predicate, 'x')])
        (gr_features, gr_nodes, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)
        predicate_index_tensor = torch.nonzero(gr_features[0])
        assert predicate_index_tensor.size() == torch.Size([1, 1]), 'Should only be one predicate in vector'
        predicate_index = predicate_index_tensor[0].item()

        # save whether head predicate is checkable
        rule_head_predicates_checkable[head_predicate] = (final_up_down_state[predicate_index] == UpDownStates.UP)

        # check neg inf
        if neg_inf_channels[predicate_index]:
            captured[rule] = RuleCaptureStates.NoNegInf
            continue

        # check if entailed by all groundings of the body
        rule_true_for_body_groundings = is_monotonic_rule_captured(
            model, model_threshold, iclr_encoder_decoder, can_encoder_decoder, rule)
        if not rule_true_for_body_groundings:
            captured[rule] = RuleCaptureStates.NoBodyNotEntail
            continue

        # check UpDown
        if final_up_down_state[predicate_index] != UpDownStates.UP:
            captured[rule] = RuleCaptureStates.CannotCheck
            continue

        # Otherwise, UpDown is UP and rule is captured, so the rule is captured
        if final_up_down_state[predicate_index] == UpDownStates.UP and rule_true_for_body_groundings:
            captured[rule] = RuleCaptureStates.Yes
            continue
    return captured, rule_head_predicates_checkable


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
