import argparse

import torch

import gnn_architectures

parser = argparse.ArgumentParser(description="Extract sound rules")
parser.add_argument('--model-path', help='Path to model file')
parser.add_argument('--extraction-algorithm', help='Algorith to use for extraction', choices=['get-stats', 'NABN'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: gnn_architectures.GNN = torch.load(args.model_path).to(device)


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


if args.extraction_algorithm == 'get-stats':
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
