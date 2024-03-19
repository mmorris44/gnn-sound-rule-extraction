#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""

import torch
import wandb
from torch_geometric.data import Data
from numpy import arange
from numpy import trapz
from numpy import nan_to_num
import argparse
import data_parser
import os.path
import rdflib as rdf
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from model_sparsity import weight_cutoff_model
import gnn_architectures
from utils import load_predicates

parser = argparse.ArgumentParser(description="Evaluate a trained GNNs")
parser.add_argument('--load-model-name',
                    help='Filename of trained model to load')
parser.add_argument('--threshold',
                    type=float,
                    default=0,
                    help='Threshold of the GNN. The default value is 0 (all facts with positive scores are derived).'
                         'Overwritten when use-optimal-threshold is 1.')
parser.add_argument('--predicates',
                    help='File with the fixed, ordered list of predicates we consider.')
parser.add_argument('--test-graph',
                    help='Filename of graph test data')
parser.add_argument('--test-positive-examples',
                    help='Filename of positive examples.')
parser.add_argument('--test-negative-examples',
                    help='Filename of negative examples.')
parser.add_argument('--output',
                    default=None,
                    help='Print the classification metrics.')
parser.add_argument('--encoding-scheme',
                    default='canonical',
                    nargs='?',
                    choices=['iclr22', 'canonical'],
                    help='Choose the encoder-decoder that will be applied to the data (canonical by default).')
parser.add_argument('--canonical-encoder-file',
                    help='File with the canonical encoder/decoder used to train the model.')
parser.add_argument('--iclr22-encoder-file',
                    default=None,
                    help='File with the iclr22 encoder/decoder used to train the model, if it was used.')
parser.add_argument('--print-entailed-facts',
                    default=None,
                    help='Print the facts that have been derived in the provided filename.')
parser.add_argument('--weight-cutoff',
                    help='Weight magnitude below which model weights are clamped to 0',
                    default=0,
                    type=float)
parser.add_argument('--use-wandb',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Log to wandb?')

parser.add_argument('--eval-threshold-key',
                    type=float,
                    default=-1,
                    help='Used as key for model.eval_thresholds')
parser.add_argument('--set-optimal-threshold',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Set the optimal threshold in the model? Done by ranging over the options'
                         'and maximising the accuracy')
parser.add_argument('--use-optimal-threshold',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Use the optimal threshold given in the model? Will override threshold param.')
args = parser.parse_args()


def precision(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + fp)
    except:
        value = float("NaN")
        value = 0  # For now, plot NaNs as zeroes
    finally:
        return value


def recall(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + fn)
    except:
        value = float("NaN")
        value = 0  # For now, plot NaNs as zeroes
    finally:
        return value


def accuracy(tp, fp, tn, fn):
    value = 0
    try:
        value = (tn + tp) / (tp + fp + tn + fn)
    except:
        value = float("NaN")
        value = 0  # For now, plot NaNs as zeroes
    finally:
        return value


def f1score(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + 0.5 * (fp + fn))
    except:
        value = float("NaN")
        value = 0  # For now, plot NaNs as zeroes
    finally:
        return value


def auprc(precision_vector, recall_vector):
    return -1 * trapz(precision_vector, recall_vector)


def parse_triple(line):
    temp_string = line[1:]
    bits = temp_string.split('>')
    ent1 = bits[0]
    print(ent1)
    ent2 = bits[1][2:]
    ent3 = bits[2][2:]
    ent4 = bits[3][1:-2]
    return ent1, ent2, ent3, ent4


if __name__ == "__main__":
    # init logging
    if args.use_wandb:
        wandb.init(project='sound-rule-extraction')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_graph_path = args.test_graph
    assert os.path.exists(test_graph_path)
    print("Loading graph data from {}".format(test_graph_path))
    test_graph_dataset = data_parser.parse(test_graph_path)

    if args.encoding_scheme == 'canonical':
        cd_dataset = test_graph_dataset
    else:
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=args.iclr22_encoder_file)
        cd_dataset = iclr_encoder_decoder.encode_dataset(test_graph_dataset)

    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=args.canonical_encoder_file)

    (test_x, test_nodes, test_edge_list, test_edge_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)

    test_data = Data(x=test_x, edge_index=test_edge_list, edge_type=test_edge_colour_list).to(device)

    given_threshold = args.threshold

    print("Evaluating model {} on dataset {} using threshold={}".format(args.load_model_name,
                                                                        args.test_graph, given_threshold))
    model: gnn_architectures.GNN = torch.load(args.load_model_name).to(device)
    model.eval()

    if args.use_optimal_threshold:
        assert args.eval_threshold_key in model.eval_thresholds, 'Optimal threshold must first be set on valid dataset'
        # set the threshold using the one provided in the model, if it is given
        given_threshold = model.eval_thresholds[args.eval_threshold_key]
        print(f'Optimal threshold specified in model: threshold set to {given_threshold} for evaluation')

    # cutoff weights of model if specified
    if args.weight_cutoff != 0:
        weight_cutoff_model(model, args.weight_cutoff)

    # gnn_output : torch.FloatTensor of size i x j, with i = num graph nodes, j = length of feature vectors
    # importantly, the ith row of gnn_output and test_x represent the same node
    gnn_output = model(test_data)

    cd_output_dataset_scores_dict = can_encoder_decoder.decode_graph(test_nodes, gnn_output, given_threshold)
    # facts_scores_dict:  a dictionary mapping triples (s,p,o) to a value (in str) score
    if args.encoding_scheme == 'canonical':
        facts_scores_dict = cd_output_dataset_scores_dict
    elif args.encoding_scheme == 'iclr22':
        facts_scores_dict = {}
        for (s, p, o) in cd_output_dataset_scores_dict:
            ss, pp, oo = iclr_encoder_decoder.decode_fact(s, p, o)
            facts_scores_dict[(ss, pp, oo)] = cd_output_dataset_scores_dict[(s, p, o)]

    # Print from the fact with the highest score to that with the least
    if args.print_entailed_facts is not None:
        to_print = []
        for (s, p, o) in facts_scores_dict:
            to_print.append((facts_scores_dict[s, p, o], (s, p, o)))
        to_print = sorted(to_print, reverse=True)
        with open(args.print_entailed_facts, 'w') as output:
            for (score, (s, p, o)) in to_print:
                output.write("{}\t{}\t{}\n".format(s, p, o))
        with open(args.print_entailed_facts + '_scored', 'w') as output2:
            for (score, (s, p, o)) in to_print:
                output2.write("{}\t{}\t{}\t{}\n".format(s, p, o, score))
        output.close()

    threshold_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] + arange(0.01, 1, 0.01).tolist()
    # If threshold was specified (i.e. not 0), then only use the given threshold
    if given_threshold != 0:
        threshold_list = [given_threshold]
    threshold_list = [round(elem, 10) for elem in threshold_list]

    number_of_positives = 0
    number_of_negatives = 0
    counter_all = 0
    counter_scored = 0
    # Each threshold is mapped to a 4-tuple containing true and false positives and negatives.
    threshold_to_counter = {0: [0, 0, 0, 0]}
    for threshold in threshold_list:
        threshold_to_counter[threshold] = [0, 0, 0, 0]
    entry_for = {"true_positives": 0, "false_positives": 1, "true_negatives": 2, "false_negatives": 3}

    test_positive_examples_path = args.test_positive_examples
    assert os.path.exists(test_positive_examples_path)
    print("Loading examples data from {}".format(test_positive_examples_path))
    test_positive_examples_dataset = data_parser.parse(test_positive_examples_path)

    test_negative_examples_path = args.test_negative_examples
    assert os.path.exists(test_negative_examples_path)
    print("Loading examples data from {}".format(test_negative_examples_path))
    test_negative_examples_dataset = data_parser.parse(test_negative_examples_path)

    test_examples_dataset = [(ex, '1') for ex in test_positive_examples_dataset] + \
                            [(ex, '0') for ex in test_negative_examples_dataset]

    for ((s, p, o), score) in test_examples_dataset:
        counter_all += 1
        # Check that the target fact has a score
        if (s, p, o) in facts_scores_dict:
            counter_scored += 1
        if score == '1':
            # Positive example
            number_of_positives += 1
            # First consider threshold 0
            # True positive
            if facts_scores_dict.get((s, p, o), 0) > 0:
                threshold_to_counter[0][entry_for["true_positives"]] += 1
            # False negative
            else:
                threshold_to_counter[0][entry_for["false_negatives"]] += 1
            # Consider all other thresholds
            for threshold in threshold_list:
                # True positive
                if facts_scores_dict.get((s, p, o), 0) > threshold:
                    threshold_to_counter[threshold][entry_for["true_positives"]] += 1
                # False negative
                else:
                    threshold_to_counter[threshold][entry_for["false_negatives"]] += 1
        # Negative example
        else:
            assert score == '0'
            number_of_negatives += 1
            # First consider threshold 0
            # False positive
            if facts_scores_dict.get((s, p, o), 0) > 0:
                threshold_to_counter[0][entry_for["false_positives"]] += 1
            # True negative
            else:
                threshold_to_counter[0][entry_for["true_negatives"]] += 1
            # Consider all other thresholds
            for threshold in threshold_list:
                # False positive
                if facts_scores_dict.get((s, p, o), 0) > threshold:
                    threshold_to_counter[threshold][entry_for["false_positives"]] += 1
                # True negative
                else:
                    threshold_to_counter[threshold][entry_for["true_negatives"]] += 1

    #  Compute and print result
    recall_vector = []
    precision_vector = []
    print("Total examples: {}".format(counter_all))
    print("Scored examples: {}".format(counter_scored))

    print('Writing test results to:', args.output)
    best_threshold, best_accuracy_v = -1, -1
    with open(args.output, 'w') as f:
        f.write("Threshold" + '\t' + "Precision" + '\t' + "Recall" + '\t' + "Accuracy" + '\t' + "F1 Score" + '\n')
        threshold_iter = threshold_to_counter
        if given_threshold != 0:
            threshold_iter = threshold_list  # Only evaluate on the single threshold if it is given
        for threshold in threshold_iter:
            tp, fp, tn, fn = threshold_to_counter[threshold]
            precision_v = precision(tp, fp, tn, fn)
            recall_v = recall(tp, fp, tn, fn)
            accuracy_v = accuracy(tp, fp, tn, fn)
            f1score_v = f1score(tp, fp, tn, fn)
            f.write("{}\t{}\t{}\t{}\t{}\n".format(threshold, precision_v, recall_v, accuracy_v, f1score_v))

            if args.use_wandb:
                wandb.log({
                    'threshold': threshold,
                    'precision': precision_v,
                    'recall': recall_v,
                    'accuracy': accuracy_v,
                    'f1score': f1score_v,
                })
            recall_vector.append(recall_v)
            precision_vector.append(precision_v)

            if accuracy_v > best_accuracy_v:
                best_accuracy_v = accuracy_v
                best_threshold = threshold
        # Add extremal points for AUC. This ensures a perfect classifier has AUC 1, a random classifier has AUC 0.5,
        # and an `always wrong' classifier has an AUC 0.
        # Without this, a perfect classifier would have a score of 0!!
        precision_vector.insert(0, 0)
        precision_vector.append(1)
        recall_vector.insert(0, 1)
        recall_vector.append(0)
        # Get rid of NaNs
        recall_vector = nan_to_num(recall_vector)
        precision_vector = nan_to_num(precision_vector)
        auprc_v = auprc(precision_vector, recall_vector)
        f.write("Area under precision recall curve: {}\n".format(auprc_v))
        if args.use_wandb:
            wandb.log({'auprc': auprc_v})

    f.close()

    # Set the threshold in the model using the one that yielded the best accuracy
    if args.set_optimal_threshold:
        print(f'Found optimal threshold of {best_threshold}, saving in model')
        model: gnn_architectures.GNN = torch.load(args.load_model_name).to(device)
        model.eval_thresholds[args.eval_threshold_key] = best_threshold
        torch.save(model, args.load_model_name)

    if args.use_wandb:
        wandb.finish()
