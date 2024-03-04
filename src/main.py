import argparse
import subprocess

link_prediction_datasets = [
    'WN18RRv1',
    'WN18RRv2',
    'WN18RRv3',
    'WN18RRv4',
    'fb237v1',
    'fb237v2',
    'fb237v3',
    'fb237v4',
    'nellv1',
    'nellv2',
    'nellv3',
    'nellv4',
    'grail',
    'kinship',
]

node_classification_datasets = [
    'aifb',
    'mutag',
]

log_infer_datasets = [
    'LogInfer-FB',
    'LogInfer-WN',
]

log_infer_patterns = [
    'comp',
    'hier',
    'inter',
    'inver'
    'sym',
]

parser = argparse.ArgumentParser(description="Main file for running experiments")
# Training
parser.add_argument('--dataset',
                    choices=link_prediction_datasets + node_classification_datasets + log_infer_datasets,
                    help='Name of the dataset')
parser.add_argument('--log-infer-pattern',  # only used for LogInfer datasets
                    default=None,
                    choices=log_infer_patterns,
                    help='Name of the dataset')
parser.add_argument('--layers',
                    default=2,
                    type=int,
                    help='Number of layers in the model')
parser.add_argument('--seed',
                    default=-1,  # -1 seed means seed will be chosen at random
                    type=int,
                    help='Seed used to init RNG')
parser.add_argument('--lr',
                    default=0.01,
                    type=float,
                    help='Learning rate')
parser.add_argument('--epochs',
                    default=10000,
                    type=int,
                    help='Number of epochs to train for')
parser.add_argument('--checkpoint-interval',
                    default=9999999,
                    type=int,
                    help='How many epochs between model checkpoints')

# Testing
parser.add_argument('--evaluation-set',
                    default='valid',
                    choices=['valid', 'test'],
                    help='Whether you should evaluate on the validation or test set')
args = parser.parse_args()

#
# TRAINING
#

model_name = f'{args.dataset}_layers_{args.layers}_lr_{args.lr}_seed_{args.seed}'  # Overwritten for LogInfer
model_folder = '../models'
encoder_folder = '../encoders'
aggregation = 'sum'
non_negative_weights = 'False'

train_graph, train_examples, predicates, train_file_full = None, None, None, None
negative_sampling_method = 'rb'  # Fixed negative sampling for LogInfer datasets

if args.dataset in node_classification_datasets:
    encoding_scheme = 'canonical'
    path_to_dataset = f'../data/node_classification/{args.dataset}'
    train_graph = f'{path_to_dataset}/graph.nt'
    train_examples = f'{path_to_dataset}/train.tsv'
    predicates = f'{path_to_dataset}/predicates.csv'
elif args.dataset in link_prediction_datasets:
    encoding_scheme = 'iclr22'
    path_to_dataset = f'../data/link_prediction/{args.dataset}'
    train_graph = f'{path_to_dataset}/train_graph.tsv'
    train_examples = f'{path_to_dataset}/train_pos.tsv'
    predicates = f'{path_to_dataset}/predicates.csv'
elif args.dataset in log_infer_datasets:
    encoding_scheme = 'iclr22'
    path_to_dataset = f'../data/LogInfer/LogInfer-benchmark/{args.dataset}-{args.log_infer_pattern}-{negative_sampling_method}'
    train_file_full = f'{path_to_dataset}/train.txt'
    model_name = f'{args.dataset}-{args.log_infer_pattern}-{negative_sampling_method}_layers_{args.layers}_lr_{args.lr}_seed_{args.seed}'
else:
    assert False, f'Dataset "{args.dataset}" not recognized'

train_command = [
    'python',
    'train.py',
    '--model-name', model_name,
    '--model-folder', model_folder,
    '--encoding-scheme', encoding_scheme,
    '--encoder-folder', encoder_folder,
    '--aggregation', aggregation,
    '--non-negative-weights', non_negative_weights,
    '--layers', args.layers,
    '--lr', args.lr,
]

if args.dataset in log_infer_datasets:
    train_command = train_command + [
        '--train-file-full', train_file_full
    ]
else:
    train_command = train_command + [
        '--train-graph', train_graph,
        '--train-examples', train_examples,
        '--predicates', predicates,
    ]

print("Training...")
subprocess.run(train_command)

#
# TESTING
#
load_model_name = f'{model_folder}/{model_name}.pt'
threshold = 0  # Fix at zero for now. TODO: change way threshold is used in test file.
weight_cutoff = 0  # TODO: range over various weight cutoffs
if args.dataset in node_classification_datasets:
    assert args.evaluation_set == 'test', f'Only the test set exists for {args.dataset}, not the valid set'
    test_graph = f'{path_to_dataset}/test.tsv'
    test_positive_examples = f'{path_to_dataset}/test_pos.tsv'
    test_negative_examples = f'{path_to_dataset}/test_neg.tsv'
elif args.dataset in link_prediction_datasets:
    test_graph = f'{path_to_dataset}/{args.evaluation_set}_graph.tsv'
    test_positive_examples = f'{path_to_dataset}/{args.evaluation_set}_pos.tsv'
    test_negative_examples = f'{path_to_dataset}/{args.evaluation_set}_neg.tsv'
else:  # log_infer_datasets:
    test_graph = train_file_full
    test_positive_examples = f'{path_to_dataset}/{args.evaluation_set}.txt'
    test_negative_examples = f'{path_to_dataset}/{args.evaluation_set}_neg_{negative_sampling_method}.txt'
output = f'../metrics/{model_name}_cutoff_{weight_cutoff}.txt'
canonical_encoder_file = f'../encoders/{model_name}_canonical.tsv'
iclr22_encoder_file = f'../encoders/{model_name}_iclr22.tsv'

test_command = [
    'python',
    'test.py',
    '--load-model-name', load_model_name,
    '--threshold', threshold,
    '--weight-cutoff', weight_cutoff,
    '--test-graph', test_graph,
    '--test-positive-examples', test_positive_examples,
    '--test-negative-examples', test_negative_examples,
    '--output', output,
    '--encoding-scheme', encoding_scheme,
    '--canonical-encoder-file', canonical_encoder_file,
    '--iclr22-encoder-file', iclr22_encoder_file,
]

print("Testing...")
subprocess.run(test_command)
