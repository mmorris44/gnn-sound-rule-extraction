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

model_name = '{}_layers_{}_lr_{}_seed_{}'.format(args.dataset, args.layers, args.lr, args.seed)
model_folder = '../models'
predicates = '../data/{}/{}/predicates.csv'.format('link_prediction', args.dataset)
encoder_folder = '../encoders'
aggregation = 'sum'
non_negative_weights = 'False'

train_graph, train_examples, train_file_full = None, None, None
negative_sampling_method = 'rb'  # Fixed negative sampling for LogInfer datasets

if args.dataset in node_classification_datasets:
    encoding_scheme = 'canonical'
    path_to_dataset = '../data/{}/{}'.format('node_classification', args.dataset)
    train_graph = '{}/graph.nt'.format(path_to_dataset)
    train_examples = '{}/train.tsv'.format(path_to_dataset)
elif args.dataset in link_prediction_datasets:
    encoding_scheme = 'iclr22'
    path_to_dataset = '../data/{}/{}'.format('link_prediction', args.dataset)
    train_graph = '{}/train_graph.tsv'.format(path_to_dataset)
    train_examples = '{}/train_pos.tsv'.format(path_to_dataset)
elif args.dataset in log_infer_datasets:
    encoding_scheme = 'iclr22'
    path_to_dataset = '../data/LogInfer/LogInfer-benchmark/{}-{}-{}'.format(
        args.dataset, args.log_infer_pattern, negative_sampling_method
    )
    train_file_full = '{}/train.txt'.format(path_to_dataset)
    model_name = '{}-{}-{}_layers_{}_lr_{}_seed_{}'.format(
        args.dataset, args.log_infer_pattern, negative_sampling_method, args.layers, args.lr, args.seed
    )
else:
    assert False, 'Dataset "{}" not recognized'.format(args.dataset)

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
load_model_name = '{}/{}.pt'.format(model_folder, model_name)
threshold = 0  # Fix at zero for now. TODO: change way threshold is used in test file.
weight_cutoff = 0  # TODO: range over various weight cutoffs
test_graph = train_file_full  # TODO: handle non-Log-Infer datasets as well
# TODO: convert all in file to format strings, as below: in future commit
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
