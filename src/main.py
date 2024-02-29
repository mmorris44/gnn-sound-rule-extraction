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
args = parser.parse_args()

model_name = '{}_layers_{}_lr_{}_seed_{}'.format(args.dataset, args.layers, args.lr, args.seed)
model_folder = '../models'
predicates = '../data/{}/{}/predicates.csv'.format('link_prediction', args.dataset)
encoder_folder = '../encoders'
aggregation = 'sum'
non_negative_weights = 'False'

train_graph, train_examples, train_file_full = None, None, None

if args.dataset in node_classification_datasets:
    encoding_scheme = 'canonical'
    train_graph = '../data/{}/{}/graph.nt'.format('node_classification', args.dataset)
    train_examples = '../data/{}/{}/train.tsv'.format('link_prediction', args.dataset)
elif args.dataset in link_prediction_datasets:
    encoding_scheme = 'iclr22'
    train_graph = '../data/{}/{}/train_graph.tsv'.format('link_prediction', args.dataset)
    train_examples = '../data/{}/{}/train_pos.tsv'.format('link_prediction', args.dataset)
elif args.dataset in log_infer_datasets:
    encoding_scheme = 'iclr22'
    negative_sampling_method = 'rb'  # Fixed
    train_file_full = '../data/LogInfer/LogInfer-benchmark/{}-{}-{}/train.txt'.format(
        args.dataset, args.log_infer_pattern, negative_sampling_method
    )
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

final_model_path = '{}/{}.pt'.format(model_folder, model_name)
