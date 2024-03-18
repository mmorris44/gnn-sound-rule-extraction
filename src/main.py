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
    # 'grail',  # exclude for now, since not established benchmark and file structure is different
    # 'kinship',  # exclude for now, different file structure
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
    # 'comp',  # not tree-like, cannot be checked
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
parser.add_argument('--train',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='If 0, the script will not train a new model, but fetch an existing trained model')
parser.add_argument('--test',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='If 0, the script will not test the model, merely train it')

# Testing
parser.add_argument('--evaluation-set',
                    default='valid',
                    choices=['valid', 'test'],
                    help='Whether you should evaluate on the validation or test set')
parser.add_argument('--threshold',
                    type=float,
                    default=0,
                    help='Threshold of the GNN.'
                         'Threshold = 0 means threshold list used. Threshold != 0 only uses given threshold')
parser.add_argument('--negative-sampling-method',
                    default='rb',
                    choices=['rb', 'rc'],
                    help='Negative sampling method for evaluation')

# Logging
parser.add_argument('--use-wandb',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Log to wandb?')
parser.add_argument('--log-interval',
                    default=1,
                    type=int,
                    help='How many epochs between model logs')

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
    assert not (args.dataset == 'LogInfer-WN' and args.log_infer_pattern == 'inter'),\
        'LogInfer pattern "inter" not supported for dataset "LogInfer-WN"'
    encoding_scheme = 'iclr22'
    path_to_dataset = f'../data/LogInfer/LogInfer-benchmark/{args.dataset}-{args.log_infer_pattern}'
    train_file_full = f'{path_to_dataset}/train.txt'
    # 'rb' hard coded into model name for now, to access old trained models. TODO: remove from model name entirely
    model_name = f'{args.dataset}-{args.log_infer_pattern}-rb_layers_{args.layers}_lr_{args.lr}_seed_{args.seed}'
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
    '--layers', str(args.layers),
    '--lr', str(args.lr),
    '--seed', str(args.seed),
    '--epochs', str(args.epochs),
    '--checkpoint-interval', str(args.checkpoint_interval),
    '--log-interval', str(args.log_interval),
    '--use-wandb', str(args.use_wandb),
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

if args.train:
    print("Training...")
    print('Running command:', train_command)
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
    test_negative_examples = f'{path_to_dataset}/{args.evaluation_set}_neg_{args.negative_sampling_method}.txt'
output = f'../metrics/{model_name}_cutoff_{weight_cutoff}.txt'
canonical_encoder_file = f'../encoders/{model_name}_canonical.tsv'
iclr22_encoder_file = f'../encoders/{model_name}_iclr22.tsv'

test_command = [
    'python',
    'test.py',
    '--load-model-name', load_model_name,
    '--threshold', str(threshold),
    '--weight-cutoff', str(weight_cutoff),
    '--test-graph', test_graph,
    '--test-positive-examples', test_positive_examples,
    '--test-negative-examples', test_negative_examples,
    '--output', output,
    '--encoding-scheme', encoding_scheme,
    '--canonical-encoder-file', canonical_encoder_file,
    '--iclr22-encoder-file', iclr22_encoder_file,
    '--use-wandb', str(args.use_wandb),
    '--threshold', str(args.threshold),
]

if args.test:
    print("Testing...")
    print('Running command:', test_command)
    subprocess.run(test_command)
