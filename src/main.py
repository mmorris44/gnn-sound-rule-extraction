import argparse
import subprocess

import torch
import wandb

from sound_rule_extraction import find_weight_cutoff_for_ratio_rule_channels, model_stats, nabn, up_down, UpDownStates, \
    neg_inf_line
import gnn_architectures
from model_sparsity import weight_cutoff_model

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
    # 'comp',  # not tree-like, cannot be checked. Triangle and diamond are also not tree-like
    'hier',
    'inter',  # inter not supported for LogInfer-WN
    'inver',
    'sym',
    'fork',
    'cup',
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
parser.add_argument('--non-negative-weights',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Restrict matrix weights during training so that they are all non-negative')

# Testing
parser.add_argument('--test',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='If 0, the script will not test the model, merely train it')
parser.add_argument('--evaluation-set',
                    default='valid',
                    choices=['valid', 'test'],
                    help='Whether you should evaluate on the validation or test set')
parser.add_argument('--negative-sampling-method',
                    default='pc',
                    choices=['rb', 'rc', 'pc'],
                    help='Negative sampling method for evaluation')
parser.add_argument('--rule-channels-min-ratio',
                    type=float,
                    default=-1,
                    help='Weight cutoff will be chosen to give a number of channels corresponding to rules'
                         'strictly greater than the ratio given, which should be in [0, 1).'
                         'Such channels are either UP or 0 (i.e. monotonic increasing or do not depend on input).'
                         'If -1, then no weight cutoff is used.')

# Rule extraction
parser.add_argument('--extract',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Run and log the outputs of the rule extraction algorithms?')

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

# init logging
if args.use_wandb:
    wandb.init(project='sound-rule-extraction')

#
# TRAINING
#

model_name = f'{args.dataset}_layers_{args.layers}_lr_{args.lr}_seed_{args.seed}'  # Overwritten for LogInfer
model_folder = '../models'
encoder_folder = '../encoders'
aggregation = 'sum'

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
    # note: 'rb' must be hard coded into model name if you want to access models trained before 20 March 2024
    # TODO: note that negative dataset not specified in model, since it does not affect training
    # TODO: however, it does affect hyperparameter tuning, so include in name if want to use different negative sets
    model_name = f'{args.dataset}-{args.log_infer_pattern}_layers_{args.layers}_lr_{args.lr}_seed_{args.seed}'
else:
    assert False, f'Dataset "{args.dataset}" not recognized'

if args.non_negative_weights:
    model_name = model_name + '_non_negative_weights'

train_command = [
    'python',
    'train.py',
    '--model-name', model_name,
    '--model-folder', model_folder,
    '--encoding-scheme', encoding_scheme,
    '--encoder-folder', encoder_folder,
    '--aggregation', aggregation,
    '--non-negative-weights', str(args.non_negative_weights),
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
    print('Training...')
    print('Running command:', train_command)
    subprocess.run(train_command)

#
# TESTING
#
load_model_name = f'{model_folder}/{model_name}.pt'

if args.dataset in node_classification_datasets:
    test_graph = f'{path_to_dataset}/graph.nt'
    test_positive_examples = f'{path_to_dataset}/{args.evaluation_set}_pos.tsv'
    test_negative_examples = f'{path_to_dataset}/{args.evaluation_set}_neg.tsv'
elif args.dataset in link_prediction_datasets:
    test_graph = f'{path_to_dataset}/{args.evaluation_set}_graph.tsv'  # Different graph given as input for testing
    test_positive_examples = f'{path_to_dataset}/{args.evaluation_set}_pos.tsv'
    test_negative_examples = f'{path_to_dataset}/{args.evaluation_set}_neg.tsv'
else:  # log_infer_datasets:
    test_graph = train_file_full
    test_positive_examples = f'{path_to_dataset}/{args.evaluation_set}.txt'
    test_negative_examples = f'{path_to_dataset}/{args.evaluation_set}_neg_{args.negative_sampling_method}.txt'
output = f'../metrics/{model_name}_rule_channels_{args.rule_channels_min_ratio}.txt'
canonical_encoder_file = f'../encoders/{model_name}_canonical.tsv'
iclr22_encoder_file = f'../encoders/{model_name}_iclr22.tsv'

test_command = [
    'python',
    'test.py',
    '--load-model-name', load_model_name,
    '--test-graph', test_graph,
    '--test-positive-examples', test_positive_examples,
    '--test-negative-examples', test_negative_examples,
    '--output', output,
    '--encoding-scheme', encoding_scheme,
    '--canonical-encoder-file', canonical_encoder_file,
    '--iclr22-encoder-file', iclr22_encoder_file,
    '--use-wandb', str(args.use_wandb),
    '--eval-threshold-key', str(args.rule_channels_min_ratio),
]

if args.evaluation_set == 'valid':
    test_command = test_command + [
        '--set-optimal-threshold', '1',
    ]
elif args.evaluation_set == 'test':
    test_command = test_command + [
        '--use-optimal-threshold', '1',
    ]

if args.test:
    print('Testing...')

    if args.rule_channels_min_ratio != -1:
        print(f'Searching for a weight cutoff to obtain a ratio of >{args.rule_channels_min_ratio} rule channels')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model: gnn_architectures.GNN = torch.load(load_model_name).to(device)
        weight_cutoff, ratio_up, ratio_zero = find_weight_cutoff_for_ratio_rule_channels(
            model,
            args.rule_channels_min_ratio,
        )
        print(f'Cutoff {weight_cutoff} found')
        test_command = test_command + [
            '--weight-cutoff', str(weight_cutoff),
        ]

    print('Running command:', test_command)
    subprocess.run(test_command)

if args.extract:
    print('Running rule extraction algorithms...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model: gnn_architectures.GNN = torch.load(load_model_name).to(device)

    if args.rule_channels_min_ratio != -1:
        print(f'Searching for a weight cutoff to obtain a ratio of >{args.rule_channels_min_ratio} rule channels')
        weight_cutoff, ratio_up, ratio_zero = find_weight_cutoff_for_ratio_rule_channels(
            model,
            args.rule_channels_min_ratio,
        )
        print(f'Cutoff {weight_cutoff} found')
        if args.use_wandb:
            wandb.log({
                'weight_cutoff': weight_cutoff,
            })
        weight_cutoff_model(model, weight_cutoff)

    print('-----\nModel stats:')
    positive_weights, negative_weights, zero_weights = model_stats(model)

    print('-----\nNABN:')
    final_state = nabn(model)
    print(final_state)
    nabn_ratio = final_state.count(0) / len(final_state)
    print('Ratio of rule heads that can be checked:', nabn_ratio)
    print('-----\nUp-Down:')
    final_state = up_down(model)
    print(final_state)
    up_down_ratio_up = final_state.count(UpDownStates.UP) / len(final_state)
    up_down_ratio_down = final_state.count(UpDownStates.DOWN) / len(final_state)
    up_down_ratio_zero = final_state.count(UpDownStates.ZERO) / len(final_state)
    up_down_ratio_unknown = final_state.count(UpDownStates.UNKNOWN) / len(final_state)
    print('Monotonically increasing:', up_down_ratio_up)
    print('Monotonically decreasing:', up_down_ratio_down)
    print('Zero:', up_down_ratio_zero)
    print('Unknown:', up_down_ratio_unknown)
    print('-----\nNeg-Inf-Line:')
    neg_inf_channels = neg_inf_line(model)
    print(neg_inf_channels)
    neg_inf_ratio = torch.count_nonzero(neg_inf_channels).item() / model.layer_dimension(model.num_layers)
    print(torch.count_nonzero(neg_inf_channels).item(), '/', model.layer_dimension(model.num_layers),
          'channels found which can be negative infinity before the final activation function')

    if args.use_wandb:
        wandb.log({
            'positive_weights': positive_weights,
            'negative_weights': negative_weights,
            'zero_weights': zero_weights,
            'nabn_ratio': nabn_ratio,
            'up_down_ratio_up': up_down_ratio_up,
            'up_down_ratio_down': up_down_ratio_down,
            'up_down_ratio_zero': up_down_ratio_zero,
            'up_down_ratio_unknown': up_down_ratio_unknown,
            'neg_inf_ratio': neg_inf_ratio,
        })

if args.use_wandb:
    wandb.finish()
