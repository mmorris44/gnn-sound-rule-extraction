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
# TODO: handle LogInfer datasets

parser = argparse.ArgumentParser(description="Main file for running experiments")
# When the below is used, it will overwrite train_graph and train_examples. TODO
parser.add_argument('--train-file-full',
                    nargs='?',
                    default=None,
                    help='Filename of full graph. Input and positive examples need to be sampled from it.')
parser.add_argument('--dataset',
                    choices=link_prediction_datasets,
                    help='Name of the dataset')
parser.add_argument('--layers',
                    default=2,
                    type=int,
                    help='Number of layers in the model')
parser.add_argument('--seed',
                    default=0,
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
parser.add_argument('--checkpoint_interval',
                    default=9999999,
                    type=int,
                    help='How many epochs between model checkpoints')
args = parser.parse_args()

model_name = '{}_layers_{}_lr_{}_seed_{}'.format(args.dataset, args.layers, args.lr, args.seed)
model_folder = '../models'
predicates = '../data/{}/{}/predicates.csv'.format('link_prediction', args.dataset)
if args.dataset in node_classification_datasets:
    encoding_scheme = 'canonical'
    train_graph = '../data/{}/{}/graph.nt'.format('node_classification', args.dataset)
    train_examples = '../data/{}/{}/train.tsv'.format('link_prediction', args.dataset)
else:
    encoding_scheme = 'iclr22'
    train_graph = '../data/{}/{}/train_graph.tsv'.format('link_prediction', args.dataset)
    train_examples = '../data/{}/{}/train_pos.tsv'.format('link_prediction', args.dataset)
encoder_folder = '../encoders'
aggregation = 'sum'
non_negative_weights = 'False'
layers = args.layers

print("Training...")
subprocess.run([
    'python',
    'train.py',
    '--model-name', model_name,
    '--model-folder', model_folder,
    '--predicates', predicates,
    '--train-graph', train_graph,
    '--train-examples', train_examples,
    '--encoding-scheme', encoding_scheme,
    '--encoder-folder', encoder_folder,
    '--aggregation', aggregation,
    '--non-negative-weights', non_negative_weights,
    '--layers', layers
])

final_model_path = '{}/{}.pt'.format(model_folder, model_name)
