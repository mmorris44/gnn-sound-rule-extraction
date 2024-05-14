# Installation
These instructions are written for Linux.

From the root of the code (the folder `code`), create a new `python3` virtual environment with:
```
python3 -m venv ./venv
```
Then activate the environment with:
```
source venv/bin/activate
```
Then install the following required packages using `pip`. If difficulties are encountered, instructions for installing `torch` can be found [here](https://pytorch.org/get-started/locally/) and instructions for `torch_geometric` can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
```
pip3 install torch torchvision torchaudio
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip3 install networkx==3.0
pip3 install wandb
pip3 install matplotlib
```

A different version of CUDA can be used, depending on your system.

# File Structure
- `data/` - datasets
- `src/` - source code. All of our code for computing channel status and whether rules are sound is found in `sound_rule_extraction.py`.
- Other folders are used for saved models, checkpoints, encoders / decoders, etc.

# Running Experiments
All experiments should be run from within the `src` directory. A full description of all hyperparameters available can be found in the argument parser in `main.py`.

## Training
The following code will train R-GCN with MGNN weight clamping (the `non-negative-weights` parameter) for 1500 epochs, stopping training if performance deteriorates for 50 epochs, with a learning rate of 0.001, with a random seed of 1, and on the LogInfer-FB-hier dataset.
```
python main.py --dataset LogInfer-FB --early-stop 50 --epochs 1500 --layers 2 --log-infer-pattern hier --lr 0.001 --non-negative-weights 1 --seed 1 --train 1
```

## Validation
To set the threshold of the model, you must next run it on the validation dataset. For example, with the following command:
```
python main.py --dataset LogInfer-FB --early-stop 50 --epochs 1500 --layers 2 --log-infer-pattern hier --lr 0.001 --non-negative-weights 1 --seed 1 --negative-sampling-method pc --test 1 --evaluation-set valid
```

## Testing and Sound Rule Extraction
Finally, to evaluate model performance on the test set, use `--evaluation-set test`. To compute which channels are safe, increasing, decreasing, unbounded, etc., use `--extract 1`. To check which of the LogInfer rules have been captured,  use `--log-infer-rule-check 1`. Here is an example of such a command:
```
python main.py --dataset LogInfer-FB --early-stop 50 --epochs 1500 --layers 2 --log-infer-pattern hier --lr 0.001 --non-negative-weights 1 --seed 1 --negative-sampling-method pc --test 1 --evaluation-set test --extract 1 --log-infer-rule-check 1
```
