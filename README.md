# Evaluating SimCLR and SimSiam for MRI Sequence Classification

In this repository, we provide a comprehensive set of frameworks written in
PyTorch Lightning to perform and evaluate self-supervised contrastive learning
using SimCLR on MRI sequence classification.

## Usage guide

### Installation

1. Clone this repository.
```bash
git clone
```

2. Create virtual environment with Python 3.10.9. Some scripts may fail on
   Python 3.11.
```bash
# Go inside repo
cd simclr-medical imaging
# Create virtual environment
python -m venv venv
# Activate virtual environment
source venv/bin/activate     # For Linux, Mac OS X
source venv/Scripts/activate # For Windows
```

3. Install required packages.
```bash
pip install -r requirements.txt
```

### Usage

The codebase provides in-depth support for SimCLR pretraining, finetuning
(downstream transfer learning), testing, data preview and feature analysis via
PCA and t-SNE.

Navigate to one of the following pages below. Each environment has a
comprehensive documentation with example usage.

Pretrain:
- Go to `pretrain/simclr` directory and see `README.md`.

Finetune with frozen encoder:
- Go to `downstream/logistic_regression` and see `README.md`.

Finetune with unfrozen encoder:
- Go to `downstream/resnet` and see `README.md`.

Baseline:
- Go to `downstream/resnet` and see `README.md`.

Regardless of the experiment, all programs search for models (`.ckpt` files) in
`models/`. For example, when performing downstream learning, the program
searches for the pretrained file in `pretrain/simclr/models/`. If you place the
model in a different folder, you need to update `MODEL_DIR` in `utils.py`.

Note that the saved model is always the latest model after training with the
specified number of epochs. To replace the model with the best-performing
version in terms of validation accuracy, read instructions in
`scripts/replace-with-best-checkpoint.sh`.


## Contribute

### Update requirements

```bash
$ pip freeze > requirements.txt
```
