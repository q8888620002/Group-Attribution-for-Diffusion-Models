# Explaining Diffusion Models via Sparsified Unlearning

## Overview
With the widespread usage of diffusion models, effective data attribution is needed
to ensure fair acknowledgment for contributors of high-quality training samples,
and to identify potential sources of harmful content. In this early work, we in-
troduce a novel framework tailored to removal-based data attribution for diffusion
models, leveraging sparsified unlearning. This approach significantly improves
the computational scalability and effectiveness of removal-based data attribution.
In our experiments, we attribute diffusion model FID back to CIFAR-10 train-
ing images with datamodel attributions, showing better linear datamodeling score
(LDS) than datamodel attributions based on naive retraining

## Training Instructions

### Install Required Packages
Set up a virtual environment with Python 3.11.5.
Before starting, ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

### Set up Directory Paths
1. Create a `src/constants.py` file with the following content:
```
"""Global constant variables for the project."""

DATASET_DIR = "/gscratch/aims/datasets"
OUTDIR = "/gscratch/aims/diffusion-attr"
LOGDIR = "/gscratch/scrubbed/logdir/diffusion-attr"
MAX_NUM_SAMPLE_IMAGES_TO_SAVE = 64

```

2. Add the repo directory to PYTHONPATH:
```
cd data_attribution
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Directory Structure

```plaintext
.
├── src/  # Files for common Python functions and classes.
│   ├── ddpm_config.py
│   ├── utils.py
│   └── # some other module files
│
├── unconditional_generation/  # Files for unconditional diffusion models.
│   ├── README.md
│   ├── main.py
│   ├── prune.py
│   ├── generate_samples.py
│   ├── calculate_local_scores.py
│   ├── cifar/
│   │   ├── calculate_global_scores.py  # Global scores can differ between different datasets.
│   │   └── results.ipynb  # Jupyter notebook(s) for plotting results.
│   ├── celeba/
│   │   ├── calculate_global_scores.py
│   │   └── results.ipynb
│   └── experiments/  # Files for managing SLURM experiments.
│
├── text_to_image/  # Files for text-to-image diffusion models.
│   ├── README.md
│   ├── train_text_to_image_lora.py
│   ├── prune.py
│   ├── generate_samples.py
│   ├── calculate_local_scores.py
│   ├── artbench/
│   │   ├── calculate_global_scores.py
│   │   └── results.ipynb
│   └── experiments/  # Files for managing SLURM experiments.
│
├── some_common_script_0.py  # Script(s) that are useful for all diffusion models.
├── some_common_script_1.py

```

## Development Guidlines
1. Git pre-commit hooks (https://pre-commit.com/) are used to automatically
check and fix formatting errors before a Git commit happens. This ensure format
consistency, which reduces the number of lines of code to review. Run
`pre-commit install` to install all the hooks.

Note: If a piece of code is adpated from some open source software, the original
formatting should be kept so it's clear what the modifications are. To commit a change
without pre-commit, run
```
commit -m "YOUR COMMIT MESSAGE" --no-verify
```
