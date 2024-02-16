# Data Attribution via Sparsified Unlearning

This README provides instructions for training diffusion models, including from retraining (exact unlearning) and **sparsified unlearning**, as well as steps for converting pre-trained checkpoints checkpoints for use with DDPMpipeline on Huggingface.

## Training Instructions

### Install Required Packages
Set up a virtual environment with Python 3.11.5.
Before starting, ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

### Set up Directory Paths
Create a `constants.py` file with the following content:
```
"""Global constant variables for the project."""

DATASET_DIR = "/gscratch/aims/datasets"
OUTDIR = "/gscratch/aims/diffusion-attr"
MAX_NUM_SAMPLE_IMAGES_TO_SAVE = 64

```

### Training a Diffusion Model from Scratch
To train a diffusion model from scratch, use the following command:
```bash
python main.py --dataset [dataset] --method [unlearning/retrain]
```

### Training with Unlearning with a removal distribution
For training with the unlearning method, use this command:
```bash
python main.py --dataset [dataset] --method [unlearning] --removal_dist [removal_dist]
```
Replace `[dataset]`, `[method]`, and `[removal_dist]` with appropriate values.

## Development Guidlines
1. Git pre-commit hooks (https://pre-commit.com/) are used to automatically
check and fix formatting errors before a Git commit happens. This ensure format
consistency, which reduces the number of lines of code to review. Run
`pre-commit install` to install all the hooks.

## References and Additional Resources

- **Understanding Diffusion Models:** A comprehensive explanation can be found in this blog: [Lilian Weng's Blog on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
- **Original DDPM Paper:** The Denoising Diffusion Probabilistic Models paper is available at [arXiv](https://arxiv.org/pdf/2006.11239.pdf).
- **PyTorch Implementation of DDPM:** A PyTorch version of DDPM can be found on GitHub: [Denoising Diffusion PyTorch Repository](https://github.com/lucidrains/denoising-diffusion-pytorch).
