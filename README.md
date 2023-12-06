# Data Attribution - MNIST Diffusion

This README provides instructions for training diffusion models, including from scratch and with unlearning, as well as steps for converting pre-trained CIFAR-10 checkpoints for use with DDPMpipeline on Huggingface.

## Training Instructions

### Install Required Packages
Before starting, ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

### Training a Diffusion Model from Scratch
To train a diffusion model from scratch, use the following command:
```bash
python main_new.py --dataset cifar --device cuda:0 --method retrain --excluded_class 0 --num_inference_steps 1000
```

### Training with Unlearning
For training with the unlearning method, use this command:
```bash
python main_new.py --dataset [dataset] --device cuda:1 --excluded_class [excluded_class] --load [path_to_pretrained]/pruned --method esd
```
Replace `[dataset]`, `[excluded_class]`, and `[path_to_pretrained]` with appropriate values.

## Converting Pre-trained CIFAR-10 Checkpoint for DDPMpipeline on Huggingface

To convert a pre-trained CIFAR-10 checkpoint for compatibility with the DDPMpipeline on Huggingface, follow these steps:

### Step 1: Prepare the Pretrained Folder
Create a directory for your pretrained model files:
```bash
mkdir -p [your_pretrained_folder]
```
Replace `[your_pretrained_folder]` with your desired folder name.

### Step 2: Download the Weights
Download the pre-trained weights from the following link:
[Download Pre-trained Weights](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/?p=%2Fdiffusion_models_converted%2Fdiffusion_cifar10_model&mode=list).

### Step 3: Convert the Checkpoint
Convert the original DDPM checkpoint to the Diffusers format for CIFAR-10:
```bash
python tools/convert_ddpm_original_checkpoint_to_diffusers_cifar10.py \
    --checkpoint_path [your_pretrained_folder]/cifar10-ema-model-790000.ckpt \
    --config_file tools/ddpm_cifar10_config.json \
    --dump_path [your_pretrained_folder]/ddpm_ema_cifar10
```
Replace `[your_pretrained_folder]` with your folder name from Step 1.

## References and Additional Resources

- **Understanding Diffusion Models:** A comprehensive explanation can be found in this blog: [Lilian Weng's Blog on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
- **Original DDPM Paper:** The Denoising Diffusion Probabilistic Models paper is available at [arXiv](https://arxiv.org/pdf/2006.11239.pdf).
- **PyTorch Implementation of DDPM:** A PyTorch version of DDPM can be found on GitHub: [Denoising Diffusion PyTorch Repository](https://github.com/lucidrains/denoising-diffusion-pytorch).

